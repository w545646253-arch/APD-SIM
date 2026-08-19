"""Shared production engine for formal APD-DMD R2 two-dimensional training.

The module is deliberately strict at its boundaries:

* input data comes only from frozen train/validation manifests;
* the sealed-test receipt is inspected as identity metadata but no test TIFF is
  ever resolved or opened;
* all learned/physical tensors are four-dimensional ``(B,C,H,W)``;
* protocol geometry and slot mappings come from the immutable registry;
* K3 waits fail-closed for a verified, completed K9 R2 checkpoint.

``run_formal_training`` is a real fixed-budget training loop.  Tests and audit
entrypoints can request preflight-only or one-batch smoke without weakening the
default no-argument behavior, which proceeds to training after preflight.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from .checkpoint_contract import architecture_hash, capture_rng_states, restore_rng_states
from .datasets import BIOSR_REQUIRED_SHAPE, BioSRGT2DDataset, BioSRManifestError
from .formal_training_gate import TrainingLock
from .model2d import APDConditionedUNet2D, assert_strictly_2d_model
from .protocols import KMAX, protocol_registry
from .protocol_runtime import checkpoint_protocol_metadata, initialization_compatibility, require_protocol
from .sim_forward_2d import (
    SIM2DConfig,
    embed_raw_to_slots_2d,
    forward_protocol_clean_2d,
    forward_protocol_sim_2d,
    masked_poisson_gaussian_likelihood_2d,
    nominal_theta_2d,
    sample_theta_2d,
)


FORMAL_CONFIG_TYPE = "APD_DMD_R2_FORMAL_2D"
EXPECTED_REGISTRY_HASH = "5186ebd2a17c5e39ccf486f3e7b61fb3cf7f86c907c9460740fbc23385fa2968"
NO_ACCESS_MARKER = "NO_ACCESS_DURING_TRAINING"
K3_WAITING_STATUS = "WAITING_FOR_VERIFIED_DMD9_CHECKPOINT"
K3_LOADED_STATUS = "LOADED_VERIFIED_DMD9_EMA_FULL_MODEL"
STANDARD_AMP_POLICY = "STANDARD_PYTORCH_DYNAMIC_LOSS_SCALING"
SCHEDULED_COUNTER_SEMANTICS = "SCHEDULED_ITERATION_SEPARATE_FROM_COMMITTED_OPTIMIZER_UPDATES"
BEST_RULE_ID = "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"
INPUT_TENSOR_DIMENSIONALITY = "4D_BCHW"
NORMALIZATION_CONTRACT: Dict[str, Any] = {
    "method": "per_image_percentile_clip",
    "lower_percentile": 0.5,
    "upper_percentile": 99.5,
    "clip": [0.0, 1.0],
    "output_dtype": "float32",
}


class Formal2DTrainingError(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = str(status)
        self.detail = str(detail)


@dataclass(frozen=True)
class OptimizerIterationResult:
    committed_optimizer_updates: int
    gradient_norm: Optional[float]
    amp_overflow_skipped: bool
    previous_scale: Optional[float]
    new_scale: Optional[float]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


# Defined after the canonical serializer so it is an invariant, not a hash of
# implementation-specific string formatting.
NORMALIZATION_CONTRACT_HASH = hashlib.sha256(
    _canonical_json_bytes(NORMALIZATION_CONTRACT)
).hexdigest()


def _file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any], excluded: Sequence[str]) -> str:
    stripped = dict(payload)
    for key in excluded:
        stripped.pop(key, None)
    return hashlib.sha256(_canonical_json_bytes(stripped)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", f"Invalid JSON {path}: {exc}") from exc


def _accepted_json_hashes(path: Path, payload: Any) -> set[str]:
    accepted = {_file_sha256(path)}
    if isinstance(payload, Mapping):
        canonical = _payload_sha256(
            payload, ("manifest_hash", "manifest_sha256", "payload_hash")
        )
        accepted.add(canonical)
        for key in ("manifest_hash", "manifest_sha256", "payload_hash"):
            embedded = payload.get(key)
            if embedded is not None and str(embedded).lower() != canonical:
                raise Formal2DTrainingError(
                    "FORMAL_TRAINING_CONFIG_BLOCKED",
                    f"Embedded {key} does not match the computed canonical JSON payload",
                )
    return accepted


def _validate_json_identity(path: Path, expected: str, label: str) -> Tuple[Any, str]:
    if not path.is_file():
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", f"Missing {label}: {path}")
    payload = _load_json(path)
    expected_l = str(expected).lower()
    accepted = _accepted_json_hashes(path, payload)
    if expected_l not in accepted:
        raise Formal2DTrainingError(
            "FORMAL_TRAINING_CONFIG_BLOCKED",
            f"{label} SHA-256 mismatch: expected {expected_l}, accepted {sorted(accepted)}",
        )
    return payload, _file_sha256(path)


def _records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, Mapping):
        raw = payload.get("samples", payload.get("records", payload.get("items")))
    else:
        raw = None
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Manifest has no records list")
    return [dict(item) for item in raw]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", f"Config key {key} must be an object")
    return value


def _resolve(path_value: Any, project_root: Path) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _sealed_receipt_is_safe(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Sealed-test receipt must be an object")
    if payload.get("runtime_policy") != NO_ACCESS_MARKER:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Sealed-test no-access marker is absent")
    if payload.get("runtime_test_payload_paths_included") is not False:
        raise Formal2DTrainingError(
            "FORMAL_TRAINING_CONFIG_BLOCKED", "Sealed-test receipt declares runtime paths"
        )
    forbidden_keys = {"absolute_path", "path", "file_path", "tiff_path", "source_path"}
    stack: List[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in forbidden_keys:
                    raise Formal2DTrainingError(
                        "FORMAL_TRAINING_CONFIG_BLOCKED",
                        f"Sealed-test runtime receipt exposes forbidden key {key!r}",
                    )
                stack.append(nested)
        elif isinstance(value, list):
            stack.extend(value)
    # Redacted path_summary strings are permitted identity metadata.  They are
    # never passed to Path, tifffile, or the datasets.


def _manifest_disjoint(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    keys = ("sample_id", "parent_id", "file_sha256", "pixel_sha256", "normalized_pixel_sha256")
    for key in keys:
        train_values = {str(record.get(key)) for record in train if record.get(key) is not None}
        val_values = {str(record.get(key)) for record in validation if record.get(key) is not None}
        overlap = train_values.intersection(val_values)
        if overlap:
            raise Formal2DTrainingError(
                "BIOSR_SPLIT_BLOCKED", f"Train/validation {key} overlap: {sorted(overlap)[:3]}"
            )


def _sealed_disjoint(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    sealed_payload: Mapping[str, Any],
) -> None:
    identities = sealed_payload.get("identities")
    if not isinstance(identities, list) or len(identities) != 30:
        raise Formal2DTrainingError("BIOSR_TEST_OVERLAP_BLOCKED", "Sealed identity list is invalid")
    for key in ("sample_id", "parent_id"):
        sealed_values = {
            str(record.get(key)) for record in identities if isinstance(record, Mapping) and record.get(key)
        }
        for split_name, records in (("train", train), ("validation", validation)):
            split_values = {str(record.get(key)) for record in records if record.get(key)}
            overlap = sealed_values.intersection(split_values)
            if overlap:
                raise Formal2DTrainingError(
                    "BIOSR_TEST_OVERLAP_BLOCKED",
                    f"{split_name}/sealed-test {key} overlap: {sorted(overlap)[:3]}",
                )


def _validate_validation_bundle(
    payload: Any,
    validation_records: Sequence[Mapping[str, Any]],
    validation_manifest_hash: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation bundle must be an object")
    expected_crops = {
        "top-left": [0, 0],
        "top-right": [0, 684],
        "center": [342, 342],
        "bottom-left": [684, 0],
        "bottom-right": [684, 684],
    }
    expected_ranges = {
        "k_ratio_xy": [0.75, 0.92],
        "mod_depth": [0.55, 1.0],
        "phase_offset_rad": [-0.25, 0.25],
        "angle_offset_deg": [-0.04, 0.04],
        "psf_sigma_scale": [0.85, 1.35],
        "background": [0.0, 0.06],
        "photon_scale_log_uniform": [1500.0, 20000.0],
        "read_noise_e": [1.2, 2.4],
    }
    if (
        payload.get("manifest_type") != "deterministic_validation_bundle_plan"
        or payload.get("validation_manifest_sha256") != validation_manifest_hash
        or payload.get("image_shape") != [1004, 1004]
        or payload.get("crop_shape") != [320, 320]
        or payload.get("crop_order") != list(expected_crops)
        or payload.get("crop_coordinates_yx") != expected_crops
        or int(payload.get("realizations_per_crop", -1)) != 2
        or payload.get("randomization_ranges") != expected_ranges
    ):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation bundle header mismatch")
    validation_ids = {str(record["sample_id"]) for record in validation_records}
    protocols = payload.get("protocols")
    if not isinstance(protocols, Mapping) or set(protocols) != set(protocol_registry.protocol_ids):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation bundle protocol set mismatch")
    shared_contract: Dict[Tuple[str, str, int], Tuple[Any, ...]] = {}
    for protocol_id in protocol_registry.protocol_ids:
        section = protocols[protocol_id]
        if not isinstance(section, Mapping):
            raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation protocol section invalid")
        entries = section.get("entries")
        if (
            section.get("protocol_hash") != require_protocol(protocol_id).protocol_hash
            or not isinstance(entries, list)
            or len(entries) != len(validation_ids) * 10
            or int(section.get("entry_count", -1)) != len(entries)
            or section.get("entries_sha256") != hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()
        ):
            raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", f"{protocol_id} bundle identity mismatch")
        observed = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation entry is not an object")
            clean = dict(entry)
            entry_hash = str(clean.pop("entry_sha256", ""))
            theta = entry.get("nongeometry_parameters")
            if (
                entry_hash != hashlib.sha256(_canonical_json_bytes(clean)).hexdigest()
                or not isinstance(theta, Mapping)
                or entry.get("nongeometry_parameters_sha256")
                != hashlib.sha256(_canonical_json_bytes(theta)).hexdigest()
            ):
                raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation entry hash mismatch")
            sample_id = str(entry.get("sample_id"))
            crop_id = str(entry.get("crop_id"))
            realization = int(entry.get("realization_id", -1))
            geometry = entry.get("geometry")
            if (
                sample_id not in validation_ids
                or crop_id not in expected_crops
                or entry.get("crop_yx") != expected_crops[crop_id]
                or entry.get("crop_shape") != [320, 320]
                or realization not in (0, 1)
                or not isinstance(geometry, Mapping)
                or geometry.get("protocol_id") != protocol_id
                or geometry.get("protocol_hash") != require_protocol(protocol_id).protocol_hash
                or not 0 <= int(entry.get("diffusion_timestep", -1)) < 1000
                or entry.get("gaussian_noise_shape") != [1, 320, 320]
                or entry.get("realization_policy")
                != ("nominal" if realization == 0 else "fixed_sha256_domain_randomization")
            ):
                raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation entry contract mismatch")
            key = (sample_id, crop_id, realization)
            if key in observed:
                raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Duplicate validation entry")
            observed.add(key)
            shared_value = (
                entry.get("object_identity_sha256"),
                entry.get("nongeometry_parameters_sha256"),
                _canonical_json_bytes(theta),
                int(entry.get("acquisition_noise_seed", -1)),
            )
            previous = shared_contract.setdefault(key, shared_value)
            if previous != shared_value:
                raise Formal2DTrainingError(
                    "FORMAL_TRAINING_CONFIG_BLOCKED",
                    "Validation object/nongeometry/acquisition seed is not shared across protocols",
                )
        expected = {
            (sample_id, crop_id, realization)
            for sample_id in validation_ids
            for crop_id in expected_crops
            for realization in (0, 1)
        }
        if observed != expected:
            raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation bundle coverage mismatch")


@dataclass
class DiffusionScheduler2D:
    total_timesteps: int
    device: torch.device
    beta_schedule: str = "cosine"

    def __post_init__(self) -> None:
        if self.beta_schedule == "cosine":
            steps = self.total_timesteps + 1
            axis = torch.linspace(0, self.total_timesteps, steps, device=self.device)
            cumulative = torch.cos(
                ((axis / self.total_timesteps) + 0.008) / 1.008 * math.pi * 0.5
            ).square()
            cumulative = cumulative / cumulative[0]
            betas = 1.0 - cumulative[1:] / cumulative[:-1]
            betas = betas.clamp(1e-6, 0.999)
        elif self.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.total_timesteps, device=self.device)
        else:
            raise ValueError(f"Unknown beta schedule {self.beta_schedule!r}")
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0)

    def q_sample(self, x0: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if x0.ndim != 4 or noise.shape != x0.shape:
            raise ValueError("DiffusionScheduler2D accepts only matching 4-D image tensors")
        alpha = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        return alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise

    def predict_x0(self, xt: torch.Tensor, timestep: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_bar[timestep].view(-1, 1, 1, 1)
        return (xt - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-8)


class EMA2D:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)


@dataclass
class FormalComponents:
    config: Dict[str, Any]
    preflight: Dict[str, Any]
    device: torch.device
    model: APDConditionedUNet2D
    scheduler: DiffusionScheduler2D
    sim_config: SIM2DConfig
    optimizer: torch.optim.Optimizer
    ema: EMA2D
    train_dataset: BioSRGT2DDataset
    train_loader: DataLoader
    loader_generator: torch.Generator


def _validate_config(config_path: Path, protocol_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config_obj = _load_json(config_path)
    if not isinstance(config_obj, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config root must be an object")
    config = dict(config_obj)
    required = (
        "schema_version",
        "config_type",
        "config_payload_hash",
        "protocol_id",
        "protocol_hash",
        "protocol_registry_hash",
        "project_root",
        "dataset_root",
        "dataset_class",
        "input_shape",
        "patch_size",
        "train_manifest_path",
        "train_manifest_hash",
        "validation_manifest_path",
        "validation_manifest_hash",
        "sealed_test_manifest_path",
        "sealed_test_manifest_hash",
        "sealed_test_identity_hash",
        "sealed_test_no_access_marker",
        "validation_bundle_manifest_path",
        "validation_bundle_hash",
        "source_snapshot_id",
        "initialization",
        "model",
        "training",
        "forward",
        "validation",
        "outputs",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config missing: " + ", ".join(missing))
    if config["config_type"] != FORMAL_CONFIG_TYPE or int(config["schema_version"]) < 1:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Unsupported formal config schema")
    actual_config_hash = _payload_sha256(config, ("config_payload_hash",))
    if str(config["config_payload_hash"]).lower() != actual_config_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config canonical payload hash mismatch")

    spec = require_protocol(protocol_id)
    if protocol_registry.registry_hash != EXPECTED_REGISTRY_HASH:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Live protocol registry hash changed")
    if config["protocol_id"] != spec.protocol_id:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config protocol ID mismatch")
    if config["protocol_hash"] != spec.protocol_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config protocol hash mismatch")
    if config["protocol_registry_hash"] != protocol_registry.registry_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config registry hash mismatch")
    if config["dataset_class"] != "BioSRGT2DDataset":
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Formal dataset class is not BioSRGT2DDataset")
    if tuple(config["input_shape"]) != BIOSR_REQUIRED_SHAPE or int(config["patch_size"]) != 320:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Input/patch geometry mismatch")
    if str(config["sealed_test_no_access_marker"]) != NO_ACCESS_MARKER:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config no-access marker mismatch")

    configured_project = Path(str(config["project_root"]))
    project_root = (config_path.parents[2] / configured_project).resolve() if not configured_project.is_absolute() else configured_project.resolve()
    dataset_root = _resolve(config["dataset_root"], project_root)
    if project_root != config_path.parents[2].resolve():
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Config project_root mismatch")
    if not dataset_root.is_dir():
        raise Formal2DTrainingError("BIOSR_GT_DATA_INVALID", f"BioSR root missing: {dataset_root}")

    train_path = _resolve(config["train_manifest_path"], project_root)
    val_path = _resolve(config["validation_manifest_path"], project_root)
    sealed_path = _resolve(config["sealed_test_manifest_path"], project_root)
    bundle_path = _resolve(config["validation_bundle_manifest_path"], project_root)
    train_payload, train_file_hash = _validate_json_identity(
        train_path, str(config["train_manifest_hash"]), "train manifest"
    )
    val_payload, val_file_hash = _validate_json_identity(
        val_path, str(config["validation_manifest_hash"]), "validation manifest"
    )
    sealed_payload, sealed_file_hash = _validate_json_identity(
        sealed_path, str(config["sealed_test_manifest_hash"]), "sealed-test receipt"
    )
    bundle_payload, bundle_file_hash = _validate_json_identity(
        bundle_path, str(config["validation_bundle_hash"]), "validation bundle"
    )
    _sealed_receipt_is_safe(sealed_payload)
    sealed_identity = str(sealed_payload.get("identity_sha256", ""))
    if sealed_identity != str(config["sealed_test_identity_hash"]):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Sealed-test identity hash mismatch")

    train_records, val_records = _records(train_payload), _records(val_payload)
    if any(str(record.get("split")) != "train" for record in train_records):
        raise Formal2DTrainingError("BIOSR_SPLIT_BLOCKED", "Train manifest contains another split")
    if any(str(record.get("split")) != "validation" for record in val_records):
        raise Formal2DTrainingError("BIOSR_SPLIT_BLOCKED", "Validation manifest contains another split")
    _manifest_disjoint(train_records, val_records)
    _sealed_disjoint(train_records, val_records, sealed_payload)
    _validate_validation_bundle(bundle_payload, val_records, str(config["validation_manifest_hash"]))
    for record in train_records + val_records:
        if tuple(record.get("shape", ())) != BIOSR_REQUIRED_SHAPE:
            raise Formal2DTrainingError("BIOSR_GT_DATA_INVALID", "Manifest declares non-1004x1004 data")
        record_path = Path(str(record.get("absolute_path", "")))
        candidate = (dataset_root / record_path).resolve() if not record_path.is_absolute() else record_path.resolve()
        if candidate.parent != dataset_root:
            raise Formal2DTrainingError("BIOSR_GT_DATA_INVALID", f"Manifest escapes dataset root: {candidate}")

    training = _require_mapping(config, "training")
    validation = _require_mapping(config, "validation")
    model = _require_mapping(config, "model")
    if int(training.get("total_steps", -1)) != 100000:
        raise Formal2DTrainingError(
            "FORMAL_TRAINING_CONFIG_BLOCKED", "Formal budget must be 100000 scheduled iterations"
        )
    if int(training.get("batch_size", -1)) != 4 or int(training.get("diffusion_steps", -1)) != 1000:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Batch/diffusion setting mismatch")
    if int(validation.get("interval", -1)) != 2000 or bool(validation.get("early_stopping", True)):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Validation policy mismatch")
    if int(model.get("in_channels", -1)) != 31 or int(model.get("kmax", -1)) != KMAX:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Fixed-slot model contract mismatch")
    amp_policy = training.get("amp_overflow_policy")
    if amp_policy is not None:
        if spec.protocol_id != "DMD_3F_1O3P" or not isinstance(amp_policy, Mapping):
            raise Formal2DTrainingError(
                "FORMAL_TRAINING_CONFIG_BLOCKED",
                "The simplified AMP overflow policy is restricted to DMD-3F",
            )
        expected_amp_policy = {
            "mode": STANDARD_AMP_POLICY,
            "full_transaction_snapshot": False,
            "initial_scale": 128.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "max_consecutive_skips": 8,
            "max_total_skips": 500,
            "max_skip_fraction": 0.005,
        }
        for key, expected in expected_amp_policy.items():
            if amp_policy.get(key) != expected:
                raise Formal2DTrainingError(
                    "FORMAL_TRAINING_CONFIG_BLOCKED",
                    f"DMD-3F AMP overflow policy mismatch for {key}",
                )
        if config.get("legacy_dmd3_resume_disabled") is not True:
            raise Formal2DTrainingError(
                "FORMAL_TRAINING_CONFIG_BLOCKED", "DMD-3F legacy resume must be explicitly disabled"
            )
        counter_contract = config.get("counter_semantics")
        if not isinstance(counter_contract, Mapping) or counter_contract.get("mode") != SCHEDULED_COUNTER_SEMANTICS:
            raise Formal2DTrainingError(
                "FORMAL_TRAINING_CONFIG_BLOCKED", "DMD-3F scheduled/committed counter contract mismatch"
            )
        if not str(config.get("run_generation_id", "")).strip():
            raise Formal2DTrainingError(
                "FORMAL_TRAINING_CONFIG_BLOCKED", "DMD-3F restart run_generation_id is required"
            )
    forward_contract = _require_mapping(config, "forward")
    if dict(forward_contract.get("normalization", {})) != {
        "clip": [0.0, 1.0],
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
    }:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "Normalization contract mismatch")

    initialization = _require_mapping(config, "initialization")
    initialization_status = "READY_FROM_SCRATCH"
    initialization_source_path: Optional[str] = None
    initialization_source_sha256: Optional[str] = None
    if spec.protocol_id == "DMD_3F_1O3P":
        if initialization.get("policy") != "full_model_from_verified_dmd9":
            raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3 initialization policy mismatch")
        checkpoint_value = initialization.get("checkpoint_path")
        receipt_value = initialization.get("completion_receipt_path")
        checkpoint = _resolve(checkpoint_value, project_root) if checkpoint_value else None
        receipt = _resolve(receipt_value, project_root) if receipt_value else None
        if checkpoint is None or receipt is None or not checkpoint.is_file() or not receipt.is_file():
            initialization_status = K3_WAITING_STATUS
        else:
            receipt_payload = _load_json(receipt)
            receipt_hash = (
                str(receipt_payload.get("checkpoint_sha256", ""))
                if isinstance(receipt_payload, Mapping)
                else ""
            )
            configured_hash = str(initialization.get("checkpoint_sha256") or "")
            expected_hash = configured_hash or receipt_hash
            if (
                not isinstance(receipt_payload, Mapping)
                or receipt_payload.get("completion_status") != "FORMAL_TRAINING_COMPLETE"
                or len(expected_hash) != 64
                or receipt_hash != expected_hash
                or _file_sha256(checkpoint) != expected_hash
            ):
                raise Formal2DTrainingError(
                    "FORMAL_TRAINING_CONFIG_BLOCKED",
                    "Present DMD9 initialization artifacts failed receipt/hash verification",
                )
            # A successful K3 preflight means the full DMD9 model was actually
            # deserialized, identity-checked and strictly loaded.  Do not label
            # a shallow path/hash check as a verified full-model load.
            verification_model = _make_model(config)
            source_identity = _load_verified_k9_initialization(
                verification_model, config, project_root
            )
            initialization_status = K3_LOADED_STATUS
            initialization_source_path = str(source_identity["checkpoint_path"])
            initialization_source_sha256 = str(source_identity["checkpoint_sha256"])
            del verification_model
    else:
        if initialization.get("policy") != "from_scratch":
            raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9/K6 must initialize from scratch")

    preflight = {
        "status": "FORMAL_2D_PREFLIGHT_PASS",
        "initialization_status": initialization_status,
        "protocol_id": spec.protocol_id,
        "protocol_hash": spec.protocol_hash,
        "protocol_registry_hash": protocol_registry.registry_hash,
        "config_path": str(config_path),
        "config_payload_hash": actual_config_hash,
        "dataset_mode": "2D BioSR GT",
        "dataset_root": str(dataset_root),
        "eligible_gt_count": len(train_records) + len(val_records),
        "training_identity_count": len(train_records),
        "validation_identity_count": len(val_records),
        "test_identities_sealed": int(sealed_payload.get("record_count", 0)),
        "input_image_shape": list(BIOSR_REQUIRED_SHAPE),
        "training_patch_shape": [320, 320],
        "initialization_policy": initialization.get("policy"),
        "initialization_source_path": initialization_source_path,
        "initialization_source_sha256": initialization_source_sha256,
        "total_steps": int(training["total_steps"]),
        "scheduled_iterations": int(training["total_steps"]),
        "counter_semantics": (
            config.get("counter_semantics", {}).get("mode")
            if isinstance(config.get("counter_semantics"), Mapping)
            else "LEGACY_MATCHED_EVENT_AND_COMMIT_COUNTERS"
        ),
        "amp_overflow_policy": (
            amp_policy.get("mode") if isinstance(amp_policy, Mapping) else "STRICT_TRANSACTIONAL_NUMERIC_GATE"
        ),
        "precision_policy": "CUDA_FP16_AUTOCAST_WITH_PYTORCH_GRADSCALER",
        "legacy_dmd3_resume_disabled": bool(config.get("legacy_dmd3_resume_disabled", False)),
        "validation_interval": int(validation["interval"]),
        "best_checkpoint_metric": "mean val_total_loss",
        "checkpoint_directory": str(_resolve(_require_mapping(config, "outputs")["checkpoint_dir"], project_root)),
        "train_manifest_file_sha256": train_file_hash,
        "validation_manifest_file_sha256": val_file_hash,
        "sealed_test_manifest_file_sha256": sealed_file_hash,
        "validation_bundle_file_sha256": bundle_file_hash,
        "sealed_test_identity_hash": sealed_identity,
        "sealed_test_runtime_tiff_accesses": 0,
        "test_files_not_accessible_to_training_runtime": True,
    }
    return config, preflight


def formal_preflight_2d(protocol_id: str, config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    candidate = _load_json(path)
    if isinstance(candidate, Mapping) and candidate.get("config_type") == "APD_DMD_R3_DMD9_RETRAIN_R1":
        if protocol_id != "DMD_9F_3O3P":
            raise Formal2DTrainingError(
                "DMD9_RETRAIN_CONFIG_BLOCKED", "The DMD9 retrain config cannot dispatch another protocol"
            )
        from .dmd9_retrain_r1 import validate_retrain_config

        _config, preflight = validate_retrain_config(path)
        return preflight
    _config, preflight = _validate_config(path, protocol_id)
    return preflight


def print_preflight_2d(result: Mapping[str, Any]) -> None:
    print("=" * 88)
    print("APD-SIM DMD R2 formal two-dimensional training preflight")
    labels = {
        "dataset_mode": "Dataset mode",
        "dataset_root": "Dataset root",
        "eligible_gt_count": "Eligible GT count",
        "training_identity_count": "Training identity count",
        "validation_identity_count": "Validation identity count",
        "test_identities_sealed": "Test identities sealed",
        "input_image_shape": "Input image shape",
        "training_patch_shape": "Training patch shape",
        "protocol_id": "Protocol ID",
        "protocol_hash": "Protocol hash",
        "initialization_policy": "Initialization policy",
        "initialization_source_path": "Initialization source",
        "initialization_source_sha256": "Initialization SHA-256",
        "initialization_status": "Initialization status",
        "scheduled_iterations": "Scheduled iteration target",
        "counter_semantics": "Counter semantics",
        "precision_policy": "Precision policy",
        "amp_overflow_policy": "AMP overflow policy",
        "legacy_dmd3_resume_disabled": "Legacy DMD3 resume disabled",
        "validation_interval": "Validation interval",
        "best_checkpoint_metric": "Best-checkpoint metric",
        "checkpoint_directory": "Checkpoint directory",
        "status": "Status",
    }
    for key, label in labels.items():
        if key in ("initialization_source_path", "initialization_source_sha256") and result.get(key) is None:
            continue
        print(f"{label}: {result.get(key)}")
    print("=" * 88)


def _make_model(config: Mapping[str, Any]) -> APDConditionedUNet2D:
    model_cfg = _require_mapping(config, "model")
    model = APDConditionedUNet2D(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(int(value) for value in model_cfg["channel_mults"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        dropout=float(model_cfg["dropout"]),
        time_dim=int(model_cfg["time_dim"]),
        groups=int(model_cfg["groups"]),
    )
    assert_strictly_2d_model(model)
    return model


def _make_sim_config(config: Mapping[str, Any]) -> SIM2DConfig:
    forward = dict(_require_mapping(config, "forward"))
    allowed = set(SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in forward.items() if key in allowed}
    for key in (
        "rand_k_ratio_xy",
        "rand_mod_depth",
        "rand_psf_sigma_scale",
        "rand_background",
        "rand_photon_scale",
        "rand_read_noise_e",
    ):
        if key in values:
            values[key] = tuple(float(v) for v in values[key])
    return SIM2DConfig(**values)


def _load_verified_k9_initialization(
    model: APDConditionedUNet2D, config: Mapping[str, Any], project_root: Path
) -> Dict[str, str]:
    initialization = _require_mapping(config, "initialization")
    checkpoint_path = _resolve(initialization["checkpoint_path"], project_root)
    receipt_path = _resolve(initialization["completion_receipt_path"], project_root)
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping) or receipt.get("completion_status") != "FORMAL_TRAINING_COMPLETE":
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 completion receipt is not final")
    configured_hash = str(initialization.get("checkpoint_sha256") or "").lower()
    receipt_hash = str(receipt.get("checkpoint_sha256") or "").lower()
    expected_hash = configured_hash or receipt_hash
    if len(expected_hash) != 64 or receipt_hash != expected_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 expected hash is not sealed by receipt")
    if _file_sha256(checkpoint_path) != expected_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 checkpoint hash mismatch")
    if receipt.get("checkpoint_sha256") != expected_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 receipt/checkpoint mismatch")
    if receipt.get("protocol_id") != "DMD_9F_3O3P":
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3 source is not DMD9")
    if receipt.get("protocol_hash") != require_protocol("DMD_9F_3O3P").protocol_hash:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 receipt protocol hash mismatch")
    if receipt.get("architecture_hash") != architecture_hash(model):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3/K9 architecture mismatch")
    if receipt.get("normalization_contract_hash") != NORMALIZATION_CONTRACT_HASH:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3/K9 normalization mismatch")
    if receipt.get("input_tensor_dimensionality") != INPUT_TENSOR_DIMENSIONALITY:
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3/K9 tensor dimensionality mismatch")
    source_spec = require_protocol("DMD_9F_3O3P")
    target_spec = require_protocol("DMD_3F_1O3P")
    if (
        target_spec.orientation_ids[0] != "X"
        or source_spec.orientation_ids[0] != "X"
        or target_spec.orientation_angles[0] != source_spec.orientation_angles[0]
        or target_spec.carrier_vectors[0] != source_spec.carrier_vectors[0]
    ):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3 X is not physically identical to K9 X")
    if (
        target_spec.phase_ids != source_spec.phase_ids
        or target_spec.nominal_phase_values != source_spec.nominal_phase_values
        or target_spec.raw_frame_order != source_spec.raw_frame_order[: target_spec.frame_count]
    ):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3/K9 phase or row order mismatch")
    classification, _reason = initialization_compatibility("DMD_9F_3O3P", "DMD_3F_1O3P")
    if classification != "full_model_initialization_from_dmd9_allowed":
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K3/DMD9 protocol compatibility failed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(checkpoint_metadata, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 checkpoint lacks metadata")
    required_checkpoint_identity = {
        "completion_status": "FORMAL_TRAINING_COMPLETE",
        "training_protocol_id": "DMD_9F_3O3P",
        "training_protocol_hash": source_spec.protocol_hash,
        "architecture_hash": architecture_hash(model),
        "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
        "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
    }
    mismatches = [
        f"{key}={checkpoint_metadata.get(key)!r}"
        for key, expected in required_checkpoint_identity.items()
        if checkpoint_metadata.get(key) != expected
    ]
    if mismatches:
        raise Formal2DTrainingError(
            "FORMAL_TRAINING_CONFIG_BLOCKED",
            "K9 checkpoint identity/complete contract mismatch: " + ", ".join(mismatches),
        )
    state = payload.get("ema", payload.get("model")) if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "K9 checkpoint lacks model state")
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": expected_hash,
    }


def build_formal_components(
    protocol_id: str,
    config_path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    allow_waiting_initialization_for_smoke: bool = False,
) -> FormalComponents:
    path = Path(config_path).resolve()
    config, preflight = _validate_config(path, protocol_id)
    if (
        preflight["initialization_status"] == K3_WAITING_STATUS
        and not allow_waiting_initialization_for_smoke
    ):
        raise Formal2DTrainingError(K3_WAITING_STATUS, "K3 actual training waits for completed verified DMD9 best.pt")
    training = _require_mapping(config, "training")
    root = _resolve(config["project_root"], Path(__file__).resolve().parents[1])
    selected_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = _make_model(config)
    if protocol_id == "DMD_3F_1O3P" and preflight["initialization_status"] != K3_WAITING_STATUS:
        _load_verified_k9_initialization(model, config, root)
    model.to(selected_device)
    scheduler = DiffusionScheduler2D(
        int(training["diffusion_steps"]), selected_device, str(training.get("beta_schedule", "cosine"))
    )
    sim_config = _make_sim_config(config)
    betas = tuple(float(value) for value in training.get("betas", (0.9, 0.999)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr_schedule"][0]["learning_rate"]),
        betas=betas,
        eps=float(training.get("eps", 1e-8)),
        weight_decay=float(training["weight_decay"]),
    )
    ema = EMA2D(model, float(training["ema_decay"]))
    train_dataset = BioSRGT2DDataset(
        _resolve(config["train_manifest_path"], root),
        patch_size=int(config["patch_size"]),
        augment=True,
        p_low=0.5,
        p_high=99.5,
        expected_manifest_hash=str(config["train_manifest_hash"]),
        expected_split="train",
        verify_file_sha256=True,
        rng_seed=seed,
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=(selected_device.type == "cuda"),
        generator=loader_generator,
    )
    return FormalComponents(
        config=config,
        preflight=preflight,
        device=selected_device,
        model=model,
        scheduler=scheduler,
        sim_config=sim_config,
        optimizer=optimizer,
        ema=ema,
        train_dataset=train_dataset,
        train_loader=train_loader,
        loader_generator=loader_generator,
    )


def _stage_scales(step: int, training: Mapping[str, Any]) -> Tuple[float, float]:
    boundaries = training.get("stage_boundaries", {})
    stage_a = int(boundaries.get("stage_a_end", 20000))
    stage_b = int(boundaries.get("stage_b_end", 50000))
    total = int(training["total_steps"])
    mismatch_b = float(training.get("stage_b_max_mismatch", 0.6))
    snr_b = float(training.get("stage_b_max_snr", 0.3))
    if step <= stage_a:
        return 0.0, 0.0
    if step <= stage_b:
        fraction = (step - stage_a) / max(1, stage_b - stage_a)
        return mismatch_b * fraction, snr_b * fraction
    fraction = (step - stage_b) / max(1, total - stage_b)
    return mismatch_b + (1.0 - mismatch_b) * fraction, snr_b + (1.0 - snr_b) * fraction


def _training_lr(step: int, training: Mapping[str, Any]) -> float:
    for stage in training["lr_schedule"]:
        if int(stage["start_step"]) <= step - 1 <= int(stage["end_step"]):
            return float(stage["learning_rate"])
    raise Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", f"No LR applies to step {step}")


def compute_training_losses(
    *,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    protocol_id: str,
    x0: torch.Tensor,
    timestep: torch.Tensor,
    noise: torch.Tensor,
    theta: Mapping[str, torch.Tensor],
    lambda_phys: float,
    physics_active: bool,
    physics_power: float = 1.0,
    acquisition_noise_generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    if x0.ndim != 4 or x0.shape[1] != 1:
        raise Formal2DTrainingError("FORMAL_2D_TRAINING_ENGINE_BLOCKED", "Training batch is not B,1,H,W")
    xt = scheduler.q_sample(x0, timestep, noise)
    with torch.no_grad():
        raw, theta_used = forward_protocol_sim_2d(
            x0,
            sim_config,
            protocol_id,
            theta=dict(theta),
            noise_generator=acquisition_noise_generator,
        )
        slotted, mask = embed_raw_to_slots_2d(raw, protocol_id)
    predicted_epsilon = model(torch.cat((xt, slotted, mask), dim=1), timestep)
    diff_loss = F.mse_loss(predicted_epsilon.float(), noise.float())
    predicted_x0 = scheduler.predict_x0(xt.float(), timestep, predicted_epsilon.float()).clamp(0.0, 1.0)
    phys_loss = diff_loss.new_zeros(())
    phys_weight = 0.0
    if physics_active:
        predicted_raw, _ = forward_protocol_clean_2d(
            predicted_x0, sim_config, protocol_id, theta=dict(theta_used)
        )
        predicted_slots, _ = embed_raw_to_slots_2d(predicted_raw, protocol_id)
        phys_loss = masked_poisson_gaussian_likelihood_2d(
            slotted.float(),
            predicted_slots.float(),
            protocol_id,
            photon_scale=theta_used["photon_scale"],
            read_noise_e=theta_used["read_noise_e"],
        )
        low_t_weight = (1.0 - timestep.float().mean() / max(1, scheduler.total_timesteps - 1)).clamp(0, 1)
        phys_weight = float(lambda_phys) * float(low_t_weight.pow(float(physics_power)).item())
    total_loss = diff_loss + phys_loss * phys_weight
    for name, value in (("diff_loss", diff_loss), ("phys_loss", phys_loss), ("total_loss", total_loss)):
        if not bool(torch.isfinite(value).all().item()):
            raise Formal2DTrainingError("APD_BIOSR_SMOKE_BLOCKED", f"Non-finite {name}")
    return {
        "diff_loss": diff_loss,
        "phys_loss": phys_loss,
        "total_loss": total_loss,
        "predicted_x0": predicted_x0,
        "raw_frames": raw,
        "slotted_frames": slotted,
        "validity_mask": mask,
    }


def run_one_batch_smoke(
    protocol_id: str,
    config_path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    real_data: bool = True,
) -> Dict[str, Any]:
    components = build_formal_components(
        protocol_id,
        config_path,
        device=device,
        # A numerical plumbing smoke is allowed before K9 finishes.  This does
        # not relax the actual K3 launch gate or create/save a K3 checkpoint.
        allow_waiting_initialization_for_smoke=True,
    )
    sample = components.train_dataset[0]
    x0 = sample["image"].unsqueeze(0).to(components.device).float()
    if not real_data:
        x0 = torch.rand_like(x0)
    training = _require_mapping(components.config, "training")
    generator = torch.Generator(device=components.device).manual_seed(int(training["seed"]) + 91)
    timestep = torch.randint(
        0,
        components.scheduler.total_timesteps,
        (x0.shape[0],),
        device=components.device,
        generator=generator,
    )
    noise = torch.randn(x0.shape, device=components.device, dtype=x0.dtype, generator=generator)
    theta = sample_theta_2d(
        components.sim_config,
        device=components.device,
        mismatch_scale=0.5,
        snr_scale=0.5,
        rng=np.random.default_rng(int(training["seed"]) + 17),
    )
    components.optimizer.zero_grad(set_to_none=True)
    losses = compute_training_losses(
        model=components.model,
        scheduler=components.scheduler,
        sim_config=components.sim_config,
        protocol_id=protocol_id,
        x0=x0,
        timestep=timestep,
        noise=noise,
        theta=theta,
        lambda_phys=float(training["lambda_phys"]),
        physics_active=True,
        physics_power=float(training.get("physics_power", 1.0)),
        acquisition_noise_generator=generator,
    )
    losses["total_loss"].backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in components.model.parameters()
    )
    if not gradient_finite:
        raise Formal2DTrainingError("APD_BIOSR_SMOKE_BLOCKED", "Model gradient is non-finite")
    components.optimizer.step()
    train_paths = {str(Path(record["absolute_path"]).resolve()) for record in components.train_dataset.records}
    accessed = set(components.train_dataset.accessed_tiff_paths)
    if not accessed or not accessed.issubset(train_paths):
        raise Formal2DTrainingError("APD_BIOSR_SMOKE_BLOCKED", "Smoke accessed TIFF outside train manifest")
    return {
        "status": "FINITE_REAL_BIOSR_OPTIMIZER_STEP",
        "protocol_id": protocol_id,
        "device": str(components.device),
        "batch_shape": list(x0.shape),
        "raw_shape": list(losses["raw_frames"].shape),
        "slotted_shape": list(losses["slotted_frames"].shape),
        "diff_loss": float(losses["diff_loss"].detach().cpu()),
        "phys_loss": float(losses["phys_loss"].detach().cpu()),
        "total_loss": float(losses["total_loss"].detach().cpu()),
        "gradient_finite": gradient_finite,
        "accessed_tiff_paths": sorted(accessed),
        "sealed_test_runtime_access_count": 0,
        "initialization_status": components.preflight["initialization_status"],
        "initialization_bypassed_for_smoke": (
            components.preflight["initialization_status"] == K3_WAITING_STATUS
        ),
        "smoke_checkpoint_written": False,
    }


def _ssim_local(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Standard Gaussian local-window SSIM for native float BCHW arrays."""
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("SSIM expects two identically shaped BCHW tensors")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be an odd integer >= 3")
    coordinate = torch.arange(window_size, device=predicted.device, dtype=predicted.dtype)
    coordinate = coordinate - window_size // 2
    gaussian = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, window_size, window_size)
    window = window.expand(predicted.shape[1], 1, window_size, window_size)
    padding = window_size // 2

    def local_mean(value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(padded, window, groups=value.shape[1])

    mu_x, mu_y = local_mean(predicted), local_mean(target)
    sigma_x = (local_mean(predicted.square()) - mu_x.square()).clamp_min(0.0)
    sigma_y = (local_mean(target.square()) - mu_y.square()).clamp_min(0.0)
    covariance = local_mean(predicted * target) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + 0.01**2) * (2.0 * covariance + 0.03**2)
    denominator = (mu_x.square() + mu_y.square() + 0.01**2) * (
        sigma_x + sigma_y + 0.03**2
    )
    return (numerator / denominator.clamp_min(torch.finfo(predicted.dtype).eps)).mean()


def _theta_from_bundle(value: Any, cfg: SIM2DConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    # The bundle generator may serialize PowerShell objects as display strings.
    # Deterministic seeds remain authoritative; reconstruct scalar parameters by
    # the same seed if a native JSON object is unavailable.
    mapping: Optional[Mapping[str, Any]] = value if isinstance(value, Mapping) else None
    if mapping is None and isinstance(value, str) and value.startswith("@{") and value.endswith("}"):
        parsed: Dict[str, float] = {}
        for item in value[2:-1].split(";"):
            if "=" not in item:
                continue
            key, raw = item.split("=", 1)
            try:
                parsed[key.strip()] = float(raw.strip())
            except ValueError:
                continue
        mapping = parsed
    if mapping is not None and all(
        key in mapping
        for key in (
            "k_ratio_xy",
            "mod_depth",
            "background",
            "psf_sigma_scale",
            "photon_scale",
            "read_noise_e",
        )
    ):
        return {
            "k_ratio_xy": torch.tensor([float(mapping["k_ratio_xy"])], device=device),
            "mod_depth": torch.tensor([float(mapping["mod_depth"])], device=device),
            "phase_offsets": torch.tensor([float(mapping.get("phase_offset_rad", 0.0))], device=device),
            "angle_offsets": torch.tensor([float(mapping.get("angle_offset_deg", 0.0))], device=device),
            "background": torch.tensor([float(mapping["background"])], device=device),
            "psf_sigma_scale": torch.tensor([float(mapping["psf_sigma_scale"])], device=device),
            "photon_scale": torch.tensor([float(mapping["photon_scale"])], device=device),
            "read_noise_e": torch.tensor([float(mapping["read_noise_e"])], device=device),
        }
    return nominal_theta_2d(cfg, device)


@torch.no_grad()
def _validate_bundle(components: FormalComponents, global_step: int) -> Dict[str, float]:
    config, device = components.config, components.device
    root = _resolve(config["project_root"], Path(__file__).resolve().parents[1])
    val_dataset = BioSRGT2DDataset(
        _resolve(config["validation_manifest_path"], root),
        patch_size=320,
        augment=False,
        expected_manifest_hash=str(config["validation_manifest_hash"]),
        expected_split="validation",
        verify_file_sha256=True,
    )
    bundle = _load_json(_resolve(config["validation_bundle_manifest_path"], root))
    protocol_bundle = bundle["protocols"][components.preflight["protocol_id"]]
    entries = protocol_bundle["entries"]
    by_sample = {str(record["sample_id"]): index for index, record in enumerate(val_dataset.records)}
    original = copy.deepcopy(components.model.state_dict())
    components.model.load_state_dict(components.ema.shadow, strict=True)
    components.model.eval()
    metrics: List[Tuple[float, float, float, float, float]] = []
    cached_full: Dict[str, np.ndarray] = {}
    try:
        for entry in entries:
            sample_id = str(entry["sample_id"])
            if sample_id not in cached_full:
                cached_full[sample_id], _record = val_dataset.load_full_normalized(by_sample[sample_id])
            yx = entry["crop_yx"]
            if isinstance(yx, str):
                top, left = (int(v) for v in yx.split())
            else:
                top, left = (int(v) for v in yx)
            crop = cached_full[sample_id][top : top + 320, left : left + 320]
            x0 = torch.from_numpy(crop[None, None].copy()).to(device=device, dtype=torch.float32)
            timestep = torch.tensor([int(entry["diffusion_timestep"])], device=device, dtype=torch.long)
            generator = torch.Generator(device=device).manual_seed(int(entry["diffusion_seed"]))
            noise = torch.randn(x0.shape, device=device, generator=generator)
            theta = _theta_from_bundle(entry.get("nongeometry_parameters"), components.sim_config, device)
            acquisition_generator = torch.Generator(device=device).manual_seed(
                int(entry["acquisition_noise_seed"])
            )
            losses = compute_training_losses(
                model=components.model,
                scheduler=components.scheduler,
                sim_config=components.sim_config,
                protocol_id=components.preflight["protocol_id"],
                x0=x0,
                timestep=timestep,
                noise=noise,
                theta=theta,
                lambda_phys=float(_require_mapping(config, "training")["lambda_phys"]),
                physics_active=True,
                physics_power=0.0,
                acquisition_noise_generator=acquisition_generator,
            )
            predicted = losses["predicted_x0"].clamp(0, 1)
            mse = F.mse_loss(predicted, x0).clamp_min(1e-12)
            psnr = 10.0 * torch.log10(1.0 / mse)
            ssim = _ssim_local(predicted, x0)
            diff = float(losses["diff_loss"].cpu())
            phys = float(losses["phys_loss"].cpu())
            total = float(losses["total_loss"].cpu())
            metrics.append((diff, phys, total, float(psnr.cpu()), float(ssim.cpu())))
    finally:
        components.model.load_state_dict(original, strict=True)
        components.model.train()
    array = np.asarray(metrics, dtype=np.float64)
    return {
        "global_step": float(global_step),
        "mean_val_diff_loss": float(array[:, 0].mean()),
        "mean_val_phys_loss": float(array[:, 1].mean()),
        "mean_val_total_loss": float(array[:, 2].mean()),
        "mean_val_x0_psnr": float(array[:, 3].mean()),
        "mean_val_x0_ssim": float(array[:, 4].mean()),
        "validation_entry_count": float(len(metrics)),
    }


def _is_better(candidate: Mapping[str, float], best: Optional[Mapping[str, float]], tolerance: float) -> bool:
    if best is None:
        return True
    loss_delta = candidate["mean_val_total_loss"] - best["mean_val_total_loss"]
    if loss_delta < -tolerance:
        return True
    if abs(loss_delta) <= tolerance:
        for key in ("mean_val_x0_psnr", "mean_val_x0_ssim"):
            if candidate[key] > best[key]:
                return True
            if candidate[key] < best[key]:
                return False
        return candidate["global_step"] < best["global_step"]
    return False


def _checkpoint_payload(
    components: FormalComponents,
    step: int,
    metrics: Mapping[str, float],
    *,
    scaler: Optional[Any] = None,
    training_generator: Optional[torch.Generator] = None,
    training_numpy_rng: Optional[np.random.Generator] = None,
    best_metrics: Optional[Mapping[str, float]] = None,
    committed_optimizer_updates: Optional[int] = None,
    amp_overflow_skips: Optional[int] = None,
    consecutive_amp_overflow_skips: Optional[int] = None,
) -> Dict[str, Any]:
    spec = require_protocol(components.preflight["protocol_id"])
    metadata = checkpoint_protocol_metadata(spec)
    committed_count = (
        _optimizer_committed_update_count(components.optimizer)
        if committed_optimizer_updates is None
        else int(committed_optimizer_updates)
    )
    overflow_count = 0 if amp_overflow_skips is None else int(amp_overflow_skips)
    scheduled_semantics = amp_overflow_skips is not None
    metadata.update(
        {
            "model_name": f"APD-SIM-{spec.protocol_id}-2D",
            "architecture_name": components.model.__class__.__qualname__,
            "architecture_hash": architecture_hash(components.model),
            "architecture_contract": components.model.architecture_contract,
            "tensor_dimensionality": "B,C,H,W_ONLY",
            "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
            "normalization_contract": dict(NORMALIZATION_CONTRACT),
            "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
            "source_snapshot_id": components.config["source_snapshot_id"],
            "train_manifest_hash": components.config["train_manifest_hash"],
            "validation_manifest_hash": components.config["validation_manifest_hash"],
            "sealed_test_no_access_hash": components.config["sealed_test_manifest_hash"],
            "validation_bundle_hash": components.config["validation_bundle_hash"],
            "training_config_hash": components.config["config_payload_hash"],
            "training_seed": int(_require_mapping(components.config, "training")["seed"]),
            "global_step": int(step),
            "loop_event_step": int(step),
            "data_event_step": int(step),
            "scheduled_iterations": int(step),
            "committed_optimizer_updates": committed_count,
            "amp_overflow_skips": overflow_count,
            "consecutive_amp_overflow_skips": int(consecutive_amp_overflow_skips or 0),
            "global_step_semantics": (
                SCHEDULED_COUNTER_SEMANTICS
                if scheduled_semantics
                else "LOOP_AND_DATA_EVENT_STEP_NOT_INFERRED_OPTIMIZER_COMMITS"
            ),
            "validation_metric": dict(metrics),
            "best_validation_metric": dict(best_metrics or {}),
            "checkpoint_selection_rule": BEST_RULE_ID,
            "initialization_source": _require_mapping(components.config, "initialization")["policy"],
            "initialization_status": components.preflight.get("initialization_status"),
            "initialization_source_path": components.preflight.get("initialization_source_path"),
            "initialization_source_sha256": components.preflight.get("initialization_source_sha256"),
            "initialization_compatibility_classification": (
                "from_scratch" if spec.protocol_id != "DMD_3F_1O3P" else "verified_DMD9_full_model"
            ),
            "completion_status": "IN_PROGRESS",
            "rng_states": capture_rng_states(components.loader_generator),
            "dataset_rng_state": components.train_dataset.get_rng_state(),
            "training_generator_state": (
                training_generator.get_state() if training_generator is not None else None
            ),
            "training_numpy_rng_state": (
                copy.deepcopy(training_numpy_rng.bit_generator.state)
                if training_numpy_rng is not None
                else None
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {
        "model": components.model.state_dict(),
        "ema": components.ema.shadow,
        "optimizer": components.optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata,
    }


def _atomic_torch_save(payload: Mapping[str, Any], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(target)


def _save_committed_checkpoint_pair(
    payload: Mapping[str, Any],
    checkpoint_dir: Path,
    *,
    event_step: int,
    committed_optimizer_updates: int,
) -> Dict[str, str]:
    """Atomically roll the sole current and last-known-good progress files.

    Both files are derived from the exact same post-iteration payload.  Legacy
    transactional runs require event==commit.  The simplified DMD-3F AMP policy
    permits scheduled iterations to exceed commits only by the explicitly
    recorded overflow count.
    """
    metadata = payload.get("metadata")
    expected = int(event_step)
    committed = int(committed_optimizer_updates)
    if not isinstance(metadata, Mapping):
        valid = False
    else:
        scheduled_mode = metadata.get("global_step_semantics") == SCHEDULED_COUNTER_SEMANTICS
        base_matches = all(
            metadata.get(key) == expected
            for key in ("global_step", "loop_event_step", "data_event_step", "scheduled_iterations")
        ) and metadata.get("committed_optimizer_updates") == committed
        if scheduled_mode:
            recorded_skips = int(metadata.get("amp_overflow_skips", -1))
            valid = base_matches and 0 <= committed <= expected and recorded_skips == expected - committed
        else:
            valid = base_matches and committed == expected
    if not valid:
        raise Formal2DTrainingError(
            "FORMAL_2D_TRANSACTION_CHECKPOINT_BLOCKED",
            "Refusing progress checkpoint write without valid scheduled/committed counters",
        )
    latest_path = checkpoint_dir / "latest.pt"
    latest_good_path = checkpoint_dir / "latest_good.pt"
    latest_hash = _atomic_torch_save(payload, latest_path)
    latest_good_hash = _atomic_torch_save(payload, latest_good_path)
    return {"latest_sha256": latest_hash, "latest_good_sha256": latest_good_hash}


def _atomic_replay_state_save(payload: Mapping[str, Any], target: Path) -> str:
    """Atomic diagnostic save without checkpoint filename semantics."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(target)


def _atomic_json_write(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _state_scalar_step(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("optimizer step is not one finite scalar")
        numeric = float(value.detach().cpu().item())
    else:
        numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"invalid optimizer step {numeric!r}")
    return int(numeric)


def _optimizer_committed_update_count(optimizer: Any) -> int:
    """Return the uniform AdamW state.step, rejecting partial/mixed commits."""
    state = optimizer.state if hasattr(optimizer, "state") else optimizer.get("state", {})
    steps = []
    for parameter_state in state.values():
        if isinstance(parameter_state, Mapping) and "step" in parameter_state:
            steps.append(_state_scalar_step(parameter_state["step"]))
    if not steps:
        return 0
    unique = sorted(set(steps))
    if len(unique) != 1:
        raise ValueError(f"optimizer state.step is not uniform: {unique}")
    return unique[0]


def _named_nonfinite_parameters(model: torch.nn.Module) -> List[str]:
    return [
        name
        for name, value in model.named_parameters()
        if not bool(torch.isfinite(value.detach()).all().item())
    ]


def _named_nonfinite_gradients(model: torch.nn.Module) -> List[str]:
    return [
        name
        for name, value in model.named_parameters()
        if value.grad is not None and not bool(torch.isfinite(value.grad.detach()).all().item())
    ]


def _nonfinite_numeric_state(value: Any, prefix: str = "state") -> List[str]:
    failures: List[str] = []
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            failures.append(prefix)
    elif isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            failures.append(prefix)
    elif isinstance(value, float):
        if not math.isfinite(value):
            failures.append(prefix)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            failures.extend(_nonfinite_numeric_state(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            failures.extend(_nonfinite_numeric_state(item, f"{prefix}[{index}]"))
    return failures


def _nonfinite_optimizer_state(optimizer: torch.optim.Optimizer) -> List[str]:
    return _nonfinite_numeric_state(optimizer.state_dict(), "optimizer")


def _finite_gradient_global_norm(model: torch.nn.Module) -> float:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        raise ValueError("no gradients were produced")
    stacked = torch.stack([value.to(dtype=torch.float64) for value in norms])
    result = torch.linalg.vector_norm(stacked)
    numeric = float(result.detach().cpu().item())
    if not math.isfinite(numeric):
        raise ValueError("gradient global norm is non-finite")
    return numeric


def _diagnostic_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        summary = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "finite": bool(torch.isfinite(detached).all().item()),
            "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
        }
        if detached.numel() <= 256:
            summary["values"] = detached.tolist()
        return summary
    if isinstance(value, np.ndarray):
        summary = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        if value.size <= 256:
            summary["values"] = value.tolist()
        return summary
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item) for item in value]
    return repr(value)


def _rng_state_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            _diagnostic_json_value(value), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except Exception:
        encoded = repr(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_optimizer_transaction(
    components: FormalComponents, scaler: Any
) -> Dict[str, Any]:
    def clone_preserving_device(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, Mapping):
            return {key: clone_preserving_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_preserving_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_preserving_device(item) for item in value)
        return copy.deepcopy(value)

    return {
        "model": {
            key: value.detach().clone()
            for key, value in components.model.state_dict().items()
        },
        "optimizer": clone_preserving_device(components.optimizer.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
    }


def _restore_optimizer_transaction(
    snapshot: Mapping[str, Any], components: FormalComponents, scaler: Any
) -> None:
    components.model.load_state_dict(snapshot["model"], strict=True)
    components.optimizer.load_state_dict(snapshot["optimizer"])
    scaler.load_state_dict(snapshot["scaler"])


def _raise_numeric_gate(
    *,
    diagnostic_dir: Path,
    protocol_id: str,
    event_step: int,
    committed_optimizer_updates: int,
    phase: str,
    detail: str,
    context: Optional[Mapping[str, Any]] = None,
    replay_state: Optional[Mapping[str, Any]] = None,
) -> None:
    safe_phase = "".join(character if character.isalnum() else "_" for character in phase)
    stem = f"numeric_gate_event_{int(event_step):06d}_{safe_phase}"
    replay_path: Optional[Path] = None
    replay_sha256: Optional[str] = None
    if replay_state is not None:
        replay_path = diagnostic_dir / f"{stem}_replay_state.pt"
        replay_sha256 = _atomic_replay_state_save(
            {
                "schema_version": 1,
                "protocol_id": protocol_id,
                "loop_event_step": int(event_step),
                "contains_gt_pixels": False,
                "pre_event_rng_states": dict(replay_state),
            },
            replay_path,
        )
    receipt = {
        "schema_version": 1,
        "status": "FORMAL_2D_NUMERIC_GATE_BLOCKED",
        "root_cause_claim": "NONE_DIAGNOSTIC_ONLY",
        "protocol_id": protocol_id,
        "loop_event_step": int(event_step),
        "data_event_step": int(event_step),
        "committed_optimizer_updates_before_event": int(committed_optimizer_updates),
        "failure_phase": str(phase),
        "detail": str(detail),
        "checkpoint_written": False,
        "batch_skipped": False,
        "rollback_note": (
            "Post-step model/optimizer/scaler state is restored before failure; "
            "non-finite gradients may remain only in the terminating in-memory process."
        ),
        "replay_state_path": str(replay_path.resolve()) if replay_path is not None else None,
        "replay_state_sha256": replay_sha256,
        "replay_state_contains_gt_pixels": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context": _diagnostic_json_value(dict(context or {})),
    }
    target = diagnostic_dir / f"{stem}.json"
    _atomic_json_write(receipt, target)
    raise Formal2DTrainingError(
        "FORMAL_2D_NUMERIC_GATE_BLOCKED",
        f"{phase}: {detail}; diagnostic_receipt={target.resolve()}",
    )


def _transactional_optimizer_update(
    *,
    loss: torch.Tensor,
    components: FormalComponents,
    scaler: Any,
    amp_enabled: bool,
    event_step: int,
    committed_optimizer_updates: int,
    diagnostic_dir: Path,
    context: Optional[Mapping[str, Any]] = None,
    replay_state: Optional[Mapping[str, Any]] = None,
    gradient_clipping: Optional[Any] = None,
) -> Tuple[int, float]:
    """Commit exactly one finite optimizer update or abort without durable progress."""

    def fail(phase: str, detail: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        merged = dict(context or {})
        merged.update(dict(extra or {}))
        _raise_numeric_gate(
            diagnostic_dir=diagnostic_dir,
            protocol_id=str(components.preflight["protocol_id"]),
            event_step=event_step,
            committed_optimizer_updates=committed_optimizer_updates,
            phase=phase,
            detail=detail,
            context=merged,
            replay_state=replay_state,
        )

    if not bool(torch.isfinite(loss.detach()).all().item()):
        fail("loss_pre_backward", "loss is non-finite", {"loss": loss})
    nonfinite_parameters = _named_nonfinite_parameters(components.model)
    nonfinite_state = _nonfinite_optimizer_state(components.optimizer)
    nonfinite_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
    if nonfinite_parameters or nonfinite_state or nonfinite_scaler:
        fail(
            "state_pre_backward",
            "model/optimizer/scaler state is non-finite",
            {
                "parameters": nonfinite_parameters,
                "optimizer_state": nonfinite_state,
                "scaler_state": nonfinite_scaler,
            },
        )
    try:
        scaler.scale(loss).backward()
        if amp_enabled:
            scaler.unscale_(components.optimizer)
    except Exception as exc:
        fail("backward_or_unscale", repr(exc))
    nonfinite_gradients = _named_nonfinite_gradients(components.model)
    if nonfinite_gradients:
        fail("gradients_pre_step", "non-finite unscaled gradients", {"gradients": nonfinite_gradients})
    try:
        gradient_norm = _finite_gradient_global_norm(components.model)
    except Exception as exc:
        fail("gradient_global_norm", str(exc))
        raise AssertionError("unreachable")
    if gradient_clipping is not None:
        if isinstance(gradient_clipping, bool) or not isinstance(gradient_clipping, (int, float)):
            fail("gradient_clipping_contract", "gradient_clipping must be null or a positive number")
        threshold = float(gradient_clipping)
        if not math.isfinite(threshold) or threshold <= 0:
            fail("gradient_clipping_contract", "gradient_clipping must be a positive finite number")
        torch.nn.utils.clip_grad_norm_(components.model.parameters(), threshold)
        post_clip_nonfinite = _named_nonfinite_gradients(components.model)
        if post_clip_nonfinite:
            fail("gradients_post_clip", "clipping produced non-finite gradients", {"gradients": post_clip_nonfinite})
    nonfinite_parameters = _named_nonfinite_parameters(components.model)
    nonfinite_state = _nonfinite_optimizer_state(components.optimizer)
    nonfinite_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
    if nonfinite_parameters or nonfinite_state or nonfinite_scaler:
        fail(
            "state_pre_step",
            "model/optimizer/scaler state is non-finite before optimizer step",
            {
                "parameters": nonfinite_parameters,
                "optimizer_state": nonfinite_state,
                "scaler_state": nonfinite_scaler,
            },
        )
    try:
        inferred_before = _optimizer_committed_update_count(components.optimizer)
    except Exception as exc:
        fail("optimizer_counter_pre_step", str(exc))
        raise AssertionError("unreachable")
    if inferred_before != int(committed_optimizer_updates):
        fail(
            "optimizer_counter_pre_step",
            f"AdamW state.step={inferred_before} but committed count={committed_optimizer_updates}",
        )
    transaction_snapshot = _snapshot_optimizer_transaction(components, scaler)
    old_scale = float(scaler.get_scale()) if amp_enabled and hasattr(scaler, "get_scale") else None
    try:
        scaler.step(components.optimizer)
        scaler.update()
    except Exception as exc:
        _restore_optimizer_transaction(transaction_snapshot, components, scaler)
        fail("optimizer_step", repr(exc), {"amp_scale_before": old_scale})
    try:
        inferred_after = _optimizer_committed_update_count(components.optimizer)
    except Exception as exc:
        _restore_optimizer_transaction(transaction_snapshot, components, scaler)
        fail("optimizer_counter_post_step", str(exc), {"amp_scale_before": old_scale})
        raise AssertionError("unreachable")
    new_scale = float(scaler.get_scale()) if amp_enabled and hasattr(scaler, "get_scale") else None
    if inferred_after != inferred_before + 1:
        _restore_optimizer_transaction(transaction_snapshot, components, scaler)
        fail(
            "amp_silent_skip" if amp_enabled else "optimizer_commit_count",
            f"optimizer update did not commit exactly once ({inferred_before}->{inferred_after})",
            {"amp_scale_before": old_scale, "amp_scale_after": new_scale},
        )
    nonfinite_parameters = _named_nonfinite_parameters(components.model)
    nonfinite_state = _nonfinite_optimizer_state(components.optimizer)
    nonfinite_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
    if nonfinite_parameters or nonfinite_state or nonfinite_scaler:
        _restore_optimizer_transaction(transaction_snapshot, components, scaler)
        fail(
            "state_post_step",
            "committed update produced non-finite model/optimizer/scaler state",
            {
                "parameters": nonfinite_parameters,
                "optimizer_state": nonfinite_state,
                "scaler_state": nonfinite_scaler,
            },
        )
    return inferred_after, gradient_norm


def _record_amp_overflow_skip(
    *,
    diagnostic_dir: Path,
    protocol_id: str,
    scheduled_iteration: int,
    committed_optimizer_updates: int,
    previous_scale: float,
    new_scale: float,
    nonfinite_gradients: Sequence[str],
    context: Optional[Mapping[str, Any]] = None,
) -> Path:
    receipt = {
        "schema_version": 1,
        "status": "AMP_OVERFLOW_SKIP",
        "protocol_id": str(protocol_id),
        "scheduled_iteration": int(scheduled_iteration),
        "committed_optimizer_updates_before": int(committed_optimizer_updates),
        "committed_optimizer_updates_after": int(committed_optimizer_updates),
        "previous_scale": float(previous_scale),
        "new_scale": float(new_scale),
        "nonfinite_unscaled_gradients": [str(name) for name in nonfinite_gradients],
        "optimizer_updated": False,
        "ema_updated": False,
        "checkpoint_written": False,
        "batch_skip_classification": "STANDARD_DYNAMIC_LOSS_SCALING_NOT_DATA_REJECTION",
        "context": _diagnostic_json_value(dict(context or {})),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = diagnostic_dir / f"amp_overflow_skip_iteration_{int(scheduled_iteration):06d}.json"
    _atomic_json_write(receipt, target)
    return target


def _standard_amp_optimizer_update(
    *,
    loss: torch.Tensor,
    components: FormalComponents,
    scaler: Any,
    amp_enabled: bool,
    scheduled_iteration: int,
    committed_optimizer_updates: int,
    diagnostic_dir: Path,
    context: Optional[Mapping[str, Any]] = None,
    replay_state: Optional[Mapping[str, Any]] = None,
    gradient_clipping: Optional[Any] = None,
) -> OptimizerIterationResult:
    """Apply standard GradScaler semantics without a full transaction snapshot.

    A finite forward followed by non-finite *unscaled* gradients is an AMP
    overflow event: AdamW and EMA remain unchanged, GradScaler lowers its scale,
    gradients are cleared, and the scheduled iteration remains auditable.
    Every other non-finite state is a fail-closed engine error.
    """

    def fail(phase: str, detail: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        merged = dict(context or {})
        merged.update(dict(extra or {}))
        _raise_numeric_gate(
            diagnostic_dir=diagnostic_dir,
            protocol_id=str(components.preflight["protocol_id"]),
            event_step=scheduled_iteration,
            committed_optimizer_updates=committed_optimizer_updates,
            phase=phase,
            detail=detail,
            context=merged,
            replay_state=replay_state,
        )

    if not bool(torch.isfinite(loss.detach()).all().item()):
        fail("loss_pre_backward", "loss is non-finite", {"loss": loss})
    pre_parameters = _named_nonfinite_parameters(components.model)
    pre_optimizer = _nonfinite_optimizer_state(components.optimizer)
    pre_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
    if pre_parameters or pre_optimizer or pre_scaler:
        fail(
            "state_pre_backward",
            "model/optimizer/scaler state is non-finite",
            {
                "parameters": pre_parameters,
                "optimizer_state": pre_optimizer,
                "scaler_state": pre_scaler,
            },
        )
    try:
        inferred_before = _optimizer_committed_update_count(components.optimizer)
    except Exception as exc:
        fail("optimizer_counter_pre_step", str(exc))
        raise AssertionError("unreachable")
    if inferred_before != int(committed_optimizer_updates):
        fail(
            "optimizer_counter_pre_step",
            f"AdamW state.step={inferred_before} but committed count={committed_optimizer_updates}",
        )
    previous_scale = float(scaler.get_scale()) if amp_enabled and hasattr(scaler, "get_scale") else None
    try:
        scaler.scale(loss).backward()
        if amp_enabled:
            scaler.unscale_(components.optimizer)
    except Exception as exc:
        fail("backward_or_unscale", repr(exc))
    nonfinite_gradients = _named_nonfinite_gradients(components.model)
    if nonfinite_gradients:
        if not amp_enabled or previous_scale is None:
            fail(
                "gradients_pre_step",
                "non-finite unscaled gradients without enabled CUDA AMP",
                {"gradients": nonfinite_gradients},
            )
        try:
            scaler.update()
        except Exception as exc:
            fail("amp_overflow_scale_update", repr(exc), {"amp_scale_before": previous_scale})
        new_scale = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else float("nan")
        post_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
        inferred_after = _optimizer_committed_update_count(components.optimizer)
        if (
            not math.isfinite(new_scale)
            or not new_scale < previous_scale
            or post_scaler
            or inferred_after != inferred_before
        ):
            fail(
                "amp_overflow_scale_update",
                "overflow did not produce one finite downward loss-scale update without an optimizer commit",
                {
                    "amp_scale_before": previous_scale,
                    "amp_scale_after": new_scale,
                    "scaler_state": post_scaler,
                    "optimizer_commits_before": inferred_before,
                    "optimizer_commits_after": inferred_after,
                },
            )
        components.optimizer.zero_grad(set_to_none=True)
        receipt_path = _record_amp_overflow_skip(
            diagnostic_dir=diagnostic_dir,
            protocol_id=str(components.preflight["protocol_id"]),
            scheduled_iteration=scheduled_iteration,
            committed_optimizer_updates=committed_optimizer_updates,
            previous_scale=previous_scale,
            new_scale=new_scale,
            nonfinite_gradients=nonfinite_gradients,
            context=context,
        )
        print(
            "AMP_OVERFLOW_SKIP "
            f"scheduled_iteration={scheduled_iteration} "
            f"committed_updates={committed_optimizer_updates} "
            f"previous_scale={previous_scale:g} new_scale={new_scale:g} "
            f"receipt={receipt_path.resolve()}"
        )
        return OptimizerIterationResult(
            committed_optimizer_updates=inferred_before,
            gradient_norm=None,
            amp_overflow_skipped=True,
            previous_scale=previous_scale,
            new_scale=new_scale,
        )

    try:
        gradient_norm = _finite_gradient_global_norm(components.model)
    except Exception as exc:
        fail("gradient_global_norm", str(exc))
        raise AssertionError("unreachable")
    if gradient_clipping is not None:
        if isinstance(gradient_clipping, bool) or not isinstance(gradient_clipping, (int, float)):
            fail("gradient_clipping_contract", "gradient_clipping must be null or a positive number")
        threshold = float(gradient_clipping)
        if not math.isfinite(threshold) or threshold <= 0:
            fail("gradient_clipping_contract", "gradient_clipping must be a positive finite number")
        torch.nn.utils.clip_grad_norm_(components.model.parameters(), threshold)
        clipped_nonfinite = _named_nonfinite_gradients(components.model)
        if clipped_nonfinite:
            fail("gradients_post_clip", "clipping produced non-finite gradients", {"gradients": clipped_nonfinite})
    try:
        scaler.step(components.optimizer)
        scaler.update()
    except Exception as exc:
        fail("optimizer_step", repr(exc), {"amp_scale_before": previous_scale})
    inferred_after = _optimizer_committed_update_count(components.optimizer)
    new_scale = float(scaler.get_scale()) if amp_enabled and hasattr(scaler, "get_scale") else previous_scale
    if inferred_after != inferred_before + 1:
        fail(
            "unexpected_optimizer_skip",
            f"finite unscaled gradients did not commit exactly once ({inferred_before}->{inferred_after})",
            {"amp_scale_before": previous_scale, "amp_scale_after": new_scale},
        )
    post_parameters = _named_nonfinite_parameters(components.model)
    post_optimizer = _nonfinite_optimizer_state(components.optimizer)
    post_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
    if post_parameters or post_optimizer or post_scaler:
        fail(
            "state_post_step",
            "optimizer update produced non-finite model/optimizer/scaler state",
            {
                "parameters": post_parameters,
                "optimizer_state": post_optimizer,
                "scaler_state": post_scaler,
            },
        )
    return OptimizerIterationResult(
        committed_optimizer_updates=inferred_after,
        gradient_norm=gradient_norm,
        amp_overflow_skipped=False,
        previous_scale=previous_scale,
        new_scale=new_scale,
    )


def _resume_expected_identities(components: FormalComponents) -> Dict[str, Any]:
    spec = require_protocol(components.preflight["protocol_id"])
    return {
        "training_protocol_id": spec.protocol_id,
        "training_protocol_hash": spec.protocol_hash,
        "architecture_hash": architecture_hash(components.model),
        "architecture_contract": components.model.architecture_contract,
        "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
        "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
        "source_snapshot_id": components.config["source_snapshot_id"],
        "train_manifest_hash": components.config["train_manifest_hash"],
        "validation_manifest_hash": components.config["validation_manifest_hash"],
        "sealed_test_no_access_hash": components.config["sealed_test_manifest_hash"],
        "validation_bundle_hash": components.config["validation_bundle_hash"],
        "training_config_hash": components.config["config_payload_hash"],
        "training_seed": int(_require_mapping(components.config, "training")["seed"]),
        "checkpoint_selection_rule": BEST_RULE_ID,
        "completion_status": "IN_PROGRESS",
    }


def _checkpoint_event_counters(
    metadata: Mapping[str, Any], optimizer_state: Mapping[str, Any], protocol_id: str
) -> Tuple[int, int]:
    """Validate loop/data events separately from durable AdamW commits."""
    try:
        global_step = int(metadata.get("global_step", -1))
        inferred_commits = _optimizer_committed_update_count(optimizer_state)
    except Exception as exc:
        status = "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED"
        raise Formal2DTrainingError(status, f"Cannot establish optimizer commit count: {exc}") from exc
    transactional_fields = (
        "loop_event_step",
        "data_event_step",
        "committed_optimizer_updates",
        "global_step_semantics",
    )
    missing = [key for key in transactional_fields if key not in metadata]
    if missing:
        detail = (
            f"Legacy checkpoint lacks transactional fields {missing}; "
            f"metadata global_step={global_step}, inferred AdamW commits={inferred_commits}"
        )
        if protocol_id == "DMD_3F_1O3P" or global_step != inferred_commits:
            status = "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED"
            raise Formal2DTrainingError(status, detail + "; trajectory must not be reinterpreted")
        return global_step, inferred_commits
    loop_step = int(metadata["loop_event_step"])
    data_step = int(metadata["data_event_step"])
    recorded_commits = int(metadata["committed_optimizer_updates"])
    semantics = metadata.get("global_step_semantics")
    if semantics == SCHEDULED_COUNTER_SEMANTICS:
        scheduled = int(metadata.get("scheduled_iterations", -1))
        overflow_skips = int(metadata.get("amp_overflow_skips", -1))
        counters_valid = (
            global_step == loop_step == data_step == scheduled
            and recorded_commits == inferred_commits
            and 0 <= recorded_commits <= scheduled
            and overflow_skips == scheduled - recorded_commits
        )
    else:
        counters_valid = (
            global_step == loop_step == data_step
            and recorded_commits == inferred_commits == loop_step
            and semantics == "LOOP_AND_DATA_EVENT_STEP_NOT_INFERRED_OPTIMIZER_COMMITS"
        )
    if not counters_valid:
        status = "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED"
        raise Formal2DTrainingError(
            status,
            "Transactional counter mismatch: "
            f"global={global_step}, loop={loop_step}, data={data_step}, "
            f"recorded_commits={recorded_commits}, inferred_commits={inferred_commits}",
        )
    return loop_step, recorded_commits


def _precheck_resume_transaction_contract(latest_path: Path, protocol_id: str) -> None:
    """Block known-invalid legacy trajectories before real-data/GPU smoke work."""
    try:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        status = "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED"
        raise Formal2DTrainingError(status, f"Cannot inspect resume checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise Formal2DTrainingError(
            "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED",
            "Resume checkpoint lacks metadata",
        )
    if not isinstance(payload.get("optimizer"), Mapping):
        raise Formal2DTrainingError(
            "DMD3_RECOVERY_NOT_READY" if protocol_id == "DMD_3F_1O3P" else "FORMAL_2D_RESUME_BLOCKED",
            "Resume checkpoint lacks optimizer state",
        )
    _checkpoint_event_counters(payload["metadata"], payload["optimizer"], protocol_id)


def _finalize_or_validate_completed_run(
    components: FormalComponents,
    *,
    final_path: Path,
    best_path: Path,
    receipt_path: Path,
) -> Dict[str, Any]:
    """Finish the narrow final-save crash window, or validate an existing run."""
    if not final_path.is_file():
        raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Completion receipt exists but final.pt is absent")
    try:
        final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        receipt = _load_json(receipt_path)
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", f"Cannot inspect completion artifacts: {exc}"
        ) from exc
    if (
        not isinstance(final_payload, Mapping)
        or not isinstance(final_payload.get("metadata"), Mapping)
        or not isinstance(best_payload, MutableMapping)
        or not isinstance(best_payload.get("metadata"), MutableMapping)
        or not isinstance(receipt, Mapping)
    ):
        raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Completion artifacts lack metadata")
    expected = _resume_expected_identities(components)
    expected.pop("completion_status", None)
    for label, metadata in (
        ("final", final_payload["metadata"]),
        ("best", best_payload["metadata"]),
    ):
        mismatches = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if mismatches:
            raise Formal2DTrainingError(
                "FORMAL_2D_RESUME_BLOCKED",
                f"{label} checkpoint identity mismatch: {', '.join(mismatches)}",
            )
    total_steps = int(_require_mapping(components.config, "training")["total_steps"])
    if (
        final_payload["metadata"].get("completion_status") != "FORMAL_TRAINING_COMPLETE"
        or int(final_payload["metadata"].get("global_step", -1)) != total_steps
        or receipt.get("protocol_id") != components.preflight["protocol_id"]
        or receipt.get("protocol_hash") != components.preflight["protocol_hash"]
    ):
        raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Final completion identity mismatch")
    final_hash = _file_sha256(final_path)
    best_metadata = best_payload["metadata"]
    if best_metadata.get("completion_status") != "FORMAL_TRAINING_COMPLETE":
        if best_metadata.get("completion_status") != "IN_PROGRESS":
            raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Unexpected best checkpoint status")
        best_metadata["completion_status"] = "FORMAL_TRAINING_COMPLETE"
        best_hash = _atomic_torch_save(best_payload, best_path)
    else:
        best_hash = _file_sha256(best_path)
    if receipt.get("completion_status") == "FORMAL_TRAINING_COMPLETE":
        if (
            receipt.get("checkpoint_sha256") != best_hash
            or receipt.get("formal_final_checkpoint_sha256") != final_hash
        ):
            raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Completed receipt hash mismatch")
    elif receipt.get("completion_status") == "BEST_CHECKPOINT_SELECTED_DURING_IN_PROGRESS_RUN":
        receipt = dict(receipt)
        receipt["completion_status"] = "FORMAL_TRAINING_COMPLETE"
        receipt["checkpoint_sha256"] = best_hash
        receipt["formal_final_checkpoint_sha256"] = final_hash
        receipt["formal_final_validation_metric"] = dict(
            final_payload["metadata"].get("final_validation_metric", {})
        )
        _atomic_json_write(receipt, receipt_path)
    else:
        raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Unexpected completion receipt status")
    return {
        "status": "FORMAL_TRAINING_COMPLETE",
        "protocol_id": components.preflight["protocol_id"],
        "total_steps": total_steps,
        "scheduled_iterations": int(final_payload["metadata"].get("scheduled_iterations", total_steps)),
        "committed_optimizer_updates": int(
            final_payload["metadata"].get("committed_optimizer_updates", total_steps)
        ),
        "amp_overflow_skips": int(final_payload["metadata"].get("amp_overflow_skips", 0)),
        "best_metrics": dict(final_payload["metadata"].get("best_validation_metric", {})),
        "final_checkpoint_sha256": final_hash,
        "sealed_test_runtime_access_count": 0,
    }


def _restore_latest_checkpoint(
    latest_path: Path,
    components: FormalComponents,
    *,
    scaler: Any,
    training_generator: torch.Generator,
    training_numpy_rng: np.random.Generator,
) -> Tuple[int, Optional[Dict[str, float]], int, int, int]:
    """Fail-closed restore of every state that can affect the next optimizer step."""
    try:
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", f"Cannot deserialize latest checkpoint {latest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise Formal2DTrainingError("FORMAL_2D_RESUME_BLOCKED", "Latest checkpoint lacks metadata")
    metadata = payload["metadata"]
    mismatches = [
        f"{key}: {metadata.get(key)!r} != {expected!r}"
        for key, expected in _resume_expected_identities(components).items()
        if metadata.get(key) != expected
    ]
    if mismatches:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", "Latest checkpoint identity mismatch: " + "; ".join(mismatches)
        )
    required_payload = ("model", "ema", "optimizer", "scaler")
    missing_payload = [key for key in required_payload if key not in payload]
    required_rng = (
        "rng_states",
        "dataset_rng_state",
        "training_generator_state",
        "training_numpy_rng_state",
    )
    missing_rng = [key for key in required_rng if metadata.get(key) is None]
    if missing_payload or missing_rng:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED",
            f"Latest checkpoint incomplete: payload={missing_payload}, rng={missing_rng}",
        )
    loop_step, committed_updates = _checkpoint_event_counters(
        metadata, payload["optimizer"], str(components.preflight["protocol_id"])
    )
    try:
        components.model.load_state_dict(payload["model"], strict=True)
        ema_state = payload["ema"]
        if set(ema_state) != set(components.ema.shadow):
            raise ValueError("EMA state schema mismatch")
        components.ema.shadow = {
            key: value.to(device=components.device) for key, value in ema_state.items()
        }
        components.optimizer.load_state_dict(payload["optimizer"])
        if payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        restore_rng_states(metadata["rng_states"], components.loader_generator)
        components.train_dataset.set_rng_state(metadata["dataset_rng_state"])
        training_generator.set_state(metadata["training_generator_state"])
        training_numpy_rng.bit_generator.state = copy.deepcopy(metadata["training_numpy_rng_state"])
    except Exception as exc:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", f"Latest checkpoint state restoration failed: {exc}"
        ) from exc
    global_step = int(metadata.get("global_step", -1))
    total_steps = int(_require_mapping(components.config, "training")["total_steps"])
    if global_step < 0 or global_step >= total_steps:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", f"Latest checkpoint step {global_step} is not resumable"
        )
    print(f"Resume source: {latest_path.resolve()}")
    print(
        f"Resume step: next_loop_event={loop_step + 1}; "
        f"committed_optimizer_updates={committed_updates}"
    )
    return (
        global_step + 1,
        None,
        committed_updates,
        int(metadata.get("amp_overflow_skips", max(0, loop_step - committed_updates))),
        int(metadata.get("consecutive_amp_overflow_skips", 0)),
    )


_HISTORY_FIELDS = [
    "global_step",
    "mean_val_diff_loss",
    "mean_val_phys_loss",
    "mean_val_total_loss",
    "mean_val_x0_psnr",
    "mean_val_x0_ssim",
    "validation_entry_count",
]


def _load_history_for_resume(history_path: Path, start_step: int) -> Optional[Dict[str, float]]:
    if not history_path.is_file():
        if start_step > 1:
            raise Formal2DTrainingError(
                "FORMAL_2D_RESUME_BLOCKED", "Latest checkpoint exists but validation history is absent"
            )
        return None
    try:
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _HISTORY_FIELDS:
                raise ValueError(f"unexpected header {reader.fieldnames}")
            rows = list(reader)
        best: Optional[Dict[str, float]] = None
        previous_step = -1
        for row in rows:
            metric = {key: float(value) for key, value in row.items()}
            step = int(metric["global_step"])
            if step <= previous_step or step >= start_step:
                raise ValueError("history step sequence conflicts with resume checkpoint")
            previous_step = step
            if _is_better(metric, best, 1e-8):
                best = metric
        return best
    except Formal2DTrainingError:
        raise
    except Exception as exc:
        raise Formal2DTrainingError(
            "FORMAL_2D_RESUME_BLOCKED", f"Validation history is corrupt: {exc}"
        ) from exc


def _reconcile_history_to_latest(history_path: Path, latest_step: int) -> None:
    """Repair only the documented crash window where history is one row ahead.

    Validation history is append-only during normal execution.  If the process
    dies after flushing validation but before atomically replacing latest.pt,
    discard rows newer than the durable checkpoint; any other malformed history
    remains fail-closed in `_load_history_for_resume`.
    """
    if not history_path.is_file():
        return
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _HISTORY_FIELDS:
            return
        rows = list(reader)
    durable = []
    changed = False
    for row in rows:
        try:
            step = int(float(row["global_step"]))
        except Exception:
            return
        if step <= int(latest_step):
            durable.append(row)
        else:
            changed = True
    if changed:
        temporary = history_path.with_name(f".{history_path.name}.tmp.{os.getpid()}")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(durable)
        os.replace(temporary, history_path)


def _deterministic_training_batch(
    dataset: BioSRGT2DDataset,
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Return the step's shuffled batch without iterator state hidden in DataLoader.

    Each epoch permutation is derived solely from the frozen seed and epoch.
    This makes a resumed step consume exactly the same identities as an
    uninterrupted run; the dataset's separately checkpointed RNG controls crop
    and augmentation draws.
    """
    batches_per_epoch = len(dataset) // int(batch_size)
    if batches_per_epoch < 1:
        raise Formal2DTrainingError(
            "FORMAL_2D_TRAINING_ENGINE_BLOCKED", "Training split is smaller than one batch"
        )
    zero_based = int(step) - 1
    epoch, batch_in_epoch = divmod(zero_based, batches_per_epoch)
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(int(seed) + 1_000_003 * int(epoch))
    permutation = torch.randperm(len(dataset), generator=permutation_generator)
    start = batch_in_epoch * int(batch_size)
    indices = permutation[start : start + int(batch_size)].tolist()
    return default_collate([dataset[int(index)] for index in indices])


def run_formal_training(
    protocol_id: str,
    config_path: Union[str, Path],
    *,
    preflight_only: bool = False,
) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    candidate = _load_json(path)
    if isinstance(candidate, Mapping) and candidate.get("config_type") == "APD_DMD_R3_DMD9_RETRAIN_R1":
        if protocol_id != "DMD_9F_3O3P":
            raise Formal2DTrainingError(
                "DMD9_RETRAIN_CONFIG_BLOCKED", "The DMD9 retrain config cannot dispatch another protocol"
            )
        from .dmd9_retrain_r1 import run_dmd9_retrain_r1

        return run_dmd9_retrain_r1(path, preflight_only=preflight_only)
    config, preflight = _validate_config(path, protocol_id)
    print_preflight_2d(preflight)
    if preflight_only:
        return preflight
    if preflight["initialization_status"] == K3_WAITING_STATUS:
        raise Formal2DTrainingError(K3_WAITING_STATUS, "K3 waits for the completed verified DMD9 best checkpoint")

    training = _require_mapping(config, "training")
    validation = _require_mapping(config, "validation")
    outputs = _require_mapping(config, "outputs")
    root = _resolve(config["project_root"], Path(__file__).resolve().parents[1])
    checkpoint_dir = _resolve(outputs["checkpoint_dir"], root)
    history_path = _resolve(outputs["validation_history_path"], root)
    receipt_path = _resolve(outputs["best_checkpoint_receipt_path"], root)
    legacy_dmd3_dir = (root / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3").resolve()
    if (
        protocol_id == "DMD_3F_1O3P"
        and bool(config.get("legacy_dmd3_resume_disabled", False))
        and checkpoint_dir.resolve() == legacy_dmd3_dir
    ):
        raise Formal2DTrainingError(
            "DMD3_RECOVERY_NOT_READY",
            "The simplified clean restart must not read or write the legacy DMD3 checkpoint directory",
        )
    validated_final_path = checkpoint_dir / "final.pt"
    validated_latest_path = checkpoint_dir / "latest.pt"
    if not validated_final_path.is_file() and validated_latest_path.is_file():
        _precheck_resume_transaction_contract(validated_latest_path, protocol_id)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lock_device = str(selected_device).replace(":", "_").replace("\\", "_").replace("/", "_")
    lock_root = root / "checkpoints" / "apd_dmd_geometry_r2" / "_device_locks" / lock_device
    lock = TrainingLock(
        lock_root,
        script_name=Path(os.environ.get("APD_DMD_ENTRYPOINT", "formal_training_2d.py")).name,
        protocol_id=protocol_id,
        gpu=str(selected_device),
        config_hash=str(config["config_payload_hash"]),
    )
    # Lock before allocating the smoke model: concurrent right-click launches
    # must not collide during the most memory-intensive preflight operation.
    lock.acquire()
    try:
        preexisting_final = checkpoint_dir / "final.pt"
        preexisting_latest = checkpoint_dir / "latest.pt"
        if not preexisting_final.is_file() and preexisting_latest.is_file():
            _precheck_resume_transaction_contract(preexisting_latest, protocol_id)
        smoke = run_one_batch_smoke(protocol_id, path, device=selected_device)
        print(f"One-batch finite smoke: {smoke['status']} loss={smoke['total_loss']:.6g}")
        components = build_formal_components(protocol_id, path, device=selected_device)
    except Exception:
        lock.release()
        raise

    try:
        best: Optional[Dict[str, float]] = None
        generator = torch.Generator(device=components.device).manual_seed(int(training["seed"]) + 313)
        rng = np.random.default_rng(int(training["seed"]) + 719)
        amp_enabled = bool(training.get("amp_cuda", False)) and components.device.type == "cuda"
        amp_policy = training.get("amp_overflow_policy")
        use_standard_amp_policy = (
            protocol_id == "DMD_3F_1O3P"
            and isinstance(amp_policy, Mapping)
            and amp_policy.get("mode") == STANDARD_AMP_POLICY
        )
        if use_standard_amp_policy and not amp_enabled:
            raise Formal2DTrainingError(
                "DMD3_AMP_POLICY_TEST_BLOCKED",
                "The simplified DMD3 formal run requires CUDA AMP with GradScaler enabled",
            )
        scaler_kwargs: Dict[str, Any] = {}
        if use_standard_amp_policy:
            scaler_kwargs = {
                "init_scale": float(amp_policy["initial_scale"]),
                "growth_factor": float(amp_policy["growth_factor"]),
                "backoff_factor": float(amp_policy["backoff_factor"]),
                "growth_interval": int(amp_policy["growth_interval"]),
            }
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, **scaler_kwargs)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, **scaler_kwargs)
    except Exception:
        lock.release()
        raise

    # `acquire` is idempotent for this owner; this context now guarantees
    # release on every exit while preserving the pre-smoke acquisition.
    with lock:
        final_path = checkpoint_dir / "final.pt"
        completed_receipt = _load_json(receipt_path) if receipt_path.is_file() else None
        if final_path.exists():
            return _finalize_or_validate_completed_run(
                components,
                final_path=final_path,
                best_path=checkpoint_dir / "best.pt",
                receipt_path=receipt_path,
            )
        if (
            isinstance(completed_receipt, Mapping)
            and completed_receipt.get("completion_status") == "FORMAL_TRAINING_COMPLETE"
        ):
            raise Formal2DTrainingError(
                "FORMAL_2D_RESUME_BLOCKED", "Completed receipt exists but final.pt is absent"
            )
        latest_path = checkpoint_dir / "latest.pt"
        start_step = 1
        committed_optimizer_updates = 0
        amp_overflow_skips = 0
        consecutive_amp_overflow_skips = 0
        if latest_path.is_file():
            (
                start_step,
                _unused_best,
                committed_optimizer_updates,
                amp_overflow_skips,
                consecutive_amp_overflow_skips,
            ) = _restore_latest_checkpoint(
                latest_path,
                components,
                scaler=scaler,
                training_generator=generator,
                training_numpy_rng=rng,
            )
            _reconcile_history_to_latest(history_path, start_step - 1)
        best = _load_history_for_resume(history_path, start_step)
        history_mode = "a" if start_step > 1 else "w"
        with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=_HISTORY_FIELDS,
            )
            if history_mode == "w":
                writer.writeheader()
            for step in range(start_step, int(training["total_steps"]) + 1):
                validation_metrics_at_step: Dict[str, float] = {}
                replay_state = {
                    "global_rng_states": capture_rng_states(components.loader_generator),
                    "dataset_rng_state": components.train_dataset.get_rng_state(),
                    "training_generator_state": generator.get_state(),
                    "training_numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                }
                batch = _deterministic_training_batch(
                    components.train_dataset,
                    step=step,
                    batch_size=int(training["batch_size"]),
                    seed=int(training["seed"]),
                )
                x0 = batch["image"].to(components.device, non_blocking=True).float()
                if not bool(torch.isfinite(x0).all().item()):
                    _raise_numeric_gate(
                        diagnostic_dir=(
                            root / "audit" / "formal_2d_numeric_diagnostics" / protocol_id.lower()
                        ),
                        protocol_id=protocol_id,
                        event_step=step,
                        committed_optimizer_updates=committed_optimizer_updates,
                        phase="input_pre_forward",
                        detail="training input is non-finite",
                        context={"sample_id": batch.get("sample_id")},
                        replay_state=replay_state,
                    )
                timestep = torch.randint(
                    0,
                    components.scheduler.total_timesteps,
                    (x0.shape[0],),
                    device=components.device,
                    generator=generator,
                )
                noise = torch.randn(x0.shape, device=components.device, generator=generator)
                mismatch_max, snr_max = _stage_scales(step, training)
                severity = float(timestep.float().mean().item()) / max(1, components.scheduler.total_timesteps - 1)
                mismatch, snr = mismatch_max * severity, snr_max * severity
                if step > int(training["stage_boundaries"]["stage_a_end"]) and rng.random() < float(
                    training.get("force_matched_probability", 0.1)
                ):
                    mismatch = snr = 0.0
                theta = sample_theta_2d(
                    components.sim_config,
                    device=components.device,
                    mismatch_scale=mismatch,
                    snr_scale=snr,
                    rng=rng,
                )
                for group in components.optimizer.param_groups:
                    group["lr"] = _training_lr(step, training)
                components.optimizer.zero_grad(set_to_none=True)
                physics_active = rng.random() < float(training["physics_activation_probability"])
                diagnostic_dir = (
                    root
                    / "audit"
                    / "formal_2d_numeric_diagnostics"
                    / protocol_id.lower()
                )
                batch_context = {
                    "sample_id": batch.get("sample_id"),
                    "parent_id": batch.get("parent_id"),
                    "class": batch.get("class"),
                    "crop_top": batch.get("crop_top"),
                    "crop_left": batch.get("crop_left"),
                    "source_path": batch.get("source_path"),
                    "augmentation_parameters": "RECONSTRUCT_FROM_PRE_EVENT_DATASET_RNG_STATE",
                    "timestep": timestep,
                    "theta": theta,
                    "physics_active": physics_active,
                    "learning_rate": float(components.optimizer.param_groups[0]["lr"]),
                    "amp_enabled": amp_enabled,
                    "scaler_scale": float(scaler.get_scale()) if hasattr(scaler, "get_scale") else None,
                    "source_snapshot_id": config["source_snapshot_id"],
                    "config_payload_hash": config["config_payload_hash"],
                    "protocol_hash": preflight["protocol_hash"],
                    "training_generator_state_hash": _rng_state_hash(
                        replay_state["training_generator_state"]
                    ),
                    "training_numpy_rng_state_hash": _rng_state_hash(
                        replay_state["training_numpy_rng_state"]
                    ),
                    "dataset_rng_state_hash": _rng_state_hash(
                        replay_state["dataset_rng_state"]
                    ),
                    "global_rng_state_hash": _rng_state_hash(
                        replay_state["global_rng_states"]
                    ),
                }
                try:
                    with torch.autocast(
                        device_type=components.device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        losses = compute_training_losses(
                            model=components.model,
                            scheduler=components.scheduler,
                            sim_config=components.sim_config,
                            protocol_id=protocol_id,
                            x0=x0,
                            timestep=timestep,
                            noise=noise,
                            theta=theta,
                            lambda_phys=float(training["lambda_phys"]),
                            physics_active=physics_active,
                            physics_power=float(training.get("physics_power", 1.0)),
                            acquisition_noise_generator=generator,
                        )
                except Exception as exc:
                    _raise_numeric_gate(
                        diagnostic_dir=diagnostic_dir,
                        protocol_id=protocol_id,
                        event_step=step,
                        committed_optimizer_updates=committed_optimizer_updates,
                        phase="forward_or_loss",
                        detail=repr(exc),
                        context=batch_context,
                        replay_state=replay_state,
                    )
                for loss_name in ("diff_loss", "phys_loss", "total_loss"):
                    loss_value = losses.get(loss_name)
                    if not isinstance(loss_value, torch.Tensor) or not bool(
                        torch.isfinite(loss_value.detach()).all().item()
                    ):
                        _raise_numeric_gate(
                            diagnostic_dir=diagnostic_dir,
                            protocol_id=protocol_id,
                            event_step=step,
                            committed_optimizer_updates=committed_optimizer_updates,
                            phase=f"{loss_name}_pre_backward",
                            detail=f"{loss_name} is non-finite",
                            context=batch_context,
                            replay_state=replay_state,
                        )
                if use_standard_amp_policy:
                    update_result = _standard_amp_optimizer_update(
                        loss=losses["total_loss"],
                        components=components,
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                        scheduled_iteration=step,
                        committed_optimizer_updates=committed_optimizer_updates,
                        diagnostic_dir=diagnostic_dir,
                        context=batch_context,
                        replay_state=replay_state,
                        gradient_clipping=training.get("gradient_clipping"),
                    )
                    committed_optimizer_updates = update_result.committed_optimizer_updates
                    gradient_norm = update_result.gradient_norm
                    iteration_committed = not update_result.amp_overflow_skipped
                    if update_result.amp_overflow_skipped:
                        amp_overflow_skips += 1
                        consecutive_amp_overflow_skips += 1
                        if consecutive_amp_overflow_skips > int(amp_policy["max_consecutive_skips"]):
                            raise Formal2DTrainingError(
                                "DMD3_AMP_POLICY_TEST_BLOCKED",
                                f"consecutive AMP overflow skips={consecutive_amp_overflow_skips}",
                            )
                        if amp_overflow_skips > int(amp_policy["max_total_skips"]):
                            raise Formal2DTrainingError(
                                "DMD3_AMP_POLICY_TEST_BLOCKED",
                                f"total AMP overflow skips={amp_overflow_skips}",
                            )
                    else:
                        consecutive_amp_overflow_skips = 0
                        components.ema.update(components.model)
                else:
                    committed_optimizer_updates, gradient_norm = _transactional_optimizer_update(
                        loss=losses["total_loss"],
                        components=components,
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                        event_step=step,
                        committed_optimizer_updates=committed_optimizer_updates,
                        diagnostic_dir=diagnostic_dir,
                        context=batch_context,
                        replay_state=replay_state,
                        gradient_clipping=training.get("gradient_clipping"),
                    )
                    iteration_committed = True
                    components.ema.update(components.model)

                if step % int(training.get("log_interval", 50)) == 0:
                    print(
                        f"step={step} committed={committed_optimizer_updates} "
                        f"amp_skips={amp_overflow_skips} "
                        f"loss={float(losses['total_loss'].detach().cpu()):.6g} "
                        f"grad_norm={gradient_norm if gradient_norm is not None else 'AMP_OVERFLOW_SKIP'}"
                    )
                if step % int(validation["interval"]) == 0 or step == int(training["total_steps"]):
                    metrics = _validate_bundle(components, step)
                    if not all(math.isfinite(float(value)) for value in metrics.values()):
                        raise Formal2DTrainingError(
                            "FORMAL_2D_NUMERIC_GATE_BLOCKED",
                            f"validation metric is non-finite at scheduled iteration {step}",
                        )
                    validation_metrics_at_step = dict(metrics)
                    writer.writerow(metrics)
                    history_file.flush()
                    if iteration_committed and _is_better(metrics, best, float(validation["tie_tolerance"])):
                        best = dict(metrics)
                        payload = _checkpoint_payload(
                            components,
                            step,
                            metrics,
                            scaler=scaler,
                            training_generator=generator,
                            training_numpy_rng=rng,
                            best_metrics=metrics,
                            committed_optimizer_updates=committed_optimizer_updates,
                            amp_overflow_skips=(amp_overflow_skips if use_standard_amp_policy else None),
                            consecutive_amp_overflow_skips=consecutive_amp_overflow_skips,
                        )
                        checkpoint_hash = _atomic_torch_save(payload, checkpoint_dir / "best.pt")
                        receipt = {
                            "schema_version": 1,
                            "completion_status": "BEST_CHECKPOINT_SELECTED_DURING_IN_PROGRESS_RUN",
                            "protocol_id": protocol_id,
                            "protocol_hash": preflight["protocol_hash"],
                            "architecture_hash": architecture_hash(components.model),
                            "architecture_contract": components.model.architecture_contract,
                            "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
                            "normalization_contract": dict(NORMALIZATION_CONTRACT),
                            "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
                            "checkpoint_path": str(checkpoint_dir / "best.pt"),
                            "checkpoint_sha256": checkpoint_hash,
                            "selection_rule": BEST_RULE_ID,
                            "metrics": metrics,
                            "scheduled_iterations": step,
                            "committed_optimizer_updates": committed_optimizer_updates,
                            "amp_overflow_skips": amp_overflow_skips,
                            "test_data_used_for_selection": False,
                        }
                        _atomic_json_write(receipt, receipt_path)

                # Save after validation/best selection so history and latest
                # represent the same optimizer-step boundary.
                if iteration_committed and step % int(training["checkpoint_interval"]) == 0:
                    progress_payload = _checkpoint_payload(
                        components,
                        step,
                        validation_metrics_at_step,
                        scaler=scaler,
                        training_generator=generator,
                        training_numpy_rng=rng,
                        best_metrics=best,
                        committed_optimizer_updates=committed_optimizer_updates,
                        amp_overflow_skips=(amp_overflow_skips if use_standard_amp_policy else None),
                        consecutive_amp_overflow_skips=consecutive_amp_overflow_skips,
                    )
                    _save_committed_checkpoint_pair(
                        progress_payload,
                        checkpoint_dir,
                        event_step=step,
                        committed_optimizer_updates=committed_optimizer_updates,
                    )

        final_validation_metrics: Dict[str, float] = {}
        scheduled_iterations = int(training["total_steps"])
        if use_standard_amp_policy:
            expected_commits = scheduled_iterations - amp_overflow_skips
            skip_fraction = amp_overflow_skips / max(1, scheduled_iterations)
            if committed_optimizer_updates != expected_commits:
                raise Formal2DTrainingError(
                    "DMD3_AMP_POLICY_TEST_BLOCKED",
                    "scheduled/committed/overflow counters are inconsistent at completion",
                )
            if skip_fraction > float(amp_policy["max_skip_fraction"]):
                raise Formal2DTrainingError(
                    "DMD3_AMP_POLICY_TEST_BLOCKED",
                    f"AMP overflow skip fraction {skip_fraction:.8f} exceeds policy",
                )
        elif committed_optimizer_updates != scheduled_iterations:
            _raise_numeric_gate(
                diagnostic_dir=(
                    root / "audit" / "formal_2d_numeric_diagnostics" / protocol_id.lower()
                ),
                protocol_id=protocol_id,
                event_step=scheduled_iterations,
                committed_optimizer_updates=committed_optimizer_updates,
                phase="final_commit_budget",
                detail=(
                    f"required {scheduled_iterations} committed optimizer updates, "
                    f"observed {committed_optimizer_updates}"
                ),
                context={
                    "source_snapshot_id": config["source_snapshot_id"],
                    "config_payload_hash": config["config_payload_hash"],
                    "protocol_hash": preflight["protocol_hash"],
                },
            )
        final_nonfinite_parameters = _named_nonfinite_parameters(components.model)
        final_nonfinite_ema = _nonfinite_numeric_state(components.ema.shadow, "ema")
        final_nonfinite_optimizer = _nonfinite_optimizer_state(components.optimizer)
        final_nonfinite_scaler = _nonfinite_numeric_state(scaler.state_dict(), "scaler")
        if (
            final_nonfinite_parameters
            or final_nonfinite_ema
            or final_nonfinite_optimizer
            or final_nonfinite_scaler
        ):
            raise Formal2DTrainingError(
                "FORMAL_2D_NUMERIC_GATE_BLOCKED",
                "model/EMA/optimizer/scaler is non-finite at completion",
            )
        if history_path.is_file():
            with history_path.open("r", newline="", encoding="utf-8") as handle:
                history_rows = list(csv.DictReader(handle))
            if history_rows:
                final_validation_metrics = {
                    key: float(value) for key, value in history_rows[-1].items()
                }
        final_metrics = final_validation_metrics or best or {}
        final_payload = _checkpoint_payload(
            components,
            int(training["total_steps"]),
            final_metrics,
            scaler=scaler,
            training_generator=generator,
            training_numpy_rng=rng,
            best_metrics=best,
            committed_optimizer_updates=committed_optimizer_updates,
            amp_overflow_skips=(amp_overflow_skips if use_standard_amp_policy else None),
            consecutive_amp_overflow_skips=consecutive_amp_overflow_skips,
        )
        final_payload["metadata"]["completion_status"] = "FORMAL_TRAINING_COMPLETE"
        final_payload["metadata"]["best_validation_metric"] = dict(best or {})
        final_payload["metadata"]["final_validation_metric"] = dict(final_validation_metrics)
        final_hash = _atomic_torch_save(final_payload, checkpoint_dir / "final.pt")
        if receipt_path.is_file():
            receipt = _load_json(receipt_path)
            best_path = checkpoint_dir / "best.pt"
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            if not isinstance(best_payload, MutableMapping) or not isinstance(
                best_payload.get("metadata"), MutableMapping
            ):
                raise Formal2DTrainingError(
                    "FORMAL_2D_TRAINING_ENGINE_BLOCKED", "Best checkpoint lacks mutable metadata"
                )
            best_payload["metadata"]["completion_status"] = "FORMAL_TRAINING_COMPLETE"
            best_hash = _atomic_torch_save(best_payload, best_path)
            receipt["completion_status"] = "FORMAL_TRAINING_COMPLETE"
            receipt["checkpoint_sha256"] = best_hash
            receipt["formal_final_checkpoint_sha256"] = final_hash
            receipt["formal_final_validation_metric"] = dict(final_validation_metrics)
            receipt["scheduled_iterations"] = scheduled_iterations
            receipt["committed_optimizer_updates"] = committed_optimizer_updates
            receipt["amp_overflow_skips"] = amp_overflow_skips
            receipt["consecutive_amp_overflow_skips_at_completion"] = consecutive_amp_overflow_skips
            receipt["amp_overflow_skip_fraction"] = (
                amp_overflow_skips / max(1, scheduled_iterations)
            )
            _atomic_json_write(receipt, receipt_path)
    return {
        "status": "FORMAL_TRAINING_COMPLETE",
        "protocol_id": protocol_id,
        "total_steps": scheduled_iterations,
        "scheduled_iterations": scheduled_iterations,
        "committed_optimizer_updates": committed_optimizer_updates,
        "amp_overflow_skips": amp_overflow_skips,
        "consecutive_amp_overflow_skips": consecutive_amp_overflow_skips,
        "best_metrics": best,
        "final_checkpoint_sha256": final_hash,
        "sealed_test_runtime_access_count": 0,
    }


__all__ = [
    "BEST_RULE_ID",
    "DiffusionScheduler2D",
    "Formal2DTrainingError",
    "FormalComponents",
    "K3_LOADED_STATUS",
    "K3_WAITING_STATUS",
    "build_formal_components",
    "compute_training_losses",
    "formal_preflight_2d",
    "print_preflight_2d",
    "run_formal_training",
    "run_one_batch_smoke",
]

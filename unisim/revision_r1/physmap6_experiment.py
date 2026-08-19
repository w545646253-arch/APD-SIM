"""Fail-closed orchestration for Reviewer #1 Comment 3.

The public entrypoint is ``./R1C3_run_physmap6_strict.py``.
This module owns provenance checks and output receipts.  Formal long-running
evaluation is deliberately reachable only after every scientific preflight
gate and a finite single-sample smoke test have passed.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile
import torch

from unisim.checkpoint_contract import architecture_hash, load_checkpoint_bound
from unisim.formal_training_2d import DiffusionScheduler2D
from unisim.model2d import APDConditionedUNet2D, assert_strictly_2d_model
from unisim.protocol_runtime import require_protocol
from unisim.protocols import protocol_registry
from unisim.sim_forward_2d import (
    ABERRATION_KEYS,
    SIM2DConfig,
    forward_protocol_clean_2d,
    forward_protocol_sim_2d,
    nominal_theta_2d,
)
from .physmap6_core import RefinementConfig, masked_refine
from .physmap6_pipeline import (
    BEST_RULE_ID,
    NORMALIZATION_HASH,
    PROTOCOL_HASH,
    PROTOCOL_ID,
    RAW_ORDER,
    STAGE1_POLICY,
    load_stage1,
    make_sim_config,
    run_four_methods,
    sha_array,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "reviewer1_physmap6_strict"
CONFIG = ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
PROTOCOL_FILE = ROOT / "protocols" / "dmd_6f_2o3p.json"
SEALED_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "sealed_test_manifest.json"
CHECKPOINT = ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt"
CHECKPOINT_RECEIPT = CHECKPOINT.parent / "best_checkpoint_receipt.json"
VALIDATION_HISTORY = CHECKPOINT.parent / "validation_history.csv"
OFFICIAL_ROOT = ROOT / "outputs" / "OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
BUNDLE_MANIFEST = OFFICIAL_ROOT / "01_shared_contract" / "test30_dmd6_manifest.tsv"
BUNDLE_ROOT = OFFICIAL_ROOT / "01_shared_contract" / "test30_dmd6_bundle"
GT_MANIFEST = ROOT / "_REVISION_R1_20260812T082048Z" / "DATASET_MANIFEST.csv"
ROBUST_MANIFEST = ROOT / "_REVISION_R1_20260812T082048Z" / "PHYSMAP6_PATCH_MANIFEST.csv"
METRICS_SOURCE = ROOT / "tools" / "official_r2_common_metrics.py"
COMMENT2_FRC_SOURCE = ROOT / "_REVISION_R1_20260812T082048Z" / "code" / "r1_metrics.py"
TRAINING_BUDGET_AUDIT = (
    ROOT / "audit" / "dmd3_nonfinite_recovery_20260814_010345"
    / "01_completed_models_audit" / "dmd6_audit.json"
)

EXPECTED = {
    "protocol_file_sha256": "dfb992d38a8c3b029d8e1d1d1b7223fc79118ec2d8d314d1e3d5c1d0211fb0d6",
    "protocol_hash": PROTOCOL_HASH,
    "registry_hash": "5186ebd2a17c5e39ccf486f3e7b61fb3cf7f86c907c9460740fbc23385fa2968",
    "sealed_manifest_sha256": "495b554a19596b299f1bc5192ee3b1eb071414fde361c2f2eae0c17f2878d794",
    "checkpoint_sha256": "10fb16662a8b71b877f2cab81bdc151dcded92f6efd1c4b006306b901a8adff7",
    "checkpoint_receipt_sha256": "c1292164356784149539c919f655f3881c59b5887a97886e35b0b0d3632b2373",
    "validation_history_sha256": "e0d989b8d7c4901d39f22d6cdf1646ce62573744c308dbb82ecd8e1c12d0d9ae",
    "metrics_sha256": "9efd7efcc6ecf126816887a710478f592ecc3b29562003a2ea452e1b93deec9a",
    "comment2_frc_sha256": "b9b2e7fb3064c6adf4e3d1de545ce7b7195674dcc196d6fd49df657ef7830e7b",
    "robust_manifest_sha256": "4ac05467274e3f6076c86a3b5a54f1c0ab5aece1c7412332521adabefaf793e4",
    "training_budget_audit_sha256": "6eb3cdf353a7dc3a7ff345bc397b29897587605ddd4bb770f85fec602b808671",
}
PROJECT_TEST_FILES = tuple(sorted((ROOT / "tests").glob("test_*.py")))
IMPLEMENTATION_FILES = (
    ROOT / "R1C3_run_physmap6_strict.py",
    ROOT / "unisim" / "revision_r1" / "physmap6_core.py",
    ROOT / "unisim" / "revision_r1" / "physmap6_pipeline.py",
    ROOT / "unisim" / "revision_r1" / "physmap6_experiment.py",
    ROOT / "unisim" / "revision_r1" / "physmap6_reporting.py",
    ROOT / "unisim" / "sim_forward_2d.py",
    ROOT / "unisim" / "formal_training_2d.py",
    ROOT / "unisim" / "model2d.py",
    ROOT / "unisim" / "checkpoint_contract.py",
    ROOT / "unisim" / "protocol_runtime.py",
    ROOT / "unisim" / "protocols.py",
) + PROJECT_TEST_FILES
VALIDITY_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
METHODS = ("WF", "DiffWS-6", "PhysMap-6", "APD-SIM-6")
OFFICIAL_STAGE1_POLICY_ID = "APD6_OFFICIAL_R2_WARMSTART_MAP_V1"
LEGACY_ROBUSTNESS_SEED = 20260812
ROBUSTNESS_FACTORS: dict[str, list[float]] = {
    "kxy_mismatch": [0.0, 0.05, 0.1, 0.15],
    "photon_scale_mul": [1.0, 0.5, 0.25, 0.125],
    "read_noise_mul": [1.0, 2.0, 4.0, 8.0],
    "background_add": [0.0, 0.01, 0.02, 0.05],
    "psf_blur": [0.0, 0.1, 0.2, 0.3],
    "mod_depth_drop": [0.0, 0.1, 0.2, 0.3],
    "phase_jitter_rad": [0.0, 0.1, 0.2, 0.4, 0.6],
    "angle_jitter_deg": [0.0, 0.5, 1.0, 2.0, 3.0],
    "aberr_defocus": [0.0, 0.025, 0.05, 0.075, 0.1],
    "aberr_astig_x": [0.0, 0.025, 0.05, 0.075, 0.1],
    "aberr_coma_x": [0.0, 0.025, 0.05, 0.075, 0.1],
    "aberr_spherical": [0.0, 0.025, 0.05, 0.075, 0.1],
}
REFINEMENT_CONFIG = RefinementConfig()


class R1C3Blocked(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_patch_sha_array(array: np.ndarray) -> str:
    """Verify the immutable legacy sample-list tensor receipt only.

    The prior fixed-patch manifest used ``dtype`` and ``shape`` string
    prefixes without separators.  New R1C3 artifacts use ``sha_array``'s
    explicit canonical header; this compatibility helper never hashes new
    formal measurements.
    """
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"JSON object required: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        name = utc_timestamp() if suffix == 0 else f"{utc_timestamp()}_{suffix:03d}"
        path = OUTPUT_BASE / name
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique UTC output directory")


def _parse_pmon_output(payload: bytes) -> tuple[list[dict[str, Any]], int]:
    """Strictly parse one pmon sample.

    Some Windows/WDDM invocations occasionally emit a single printable
    fragment (observed: ``R``) after truncating a process-name row.  Such a
    sample is unusable: callers must discard the *entire* sample and retry.
    It is never accepted as evidence that the GPU was contention-free.
    """
    text = payload.decode("utf-8", errors="replace")
    pure_compute: list[dict[str, Any]] = []
    parsed_rows = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1 and set(parts[0]) == {"\ufffd"}:
            continue
        if len(parts) < 3:
            raise ValueError(f"unparseable nvidia-smi pmon row: {line!r}")
        try:
            gpu_index, pid = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid pmon GPU/PID row: {line!r}") from exc
        process_type = parts[2]
        if process_type not in {"C", "G", "C+G"}:
            raise ValueError(f"unknown pmon process type: {line!r}")
        parsed_rows += 1
        if gpu_index == 0 and process_type == "C" and pid != os.getpid():
            pure_compute.append({"pid": pid, "command": parts[-1]})
    return pure_compute, parsed_rows


def assert_no_external_cuda_compute() -> dict[str, Any]:
    """Fail closed when another pure-compute process occupies GPU 0.

    Windows/WDDM lists desktop applications as ``C+G``; those are recorded
    but do not trigger this gate.  A pure ``C`` row always blocks the formal
    run because timing and peak-memory results would no longer be comparable.
    """
    samples: list[dict[str, Any]] = []
    pure_compute: list[dict[str, Any]] = []
    parsed_rows = 0
    for attempt in range(1, 6):
        try:
            completed = subprocess.run(
                ["nvidia-smi", "pmon", "-i", "0", "-c", "1"],
                check=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise R1C3Blocked(
                "R1C3_FORMAL_EVALUATION_INCOMPLETE", f"GPU contention gate failed closed: {exc}"
            ) from exc
        sample_sha = hashlib.sha256(completed.stdout).hexdigest()
        try:
            current_compute, current_rows = _parse_pmon_output(completed.stdout)
        except ValueError as exc:
            samples.append({"attempt": attempt, "status": "DISCARDED_UNPARSEABLE", "sha256": sample_sha})
            if attempt == 5:
                raise R1C3Blocked(
                    "R1C3_FORMAL_EVALUATION_INCOMPLETE",
                    f"GPU contention gate had no parseable sample after 5 attempts: {exc}",
                ) from exc
            continue
        samples.append({"attempt": attempt, "status": "PARSED", "sha256": sample_sha})
        pure_compute = current_compute
        parsed_rows = current_rows
        break
    if pure_compute:
        raise R1C3Blocked(
            "R1C3_FORMAL_EVALUATION_INCOMPLETE",
            f"external pure-C CUDA process detected: {pure_compute}",
        )
    return {
        "status": "PASS",
        "gpu_index": 0,
        "sample_kind": "nvidia-smi pmon point-in-time gate",
        "parsed_process_rows": parsed_rows,
        "external_pure_compute": pure_compute,
        "sample_attempts": samples,
        "accepted_raw_output_sha256": samples[-1]["sha256"],
    }


def assert_implementation_unchanged(frozen: Mapping[str, str]) -> None:
    current = {str(path): sha_file(path) for path in IMPLEMENTATION_FILES}
    if current != dict(frozen):
        changed = sorted(path for path in set(current) | set(frozen) if current.get(path) != frozen.get(path))
        raise R1C3Blocked(
            "R1C3_FORMAL_EVALUATION_INCOMPLETE",
            f"implementation source changed after preflight: {changed}",
        )


def normalization(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"GT must be 2-D, got {value.shape}")
    low, high = np.percentile(value, [0.5, 99.5])
    return np.clip((value - low) / (high - low + 1e-8), 0.0, 1.0).astype(np.float32)


def protocol_receipt() -> dict[str, Any]:
    if sha_file(PROTOCOL_FILE) != EXPECTED["protocol_file_sha256"]:
        raise R1C3Blocked("R1C3_PROTOCOL_UNRESOLVED", "protocol file SHA changed")
    spec = require_protocol(PROTOCOL_ID)
    checks = {
        "protocol_id": spec.protocol_id == PROTOCOL_ID,
        "protocol_hash": spec.protocol_hash == PROTOCOL_HASH,
        "registry_hash": protocol_registry.registry_hash == EXPECTED["registry_hash"],
        "frame_count": spec.frame_count == 6,
        "orientation_count": spec.orientation_count == 2,
        "phases_per_orientation": spec.phases_per_orientation == 3,
        "raw_frame_order": tuple(spec.raw_frame_order) == RAW_ORDER,
        "raw_to_slot_mapping": tuple(spec.raw_to_slot_mapping) == tuple(range(6)),
        "validity_mask": tuple(spec.validity_mask) == VALIDITY_MASK,
        "orientation_angles": tuple(spec.orientation_angles) == (90.0, 0.0),
        "nominal_phases": np.allclose(spec.nominal_phase_values, (0.0, 2 * math.pi / 3, 4 * math.pi / 3)),
    }
    if not all(checks.values()):
        raise R1C3Blocked("R1C3_PROTOCOL_UNRESOLVED", f"protocol checks failed: {checks}")
    source = read_json(PROTOCOL_FILE)
    return {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "protocol_file": str(PROTOCOL_FILE),
        "protocol_file_sha256": sha_file(PROTOCOL_FILE),
        "protocol_registry_hash": protocol_registry.registry_hash,
        "raw_frame_order": list(RAW_ORDER),
        "raw_to_slot_mapping": list(spec.raw_to_slot_mapping),
        "validity_mask_15_slots": list(VALIDITY_MASK),
        "prompt_3x3_projection": [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
        "orientation_angles_degree_mod_180": list(spec.orientation_angles),
        "nominal_phase_values_radian": list(spec.nominal_phase_values),
        "evidence_level": source.get("evidence_level"),
        "claim_level": source.get("claim_level"),
        "historical_acquisition_receipt": source.get("historical_acquisition_receipt"),
        "scope_limitation": "synthetic controller-defined nominal DMD geometry; not a historical acquisition receipt",
        "checks": checks,
    }


def _best_history_row() -> dict[str, str]:
    with VALIDATION_HISTORY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "validation history count changed")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["mean_val_total_loss"]),
            -float(row["mean_val_x0_psnr"]),
            -float(row["mean_val_x0_ssim"]),
            float(row["global_step"]),
        ),
    )
    return ordered[0]


def _optimizer_committed_step(payload: Mapping[str, Any]) -> int:
    """Recover the unique committed AdamW update count from checkpoint state.

    ``global_step`` in the historical checkpoint records a loop/data event,
    whereas AMP may skip an optimizer commit.  The distinction is material to
    the manuscript wording even though the selected EMA remains usable.
    """
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise R1C3Blocked(
            "R1C3_APD6_CHECKPOINT_UNRESOLVED", "optimizer state is absent from best.pt"
        )
    steps: list[int] = []
    for state in optimizer["state"].values():
        if not isinstance(state, Mapping) or "step" not in state:
            continue
        value = state["step"]
        if torch.is_tensor(value):
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise R1C3Blocked(
                    "R1C3_APD6_CHECKPOINT_UNRESOLVED", "invalid optimizer step tensor"
                )
            number = float(value.detach().cpu().item())
        else:
            number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise R1C3Blocked(
                "R1C3_APD6_CHECKPOINT_UNRESOLVED", "non-integral optimizer committed step"
            )
        steps.append(int(number))
    if len(steps) != 270 or len(set(steps)) != 1:
        raise R1C3Blocked(
            "R1C3_APD6_CHECKPOINT_UNRESOLVED",
            f"optimizer committed-step identity is not unique across 270 tensors: {len(steps)}",
        )
    return steps[0]


def checkpoint_receipt(config: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if sha_file(CHECKPOINT) != EXPECTED["checkpoint_sha256"]:
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "best.pt SHA mismatch")
    if sha_file(CHECKPOINT_RECEIPT) != EXPECTED["checkpoint_receipt_sha256"]:
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "selection receipt SHA mismatch")
    if sha_file(VALIDATION_HISTORY) != EXPECTED["validation_history_sha256"]:
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "validation history SHA mismatch")
    receipt = read_json(CHECKPOINT_RECEIPT)
    best = _best_history_row()
    if (
        receipt.get("completion_status") != "FORMAL_TRAINING_COMPLETE"
        or receipt.get("test_data_used_for_selection") is not False
        or receipt.get("selection_rule") != BEST_RULE_ID
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("protocol_hash") != PROTOCOL_HASH
        or receipt.get("checkpoint_sha256") != EXPECTED["checkpoint_sha256"]
        or int(float(best["global_step"])) != int(receipt["metrics"]["global_step"])
    ):
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "validation-only identity mismatch")
    model_cfg = config["model"]
    model = APDConditionedUNet2D(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        dropout=float(model_cfg["dropout"]),
        time_dim=int(model_cfg["time_dim"]),
        groups=int(model_cfg["groups"]),
    )
    assert_strictly_2d_model(model)
    expected_identities = {
        "training_protocol_id": PROTOCOL_ID,
        "training_protocol_hash": PROTOCOL_HASH,
        "architecture_hash": architecture_hash(model),
        "architecture_contract": model.architecture_contract,
        "input_tensor_dimensionality": "4D_BCHW",
        "normalization_contract_hash": NORMALIZATION_HASH,
        "source_snapshot_id": config["source_snapshot_id"],
        "train_manifest_hash": config["train_manifest_hash"],
        "validation_manifest_hash": config["validation_manifest_hash"],
        "sealed_test_no_access_hash": config["sealed_test_manifest_hash"],
        "validation_bundle_hash": config["validation_bundle_hash"],
        "training_config_hash": config["config_payload_hash"],
        "training_seed": int(config["training"]["seed"]),
        "checkpoint_selection_rule": BEST_RULE_ID,
        "completion_status": "FORMAL_TRAINING_COMPLETE",
    }
    payload = load_checkpoint_bound(
        CHECKPOINT,
        protocol=PROTOCOL_ID,
        expected_sha256=EXPECTED["checkpoint_sha256"],
        expected_identities=expected_identities,
    )
    if sha_file(TRAINING_BUDGET_AUDIT) != EXPECTED["training_budget_audit_sha256"]:
        raise R1C3Blocked(
            "R1C3_APD6_CHECKPOINT_UNRESOLVED", "training-budget audit SHA mismatch"
        )
    budget_audit = read_json(TRAINING_BUDGET_AUDIT)
    budget_best = budget_audit.get("checkpoints", {}).get("best", {})
    committed_step = _optimizer_committed_step(payload)
    event_step = int(float(best["global_step"]))
    if (
        budget_best.get("sha256") != EXPECTED["checkpoint_sha256"]
        or int(budget_best.get("global_step", -1)) != event_step
        or int(budget_best.get("optimizer_committed_step", -1)) != committed_step
        or committed_step != 95956
        or event_step - committed_step != 44
    ):
        raise R1C3Blocked(
            "R1C3_APD6_CHECKPOINT_UNRESOLVED", "training-budget semantics audit mismatch"
        )
    counts: dict[str, int] = {}
    finite: dict[str, bool] = {}
    for state_name in ("model", "ema"):
        state = payload.get(state_name)
        if not isinstance(state, Mapping):
            raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", f"{state_name} absent")
        tensors = [value for value in state.values() if torch.is_tensor(value)]
        counts[state_name] = len(tensors)
        finite[state_name] = all(bool(torch.isfinite(value).all()) for value in tensors)
    if counts != {"model": 270, "ema": 270} or not all(finite.values()):
        raise R1C3Blocked("R1C3_APD6_CHECKPOINT_UNRESOLVED", "checkpoint parameters invalid")
    generated = {
        "schema_version": 1,
        "status": "PASS",
        "checkpoint_absolute_path": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED["checkpoint_sha256"],
        "selected_step": event_step,
        "selected_step_semantics": "loop/data event step; not the number of committed optimizer updates",
        "optimizer_committed_updates_at_selected_checkpoint": committed_step,
        "global_minus_optimizer_commits": event_step - committed_step,
        "training_budget_semantics_status": (
            "P1_DISCLOSED; checkpoint identity, finiteness, EMA inference, and "
            "validation-only selection remain valid"
        ),
        "training_budget_audit": str(TRAINING_BUDGET_AUDIT),
        "training_budget_audit_sha256": sha_file(TRAINING_BUDGET_AUDIT),
        "validation_metric_name": "mean_val_total_loss",
        "validation_metric_value": float(best["mean_val_total_loss"]),
        "validation_psnr": float(best["mean_val_x0_psnr"]),
        "validation_ssim": float(best["mean_val_x0_ssim"]),
        "validation_entry_count": int(float(best["validation_entry_count"])),
        "selection_rule": BEST_RULE_ID,
        "test_data_used_for_selection": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "model_architecture_identity": model.architecture_contract,
        "model_architecture_hash": architecture_hash(model),
        "model_parameter_tensors": counts["model"],
        "ema_parameter_tensors": counts["ema"],
        "all_model_parameters_finite": finite["model"],
        "all_ema_parameters_finite": finite["ema"],
        "source_selection_receipt": str(CHECKPOINT_RECEIPT),
        "source_selection_receipt_sha256": sha_file(CHECKPOINT_RECEIPT),
        "validation_history": str(VALIDATION_HISTORY),
        "validation_history_sha256": sha_file(VALIDATION_HISTORY),
    }
    return generated, payload["metadata"]


def load_bundle_rows(*, verify_payloads: bool) -> list[dict[str, str]]:
    with BUNDLE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 30 or [int(row["order"]) for row in rows] != list(range(30)):
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "test30 manifest ordering changed")
    if {row["structure_class"] for row in rows} != {"CCP", "ER", "MT"}:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "test class labels changed")
    for row in rows:
        path = BUNDLE_MANIFEST.parent / row["npz_path"]
        if not path.is_file() or sha_file(path) != row["npz_sha256"]:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"bundle drift: {path}")
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["protocol_hash"] != PROTOCOL_HASH
            or tuple(row["frame_order"].split("/")) != RAW_ORDER
            or row["test_gt_embedded_in_npz"].lower() != "false"
        ):
            raise R1C3Blocked("R1C3_PROTOCOL_UNRESOLVED", f"bundle protocol drift: {row['sample_id']}")
        if verify_payloads:
            with np.load(path, allow_pickle=False) as archive:
                if any("gt" in key.lower() for key in archive.files):
                    raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "GT leaked into raw bundle")
                raw = np.asarray(archive["raw_stack"], dtype=np.float32)
                if raw.shape != (6, 1004, 1004) or sha_array(raw) != row["raw_stack_sha256"]:
                    raise R1C3Blocked("R1C3_INPUT_IDENTITY_MISMATCH", row["sample_id"])
    return rows


def load_gt_mapping() -> dict[str, dict[str, str]]:
    with GT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == "30-FOV GT benchmark"]
    if len(rows) != 30:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "GT mapping is not 30 rows")
    result = {row["sample_id"]: row for row in rows}
    for sample, row in result.items():
        path = Path(row["absolute_path"])
        if not path.is_file() or sha_file(path) != row["sha256"]:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"GT hash drift: {sample}")
    return result


def load_robust_rows() -> list[dict[str, str]]:
    if sha_file(ROBUST_MANIFEST) != EXPECTED["robust_manifest_sha256"]:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "robustness sample receipt changed")
    with ROBUST_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or [int(row["sample_order"]) for row in rows] != list(range(20)):
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "robustness manifest is not fixed 20")
    for row in rows:
        path = Path(row["absolute_path"])
        if not path.is_file() or sha_file(path) != row["file_sha256"]:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"robustness GT drift: {path}")
        source = tifffile.imread(path)
        source = source[0] if source.ndim == 3 else source
        if source.shape != (int(row["source_height"]), int(row["source_width"])):
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"robustness GT shape drift: {path}")
        if row["normalization"] != "full-source percentile 0.5/99.5 then fixed crop":
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "robustness normalization drift")
        y, x = int(row["crop_y"]), int(row["crop_x"])
        height, width = int(row["crop_height"]), int(row["crop_width"])
        if height != 320 or width != 320 or y < 0 or x < 0 or y + height > source.shape[0] or x + width > source.shape[1]:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "robustness crop drift")
        patch = np.ascontiguousarray(normalization(source)[y : y + height, x : x + width])
        if legacy_patch_sha_array(patch) != row["gt_patch_sha256"]:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"robustness patch hash drift: {path}")
    return rows


def _metrics_module() -> Any:
    if sha_file(METRICS_SOURCE) != EXPECTED["metrics_sha256"]:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "official FRC/metric source changed")
    spec = importlib.util.spec_from_file_location("r1c3_official_metrics", METRICS_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load official metrics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comment2_frc_module() -> Any:
    """Load the exact Reviewer Comment 2 FRC implementation by frozen SHA."""
    if sha_file(COMMENT2_FRC_SOURCE) != EXPECTED["comment2_frc_sha256"]:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "Comment 2 FRC source changed")
    spec = importlib.util.spec_from_file_location("r1c3_comment2_frc", COMMENT2_FRC_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Comment 2 FRC")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _theta_from_archive(archive: Any, device: torch.device) -> dict[str, torch.Tensor]:
    values = dict(zip((str(item) for item in archive["theta_fields"].tolist()), archive["theta_values"].tolist()))
    theta = {
        "k_ratio_xy": torch.tensor([values["k_ratio_xy"]], device=device),
        "mod_depth": torch.tensor([values["mod_depth"]], device=device),
        "phase_offsets": torch.tensor([values["phase_offset_rad"]], device=device),
        "angle_offsets": torch.tensor([values["angle_offset_deg"]], device=device),
        "background": torch.tensor([values["background"]], device=device),
        "psf_sigma_scale": torch.tensor([values["psf_sigma_scale"]], device=device),
        "photon_scale": torch.tensor([values["photon_scale"]], device=device),
        "read_noise_e": torch.tensor([values["read_noise_e"]], device=device),
    }
    for key in ABERRATION_KEYS:
        theta[key] = torch.zeros((1,), device=device)
    return theta


def _clone_theta(theta: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in theta.items()}


def perturb_theta(
    nominal: Mapping[str, torch.Tensor],
    factor: str,
    severity: float,
    *,
    sample_order: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply one preregistered perturbation with a fixed sample-wise direction."""
    theta = _clone_theta(nominal)
    level = float(severity)
    seed = (
        LEGACY_ROBUSTNESS_SEED
        + int(sample_order) * 1009
        + (int(hashlib.sha256(factor.encode("utf-8")).hexdigest()[:8], 16) % 100000)
    )
    rng = np.random.default_rng(seed)
    if factor == "kxy_mismatch":
        theta["k_ratio_xy"] = theta["k_ratio_xy"] * (1.0 - level)
    elif factor == "photon_scale_mul":
        theta["photon_scale"] = theta["photon_scale"] * level
    elif factor == "read_noise_mul":
        theta["read_noise_e"] = theta["read_noise_e"] * level
    elif factor == "background_add":
        theta["background"] = theta["background"] + level
    elif factor == "psf_blur":
        theta["psf_sigma_scale"] = theta["psf_sigma_scale"] * (1.0 + level)
    elif factor == "mod_depth_drop":
        theta["mod_depth"] = (theta["mod_depth"] * (1.0 - level)).clamp_min(0.0)
    elif factor == "phase_jitter_rad":
        vector = rng.normal(size=6)
        vector -= vector.mean()
        vector /= np.max(np.abs(vector)) + 1e-12
        theta["phase_offsets"] = torch.tensor(vector * level, device=device, dtype=torch.float32)
    elif factor == "angle_jitter_deg":
        vector = rng.normal(size=2)
        vector -= vector.mean()
        vector /= np.max(np.abs(vector)) + 1e-12
        theta["angle_offsets"] = torch.tensor(vector * level, device=device, dtype=torch.float32)
    elif factor in {"aberr_defocus", "aberr_astig_x", "aberr_coma_x", "aberr_spherical"}:
        theta[factor] = torch.tensor([level], device=device, dtype=torch.float32)
    else:
        raise KeyError(factor)
    return theta


def inverse_theta_for_robustness(
    true_theta: Mapping[str, torch.Tensor], nominal: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Freeze the historical non-oracle inverse contract with known background."""
    inverse = _clone_theta(nominal)
    inverse["background"] = true_theta["background"].detach().clone()
    return inverse


def _measurement_seed(factor: str, sample_order: int) -> int:
    return (
        LEGACY_ROBUSTNESS_SEED
        + int(hashlib.sha256(factor.encode("utf-8")).hexdigest()[:8], 16)
        + int(sample_order) * 131
    ) % 2147483647


def _diffusion_seed(label: str, sample_order: int) -> int:
    if label == "robustness":
        return (LEGACY_ROBUSTNESS_SEED + int(sample_order) * 17) % 2147483647
    digest = hashlib.sha256(f"R1C3_DIFFUSION|{label}|{sample_order}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _official_nominal_diffusion_seed(raw_hash: str) -> int:
    label = (
        f"{OFFICIAL_STAGE1_POLICY_ID}|{EXPECTED['checkpoint_sha256']}|{raw_hash}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") & ((1 << 63) - 1)


def _geometry_receipt(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "raw_frame_order": list(RAW_ORDER),
        "validity_mask": list(VALIDITY_MASK),
        "orientation_angles": protocol["orientation_angles_degree_mod_180"],
        "nominal_phase_values": protocol["nominal_phase_values_radian"],
    }


def smoke_test(
    rows: Sequence[Mapping[str, str]],
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    row = rows[0]
    path = BUNDLE_MANIFEST.parent / row["npz_path"]
    with np.load(path, allow_pickle=False) as archive:
        raw = torch.from_numpy(np.asarray(archive["raw_stack"], dtype=np.float32))[None].to(device)
        theta = _theta_from_archive(archive, device)
    result = run_four_methods(
        raw,
        model,
        scheduler,
        sim_config,
        theta,
        diffusion_seed=2026081401,
        refinement_config=REFINEMENT_CONFIG,
        geometry_receipt=geometry,
    )
    checks = {
        "same_refinement_function": result["shared_refinement_function"].endswith("masked_refine"),
        "same_refinement_config_object": isinstance(result["shared_refinement_config_object_id"], int),
        "exact_method_set": set(METHODS).issubset(result),
        "all_finite": all(bool(torch.isfinite(result[method]["image"]).all()) for method in METHODS),
        "all_shapes_equal": len({tuple(result[method]["image"].shape) for method in METHODS}) == 1,
        "physmap_history_length_41": len(result["PhysMap-6"]["refinement"].objective_history) == 41,
        "apd_history_length_41": len(result["APD-SIM-6"]["refinement"].objective_history) == 41,
        "config_receipts_equal": result["PhysMap-6"]["refinement"].configuration_receipt
        == result["APD-SIM-6"]["refinement"].configuration_receipt,
    }
    if not all(checks.values()):
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"smoke failed: {checks}")
    return {
        "status": "PASS",
        "sample_id": row["sample_id"],
        "raw_stack_sha256": result["raw_stack_sha256"],
        "checks": checks,
        "refinement_config_receipt": REFINEMENT_CONFIG.receipt(),
    }


NOMINAL_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "method",
    "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
    "forward_parameters_sha256", "normalization_sha256", "gt_identity_sha256",
    "noise_seed", "diffusion_seed", "refinement_config_sha256", "psnr", "ssim",
    "frc_status", "frc_cutoff_cycles_per_pixel", "frc_spatial_period_px",
    "observed_nrmse", "poisson_gaussian_objective", "runtime_seconds",
    "peak_gpu_memory_bytes", "gradient_finite", "output_finite", "prediction_sha256",
)


ROBUST_FIELDS = (
    "factor", "severity", "sample_order", "sample_id", "parent_id", "structure", "method",
    "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256", "forward_parameters_sha256",
    "normalization_sha256", "gt_identity_sha256", "noise_seed", "diffusion_seed",
    "refinement_config_sha256", "theta_true_json", "theta_inverse_json", "psnr", "ssim",
    "observed_nrmse", "poisson_gaussian_objective", "runtime_seconds",
    "peak_gpu_memory_bytes", "gradient_finite", "output_finite", "prediction_sha256", "status",
)


RUNTIME_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "repeat_index", "method", "component",
    "measurement_kind", "warmup_runs_before_measurement", "raw_stack_sha256", "validity_mask_sha256",
    "geometry_sha256", "forward_parameters_sha256", "normalization_sha256", "noise_seed",
    "diffusion_seed", "refinement_config_sha256", "runtime_seconds", "peak_gpu_memory_bytes",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _tensor_theta_json(theta: Mapping[str, torch.Tensor]) -> str:
    payload = {
        key: [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        for key, value in sorted(theta.items())
    }
    return canonical_bytes(payload).decode("utf-8")


def _method_payload(method: str, result: Mapping[str, Any]) -> tuple[torch.Tensor, float, int, float | None, float | None, bool, bool]:
    item = result[method]
    if method in {"PhysMap-6", "APD-SIM-6"}:
        refinement = item["refinement"]
        return (
            item["image"],
            float(refinement.runtime_seconds),
            int(refinement.peak_gpu_memory_bytes),
            float(refinement.objective_history[-1]),
            float(refinement.observed_nrmse_history[-1]),
            bool(refinement.gradient_finite),
            bool(refinement.output_finite),
        )
    return (
        item["image"],
        float(item["runtime_seconds"]),
        int(item["peak_gpu_memory_bytes"]),
        float(item["observed_fit"].objective),
        float(item["observed_fit"].observed_nrmse),
        True,
        bool(torch.isfinite(item["image"]).all()),
    )


def _late_gt(gt_row: Mapping[str, str]) -> np.ndarray:
    return normalization(tifffile.imread(Path(gt_row["absolute_path"])))


def _compute_metrics(gt: np.ndarray, prediction: np.ndarray, metrics: Any) -> tuple[float, float, dict[str, Any]]:
    psnr = float(metrics.psnr_native(gt, prediction))
    ssim = float(metrics.ssim_native(gt, prediction))
    frc, _curves = _comment2_frc_module().reference_frc_1over7(gt, prediction)
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise R1C3Blocked("R1C3_NONFINITE_RESULT", "non-finite nominal PSNR/SSIM")
    return psnr, ssim, frc


def nominal_evaluation(
    run_dir: Path,
    bundle_rows: Sequence[Mapping[str, str]],
    gt_mapping: Mapping[str, Mapping[str, str]],
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metrics = _metrics_module()
    mask_hash = sha_array(np.asarray(VALIDITY_MASK, dtype=np.float32))
    geometry_hash = _json_hash(geometry)
    normalization_hash = NORMALIZATION_HASH
    config_hash = REFINEMENT_CONFIG.receipt()["config_sha256"]
    output_rows: list[dict[str, Any]] = []
    prediction_dir = run_dir / "nominal_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for sample_order, row in enumerate(bundle_rows):
        assert_no_external_cuda_compute()
        bundle_path = BUNDLE_MANIFEST.parent / row["npz_path"]
        with np.load(bundle_path, allow_pickle=False) as archive:
            raw_np = np.asarray(archive["raw_stack"], dtype=np.float32)
            raw = torch.from_numpy(raw_np)[None].to(device)
            theta = _theta_from_archive(archive, device)
            noise_seed = int(np.asarray(archive["acquisition_noise_seed"]).reshape(-1)[0])
        diffusion_seed = _official_nominal_diffusion_seed(row["raw_stack_sha256"])
        result = run_four_methods(
            raw, model, scheduler, sim_config, theta,
            diffusion_seed=diffusion_seed,
            refinement_config=REFINEMENT_CONFIG,
            geometry_receipt=geometry,
        )
        if result["raw_stack_sha256"] != row["raw_stack_sha256"]:
            raise R1C3Blocked("R1C3_INPUT_IDENTITY_MISMATCH", row["sample_id"])
        gt_row = gt_mapping[row["sample_id"]]
        gt = _late_gt(gt_row)
        forward_hash = _json_hash({"theta": _tensor_theta_json(theta), "protocol_hash": PROTOCOL_HASH})
        for method in METHODS:
            image, runtime, peak, objective, nrmse, grad_finite, output_finite = _method_payload(method, result)
            prediction = np.ascontiguousarray(image[0, 0].detach().cpu().numpy(), dtype=np.float32)
            psnr, ssim, frc = _compute_metrics(gt, prediction, metrics)
            pred_path = prediction_dir / f"{sample_order:03d}_{row['sample_id']}_{method.replace(' ', '_')}.npy"
            stream = io.BytesIO(); np.save(stream, prediction, allow_pickle=False); atomic_write(pred_path, stream.getvalue())
            frc_status = "CUTOFF" if frc["cutoff_cycles_per_pixel"] is not None else (
                "RIGHT_CENSORED" if frc["right_censored_at_nyquist"] else "UNRESOLVED"
            )
            output_rows.append({
                "sample_order": sample_order, "sample_id": row["sample_id"], "parent_id": row["parent_id"],
                "structure": row["structure_class"], "method": method,
                "raw_stack_sha256": row["raw_stack_sha256"], "validity_mask_sha256": mask_hash,
                "geometry_sha256": geometry_hash, "forward_parameters_sha256": forward_hash,
                "normalization_sha256": normalization_hash, "gt_identity_sha256": gt_row["sha256"],
                "noise_seed": noise_seed, "diffusion_seed": diffusion_seed,
                "refinement_config_sha256": config_hash if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                "psnr": psnr, "ssim": ssim, "frc_status": frc_status,
                "frc_cutoff_cycles_per_pixel": frc["cutoff_cycles_per_pixel"] if frc["cutoff_cycles_per_pixel"] is not None else "",
                "frc_spatial_period_px": frc["cutoff_derived_spatial_period_px"] if frc["cutoff_derived_spatial_period_px"] is not None else "",
                "observed_nrmse": nrmse if nrmse is not None else "",
                "poisson_gaussian_objective": objective if objective is not None else "",
                "runtime_seconds": runtime, "peak_gpu_memory_bytes": peak,
                "gradient_finite": grad_finite, "output_finite": output_finite,
                "prediction_sha256": sha_array(prediction),
            })
        case_rows = output_rows[-len(METHODS):]
        for field in (
            "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
            "forward_parameters_sha256", "normalization_sha256", "gt_identity_sha256",
            "noise_seed", "diffusion_seed",
        ):
            if len({str(item[field]) for item in case_rows}) != 1:
                raise R1C3Blocked(
                    "R1C3_INPUT_IDENTITY_MISMATCH", f"{row['sample_id']}:{field}"
                )
        atomic_write(run_dir / "R1C3_NOMINAL_PER_FOV.csv", _csv_bytes(output_rows, NOMINAL_FIELDS))
    if len(output_rows) != 120:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "nominal grid incomplete")
    return output_rows


def robustness_evaluation(
    run_dir: Path,
    sample_rows: Sequence[Mapping[str, str]],
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metrics = _metrics_module()
    mask_hash = sha_array(np.asarray(VALIDITY_MASK, dtype=np.float32))
    geometry_hash = _json_hash(geometry)
    config_hash = REFINEMENT_CONFIG.receipt()["config_sha256"]
    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    visual_dir = run_dir / "robustness_visual_arrays"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_levels = {
        "phase_jitter_rad": {0.1, 0.4, 0.6},
        "psf_blur": {0.1, 0.2, 0.3},
        "photon_scale_mul": {0.5, 0.25, 0.125},
    }
    nominal = nominal_theta_2d(sim_config, device)
    for factor, levels in ROBUSTNESS_FACTORS.items():
        for severity in levels:
            for sample in sample_rows:
                assert_no_external_cuda_compute()
                order = int(sample["sample_order"])
                source = normalization(tifffile.imread(Path(sample["absolute_path"])))
                y, x = int(sample["crop_y"]), int(sample["crop_x"])
                gt = np.ascontiguousarray(source[y : y + 320, x : x + 320], dtype=np.float32)
                gt_tensor = torch.from_numpy(gt)[None, None].to(device)
                theta_true = perturb_theta(nominal, factor, severity, sample_order=order, device=device)
                theta_inverse = inverse_theta_for_robustness(theta_true, nominal)
                noise_seed = _measurement_seed(factor, order)
                generator = torch.Generator(device=device).manual_seed(noise_seed)
                raw, _ = forward_protocol_sim_2d(
                    gt_tensor, sim_config, PROTOCOL_ID, theta=theta_true, randomize=False,
                    noise_generator=generator,
                )
                diffusion_seed = _diffusion_seed("robustness", order)
                try:
                    result = run_four_methods(
                        raw, model, scheduler, sim_config, theta_inverse,
                        diffusion_seed=diffusion_seed, refinement_config=REFINEMENT_CONFIG,
                        geometry_receipt=geometry,
                    )
                    forward_hash = _json_hash({"true": _tensor_theta_json(theta_true), "inverse": _tensor_theta_json(theta_inverse)})
                    case_rows: list[dict[str, Any]] = []
                    for method in METHODS:
                        image, runtime, peak, objective, nrmse, grad_finite, output_finite = _method_payload(method, result)
                        prediction = np.ascontiguousarray(image[0, 0].detach().cpu().numpy(), dtype=np.float32)
                        psnr = float(metrics.psnr_native(gt, prediction)); ssim = float(metrics.ssim_native(gt, prediction))
                        if not math.isfinite(psnr) or not math.isfinite(ssim):
                            raise R1C3Blocked("R1C3_NONFINITE_RESULT", f"{factor}/{severity}/{sample['sample_id']}/{method}")
                        case_rows.append({
                            "factor": factor, "severity": severity, "sample_order": order,
                            "sample_id": sample["sample_id"], "parent_id": sample["sample_id"],
                            "structure": sample["structure"], "method": method,
                            "raw_stack_sha256": result["raw_stack_sha256"], "validity_mask_sha256": mask_hash,
                            "geometry_sha256": geometry_hash, "forward_parameters_sha256": forward_hash,
                            "normalization_sha256": NORMALIZATION_HASH, "gt_identity_sha256": sample["gt_patch_sha256"],
                            "noise_seed": noise_seed, "diffusion_seed": diffusion_seed,
                            "refinement_config_sha256": config_hash if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                            "theta_true_json": _tensor_theta_json(theta_true), "theta_inverse_json": _tensor_theta_json(theta_inverse),
                            "psnr": psnr, "ssim": ssim, "observed_nrmse": nrmse if nrmse is not None else "",
                            "poisson_gaussian_objective": objective if objective is not None else "",
                            "runtime_seconds": runtime, "peak_gpu_memory_bytes": peak,
                            "gradient_finite": grad_finite, "output_finite": output_finite,
                            "prediction_sha256": sha_array(prediction), "status": "PASS",
                        })
                        if order == 0 and factor in visual_levels and float(severity) in visual_levels[factor] and method != "WF":
                            path = visual_dir / f"{factor}_{severity:g}_{method.replace(' ', '_')}.npz"
                            stream = io.BytesIO(); np.savez_compressed(stream, prediction=prediction, gt=gt); atomic_write(path, stream.getvalue())
                    # Identity is asserted across all four rows before commit.
                    for field in ("raw_stack_sha256", "validity_mask_sha256", "geometry_sha256", "forward_parameters_sha256", "noise_seed"):
                        if len({str(item[field]) for item in case_rows}) != 1:
                            raise R1C3Blocked("R1C3_INPUT_IDENTITY_MISMATCH", f"{factor}/{severity}/{sample['sample_id']}:{field}")
                    rows_out.extend(case_rows)
                except Exception as exc:
                    failures.append({"factor": factor, "severity": severity, "sample_id": sample["sample_id"], "error": f"{type(exc).__name__}: {exc}"})
                    raise
                atomic_write(run_dir / "R1C3_ROBUSTNESS_PER_SAMPLE.csv", _csv_bytes(rows_out, ROBUST_FIELDS))
                atomic_write(run_dir / "R1C3_FAILED_CASES.csv", _csv_bytes(failures, ("factor", "severity", "sample_id", "error")))
    expected_rows = sum(len(levels) for levels in ROBUSTNESS_FACTORS.values()) * 20 * 4
    if len(rows_out) != expected_rows or failures:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "robustness grid incomplete")
    return rows_out


def runtime_benchmark(
    run_dir: Path,
    bundle_rows: Sequence[Mapping[str, str]],
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One warm-up then three recorded repeats per FOV on the frozen raw stack."""
    rows_out: list[dict[str, Any]] = []
    mask_hash = sha_array(np.asarray(VALIDITY_MASK, dtype=np.float32))
    geometry_hash = _json_hash(geometry)
    config_hash = REFINEMENT_CONFIG.receipt()["config_sha256"]
    for sample_order, row in enumerate(bundle_rows):
        assert_no_external_cuda_compute()
        path = BUNDLE_MANIFEST.parent / row["npz_path"]
        with np.load(path, allow_pickle=False) as archive:
            raw = torch.from_numpy(np.asarray(archive["raw_stack"], dtype=np.float32))[None].to(device)
            theta = _theta_from_archive(archive, device)
            noise_seed = int(np.asarray(archive["acquisition_noise_seed"]).reshape(-1)[0])
        diffusion_seed = _official_nominal_diffusion_seed(row["raw_stack_sha256"])
        forward_hash = _json_hash({"theta": _tensor_theta_json(theta), "protocol_hash": PROTOCOL_HASH})
        # Full unrecorded warm-up for this FOV.
        run_four_methods(
            raw, model, scheduler, sim_config, theta,
            diffusion_seed=diffusion_seed, refinement_config=REFINEMENT_CONFIG,
            geometry_receipt=geometry,
        )
        for repeat in range(3):
            result = run_four_methods(
                raw, model, scheduler, sim_config, theta,
                diffusion_seed=diffusion_seed, refinement_config=REFINEMENT_CONFIG,
                geometry_receipt=geometry,
            )
            wf = result["WF"]; diff = result["DiffWS-6"]
            phys = result["PhysMap-6"]["refinement"]; stage2 = result["APD-SIM-6"]["refinement"]
            entries = (
                ("WF", "six-frame mean", "direct_cuda_timing", wf["runtime_seconds"], wf["peak_gpu_memory_bytes"]),
                ("DiffWS-6", "Stage 1", "direct_cuda_timing", diff["runtime_seconds"], diff["peak_gpu_memory_bytes"]),
                ("PhysMap-6", "total", "direct_cuda_timing", phys.runtime_seconds, phys.peak_gpu_memory_bytes),
                ("APD-SIM-6", "Stage 1", "alias_of_same_repeat_diffws_stage1", diff["runtime_seconds"], diff["peak_gpu_memory_bytes"]),
                ("APD-SIM-6", "Stage 2", "direct_cuda_timing", stage2.runtime_seconds, stage2.peak_gpu_memory_bytes),
                ("APD-SIM-6", "total", "derived_same_repeat_component_sum", diff["runtime_seconds"] + stage2.runtime_seconds, max(diff["peak_gpu_memory_bytes"], stage2.peak_gpu_memory_bytes)),
            )
            for method, component, kind, elapsed, peak in entries:
                rows_out.append({
                    "sample_order": sample_order, "sample_id": row["sample_id"], "parent_id": row["parent_id"],
                    "structure": row["structure_class"], "repeat_index": repeat, "method": method,
                    "component": component, "measurement_kind": kind, "warmup_runs_before_measurement": 1,
                    "raw_stack_sha256": row["raw_stack_sha256"], "validity_mask_sha256": mask_hash,
                    "geometry_sha256": geometry_hash, "forward_parameters_sha256": forward_hash,
                    "normalization_sha256": NORMALIZATION_HASH, "noise_seed": noise_seed,
                    "diffusion_seed": diffusion_seed,
                    "refinement_config_sha256": config_hash if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                    "runtime_seconds": float(elapsed), "peak_gpu_memory_bytes": int(peak),
                })
            atomic_write(run_dir / "R1C3_RUNTIME_PER_RUN.csv", _csv_bytes(rows_out, RUNTIME_FIELDS))
    if len(rows_out) != 30 * 3 * 6:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "runtime grid incomplete")
    return rows_out


def _descriptive_by_method(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result[method] = {}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            result[method][metric] = {
                "n": int(values.size), "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)), "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)), "q3": float(np.quantile(values, 0.75)),
            }
    return result


def nominal_postprocess(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metrics = _metrics_module()
    stats = {
        "schema_version": 1, "status": "COMPLETE_VALIDATED", "n_fov": 30,
        "method_summaries": _descriptive_by_method(rows, ("psnr", "ssim")),
        "frc_status_counts": {
            method: {status: sum(row["method"] == method and row["frc_status"] == status for row in rows) for status in ("CUTOFF", "RIGHT_CENSORED", "UNRESOLVED")}
            for method in METHODS
        },
        "source_csv_sha256": sha_file(run_dir / "R1C3_NOMINAL_PER_FOV.csv"),
    }
    paired_rows: list[dict[str, Any]] = []
    for comparator in ("PhysMap-6", "DiffWS-6"):
        for metric in ("psnr", "ssim"):
            apd = sorted((row for row in rows if row["method"] == "APD-SIM-6"), key=lambda row: int(row["sample_order"]))
            other = sorted((row for row in rows if row["method"] == comparator), key=lambda row: int(row["sample_order"]))
            ci = metrics.parent_image_bootstrap_ci(
                [float(row[metric]) for row in apd], [float(row[metric]) for row in other],
                [row["parent_id"] for row in apd], class_labels=[row["structure"] for row in apd],
                n_resamples=10_000, seed=20260814,
            )
            for left, right in zip(apd, other):
                paired_rows.append({
                    "sample_id": left["sample_id"], "parent_id": left["parent_id"], "structure": left["structure"],
                    "contrast": f"APD-SIM-6_minus_{comparator}", "metric": metric,
                    "paired_difference": float(left[metric]) - float(right[metric]),
                    "bootstrap_mean_difference": ci["estimate"], "bootstrap_ci_low": ci["confidence_interval"][0],
                    "bootstrap_ci_high": ci["confidence_interval"][1], "bootstrap_resamples": ci["n_resamples"],
                })
    fields = ("sample_id", "parent_id", "structure", "contrast", "metric", "paired_difference", "bootstrap_mean_difference", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples")
    atomic_write(run_dir / "R1C3_NOMINAL_PAIRED_DIFFERENCES.csv", _csv_bytes(paired_rows, fields))
    write_json(run_dir / "R1C3_NOMINAL_STATS.json", stats)


def robustness_postprocess(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[str, Any] = {}
    for factor, levels in ROBUSTNESS_FACTORS.items():
        grouped[factor] = {}
        for level in levels:
            grouped[factor][str(level)] = {}
            for method in METHODS:
                selected = [row for row in rows if row["factor"] == factor and float(row["severity"]) == level and row["method"] == method]
                if len(selected) != 20:
                    raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"robust group incomplete: {factor}/{level}/{method}")
                grouped[factor][str(level)][method] = {
                    metric: {"mean": float(np.mean([float(row[metric]) for row in selected])), "sample_sd": float(np.std([float(row[metric]) for row in selected], ddof=1))}
                    for metric in ("psnr", "ssim")
                }
    write_json(run_dir / "R1C3_ROBUSTNESS_STATS.json", {
        "schema_version": 1, "status": "COMPLETE_VALIDATED", "factor_count": 12,
        "factor_level_count": sum(len(levels) for levels in ROBUSTNESS_FACTORS.values()),
        "sample_count": 20, "method_count": 4, "row_count": len(rows), "groups": grouped,
        "source_csv_sha256": sha_file(run_dir / "R1C3_ROBUSTNESS_PER_SAMPLE.csv"),
    })


def runtime_postprocess(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    summaries = []
    for method, component in (
        ("WF", "six-frame mean"), ("DiffWS-6", "Stage 1"), ("PhysMap-6", "total"),
        ("APD-SIM-6", "Stage 1"), ("APD-SIM-6", "Stage 2"), ("APD-SIM-6", "total"),
    ):
        selected = [row for row in rows if row["method"] == method and row["component"] == component]
        if len(selected) != 90:
            raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", f"runtime group {method}/{component}")
        by_fov: dict[str, list[float]] = {}
        for row in selected:
            by_fov.setdefault(str(row["sample_id"]), []).append(float(row["runtime_seconds"]))
        fov_means = np.asarray([np.mean(values) for values in by_fov.values()], dtype=np.float64)
        summaries.append({
            "method": method, "component": component, "n_fov": 30, "repeats_per_fov": 3,
            "mean_of_fov_means_seconds": float(fov_means.mean()),
            "sample_sd_of_fov_means_seconds": float(fov_means.std(ddof=1)),
            "peak_gpu_memory_bytes": int(max(int(row["peak_gpu_memory_bytes"]) for row in selected)),
            "measurement_kind": selected[0]["measurement_kind"],
        })
    summary = {"schema_version": 1, "status": "COMPLETE_VALIDATED", "n_fov": 30, "recorded_repeats": 3, "warmup_per_fov": 1, "summaries": summaries, "source_csv_sha256": sha_file(run_dir / "R1C3_RUNTIME_PER_RUN.csv")}
    write_json(run_dir / "R1C3_RUNTIME_STATS.json", summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Strict DMD six-frame runtime benchmark (30 FOVs, one warm-up and three recorded CUDA-synchronized repeats per FOV).}",
        r"\begin{tabular}{llrr}",
        r"\hline",
        "Method & Component & Time (s) & Peak memory (MiB) \\\\",
        r"\hline",
    ]
    for item in summaries:
        lines.append(
            f"{item['method']} & {item['component']} & "
            f"{item['mean_of_fov_means_seconds']:.3f} $\\pm$ "
            f"{item['sample_sd_of_fov_means_seconds']:.3f} & "
            f"{item['peak_gpu_memory_bytes']/2**20:.1f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    atomic_write(run_dir / "TABLE_RUNTIME_PHYSMAP6_STRICT.tex", "\n".join(lines).encode("utf-8"))


def _source_audit() -> dict[str, Any]:
    core_source = inspect.getsource(masked_refine)
    return {
        "physmap_core_has_no_checkpoint_parameter": "checkpoint" not in inspect.signature(masked_refine).parameters,
        "physmap_core_has_no_gt_parameter": "gt" not in inspect.signature(masked_refine).parameters,
        "physmap_core_imports_no_diffusion_model": "load_checkpoint" not in core_source,
        "physmap9_excluded": "PhysMap-9" not in core_source,
        "lambda_prior_zero": REFINEMENT_CONFIG.lambda_prior == 0.0,
        "shared_function": f"{masked_refine.__module__}.{masked_refine.__qualname__}",
    }


def _aberration_zero_regression(sim_config: SIM2DConfig) -> dict[str, Any]:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(301)
    image = torch.rand((1, 1, 48, 52), generator=generator)
    nominal = nominal_theta_2d(sim_config, device)
    explicit = {key: value.clone() for key, value in nominal.items()}
    for key in ABERRATION_KEYS:
        explicit[key] = torch.zeros((1,), device=device)
    a, _ = forward_protocol_clean_2d(image, sim_config, PROTOCOL_ID, theta=nominal)
    b, _ = forward_protocol_clean_2d(image, sim_config, PROTOCOL_ID, theta=explicit)
    exact = torch.equal(a, b)
    if not exact:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "zero-aberration regression failed")
    nonzero = {key: value.clone() for key, value in explicit.items()}
    nonzero["aberr_defocus"] = torch.tensor([0.1])
    c, _ = forward_protocol_clean_2d(image, sim_config, PROTOCOL_ID, theta=nonzero)
    changed = not torch.equal(a, c) and bool(torch.isfinite(c).all())
    if not changed:
        raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "aberration operator inactive")
    return {
        "zero_wave_exact_regression": exact,
        "nonzero_defocus_changes_forward": changed,
        "zero_wave_output_sha256": sha_array(a.numpy().astype(np.float32)),
        "defocus_0p1_output_sha256": sha_array(c.numpy().astype(np.float32)),
    }


def preflight(run_dir: Path) -> dict[str, Any]:
    protocol = protocol_receipt()
    config = read_json(CONFIG)
    generated_checkpoint, checkpoint_metadata = checkpoint_receipt(config)
    rows = load_bundle_rows(verify_payloads=False)
    robust_rows = load_robust_rows()
    metrics = _metrics_module()
    sim_config = make_sim_config(config)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "interpreter": sys.executable,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "protocol": protocol,
        "checkpoint": generated_checkpoint,
        "sealed_test_manifest": str(SEALED_MANIFEST),
        "sealed_test_manifest_sha256": sha_file(SEALED_MANIFEST),
        "test30_bundle_manifest": str(BUNDLE_MANIFEST),
        "test30_bundle_manifest_sha256": sha_file(BUNDLE_MANIFEST),
        "test30_count": len(rows),
        "robustness_manifest": str(ROBUST_MANIFEST),
        "robustness_manifest_sha256": sha_file(ROBUST_MANIFEST),
        "robustness_sample_count": len(robust_rows),
        "robustness_factor_levels": ROBUSTNESS_FACTORS,
        "robustness_seed_base": LEGACY_ROBUSTNESS_SEED,
        "nominal_stage1_seed_policy": (
            "uint64_be(SHA256(APD6_OFFICIAL_R2_WARMSTART_MAP_V1|checkpoint_sha256|"
            "raw_stack_sha256)[:8]) mod 2^63"
        ),
        "official_metric_source": str(METRICS_SOURCE),
        "official_metric_source_sha256": sha_file(METRICS_SOURCE),
        "comment2_frc_source": str(COMMENT2_FRC_SOURCE),
        "comment2_frc_source_sha256": sha_file(COMMENT2_FRC_SOURCE),
        "official_frc_function": _comment2_frc_module().reference_frc_1over7.__name__,
        "refinement_config": REFINEMENT_CONFIG.receipt(),
        "source_contract_audit": _source_audit(),
        "aberration_forward_regression": _aberration_zero_regression(sim_config),
        "checkpoint_metadata_global_step": int(checkpoint_metadata["global_step"]),
        "formal_outputs_started": False,
        "implementation_sha256": {str(path): sha_file(path) for path in IMPLEMENTATION_FILES},
        "robustness_sample_reuse_scope": (
            "Only the immutable 20-sample center-crop identities, crop coordinates, "
            "normalization rule, factors, severities, and seeds are reused. No legacy "
            "3O2P measurement, reconstruction, metric, figure, or runtime value is reused."
        ),
    }
    write_json(run_dir / "DMD6_PROTOCOL_RECEIPT.json", protocol)
    write_json(run_dir / "APD6_CHECKPOINT_RECEIPT.json", generated_checkpoint)
    write_json(run_dir / "R1C3_PREFLIGHT.json", result)
    return result


def run_project_test_gate(run_dir: Path) -> dict[str, Any]:
    """Run the repository test suite before a full run may publish READY."""
    command = [sys.executable, "-B", "-m", "pytest", "tests", "-q"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            timeout=15 * 60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise R1C3Blocked(
            "R1C3_FORMAL_EVALUATION_INCOMPLETE", f"project test gate failed to execute: {exc}"
        ) from exc
    log = completed.stdout + (b"\n--- STDERR ---\n" + completed.stderr if completed.stderr else b"")
    atomic_write(run_dir / "R1C3_TEST_LOG.txt", log)
    decoded = log.decode("utf-8", errors="replace")
    match = re.search(r"(?P<count>\d+) passed(?:[,\s]|$)", decoded)
    passed = int(match.group("count")) if match else 0
    receipt = {
        "schema_version": 1,
        "status": "PASS" if completed.returncode == 0 and passed > 0 else "FAIL",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "command": command,
        "working_directory": str(ROOT),
        "exit_code": int(completed.returncode),
        "passed": passed,
        "failed": 0 if completed.returncode == 0 else None,
        "stdout_stderr_sha256": hashlib.sha256(log).hexdigest(),
        "log": str(run_dir / "R1C3_TEST_LOG.txt"),
        "test_source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha_file(path)
            for path in PROJECT_TEST_FILES
        },
    }
    write_json(run_dir / "R1C3_TEST_RECEIPT.json", receipt)
    if receipt["status"] != "PASS":
        raise R1C3Blocked(
            "R1C3_FORMAL_EVALUATION_INCOMPLETE",
            f"project test gate failed: exit={completed.returncode}, passed={passed}",
        )
    return receipt


def _markdown_audit(
    preflight_result: Mapping[str, Any], final_status: str, final_result: Mapping[str, Any]
) -> str:
    checkpoint = preflight_result["checkpoint"]
    protocol = preflight_result["protocol"]
    return f"""# Strict PhysMap-6 implementation audit

Status: `{final_status}`

## Protocol

- ID: `{protocol['protocol_id']}`
- Raw order: `{', '.join(protocol['raw_frame_order'])}`
- Protocol SHA-256: `{protocol['protocol_hash']}`
- Validity mask (15 slots): `{protocol['validity_mask_15_slots']}`
- Evidence boundary: controller-defined nominal synthetic DMD geometry; historical acquisition receipt is absent.

## Frozen APD-SIM-6 checkpoint

- Path: `{checkpoint['checkpoint_absolute_path']}`
- SHA-256: `{checkpoint['checkpoint_sha256']}`
- Validation-selected loop/data event step: `{checkpoint['selected_step']}`
- Committed AdamW updates at the selected checkpoint: `{checkpoint['optimizer_committed_updates_at_selected_checkpoint']}`
- Selection metric: `{checkpoint['validation_metric_name']}={checkpoint['validation_metric_value']}`
- Test data used for selection: `{checkpoint['test_data_used_for_selection']}`
- Model and EMA tensors finite: `{checkpoint['all_model_parameters_finite'] and checkpoint['all_ema_parameters_finite']}`

## Strict ablation contract

`PhysMap-6` is not a trained model. It is per-sample physics-only masked-likelihood optimization.
It does not load a checkpoint and cannot read GT. PhysMap-6 and APD-SIM-6 Stage 2 call the same
`unisim.revision_r1.physmap6_core.masked_refine` function with the same frozen configuration;
only the initialization differs (six-frame mean versus `x_ws`). PhysMap-9 is additional-data
reference only and is excluded from primary outputs.

## Final gate

The status is READY only after finite smoke, the 30-FOV nominal grid, all 12 robustness factors,
the 30x3 runtime grid, independent report recomputation, and the repository test suite pass.
Recorded rows: nominal `{final_result.get('nominal_rows', 0)}`, robustness
`{final_result.get('robustness_rows', 0)}`, runtime `{final_result.get('runtime_rows', 0)}`.
Tests all pass: `{final_result.get('tests_all_pass', False)}`.

The selected checkpoint is finite and validation-only selected, but event step 96000 corresponds
to 95956 committed AdamW updates.  Do not describe it as 96000 committed optimizer updates.
"""


def run(preflight_only: bool = False) -> dict[str, Any]:
    if not __debug__:
        raise RuntimeError("R1C3 refuses python -O because assertions/provenance checks must remain active")
    run_dir = create_run_dir()
    status = "R1C3_FORMAL_EVALUATION_INCOMPLETE"
    preflight_result: dict[str, Any] = {}
    try:
        preflight_result = preflight(run_dir)
        if preflight_only:
            status = "R1C3_PREFLIGHT_PASS"
            result = {
                "status": status,
                "run_dir": str(run_dir),
                "preflight_only": True,
                "formal_evaluation_executed": False,
            }
        else:
            if not torch.cuda.is_available():
                raise R1C3Blocked("R1C3_FORMAL_EVALUATION_INCOMPLETE", "CUDA unavailable for formal smoke")
            gpu_gate_before = assert_no_external_cuda_compute()
            write_json(run_dir / "R1C3_GPU_GATE_BEFORE.json", gpu_gate_before)
            config = read_json(CONFIG)
            device = torch.device("cuda:0")
            model, scheduler, _metadata = load_stage1(
                config, CHECKPOINT, EXPECTED["checkpoint_sha256"], device
            )
            sim_config = make_sim_config(config)
            rows = load_bundle_rows(verify_payloads=True)
            geometry = _geometry_receipt(preflight_result["protocol"])
            smoke = smoke_test(rows, model, scheduler, sim_config, device, geometry)
            write_json(run_dir / "R1C3_SMOKE_TEST.json", smoke)
            gt_mapping = load_gt_mapping()
            nominal_rows = nominal_evaluation(
                run_dir, rows, gt_mapping, model, scheduler, sim_config, device, geometry
            )
            assert_implementation_unchanged(preflight_result["implementation_sha256"])
            robust_samples = load_robust_rows()
            robustness_rows = robustness_evaluation(
                run_dir, robust_samples, model, scheduler, sim_config, device, geometry
            )
            assert_implementation_unchanged(preflight_result["implementation_sha256"])
            runtime_rows = runtime_benchmark(
                run_dir, rows, model, scheduler, sim_config, device, geometry
            )
            assert_implementation_unchanged(preflight_result["implementation_sha256"])
            gpu_gate_after = assert_no_external_cuda_compute()
            write_json(run_dir / "R1C3_GPU_GATE_AFTER.json", gpu_gate_after)
            # Reporting is imported only after the three complete sample-level
            # grids exist; it reloads those files and independently recomputes
            # every statistic, table, caption, figure receipt and runtime row.
            from .physmap6_reporting import generate_all_reports, independent_audit

            report = generate_all_reports(
                run_dir,
                protocol_receipt=preflight_result["protocol"],
                checkpoint_receipt=preflight_result["checkpoint"],
                factor_levels=ROBUSTNESS_FACTORS,
            )
            independent = independent_audit(
                run_dir,
                protocol_receipt=preflight_result["protocol"],
                checkpoint_receipt=preflight_result["checkpoint"],
                factor_levels=ROBUSTNESS_FACTORS,
            )
            assert_implementation_unchanged(preflight_result["implementation_sha256"])
            if report.get("status") != "PASS" or independent.get("status") != "PASS":
                raise R1C3Blocked(
                    "R1C3_FORMAL_EVALUATION_INCOMPLETE", "independent reporting audit failed"
                )
            test_receipt = run_project_test_gate(run_dir)
            assert_implementation_unchanged(preflight_result["implementation_sha256"])
            status = "R1C3_PHYSMAP6_STRICT_READY"
            result = {
                "status": status,
                "run_dir": str(run_dir),
                "preflight_only": False,
                "formal_evaluation_executed": True,
                "smoke_passed": True,
                "nominal_rows": len(nominal_rows),
                "robustness_rows": len(robustness_rows),
                "runtime_rows": len(runtime_rows),
                "tests_all_pass": True,
                "project_tests_passed": int(test_receipt["passed"]),
                "test_receipt": str(run_dir / "R1C3_TEST_RECEIPT.json"),
                "reporting_audit": independent,
            }
    except R1C3Blocked as exc:
        status = exc.status
        result = {
            "status": status,
            "detail": exc.detail,
            "run_dir": str(run_dir),
            "preflight_only": preflight_only,
            "formal_evaluation_executed": False,
        }
    except Exception as exc:
        result = {
            "status": status,
            "detail": f"{type(exc).__name__}: {exc}",
            "run_dir": str(run_dir),
            "preflight_only": preflight_only,
            "formal_evaluation_executed": False,
        }
    if preflight_result:
        audit_json = {
            "schema_version": 1,
            "status": status,
            "preflight": preflight_result,
            "result": result,
            "primary_physmap9_values_used": False,
            "formal_artifacts_published": status == "R1C3_PHYSMAP6_STRICT_READY",
        }
        write_json(run_dir / "AUDIT_PHYSMAP6_IMPLEMENTATION.json", audit_json)
        atomic_write(
            run_dir / "AUDIT_PHYSMAP6_IMPLEMENTATION.md",
            _markdown_audit(preflight_result, status, result).encode("utf-8"),
        )
    write_json(run_dir / "STATUS.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


__all__ = [
    "REFINEMENT_CONFIG",
    "ROBUSTNESS_FACTORS",
    "R1C3Blocked",
    "preflight",
    "run",
    "run_project_test_gate",
]

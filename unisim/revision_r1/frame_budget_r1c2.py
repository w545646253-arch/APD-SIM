"""Formal 30-FOV DMD acquisition-budget evaluation for Reviewer #1 C2.

This module is intentionally isolated from the sealed R1C3 implementation.
It binds the already-audited 2-D forward, 15-slot conditioning, EMA checkpoint
loader, DDIM schedule, masked Poisson-Gaussian objective, normalization, and
native metrics to each controller protocol independently.  No training or
checkpoint mutation is reachable from this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile
import torch

from unisim.checkpoint_contract import architecture_hash, load_checkpoint_bound
from unisim.formal_training_2d import DiffusionScheduler2D
from unisim.model2d import APDConditionedUNet2D, assert_strictly_2d_model
from unisim.protocol_runtime import require_protocol
from unisim.revision_r1.dataset_fig5_audit_identity import (
    CESHIJI_ROOT,
    enumerate_ceshiji,
    normalize_image,
    normalized_pixel_sha256,
)
from unisim.revision_r1.physmap6_core import RefinementConfig, masked_refine
from unisim.revision_r1.physmap6_pipeline import (
    BEST_RULE_ID,
    NORMALIZATION_HASH,
    load_stage1_registered,
    sha_array,
    stage1_reconstruct_registered,
    stage1_reconstruct_registered_tiled,
)
from unisim.sim_forward_2d import (
    SIM2DConfig,
    embed_raw_to_slots_2d,
    forward_protocol_clean_2d,
    forward_protocol_sim_2d,
    masked_poisson_gaussian_likelihood_2d,
    nominal_theta_2d,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "reviewer1_frame_budget_30fov"
PRIOR_IDENTITY_RUN = (
    ROOT / "outputs" / "reviewer1_physmap6_dataset_fig5_audit" / "20260814T031054Z"
)
PRIOR_CESHIJI_MANIFEST = PRIOR_IDENTITY_RUN / "CESHIJI_MANIFEST.csv"
FORMAL_R1C3_RUN = ROOT / "outputs" / "reviewer1_physmap6_strict" / "20260813T183229Z"
MANUSCRIPT_CANDIDATE = Path(
    r"docs/manuscript_candidate"
)
METRICS_SOURCE = ROOT / "tools" / "official_r2_common_metrics.py"
METRICS_SHA256 = "9efd7efcc6ecf126816887a710478f592ecc3b29562003a2ea452e1b93deec9a"
COMMON_MEASUREMENT_SEED_BASE = 2026081500
COMMON_DIFFUSION_SEED_BASE = 2026082500
REPRESENTATIVE_SAMPLE_ORDER = 19
REPRESENTATIVE_ROI = {"y": 342, "x": 342, "height": 320, "width": 320}
REFINEMENT_CONFIG = RefinementConfig()
STAGE1_POLICY = {
    "weights": "ema",
    "diffusion_init_t": 600,
    "diffusion_steps_including_endpoints": 80,
    "ddim_eta": 0.0,
    "padding": "reflect_bottom_right_to_multiple_16_then_exact_unpad",
    "precision": "float32_model_and_scheduler_no_autocast",
}


@dataclass(frozen=True)
class ProtocolPlan:
    label: str
    protocol_id: str
    protocol_file: Path
    config: Path
    checkpoint: Path
    receipt: Path
    history: Path
    raw_order: tuple[str, ...]
    validity_mask: tuple[int, ...]


PLANS = (
    ProtocolPlan(
        "DMD-3F",
        "DMD_3F_1O3P",
        ROOT / "protocols" / "dmd_3f_1o3p.json",
        ROOT / "configs" / "apd_dmd_r2" / "train3_formal_restart_simple_r1.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "best.pt",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "best_checkpoint_receipt.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "validation_history.csv",
        ("X0", "X120", "X240"),
        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    ProtocolPlan(
        "DMD-6F",
        "DMD_6F_2O3P",
        ROOT / "protocols" / "dmd_6f_2o3p.json",
        ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best_checkpoint_receipt.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "validation_history.csv",
        ("H0", "H120", "H240", "V0", "V120", "V240"),
        (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    ProtocolPlan(
        "DMD-9F",
        "DMD_9F_3O3P",
        ROOT / "protocols" / "dmd_9f_3o3p.json",
        ROOT / "configs" / "apd_dmd_r2" / "train9_formal.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r4" / "dmd9" / "best.pt",
        ROOT / "archive" / "dmd9_superseded_20260817_104841" / "checkpoints" / "apd_dmd_geometry_r2_dmd9" / "best_checkpoint_receipt.json",
        ROOT / "archive" / "dmd9_superseded_20260817_104841" / "checkpoints" / "apd_dmd_geometry_r2_dmd9" / "validation_history.csv",
        ("X0", "X120", "X240", "Y0", "Y120", "Y240", "Z0", "Z120", "Z240"),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    ),
)


METHODS = ("WF", "APD-SIM")
PER_FOV_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "protocol_label",
    "protocol_id", "protocol_hash", "frame_count", "raw_frame_order", "method",
    "gt_path", "gt_file_sha256", "gt_normalized_sha256", "checkpoint_path",
    "checkpoint_sha256", "checkpoint_selected_event", "checkpoint_committed_updates",
    "inference_weight_branch", "measurement_seed", "diffusion_seed", "raw_stack_sha256",
    "raw_npz_path", "raw_npz_file_sha256",
    "validity_mask_sha256", "forward_parameters_sha256", "prediction_path",
    "prediction_file_sha256", "prediction_array_sha256", "psnr", "ssim", "frc_status",
    "frc_cutoff_cycles_per_pixel", "frc_spatial_period_px", "frc_right_censored",
    "frc_unresolved", "runtime_seconds", "stage1_runtime_seconds", "stage2_runtime_seconds",
    "final_objective", "observed_frame_nrmse", "finite", "status",
)


class R1C2Blocked(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class CudaContentionMonitor:
    """Sample GPU 0 for external pure-compute processes during formal inference."""

    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._samples = 0
        self._thread = threading.Thread(target=self._run, name="r1c2-gpu-contention-monitor", daemon=True)

    def _run(self) -> None:
        from unisim.revision_r1.physmap6_experiment import assert_no_external_cuda_compute

        while not self._stop.is_set():
            try:
                assert_no_external_cuda_compute()
                self._samples += 1
            except Exception as error:  # fail closed and preserve the first detection
                self._error = error
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def checkpoint(self) -> None:
        if self._error is not None:
            raise R1C2Blocked("R1C2_GPU_CONTENTION_INVALIDATED", str(self._error))
        if not self._thread.is_alive() and not self._stop.is_set():
            raise R1C2Blocked("R1C2_GPU_CONTENTION_INVALIDATED", "contention monitor stopped unexpectedly")

    def stop_and_validate(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise R1C2Blocked("R1C2_GPU_CONTENTION_INVALIDATED", "contention monitor did not stop")
        self.checkpoint()
        from unisim.revision_r1.physmap6_experiment import assert_no_external_cuda_compute

        final = assert_no_external_cuda_compute()
        return {"status": "PASS", "sampling_interval_seconds": self.interval_seconds, "completed_samples": self._samples, "final_synchronous_gate": final}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"JSON object required: {path}")
    return value


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_replace(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n")


def save_npy(path: Path, array: np.ndarray) -> None:
    stream = io.BytesIO()
    value = np.ascontiguousarray(array, dtype=np.float32)
    np.save(stream, value, allow_pickle=False)
    atomic_replace(path, stream.getvalue())
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.float32 or loaded.shape != value.shape or not np.array_equal(loaded, value):
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"NPY verification failed: {path}")


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "exists": False, "file_count": 0, "aggregate_sha256": None, "files": []}
    files = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix().lower()):
        files.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha_file(path),
        })
    return {
        "root": str(root.resolve()),
        "exists": True,
        "file_count": len(files),
        "aggregate_sha256": hashlib.sha256(canonical_bytes(files)).hexdigest(),
        "files": files,
    }


def _config_for_sim(config: Mapping[str, Any]) -> SIM2DConfig:
    forward = dict(config["forward"])
    allowed = set(SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in forward.items() if key in allowed}
    for key in (
        "rand_k_ratio_xy", "rand_mod_depth", "rand_psf_sigma_scale",
        "rand_background", "rand_photon_scale", "rand_read_noise_e",
    ):
        if key in values:
            values[key] = tuple(float(item) for item in values[key])
    return SIM2DConfig(**values)


def _model_from_config(config: Mapping[str, Any]) -> APDConditionedUNet2D:
    model_cfg = config["model"]
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


def _expected_checkpoint_identities(
    config: Mapping[str, Any], plan: ProtocolPlan, model: APDConditionedUNet2D
) -> dict[str, Any]:
    spec = require_protocol(plan.protocol_id)
    return {
        "training_protocol_id": plan.protocol_id,
        "training_protocol_hash": spec.protocol_hash,
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


def _validation_best(history: Path) -> tuple[dict[str, str], int]:
    with history.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"history rows: {history}")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["mean_val_total_loss"]),
            -float(row["mean_val_x0_psnr"]),
            -float(row["mean_val_x0_ssim"]),
            float(row["global_step"]),
        ),
    )
    return ordered[0], len(rows)


def _optimizer_committed_step(payload: Mapping[str, Any]) -> int:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", "optimizer state absent")
    steps: set[int] = set()
    for state in optimizer["state"].values():
        if not isinstance(state, Mapping) or "step" not in state:
            continue
        value = state["step"]
        steps.add(int(value.item()) if torch.is_tensor(value) else int(value))
    if len(steps) != 1:
        raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"optimizer steps: {steps}")
    return next(iter(steps))


def _state_receipt(state: Mapping[str, Any]) -> tuple[int, bool, str]:
    digest = hashlib.sha256()
    count = 0
    finite = True
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        count += 1
        finite = finite and bool(torch.isfinite(tensor).all())
        digest.update(canonical_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
        digest.update(b"\n")
        digest.update(tensor.numpy().tobytes(order="C"))
    return count, finite, digest.hexdigest()


def audit_checkpoints(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for plan in PLANS:
        for path in (plan.config, plan.checkpoint, plan.receipt, plan.history, plan.protocol_file):
            if not path.is_file():
                raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", str(path))
        config = read_json(plan.config)
        receipt = read_json(plan.receipt)
        spec = require_protocol(plan.protocol_id)
        checkpoint_sha = sha_file(plan.checkpoint)
        best, history_count = _validation_best(plan.history)
        if (
            receipt.get("completion_status") != "FORMAL_TRAINING_COMPLETE"
            or receipt.get("selection_rule") != BEST_RULE_ID
            or receipt.get("test_data_used_for_selection") is not False
            or receipt.get("protocol_id") != plan.protocol_id
            or receipt.get("protocol_hash") != spec.protocol_hash
            or receipt.get("checkpoint_sha256") != checkpoint_sha
            or int(float(best["global_step"])) != int(float(receipt["metrics"]["global_step"]))
            or not math.isclose(float(best["mean_val_total_loss"]), float(receipt["metrics"]["mean_val_total_loss"]), rel_tol=0.0, abs_tol=1e-15)
        ):
            raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", plan.label)
        model = _model_from_config(config)
        payload = load_checkpoint_bound(
            plan.checkpoint,
            protocol=plan.protocol_id,
            expected_sha256=checkpoint_sha,
            expected_identities=_expected_checkpoint_identities(config, plan, model),
        )
        model_count, model_finite, model_hash = _state_receipt(payload.get("model", {}))
        ema_count, ema_finite, ema_hash = _state_receipt(payload.get("ema", {}))
        committed = _optimizer_committed_step(payload)
        selected_event = int(float(best["global_step"]))
        if model_count != 270 or ema_count != 270 or not model_finite or not ema_finite:
            raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"state invalid: {plan.label}")
        if payload["metadata"].get("global_step") != selected_event:
            raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"event mismatch: {plan.label}")
        row = {
            "protocol_label": plan.label,
            "protocol_id": plan.protocol_id,
            "protocol_hash": spec.protocol_hash,
            "raw_frame_order": "/".join(plan.raw_order),
            "validity_mask": "".join(str(value) for value in plan.validity_mask),
            "checkpoint_absolute_path": str(plan.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "selection_receipt_absolute_path": str(plan.receipt.resolve()),
            "selection_receipt_sha256": sha_file(plan.receipt),
            "validation_history_absolute_path": str(plan.history.resolve()),
            "validation_history_sha256": sha_file(plan.history),
            "validation_history_rows": history_count,
            "selected_loop_data_event": selected_event,
            "actual_committed_optimizer_updates": committed,
            "validation_metric_name": "mean_val_total_loss",
            "validation_metric_value": float(best["mean_val_total_loss"]),
            "validation_psnr": float(best["mean_val_x0_psnr"]),
            "validation_ssim": float(best["mean_val_x0_ssim"]),
            "model_state_sha256": model_hash,
            "ema_state_sha256": ema_hash,
            "formal_inference_weight_branch": "ema",
            "architecture_identity": model.architecture_contract,
            "architecture_hash": architecture_hash(model),
            "model_tensor_count": model_count,
            "ema_tensor_count": ema_count,
            "model_finite": model_finite,
            "ema_finite": ema_finite,
            "training_validation_test_separation": "PASS; receipt test_data_used_for_selection=false",
            "status": "PASS",
        }
        rows.append(row)
        details[plan.protocol_id] = {
            "plan": plan,
            "config": config,
            "checkpoint": row,
            "metadata": dict(payload["metadata"]),
        }
        del payload, model
    fields = tuple(rows[0])
    atomic_replace(run_dir / "R1C2_CHECKPOINT_MANIFEST.csv", csv_bytes(rows, fields))
    return rows, details


def audit_protocols(run_dir: Path) -> dict[str, Any]:
    protocols = []
    for plan in PLANS:
        source = read_json(plan.protocol_file)
        spec = require_protocol(plan.protocol_id)
        checks = {
            "protocol_id": spec.protocol_id == plan.protocol_id,
            "protocol_hash": spec.protocol_hash == source.get("protocol_hash"),
            "raw_order": tuple(spec.raw_frame_order) == plan.raw_order,
            "validity_mask": tuple(spec.validity_mask) == plan.validity_mask,
            "raw_to_slot_mapping": tuple(spec.raw_to_slot_mapping) == tuple(range(spec.frame_count)),
            "frame_count": spec.frame_count == len(plan.raw_order),
            "three_phases_per_orientation": spec.phases_per_orientation == 3,
            "direct_synthesis_allowed": source.get("simulation_training_blocked") is False,
        }
        if not all(checks.values()):
            raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"{plan.label}: {checks}")
        protocols.append({
            "label": plan.label,
            "protocol_id": spec.protocol_id,
            "protocol_hash": spec.protocol_hash,
            "protocol_file": str(plan.protocol_file.resolve()),
            "protocol_file_sha256": sha_file(plan.protocol_file),
            "frame_count": spec.frame_count,
            "orientation_count": spec.orientation_count,
            "phases_per_orientation": spec.phases_per_orientation,
            "orientation_ids": list(source["orientation_ids"]),
            "orientation_angles_degree_mod_180": list(spec.orientation_angles),
            "nominal_phases_radian": list(spec.nominal_phase_values),
            "raw_frame_order": list(spec.raw_frame_order),
            "raw_to_slot_mapping": list(spec.raw_to_slot_mapping),
            "validity_mask": list(spec.validity_mask),
            "evidence_level": source.get("evidence_level"),
            "historical_acquisition_receipt": source.get("historical_acquisition_receipt"),
            "direct_protocol_specific_synthesis": True,
            "not_retrospective_subset": True,
            "checks": checks,
        })
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "comparison_scope": (
            "controller-defined acquisition-budget comparison; frame count and orientation support both change; "
            "not a single-factor causal frame-count experiment"
        ),
        "protocols": protocols,
    }
    write_json(run_dir / "R1C2_PROTOCOL_RECEIPT.json", receipt)
    return receipt


def audit_dataset(run_dir: Path) -> list[dict[str, Any]]:
    current = enumerate_ceshiji(CESHIJI_ROOT)
    if len(current) != 30:
        raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", f"TIFF count={len(current)}")
    counts = {name: sum(row["structure_class"] == name for row in current) for name in ("CCPs", "ER", "microtubules")}
    if counts != {"CCPs": 10, "ER": 10, "microtubules": 10}:
        raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", str(counts))
    with PRIOR_CESHIJI_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        prior = list(csv.DictReader(handle))
    if len(prior) != 30:
        raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", "prior CESHIJI manifest")
    rows: list[dict[str, Any]] = []
    for index, (now, old) in enumerate(zip(current, prior)):
        checks = {
            "order": int(now["order"]) == index == int(old["order"]),
            "sample_id": now["sample_id"] == old["sample_id"],
            "absolute_path": Path(now["absolute_path"]).resolve() == Path(old["absolute_path"]).resolve(),
            "file_sha256": now["sha256"] == old["sha256"],
            "normalized_pixel_sha256": now["normalized_pixel_sha256"] == old["normalized_pixel_sha256"],
            "normalized_array_sha256": now["normalized_array_sha256"] == old["normalized_array_sha256"],
            "shape": now["image_shape"] == old["image_shape"] == "1004x1004",
            "dtype": now["dtype"] == old["dtype"],
        }
        if not all(checks.values()):
            raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", f"{now['sample_id']}: {checks}")
        structure = {"CCPs": "CCP", "ER": "ER", "microtubules": "MT"}[now["structure_class"]]
        rows.append({
            "order": index,
            "sample_id": now["sample_id"],
            "parent_id": f"{structure}:{now['sample_id'].split('Cell_')[-1].split('_')[0]}",
            "class": structure,
            "absolute_path": now["absolute_path"],
            "relative_path": now["relative_path"],
            "shape": now["image_shape"],
            "dtype": now["dtype"],
            "size_bytes": now["file_size_bytes"],
            "file_sha256": now["sha256"],
            "normalized_pixel_sha256": now["normalized_pixel_sha256"],
            "normalized_array_sha256": now["normalized_array_sha256"],
            "prior_ceshiji_exact_match": True,
            "common_order_for_3f_6f_9f": True,
        })
    atomic_replace(run_dir / "R1C2_DATASET_MANIFEST.csv", csv_bytes(rows, tuple(rows[0])))
    return rows


def audit_cross_protocol_config(details: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    configs = [details[plan.protocol_id]["config"] for plan in PLANS]
    forward_payloads = [config["forward"] for config in configs]
    training_inference = [
        {
            "diffusion_steps": config["training"]["diffusion_steps"],
            "beta_schedule": config["training"]["beta_schedule"],
            "model": config["model"],
        }
        for config in configs
    ]
    checks = {
        "same_non_geometry_forward_config": forward_payloads[0] == forward_payloads[1] == forward_payloads[2],
        "same_model_and_diffusion_config": training_inference[0] == training_inference[1] == training_inference[2],
        "same_normalization": all(config["forward"]["normalization"] == configs[0]["forward"]["normalization"] for config in configs),
        "same_reconstruction_grid": all(int(config["forward"]["upsample"]) == int(configs[0]["forward"]["upsample"]) for config in configs),
    }
    if not all(checks.values()):
        raise R1C2Blocked("R1C2_CHECKPOINT_OR_PROTOCOL_UNRESOLVED", f"cross-protocol config: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "shared_forward_config_sha256": hashlib.sha256(canonical_bytes(forward_payloads[0])).hexdigest(),
        "shared_stage1_policy": STAGE1_POLICY,
        "shared_refinement_config": REFINEMENT_CONFIG.receipt(),
        "measurement_seed_policy": "COMMON_MEASUREMENT_SEED_BASE + sample_order; same base seed for 3F/6F/9F",
        "diffusion_seed_policy": "COMMON_DIFFUSION_SEED_BASE + sample_order; same seed for 3F/6F/9F",
    }


def load_ema_model(
    plan: ProtocolPlan, detail: Mapping[str, Any], device: torch.device
) -> tuple[APDConditionedUNet2D, DiffusionScheduler2D]:
    model, scheduler, _metadata = load_stage1_registered(
        detail["config"],
        plan.checkpoint,
        detail["checkpoint"]["checkpoint_sha256"],
        device,
        protocol_id=plan.protocol_id,
    )
    return model, scheduler


@torch.no_grad()
def stage1_reconstruct(
    raw: torch.Tensor,
    protocol_id: str,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    seed: int,
) -> tuple[torch.Tensor, float]:
    if protocol_id == "DMD_9F_3O3P":
        output, elapsed, _peak = stage1_reconstruct_registered_tiled(
            raw, model, scheduler, protocol_id=protocol_id, seed=seed,
            tile_size=320, core_size=160, tile_batch_size=4,
        )
    else:
        output, elapsed, _peak = stage1_reconstruct_registered(
            raw, model, scheduler, protocol_id=protocol_id, seed=seed
        )
    return output, elapsed


def refine_protocol(
    initial: torch.Tensor,
    raw: torch.Tensor,
    protocol_id: str,
    sim_config: SIM2DConfig,
    theta: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, float, float, float]:
    spec = require_protocol(protocol_id)
    if raw.shape[1] != spec.frame_count or initial.shape[-2:] != raw.shape[-2:]:
        raise ValueError("protocol refinement support mismatch")
    _observed_slotted, mask = embed_raw_to_slots_2d(raw, protocol_id)
    mask_vector = tuple(int(value) for value in mask[0, :, 0, 0].detach().cpu().tolist())
    if mask_vector != tuple(spec.validity_mask):
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "validity mask drift")
    result = masked_refine(
        initial,
        raw,
        mask[0, :, 0, 0],
        {
            "protocol_id": protocol_id,
            "protocol_hash": spec.protocol_hash,
            "raw_frame_order": list(spec.raw_frame_order),
        },
        {"sim_config": sim_config, "theta": theta},
        REFINEMENT_CONFIG,
    )
    if len(result.objective_history) != 41 or len(result.observed_nrmse_history) != 41:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "shared masked_refine history")
    return (
        result.final_reconstruction,
        result.runtime_seconds,
        result.objective_history[-1],
        result.observed_nrmse_history[-1],
    )


def _metrics_module() -> Any:
    import importlib.util

    if sha_file(METRICS_SOURCE) != METRICS_SHA256:
        raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", "metrics source SHA mismatch")
    spec = importlib.util.spec_from_file_location("r1c2_official_metrics", METRICS_SOURCE)
    if spec is None or spec.loader is None:
        raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", "metrics import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _theta_payload(theta: Mapping[str, torch.Tensor]) -> dict[str, list[float]]:
    return {
        key: [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
        for key, value in sorted(theta.items())
    }


def run_inference(
    run_dir: Path,
    dataset: Sequence[Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from unisim.revision_r1.physmap6_experiment import assert_no_external_cuda_compute

    assert_no_external_cuda_compute()
    if not torch.cuda.is_available():
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "CUDA unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    monitor = CudaContentionMonitor(interval_seconds=1.0)
    monitor.start()
    metrics = _metrics_module()
    rows: list[dict[str, Any]] = []
    per_fov_path = run_dir / "R1C2_FRAME_BUDGET_PER_FOV.csv"
    for plan in PLANS:
        detail = details[plan.protocol_id]
        config = detail["config"]
        model, scheduler = load_ema_model(plan, detail, device)
        sim_config = _config_for_sim(config)
        theta = nominal_theta_2d(sim_config, device)
        theta_hash = hashlib.sha256(canonical_bytes(_theta_payload(theta))).hexdigest()
        spec = require_protocol(plan.protocol_id)
        mask_hash = sha_array(np.asarray(spec.validity_mask, dtype=np.float32))
        for sample in dataset:
            monitor.checkpoint()
            order = int(sample["order"])
            gt = normalize_image(tifffile.imread(Path(sample["absolute_path"])))
            if sha_array(gt) != sample["normalized_array_sha256"]:
                raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", sample["sample_id"])
            gt_tensor = torch.from_numpy(gt)[None, None].to(device=device, dtype=torch.float32)
            measurement_seed = COMMON_MEASUREMENT_SEED_BASE + order
            diffusion_seed = COMMON_DIFFUSION_SEED_BASE + order
            generator = torch.Generator(device=device).manual_seed(measurement_seed)
            with torch.no_grad():
                raw, _ = forward_protocol_sim_2d(
                    gt_tensor,
                    sim_config,
                    plan.protocol_id,
                    theta=dict(theta),
                    randomize=False,
                    noise_generator=generator,
                )
            if raw.shape != (1, spec.frame_count, 1004, 1004) or not bool(torch.isfinite(raw).all()):
                raise R1C2Blocked("R1C2_NONFINITE_RESULT", f"raw {plan.label}/{sample['sample_id']}")
            raw_np = np.ascontiguousarray(raw[0].detach().cpu().numpy(), dtype=np.float32)
            raw_path = run_dir / "measurements" / plan.label / f"{order:03d}_{sample['sample_id']}.npz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            stream = io.BytesIO()
            np.savez_compressed(
                stream,
                raw_stack=raw_np,
                raw_frame_order=np.asarray(plan.raw_order),
                protocol_id=np.asarray(plan.protocol_id),
                measurement_seed=np.asarray(measurement_seed, dtype=np.int64),
                theta_json=np.asarray(json.dumps(_theta_payload(theta), sort_keys=True)),
            )
            atomic_replace(raw_path, stream.getvalue())
            if raw.device.type == "cuda":
                torch.cuda.synchronize(device)
            wf_started = time.perf_counter()
            wf = raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            torch.cuda.synchronize(device)
            wf_runtime = time.perf_counter() - wf_started
            x_ws, stage1_runtime = stage1_reconstruct(
                raw, plan.protocol_id, model, scheduler, diffusion_seed
            )
            apd, stage2_runtime, objective, nrmse = refine_protocol(
                x_ws, raw, plan.protocol_id, sim_config, theta
            )
            method_values = (
                ("WF", wf, wf_runtime, 0.0, 0.0, 0.0),
                ("APD-SIM", apd, stage1_runtime + stage2_runtime, stage1_runtime, stage2_runtime, objective),
            )
            for method, tensor, runtime, s1_runtime, s2_runtime, final_objective in method_values:
                prediction = np.ascontiguousarray(tensor[0, 0].detach().cpu().numpy(), dtype=np.float32)
                if prediction.shape != gt.shape or not bool(np.isfinite(prediction).all()):
                    raise R1C2Blocked("R1C2_NONFINITE_RESULT", f"{plan.label}/{sample['sample_id']}/{method}")
                psnr = float(metrics.psnr_native(gt, prediction))
                ssim = float(metrics.ssim_native(gt, prediction))
                frc, curves = metrics.reference_frc_1over7(gt, prediction)
                numeric = [psnr, ssim, runtime]
                if not all(math.isfinite(value) for value in numeric):
                    raise R1C2Blocked("R1C2_NONFINITE_RESULT", f"metric {plan.label}/{sample['sample_id']}/{method}")
                frc_status = "CUTOFF"
                if frc["right_censored_at_nyquist"]:
                    frc_status = "RIGHT_CENSORED"
                elif frc["unresolved_no_crossing"]:
                    frc_status = "UNRESOLVED"
                prediction_path = (
                    run_dir / "predictions" / plan.label / method.replace("-", "")
                    / f"{order:03d}_{sample['sample_id']}.npy"
                )
                save_npy(prediction_path, prediction)
                if order == REPRESENTATIVE_SAMPLE_ORDER:
                    curve_path = (
                        run_dir / "representative" / "frc_curves"
                        / f"{plan.label}_{method.replace('-', '')}.npz"
                    )
                    curve_path.parent.mkdir(parents=True, exist_ok=True)
                    curve_stream = io.BytesIO()
                    np.savez_compressed(curve_stream, **curves)
                    atomic_replace(curve_path, curve_stream.getvalue())
                rows.append({
                    "sample_order": order,
                    "sample_id": sample["sample_id"],
                    "parent_id": sample["parent_id"],
                    "structure": sample["class"],
                    "protocol_label": plan.label,
                    "protocol_id": plan.protocol_id,
                    "protocol_hash": spec.protocol_hash,
                    "frame_count": spec.frame_count,
                    "raw_frame_order": "/".join(plan.raw_order),
                    "method": method,
                    "gt_path": sample["absolute_path"],
                    "gt_file_sha256": sample["file_sha256"],
                    "gt_normalized_sha256": sample["normalized_array_sha256"],
                    "checkpoint_path": str(plan.checkpoint.resolve()) if method == "APD-SIM" else "NA",
                    "checkpoint_sha256": detail["checkpoint"]["checkpoint_sha256"] if method == "APD-SIM" else "NA",
                    "checkpoint_selected_event": detail["checkpoint"]["selected_loop_data_event"] if method == "APD-SIM" else "NA",
                    "checkpoint_committed_updates": detail["checkpoint"]["actual_committed_optimizer_updates"] if method == "APD-SIM" else "NA",
                    "inference_weight_branch": "ema" if method == "APD-SIM" else "NA",
                    "measurement_seed": measurement_seed,
                    "diffusion_seed": diffusion_seed if method == "APD-SIM" else "NA",
                    "raw_stack_sha256": sha_array(raw_np),
                    "raw_npz_path": str(raw_path.resolve()),
                    "raw_npz_file_sha256": sha_file(raw_path),
                    "validity_mask_sha256": mask_hash,
                    "forward_parameters_sha256": theta_hash,
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_file_sha256": sha_file(prediction_path),
                    "prediction_array_sha256": sha_array(prediction),
                    "psnr": psnr,
                    "ssim": ssim,
                    "frc_status": frc_status,
                    "frc_cutoff_cycles_per_pixel": "" if frc["cutoff_cycles_per_pixel"] is None else frc["cutoff_cycles_per_pixel"],
                    "frc_spatial_period_px": "" if frc["cutoff_derived_spatial_period_px"] is None else frc["cutoff_derived_spatial_period_px"],
                    "frc_right_censored": bool(frc["right_censored_at_nyquist"]),
                    "frc_unresolved": bool(frc["unresolved_no_crossing"]),
                    "runtime_seconds": runtime,
                    "stage1_runtime_seconds": s1_runtime,
                    "stage2_runtime_seconds": s2_runtime,
                    "final_objective": final_objective if method == "APD-SIM" else "NA",
                    "observed_frame_nrmse": nrmse if method == "APD-SIM" else "NA",
                    "finite": True,
                    "status": "PASS",
                })
            atomic_replace(per_fov_path, csv_bytes(rows, PER_FOV_FIELDS))
            monitor.checkpoint()
            del gt_tensor, raw, wf, x_ws, apd
            torch.cuda.empty_cache()
        del model, scheduler
        torch.cuda.empty_cache()
    expected = {
        (order, plan.label, method)
        for order in range(30)
        for plan in PLANS
        for method in METHODS
    }
    observed = {(int(row["sample_order"]), row["protocol_label"], row["method"]) for row in rows}
    if len(rows) != 180 or observed != expected:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "180-row grid incomplete")
    monitor_receipt = monitor.stop_and_validate()
    write_json(run_dir / "R1C2_GPU_CONTENTION_MONITOR.json", monitor_receipt)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise R1C2Blocked("R1C2_NONFINITE_RESULT", "invalid descriptive vector")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate_per_fov(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "R1C2_FRAME_BUDGET_PER_FOV.csv"
    rows = _read_csv(path)
    if len(rows) != 180 or tuple(rows[0]) != PER_FOV_FIELDS:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "per-FOV schema/count")
    expected = {
        (order, plan.label, method)
        for order in range(30)
        for plan in PLANS
        for method in METHODS
    }
    keys = {(int(row["sample_order"]), row["protocol_label"], row["method"]) for row in rows}
    if keys != expected or len(keys) != len(rows):
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "per-FOV grid/duplicates")
    identity_by_order: dict[int, tuple[str, str, str, str]] = {}
    raw_by_protocol_order: dict[tuple[int, str], tuple[str, str, str]] = {}
    validated_raw_files: set[Path] = set()
    seed_by_order: dict[int, str] = {}
    diffusion_seed_by_order: dict[int, str] = {}
    for row in rows:
        order = int(row["sample_order"])
        identity = (row["sample_id"], row["parent_id"], row["structure"], row["gt_normalized_sha256"])
        if order in identity_by_order and identity_by_order[order] != identity:
            raise R1C2Blocked("R1C2_DATASET_IDENTITY_MISMATCH", f"order={order}")
        identity_by_order[order] = identity
        raw_key = (order, row["protocol_label"])
        raw_identity = (row["raw_stack_sha256"], row["validity_mask_sha256"], row["forward_parameters_sha256"])
        if raw_key in raw_by_protocol_order and raw_by_protocol_order[raw_key] != raw_identity:
            raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"shared raw drift: {raw_key}")
        raw_by_protocol_order[raw_key] = raw_identity
        if order in seed_by_order and seed_by_order[order] != row["measurement_seed"]:
            raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"measurement seed drift: {order}")
        seed_by_order[order] = row["measurement_seed"]
        if row["method"] == "APD-SIM":
            if order in diffusion_seed_by_order and diffusion_seed_by_order[order] != row["diffusion_seed"]:
                raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"diffusion seed drift: {order}")
            diffusion_seed_by_order[order] = row["diffusion_seed"]
        raw_path = Path(row["raw_npz_path"])
        if not raw_path.is_file() or sha_file(raw_path) != row["raw_npz_file_sha256"]:
            raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"raw NPZ: {raw_key}")
        if raw_path not in validated_raw_files:
            with np.load(raw_path, allow_pickle=False) as archive:
                raw_array = np.asarray(archive["raw_stack"])
                protocol_text = str(np.asarray(archive["protocol_id"]).item())
            if raw_array.dtype != np.float32 or raw_array.shape != (int(row["frame_count"]), 1004, 1004) or not np.isfinite(raw_array).all() or sha_array(raw_array) != row["raw_stack_sha256"] or protocol_text != row["protocol_id"]:
                raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"raw NPZ payload: {raw_key}")
            validated_raw_files.add(raw_path)
        if row["status"] != "PASS" or row["finite"].lower() != "true":
            raise R1C2Blocked("R1C2_NONFINITE_RESULT", f"row status: {raw_key}")
        for field in ("psnr", "ssim", "runtime_seconds"):
            if not math.isfinite(float(row[field])):
                raise R1C2Blocked("R1C2_NONFINITE_RESULT", f"{field}: {raw_key}")
        prediction_path = Path(row["prediction_path"])
        prediction = np.load(prediction_path, allow_pickle=False)
        if (
            sha_file(prediction_path) != row["prediction_file_sha256"]
            or sha_array(prediction) != row["prediction_array_sha256"]
            or prediction.dtype != np.float32
            or prediction.shape != (1004, 1004)
            or not np.isfinite(prediction).all()
        ):
            raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", str(prediction_path))
        status = row["frc_status"]
        if status == "CUTOFF":
            if not row["frc_spatial_period_px"] or not math.isfinite(float(row["frc_spatial_period_px"])):
                raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", f"cutoff row: {raw_key}")
        elif status == "RIGHT_CENSORED":
            if row["frc_spatial_period_px"] or row["frc_right_censored"].lower() != "true":
                raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", f"censor row: {raw_key}")
        elif status == "UNRESOLVED":
            if row["frc_spatial_period_px"] or row["frc_unresolved"].lower() != "true":
                raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", f"unresolved row: {raw_key}")
        else:
            raise R1C2Blocked("R1C2_FRC_ANALYSIS_INVALID", status)
    if len(identity_by_order) != 30 or len(raw_by_protocol_order) != 90:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "identity/raw cardinality")
    return rows


def analyze_results(run_dir: Path) -> dict[str, Any]:
    rows = validate_per_fov(run_dir)
    metrics = _metrics_module()
    stats: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE_VALIDATED",
        "source_csv": str((run_dir / "R1C2_FRAME_BUDGET_PER_FOV.csv").resolve()),
        "source_csv_sha256": sha_file(run_dir / "R1C2_FRAME_BUDGET_PER_FOV.csv"),
        "n_fov": 30,
        "n_rows": 180,
        "unit": "parent/FOV; 30 unique manifest-derived parent IDs",
        "biological_independence": "UNRESOLVED beyond file-level parent identity",
        "protocol_scope": "frame count and orientation support both change",
        "groups": {},
        "frc_policy": {
            "threshold": "1/7",
            "dc_excluded": True,
            "crossing": "first downward crossing with linear interpolation",
            "right_censoring": "Nyquist; censored/unresolved values excluded from ordinary exact-value means",
        },
    }
    class_rows: list[dict[str, Any]] = []
    censor_rows: list[dict[str, Any]] = []
    for plan in PLANS:
        for method in METHODS:
            group = [row for row in rows if row["protocol_label"] == plan.label and row["method"] == method]
            key = f"{plan.label}/{method}"
            payload: dict[str, Any] = {}
            for metric_name in ("psnr", "ssim"):
                values = [float(row[metric_name]) for row in group]
                payload[metric_name] = _descriptive(values)
                summaries = metrics.class_summaries(
                    values,
                    [row["structure"] for row in group],
                    parent_ids=[row["parent_id"] for row in group],
                    class_order=["CCP", "ER", "MT"],
                )
                payload[f"{metric_name}_by_structure"] = summaries["classes"]
                for structure, summary in summaries["classes"].items():
                    class_rows.append({
                        "protocol": plan.label,
                        "method": method,
                        "metric": metric_name.upper(),
                        "structure": structure,
                        **{name: summary[name] for name in ("n", "mean", "sd", "median", "q1", "q3", "iqr")},
                        "exact_n": summary["n"], "right_censored_n": 0, "unresolved_n": 0,
                        "status": "PASS_EXACT",
                    })
            exact_periods = [float(row["frc_spatial_period_px"]) for row in group if row["frc_status"] == "CUTOFF"]
            status_counts = {status: sum(row["frc_status"] == status for row in group) for status in ("CUTOFF", "RIGHT_CENSORED", "UNRESOLVED")}
            payload["frc_spatial_period_px"] = {
                "status_counts": status_counts,
                "exact_cutoff_descriptive": _descriptive(exact_periods) if exact_periods else None,
                "ordinary_mean_all_30_not_estimated": status_counts["CUTOFF"] != 30,
            }
            for structure in ("CCP", "ER", "MT"):
                selected_structure = [row for row in group if row["structure"] == structure]
                exact_structure = [float(row["frc_spatial_period_px"]) for row in selected_structure if row["frc_status"] == "CUTOFF"]
                censored_n = sum(row["frc_status"] == "RIGHT_CENSORED" for row in selected_structure)
                unresolved_n = sum(row["frc_status"] == "UNRESOLVED" for row in selected_structure)
                exact_summary = _descriptive(exact_structure) if exact_structure and not censored_n and not unresolved_n else None
                class_rows.append({
                    "protocol": plan.label, "method": method, "metric": "FRC_SPATIAL_PERIOD_PX",
                    "structure": structure, "n": len(selected_structure),
                    "mean": "" if exact_summary is None else exact_summary["mean"],
                    "sd": "" if exact_summary is None else exact_summary["sd"],
                    "median": "" if exact_summary is None else exact_summary["median"],
                    "q1": "" if exact_summary is None else exact_summary["q1"],
                    "q3": "" if exact_summary is None else exact_summary["q3"],
                    "iqr": "" if exact_summary is None else exact_summary["iqr"],
                    "exact_n": len(exact_structure), "right_censored_n": censored_n,
                    "unresolved_n": unresolved_n,
                    "status": "PASS_EXACT" if exact_summary is not None else "NOT_SUMMARIZED_AS_ORDINARY_EXACT_VALUES",
                })
            stats["groups"][key] = payload
            for row in group:
                censor_rows.append({
                    "sample_order": row["sample_order"], "sample_id": row["sample_id"],
                    "parent_id": row["parent_id"], "structure": row["structure"],
                    "protocol": plan.label, "method": method, "frc_status": row["frc_status"],
                    "cutoff_cycles_per_pixel": row["frc_cutoff_cycles_per_pixel"],
                    "spatial_period_px": row["frc_spatial_period_px"],
                    "right_censoring_boundary_period_px": 2.0 if row["frc_status"] == "RIGHT_CENSORED" else "",
                    "used_as_exact_value_in_mean": row["frc_status"] == "CUTOFF",
                })

    paired_rows: list[dict[str, Any]] = []
    apd = {(int(row["sample_order"]), row["protocol_label"]): row for row in rows if row["method"] == "APD-SIM"}
    for left_label, right_label in (("DMD-3F", "DMD-6F"), ("DMD-6F", "DMD-9F")):
        for metric_name in ("psnr", "ssim"):
            left = np.asarray([float(apd[(order, left_label)][metric_name]) for order in range(30)], dtype=np.float64)
            right = np.asarray([float(apd[(order, right_label)][metric_name]) for order in range(30)], dtype=np.float64)
            parents = [apd[(order, left_label)]["parent_id"] for order in range(30)]
            classes = [apd[(order, left_label)]["structure"] for order in range(30)]
            # The reviewer-requested direction is the lower-budget/left protocol minus the
            # higher-budget/right protocol (3F-6F and 6F-9F).
            bootstrap = metrics.parent_image_bootstrap_ci(
                left, right, parents, class_labels=classes, n_resamples=10_000,
                seed=20260815 + (0 if metric_name == "psnr" else 10) + (0 if left_label == "DMD-3F" else 1),
            )
            differences = left - right
            for order, difference in enumerate(differences):
                paired_rows.append({
                    "row_type": "PER_FOV", "contrast": f"{left_label}_minus_{right_label}",
                    "metric": metric_name.upper(), "sample_order": order,
                    "sample_id": apd[(order, left_label)]["sample_id"],
                    "parent_id": parents[order], "structure": classes[order],
                    "difference": float(difference), "estimate": "", "ci_low": "", "ci_high": "",
                    "n": 30, "bootstrap_resamples": 10000, "status": "PASS",
                })
            paired_rows.append({
                "row_type": "SUMMARY", "contrast": f"{left_label}_minus_{right_label}",
                "metric": metric_name.upper(), "sample_order": "", "sample_id": "", "parent_id": "",
                "structure": "ALL", "difference": "", "estimate": bootstrap["estimate"],
                "ci_low": bootstrap["confidence_interval"][0], "ci_high": bootstrap["confidence_interval"][1],
                "n": 30, "bootstrap_resamples": 10000, "status": "PASS",
            })
    # FRC differences are not silently complete-case analyzed when any value is censored/unresolved.
    for left_label, right_label in (("DMD-3F", "DMD-6F"), ("DMD-6F", "DMD-9F")):
        statuses = [apd[(order, label)]["frc_status"] for order in range(30) for label in (left_label, right_label)]
        status = "NOT_ESTIMATED_DUE_TO_CENSORING" if any(item != "CUTOFF" for item in statuses) else "PASS"
        if status == "PASS":
            left = np.asarray([float(apd[(order, left_label)]["frc_spatial_period_px"]) for order in range(30)])
            right = np.asarray([float(apd[(order, right_label)]["frc_spatial_period_px"]) for order in range(30)])
            bootstrap = metrics.parent_image_bootstrap_ci(
                left, right, [apd[(order, left_label)]["parent_id"] for order in range(30)],
                class_labels=[apd[(order, left_label)]["structure"] for order in range(30)],
                n_resamples=10_000, seed=20260835,
            )
            estimate, (low, high) = bootstrap["estimate"], bootstrap["confidence_interval"]
        else:
            estimate = low = high = ""
        paired_rows.append({
            "row_type": "SUMMARY", "contrast": f"{left_label}_minus_{right_label}",
            "metric": "FRC_SPATIAL_PERIOD_PX", "sample_order": "", "sample_id": "", "parent_id": "",
            "structure": "ALL", "difference": "", "estimate": estimate, "ci_low": low, "ci_high": high,
            "n": 30, "bootstrap_resamples": 10000 if status == "PASS" else 0, "status": status,
        })

    class_fields = ("protocol", "method", "metric", "structure", "n", "mean", "sd", "median", "q1", "q3", "iqr", "exact_n", "right_censored_n", "unresolved_n", "status")
    paired_fields = ("row_type", "contrast", "metric", "sample_order", "sample_id", "parent_id", "structure", "difference", "estimate", "ci_low", "ci_high", "n", "bootstrap_resamples", "status")
    censor_fields = tuple(censor_rows[0])
    atomic_replace(run_dir / "R1C2_FRAME_BUDGET_CLASS_STRATIFIED.csv", csv_bytes(class_rows, class_fields))
    atomic_replace(run_dir / "R1C2_FRAME_BUDGET_PAIRED_DIFFERENCES.csv", csv_bytes(paired_rows, paired_fields))
    atomic_replace(run_dir / "R1C2_FRC_CENSORING.csv", csv_bytes(censor_rows, censor_fields))
    stats["paired_differences"] = [row for row in paired_rows if row["row_type"] == "SUMMARY"]
    write_json(run_dir / "R1C2_FRAME_BUDGET_STATS.json", stats)
    return stats


def _fixed_pdf_metadata(title: str) -> dict[str, Any]:
    fixed = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return {
        "Title": title,
        "Creator": "R1C2_FRAME_BUDGET_30FOV_V1",
        "CreationDate": fixed,
        "ModDate": fixed,
    }


def generate_figures(run_dir: Path, dataset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    from unisim.revision_r1.physmap6_fig5_prespecified import select_gt_profile

    rows = validate_per_fov(run_dir)
    representative = dataset[REPRESENTATIVE_SAMPLE_ORDER]
    gt = normalize_image(tifffile.imread(Path(representative["absolute_path"])))
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for row in rows:
        if int(row["sample_order"]) == REPRESENTATIVE_SAMPLE_ORDER:
            predictions[(row["protocol_label"], row["method"])] = np.load(row["prediction_path"], allow_pickle=False)
    required = {(plan.label, method) for plan in PLANS for method in METHODS}
    if set(predictions) != required:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "representative predictions")
    y = REPRESENTATIVE_ROI["y"]; x = REPRESENTATIVE_ROI["x"]
    height = REPRESENTATIVE_ROI["height"]; width = REPRESENTATIVE_ROI["width"]
    values = [gt, predictions[("DMD-3F", "APD-SIM")], predictions[("DMD-6F", "APD-SIM")],
              predictions[("DMD-9F", "APD-SIM")], predictions[("DMD-9F", "WF")]]
    labels = ["GT", "APD-SIM-3", "APD-SIM-6", "APD-SIM-9", "DMD-9F WF"]
    figure = plt.figure(figsize=(14.5, 10.8), constrained_layout=True)
    grid = GridSpec(4, 6, figure=figure, height_ratios=[1.0, 1.0, 0.9, 0.9])
    for index, (image, label) in enumerate(zip(values, labels)):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ffcc00", linewidth=1.0))
        axis.set_title(label, fontsize=10, weight="bold"); axis.set_xticks([]); axis.set_yticks([])
        if index == 0:
            axis.set_ylabel("(a) full FOV")
        zoom = figure.add_subplot(grid[1, index])
        zoom.imshow(image[y:y + height, x:x + width], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        zoom.set_xticks([]); zoom.set_yticks([])
        if index == 0:
            zoom.set_ylabel("(b) fixed ROI")
    note = figure.add_subplot(grid[:2, 5]); note.axis("off")
    note.text(0.0, 0.95, "Prespecified field\nER Cell 068", va="top", weight="bold")
    note.text(0.0, 0.72, "Fixed display range\n[0, 1] for every panel", va="top")
    note.text(0.0, 0.48, "Direct protocol-specific\nsynthesis; not retrospective\nsubsampling", va="top")
    colors = {"DMD-3F": "#d62728", "DMD-6F": "#1f77b4", "DMD-9F": "#2ca02c"}
    for metric_index, metric_name in enumerate(("psnr", "ssim")):
        axis = figure.add_subplot(grid[2, metric_index * 2:(metric_index + 1) * 2])
        for order in range(30):
            values_metric = [float(next(row[metric_name] for row in rows if int(row["sample_order"]) == order and row["protocol_label"] == label and row["method"] == "APD-SIM")) for label in colors]
            axis.plot(range(3), values_metric, color="#888888", alpha=0.30, linewidth=0.65)
        for protocol_index, label in enumerate(colors):
            vector = [float(row[metric_name]) for row in rows if row["protocol_label"] == label and row["method"] == "APD-SIM"]
            axis.scatter(np.full(30, protocol_index), vector, color=colors[label], s=14, alpha=0.75)
            axis.plot(protocol_index, np.mean(vector), marker="D", color="black", markersize=5)
        axis.set_xticks(range(3), ["3F", "6F", "9F"])
        axis.set_ylabel("PSNR (dB)" if metric_name == "psnr" else "SSIM")
        axis.set_title("(c) Paired PSNR" if metric_name == "psnr" else "(d) Paired SSIM", loc="left", weight="bold")
        axis.grid(axis="y", alpha=0.2)
    frc_axis = figure.add_subplot(grid[2:, 4:])
    for order in range(30):
        display_values = []
        unresolved_present = False
        for label in colors:
            row = next(row for row in rows if int(row["sample_order"]) == order and row["protocol_label"] == label and row["method"] == "APD-SIM")
            unresolved_present = unresolved_present or row["frc_status"] == "UNRESOLVED"
            value = float(row["frc_spatial_period_px"]) if row["frc_status"] == "CUTOFF" else 2.0
            display_values.append(value)
        if not unresolved_present:
            frc_axis.plot(range(3), display_values, color="#888888", alpha=0.25, linewidth=0.65)
    for protocol_index, label in enumerate(colors):
        group = [row for row in rows if row["protocol_label"] == label and row["method"] == "APD-SIM"]
        exact = [float(row["frc_spatial_period_px"]) for row in group if row["frc_status"] == "CUTOFF"]
        censored = sum(row["frc_status"] == "RIGHT_CENSORED" for row in group)
        unresolved = sum(row["frc_status"] == "UNRESOLVED" for row in group)
        frc_axis.scatter(np.full(len(exact), protocol_index), exact, color=colors[label], s=16, alpha=0.8)
        if censored:
            frc_axis.scatter(np.full(censored, protocol_index), np.full(censored, 2.0), marker="v", color=colors[label], s=30, label="Nyquist right-censored" if protocol_index == 0 else None)
        if unresolved:
            frc_axis.scatter(np.full(unresolved, protocol_index), np.full(unresolved, 2.15), marker="x", color=colors[label], s=28, label="unresolved" if protocol_index == 0 else None)
    frc_axis.axhline(2.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    frc_axis.set_xticks(range(3), ["3F", "6F", "9F"]); frc_axis.set_ylabel("FRC spatial period (px)")
    frc_axis.set_title("(e) Paired GT-referenced FRC", loc="left", weight="bold")
    frc_axis.grid(axis="y", alpha=0.2); frc_axis.legend(fontsize=7, frameon=False)
    figure.suptitle("Direct DMD acquisition-budget comparison across 30 fixed FOVs", weight="bold")
    png_path = run_dir / "FIG3_FRAME_BUDGET_30FOV.png"
    pdf_path = run_dir / "FIG3_FRAME_BUDGET_30FOV.pdf"
    figure.savefig(png_path, dpi=220, metadata={"Software": "matplotlib"})
    figure.savefig(pdf_path, metadata=_fixed_pdf_metadata("Reviewer 1 Comment 2 direct frame-budget Figure 3"))
    plt.close(figure)

    # Supplementary spectra, a GT-only selected profile, and the stored single-FOV FRC curves.
    profile = select_gt_profile(gt[y:y + height, x:x + width])
    profile_row = y + int(profile["row_index"])
    supplementary = plt.figure(figsize=(14, 10), constrained_layout=True)
    supplemental_grid = GridSpec(3, 3, figure=supplementary)
    for index, (image, label) in enumerate(zip(values[1:4], labels[1:4])):
        axis = supplementary.add_subplot(supplemental_grid[0, index])
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image - np.mean(image)))))
        axis.imshow(spectrum, cmap="magma", interpolation="nearest")
        axis.set_title(f"{label} spectrum"); axis.set_xticks([]); axis.set_yticks([])
    profile_axis = supplementary.add_subplot(supplemental_grid[1, :])
    for image, label, color in zip(values[:4], labels[:4], ("black", "#d62728", "#1f77b4", "#2ca02c")):
        profile_axis.plot(image[profile_row], label=label, color=color, linewidth=0.9)
    profile_axis.set_title("GT-only selected full-width line profile"); profile_axis.set_xlabel("position (px)"); profile_axis.set_ylabel("native intensity")
    profile_axis.set_ylim(0.0, 1.0); profile_axis.legend(ncol=4, fontsize=8, frameon=False); profile_axis.grid(alpha=0.15)
    for protocol_index, plan in enumerate(PLANS):
        axis = supplementary.add_subplot(supplemental_grid[2, protocol_index])
        for method, linestyle in (("WF", "--"), ("APD-SIM", "-")):
            curve_path = run_dir / "representative" / "frc_curves" / f"{plan.label}_{method.replace('-', '')}.npz"
            with np.load(curve_path, allow_pickle=False) as curves:
                frequency = curves["frequency_cycles_per_pixel"]
                frc = curves["frc"]
            valid = np.isfinite(frequency) & np.isfinite(frc) & (frequency > 0.0)
            axis.plot(frequency[valid], frc[valid], linestyle=linestyle, label=method, linewidth=1.0)
        axis.axhline(1.0 / 7.0, color="black", linestyle=":", linewidth=0.8)
        axis.set_xlim(0.0, 0.5); axis.set_ylim(-0.1, 1.05); axis.set_title(plan.label)
        axis.set_xlabel("cycles/pixel"); axis.set_ylabel("FRC"); axis.legend(frameon=False, fontsize=8)
    supplementary.suptitle("Supplementary frame-budget diagnostics: fixed representative FOV", weight="bold")
    supp_pdf = run_dir / "FIGS_FRAME_BUDGET_REPRESENTATIVE_DETAILS.pdf"
    supplementary.savefig(supp_pdf, metadata=_fixed_pdf_metadata("Frame-budget supplementary spectra, profile, and FRC"))
    plt.close(supplementary)
    supplementary_compatibility_alias = run_dir / "FIGS_FRAME_BUDGET_SPECTRA_PROFILE_FRC.pdf"
    atomic_replace(supplementary_compatibility_alias, supp_pdf.read_bytes())
    receipt = {
        "status": "PASS", "representative_selection": "prespecified before inference",
        "sample_order": REPRESENTATIVE_SAMPLE_ORDER, "sample_id": representative["sample_id"],
        "selection_not_performance_based": True, "roi": REPRESENTATIVE_ROI,
        "display_range": [0.0, 1.0], "method_specific_remapping": False,
        "profile_selection": "GT-only maximum row variance, first tie",
        "profile_row_full_fov": profile_row, "pixel_size": "UNRESOLVED; positions reported in pixels",
        "figure_png_sha256": sha_file(png_path), "figure_pdf_sha256": sha_file(pdf_path),
        "supplementary_pdf_sha256": sha_file(supp_pdf),
        "supplementary_compatibility_alias_sha256": sha_file(supplementary_compatibility_alias),
    }
    write_json(run_dir / "R1C2_FIGURE_RECEIPT.json", receipt)
    return receipt


def _tex_write(run_dir: Path, name: str, text: str) -> None:
    forbidden = ("\\input", "TODO", "TBD", "PLACEHOLDER", "[INSERT]")
    if any(token.lower() in text.lower() for token in forbidden):
        raise R1C2Blocked("R1C2_MANUSCRIPT_TEXT_INCOMPLETE", f"{name}: forbidden token")
    atomic_replace(run_dir / name, (text.rstrip() + "\n").encode("utf-8"))


def generate_latex(run_dir: Path, stats: Mapping[str, Any]) -> dict[str, Any]:
    def group(protocol: str, metric: str) -> Mapping[str, Any]:
        return stats["groups"][f"{protocol}/APD-SIM"][metric]

    values = {}
    for protocol in ("DMD-3F", "DMD-6F", "DMD-9F"):
        values[protocol] = {
            "psnr": group(protocol, "psnr"), "ssim": group(protocol, "ssim"),
            "frc": stats["groups"][f"{protocol}/APD-SIM"]["frc_spatial_period_px"],
        }
    contrasts = {(row["contrast"], row["metric"]): row for row in stats["paired_differences"]}
    scope = (
        "This controlled comparison uses three directly synthesized controller protocols; both the number of "
        "frames and the supported illumination orientations change, so it is not interpreted as a single-factor "
        "causal estimate of frame count alone."
    )
    abstract = (
        "Across the same 30 fixed GT-available FOV files, validation-selected protocol-specific EMA models were "
        "evaluated under direct DMD-3F, DMD-6F, and DMD-9F synthesis, with paired native PSNR, SSIM, and "
        "GT-referenced 1/7-FRC summaries; the acquisition-budget comparison changes both frame count and "
        "orientation support and therefore is not a frame-count-only causal experiment."
    )
    results = (
        f"The direct acquisition-budget experiment used 30 fixed FOV files (10 CCPs, 10 ER, and 10 "
        f"microtubules) in identical order for all three protocols. APD-SIM-3 achieved "
        f"{values['DMD-3F']['psnr']['mean']:.2f} $\\pm$ {values['DMD-3F']['psnr']['sd']:.2f} dB and "
        f"SSIM {values['DMD-3F']['ssim']['mean']:.4f} $\\pm$ {values['DMD-3F']['ssim']['sd']:.4f}; "
        f"APD-SIM-6 achieved {values['DMD-6F']['psnr']['mean']:.2f} $\\pm$ {values['DMD-6F']['psnr']['sd']:.2f} dB "
        f"and {values['DMD-6F']['ssim']['mean']:.4f} $\\pm$ {values['DMD-6F']['ssim']['sd']:.4f}; and "
        f"APD-SIM-9 achieved {values['DMD-9F']['psnr']['mean']:.2f} $\\pm$ {values['DMD-9F']['psnr']['sd']:.2f} dB "
        f"and {values['DMD-9F']['ssim']['mean']:.4f} $\\pm$ {values['DMD-9F']['ssim']['sd']:.4f}. "
        f"The paired 3F-minus-6F PSNR difference was {contrasts[('DMD-3F_minus_DMD-6F','PSNR')]['estimate']:+.2f} dB "
        f"(95\\% CI {contrasts[('DMD-3F_minus_DMD-6F','PSNR')]['ci_low']:+.2f} to "
        f"{contrasts[('DMD-3F_minus_DMD-6F','PSNR')]['ci_high']:+.2f}); the paired 6F-minus-9F difference was "
        f"{contrasts[('DMD-6F_minus_DMD-9F','PSNR')]['estimate']:+.2f} dB "
        f"(95\\% CI {contrasts[('DMD-6F_minus_DMD-9F','PSNR')]['ci_low']:+.2f} to "
        f"{contrasts[('DMD-6F_minus_DMD-9F','PSNR')]['ci_high']:+.2f}). " + scope
    )
    discussion = (
        "These results quantify protocol-specific reconstruction pipelines under direct controller-defined "
        "DMD acquisition geometries. The paired design controls FOV identity and non-geometric simulation settings, "
        "but DMD-3F, DMD-6F, and DMD-9F differ jointly in exposure count and orientation support; consequently, "
        "the observed differences should not be attributed exclusively to frame count. FRC observations that did "
        "not yield an exact first downward 1/7 crossing were retained as right-censored at Nyquist or unresolved and "
        "were not converted into artificial exact resolution values."
    )
    conclusion = (
        "Direct protocol-specific evaluation on the same 30 FOV files establishes the measured acquisition-budget "
        "behavior of APD-SIM-3/6/9 while preserving the important limitation that both frame count and orientation "
        "support change across protocols."
    )
    caption = (
        "\\textbf{Figure 3. Direct DMD acquisition-budget comparison on 30 fixed FOV files.} "
        "(a) A representative field selected before inference (ER Cell 068), shown as normalized GT, APD-SIM-3, "
        "APD-SIM-6, APD-SIM-9, and the DMD-9F wide-field mean. (b) The same fixed center ROI is shown for all images "
        "with one method-independent display range [0,1]. (c,d) Per-FOV paired native PSNR and SSIM for the "
        "protocol-specific APD reconstructions; gray lines connect the same FOV. (e) GT-referenced radial FRC "
        "spatial periods using the prespecified 1/7 threshold, excluded DC, and the first downward interpolated "
        "crossing. Downward triangles at 2 pixels denote right-censoring at Nyquist and crosses denote unresolved "
        "no-crossing curves. DMD-3F, DMD-6F, and DMD-9F were synthesized directly from their own controller "
        "protocols rather than retrospectively subset from nine frames. Frame count and orientation support both "
        "change, so this is not a frame-count-only causal comparison."
    )
    table_lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Direct 30-FOV DMD acquisition-budget results. Values are mean $\\pm$ sample SD and median [IQR] across the same 30 FOV files.}",
        "\\label{tab:r1c2-frame-budget}", "\\begin{tabular}{lcc}", "\\toprule",
        "Protocol & PSNR (dB) & SSIM \\\\", "\\midrule",
    ]
    for protocol in ("DMD-3F", "DMD-6F", "DMD-9F"):
        psnr = values[protocol]["psnr"]; ssim = values[protocol]["ssim"]
        table_lines.append(
            f"{protocol.replace('DMD-', 'APD-SIM-')} & {psnr['mean']:.2f} $\\pm$ {psnr['sd']:.2f}; "
            f"{psnr['median']:.2f} [{psnr['q1']:.2f}, {psnr['q3']:.2f}] & "
            f"{ssim['mean']:.4f} $\\pm$ {ssim['sd']:.4f}; {ssim['median']:.4f} "
            f"[{ssim['q1']:.4f}, {ssim['q3']:.4f}] \\\\"
        )
    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    response = (
        "\\textbf{Response to Reviewer 1, Comment 2.} We thank the reviewer for requesting a formal acquisition-budget "
        "comparison. We evaluated validation-selected protocol-specific EMA checkpoints for direct controller-defined "
        "DMD-3F, DMD-6F, and DMD-9F measurements on the same prespecified 30-FOV cohort (10 files per structure). "
        "All protocols used the same non-geometric forward settings, normalization, reconstruction grid, native "
        "metric support, and paired FOV order. We report per-FOV PSNR, SSIM, and GT-referenced radial 1/7-FRC with "
        "explicit Nyquist right-censoring, together with 10,000-resample class-stratified paired-parent bootstrap "
        "intervals. Figure 3 and the accompanying table provide the new results. " + scope +
        " The sealed Reviewer 1 Comment 3 robustness and runtime tables were copied only after independent source "
        "validation; no former PhysMap-9 comparison was introduced."
    )
    figure_tex = "\n".join([
        "\\begin{figure*}[t]", "\\centering",
        "\\includegraphics[width=\\textwidth]{FIG3_FRAME_BUDGET_30FOV.pdf}",
        f"\\caption{{{caption}}}", "\\label{fig:r1c2-frame-budget}", "\\end{figure*}",
    ])
    outputs = {
        "R1C2_ABSTRACT_SENTENCE_DIRECT.tex": abstract,
        "R1C2_RESULTS_DIRECT.tex": results,
        "R1C2_DISCUSSION_DIRECT.tex": discussion,
        "R1C2_CONCLUSION_SENTENCE_DIRECT.tex": conclusion,
        "R1C2_FIG3_DIRECT.tex": figure_tex,
        "FIG3_FRAME_BUDGET_30FOV_CAPTION.tex": caption,
        "R1C2_FRAME_BUDGET_TABLE_DIRECT.tex": "\n".join(table_lines),
        "R1C2_RESPONSE_TO_REVIEWER1_COMMENT2_DIRECT.tex": response,
    }
    for name, text in outputs.items():
        _tex_write(run_dir, name, text)
    return {name: sha_file(run_dir / name) for name in outputs}


def export_validated_r1c3_tables(run_dir: Path) -> dict[str, Any]:
    robust_path = FORMAL_R1C3_RUN / "R1C3_ROBUSTNESS_PER_SAMPLE.csv"
    robust_rows = _read_csv(robust_path)
    if len(robust_rows) != 4320 or len({(row["factor"], row["severity"], row["sample_order"], row["method"]) for row in robust_rows}) != 4320:
        raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "robustness grid")
    if any(row["status"] != "PASS" or row["gradient_finite"].lower() != "true" or row["output_finite"].lower() != "true" for row in robust_rows):
        raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "robustness finite/status")
    for row in robust_rows:
        if not all(math.isfinite(float(row[field])) for field in ("psnr", "ssim", "observed_nrmse", "poisson_gaussian_objective", "runtime_seconds")):
            raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "robustness numeric")
    runtime = read_json(FORMAL_R1C3_RUN / "R1C3_RUNTIME_STATS.json")
    if runtime.get("status") != "COMPLETE_VALIDATED" or runtime.get("n_fov") != 30 or runtime.get("recorded_repeats_per_fov") != 3:
        raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "runtime header")
    expected = {
        ("DiffWS-6", "Stage 1"): (15.093118044446197, 0.011296211400712387),
        ("PhysMap-6", "total"): (0.9132903544448912, 0.0032802692406789418),
        ("APD-SIM-6", "Stage 2"): (0.9142394233330075, 0.0028469444545487464),
        ("APD-SIM-6", "total"): (16.007357467779205, 0.011760556063957565),
    }
    summaries = {(row["method"], row["component"]): row for row in runtime["summaries"]}
    for key, (mean, sd) in expected.items():
        row = summaries.get(key)
        if row is None or not math.isclose(row["mean_of_fov_means_seconds"], mean, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(row["sample_sd_of_fov_means_seconds"], sd, rel_tol=0.0, abs_tol=1e-12):
            raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", f"runtime {key}")
    sources = {
        "R1C3_ROBUSTNESS_TABLE_DIRECT.tex": FORMAL_R1C3_RUN / "TABLE2_PHYSMAP6_STRICT.tex",
        "R1C3_RUNTIME_TABLE_DIRECT.tex": FORMAL_R1C3_RUN / "TABLE_RUNTIME_PHYSMAP6_STRICT.tex",
    }
    receipt = {
        "status": "PASS", "robustness_rows": 4320, "robustness_unique_keys": 4320,
        "robustness_source_sha256": sha_file(robust_path),
        "runtime_source_sha256": sha_file(FORMAL_R1C3_RUN / "R1C3_RUNTIME_STATS.json"),
        "physmap9_present": False, "runtime_expected_values_exact_match": True, "outputs": {},
    }
    table2_rows = _read_csv(FORMAL_R1C3_RUN / "TABLE2_PHYSMAP6_STRICT.csv")
    if len(table2_rows) != 24:
        raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "Table 2 row count")
    method_columns = {
        "WF": ("wf_mean", "wf_sample_sd"),
        "DiffWS-6": ("diffws6_mean", "diffws6_sample_sd"),
        "PhysMap-6": ("physmap6_mean", "physmap6_sample_sd"),
        "APD-SIM-6": ("apd6_mean", "apd6_sample_sd"),
    }
    metrics = _metrics_module()
    for table_row in table2_rows:
        selected_by_method: dict[str, list[dict[str, str]]] = {}
        for method in method_columns:
            selected = [
                row for row in robust_rows
                if row["factor"] == table_row["factor"]
                and math.isclose(float(row["severity"]), float(table_row["severity"]), rel_tol=0.0, abs_tol=1e-12)
                and row["method"] == method
            ]
            if len(selected) != 20:
                raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", f"Table 2 group {table_row['factor']}/{method}")
            selected_by_method[method] = selected
            values = np.asarray([float(row[table_row["metric"]]) for row in selected], dtype=np.float64)
            mean_column, sd_column = method_columns[method]
            if not math.isclose(float(np.mean(values)), float(table_row[mean_column]), rel_tol=0.0, abs_tol=1e-12) or not math.isclose(float(np.std(values, ddof=1)), float(table_row[sd_column]), rel_tol=0.0, abs_tol=1e-12):
                raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", f"Table 2 descriptive {table_row['factor']}/{method}/{table_row['metric']}")
        baseline = table_row["best_matched_six_frame_baseline"]
        apd_rows = selected_by_method["APD-SIM-6"]
        base_rows = selected_by_method[baseline]
        if [row["parent_id"] for row in apd_rows] != [row["parent_id"] for row in base_rows]:
            raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", "Table 2 parent pairing")
        ci = metrics.parent_image_bootstrap_ci(
            [float(row[table_row["metric"]]) for row in apd_rows],
            [float(row[table_row["metric"]]) for row in base_rows],
            [row["parent_id"] for row in apd_rows],
            class_labels=[row["structure"] for row in apd_rows],
            n_resamples=int(table_row["bootstrap_resamples"]),
            seed=int(table_row["bootstrap_seed"]),
        )
        expected_ci = (float(table_row["apd_minus_best_baseline_mean"]), float(table_row["apd_minus_best_baseline_ci_low"]), float(table_row["apd_minus_best_baseline_ci_high"]))
        actual_ci = (float(ci["estimate"]), float(ci["confidence_interval"][0]), float(ci["confidence_interval"][1]))
        if any(not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12) for actual, expected_value in zip(actual_ci, expected_ci)):
            raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", f"Table 2 bootstrap {table_row['factor']}/{table_row['metric']}")
    receipt["table2_rows_independently_recomputed"] = 24
    receipt["table2_csv_sha256"] = sha_file(FORMAL_R1C3_RUN / "TABLE2_PHYSMAP6_STRICT.csv")
    for name, source in sources.items():
        text = source.read_text(encoding="utf-8")
        if "PhysMap-9" in text or "\\input" in text or any(token in text for token in ("TODO", "TBD", "[INSERT]")):
            raise R1C2Blocked("R1C3_TABLE_VALIDATION_FAILED", source.name)
        atomic_replace(run_dir / name, text.encode("utf-8"))
        receipt["outputs"][name] = {"source": str(source), "source_sha256": sha_file(source), "output_sha256": sha_file(run_dir / name)}
    write_json(run_dir / "R1C3_TABLE_EXPORT_RECEIPT.json", receipt)
    return receipt


def implementation_snapshot(entry_path: Path) -> dict[str, str]:
    paths = [
        entry_path, Path(__file__).resolve(), ROOT / "unisim" / "checkpoint_contract.py",
        ROOT / "unisim" / "formal_training_2d.py", ROOT / "unisim" / "model2d.py",
        ROOT / "unisim" / "protocols.py", ROOT / "unisim" / "protocol_runtime.py",
        ROOT / "unisim" / "sim_forward_2d.py", ROOT / "unisim" / "revision_r1" / "physmap6_core.py",
        ROOT / "unisim" / "revision_r1" / "physmap6_pipeline.py",
        ROOT / "unisim" / "revision_r1" / "dataset_fig5_audit_identity.py", METRICS_SOURCE,
    ]
    paths.extend(sorted((ROOT / "tests").glob("test*.py")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", f"implementation missing: {missing}")
    return {str(path.resolve()): sha_file(path) for path in paths}


def independent_audit(run_dir: Path, dataset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = validate_per_fov(run_dir)
    metrics = _metrics_module()
    mismatches: list[str] = []
    for row in rows:
        gt = normalize_image(tifffile.imread(Path(row["gt_path"])))
        prediction = np.load(row["prediction_path"], allow_pickle=False)
        psnr = float(metrics.psnr_native(gt, prediction))
        ssim = float(metrics.ssim_native(gt, prediction))
        frc, _curves = metrics.reference_frc_1over7(gt, prediction)
        expected_status = "RIGHT_CENSORED" if frc["right_censored_at_nyquist"] else ("UNRESOLVED" if frc["unresolved_no_crossing"] else "CUTOFF")
        if not math.isclose(psnr, float(row["psnr"]), rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"PSNR:{row['sample_order']}:{row['protocol_label']}:{row['method']}")
        if not math.isclose(ssim, float(row["ssim"]), rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"SSIM:{row['sample_order']}:{row['protocol_label']}:{row['method']}")
        if expected_status != row["frc_status"]:
            mismatches.append(f"FRC_STATUS:{row['sample_order']}:{row['protocol_label']}:{row['method']}")
        if expected_status == "CUTOFF" and not math.isclose(float(frc["cutoff_derived_spatial_period_px"]), float(row["frc_spatial_period_px"]), rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(f"FRC_VALUE:{row['sample_order']}:{row['protocol_label']}:{row['method']}")
    if mismatches:
        raise R1C2Blocked("R1C2_INDEPENDENT_AUDIT_FAILED", str(mismatches[:10]))
    stats = read_json(run_dir / "R1C2_FRAME_BUDGET_STATS.json")
    if stats.get("status") != "COMPLETE_VALIDATED" or stats.get("n_rows") != 180:
        raise R1C2Blocked("R1C2_INDEPENDENT_AUDIT_FAILED", "stats")
    receipt = {
        "schema_version": 1, "status": "PASS", "per_fov_rows_recomputed": 180,
        "psnr_ssim_frc_recomputed_from_native_arrays": True, "mismatch_count": 0,
        "dataset_count": len(dataset), "source_csv_sha256": sha_file(run_dir / "R1C2_FRAME_BUDGET_PER_FOV.csv"),
        "metrics_source_sha256": sha_file(METRICS_SOURCE),
        "right_censored_values_not_coerced_to_exact_periods": True,
    }
    write_json(run_dir / "R1C2_INDEPENDENT_AUDIT.json", receipt)
    return receipt


def run_test_gate(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "R1C2_TEST_LOG.txt"
    command = [sys.executable, "-B", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    atomic_replace(log_path, output.encode("utf-8", errors="replace"))
    if completed.returncode != 0 or " failed" in output.lower():
        raise R1C2Blocked("R1C2_TEST_GATE_FAILED", f"exit={completed.returncode}")
    import re
    matches = re.findall(r"(\d+) passed", output)
    if not matches:
        raise R1C2Blocked("R1C2_TEST_GATE_FAILED", "pytest pass count absent")
    receipt = {
        "status": "PASS", "interpreter": sys.executable, "command": command,
        "exit_code": completed.returncode, "passed": int(matches[-1]), "failed": 0,
        "log_path": str(log_path.resolve()), "log_sha256": sha_file(log_path),
    }
    write_json(run_dir / "R1C2_TEST_RECEIPT.json", receipt)
    return receipt


def visual_qa(run_dir: Path) -> dict[str, Any]:
    tool_names = ("pdfinfo.exe", "pdftotext.exe", "pdftoppm.exe")
    tools: dict[str, str] = {}
    for name in tool_names:
        result = subprocess.run(["where", name], capture_output=True, text=True, timeout=30, check=False)
        candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or not candidates:
            raise R1C2Blocked("R1C2_VISUAL_QA_FAILED", f"{name} unavailable")
        tools[name] = candidates[0]
    render_dir = run_dir / "pdf_render"
    render_dir.mkdir(exist_ok=False)
    required_text = {
        "FIG3_FRAME_BUDGET_30FOV.pdf": ("Direct DMD acquisition-budget", "Paired PSNR", "Paired SSIM", "FRC"),
        "FIGS_FRAME_BUDGET_REPRESENTATIVE_DETAILS.pdf": ("Supplementary", "spectrum", "line profile", "FRC"),
    }
    pages = {}
    for filename, labels in required_text.items():
        pdf_path = run_dir / filename
        info = subprocess.run([tools["pdfinfo.exe"], str(pdf_path)], capture_output=True, text=True, timeout=60, check=False)
        text_result = subprocess.run([tools["pdftotext.exe"], str(pdf_path), "-"], capture_output=True, text=True, timeout=60, check=False)
        import re
        if info.returncode != 0 or text_result.returncode != 0 or re.search(r"^Pages:\s+1\s*$", info.stdout, flags=re.MULTILINE) is None:
            raise R1C2Blocked("R1C2_VISUAL_QA_FAILED", filename)
        if any(label.lower() not in text_result.stdout.lower() for label in labels):
            raise R1C2Blocked("R1C2_VISUAL_QA_FAILED", f"labels: {filename}")
        prefix = render_dir / pdf_path.stem
        rendered = subprocess.run([tools["pdftoppm.exe"], "-png", "-r", "120", "-singlefile", str(pdf_path), str(prefix)], capture_output=True, text=True, timeout=180, check=False)
        image_path = prefix.with_suffix(".png")
        if rendered.returncode != 0 or not image_path.is_file() or image_path.stat().st_size < 10_000:
            raise R1C2Blocked("R1C2_VISUAL_QA_FAILED", f"render: {filename}")
        pages[filename] = {"pdf_sha256": sha_file(pdf_path), "rendered_png": str(image_path.resolve()), "rendered_png_sha256": sha_file(image_path), "required_labels": list(labels)}
    receipt = {"status": "PASS", "page_count_each": 1, "poppler_tools": tools, "files": pages, "human_visual_inspection_required_before_final_handoff": True}
    write_json(run_dir / "R1C2_VISUAL_QA_RECEIPT.json", receipt)
    return receipt


def artifact_manifest(run_dir: Path) -> dict[str, Any]:
    excluded = {"R1C2_ARTIFACT_MANIFEST.json", "STATUS.json", "RUNNING.lock"}
    files = []
    for path in sorted((item for item in run_dir.rglob("*") if item.is_file() and item.name not in excluded), key=lambda item: item.as_posix().lower()):
        files.append({"relative_path": path.relative_to(run_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {"schema_version": 1, "status": "SEALED", "file_count": len(files), "aggregate_sha256": hashlib.sha256(canonical_bytes(files)).hexdigest(), "files": files}
    write_json(run_dir / "R1C2_ARTIFACT_MANIFEST.json", manifest)
    return manifest


def allocate_run_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = OUTPUT_ROOT / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "cannot allocate run directory")


def main(entry_path: Path) -> int:
    if not __debug__:
        raise RuntimeError("python -O / PYTHONOPTIMIZE is forbidden")
    run_dir = allocate_run_dir()
    lock_path = run_dir / "RUNNING.lock"
    lock_handle = lock_path.open("x", encoding="utf-8")
    lock_handle.write(f"pid={os.getpid()}\nutc={utc_now()}\n"); lock_handle.flush(); os.fsync(lock_handle.fileno())
    protected_before = {
        "formal_r1c3": tree_snapshot(FORMAL_R1C3_RUN),
        "manuscript_candidate": tree_snapshot(MANUSCRIPT_CANDIDATE),
    }
    source_before: dict[str, str] = {}
    dataset: list[dict[str, Any]] = []
    status: dict[str, Any] = {"status": "R1C2_FORMAL_EVALUATION_INCOMPLETE", "run_dir": str(run_dir.resolve()), "started_utc": utc_now()}
    try:
        source_before = implementation_snapshot(entry_path)
        checkpoint_rows, details = audit_checkpoints(run_dir)
        protocol_receipt = audit_protocols(run_dir)
        dataset = audit_dataset(run_dir)
        config_receipt = audit_cross_protocol_config(details)
        preflight = {
            "status": "PASS", "utc": utc_now(), "interpreter": sys.executable,
            "project_root": str(ROOT), "run_dir": str(run_dir.resolve()),
            "checkpoint_count": len(checkpoint_rows), "protocol_receipt": protocol_receipt,
            "dataset_count": len(dataset), "cross_protocol_contract": config_receipt,
            "implementation_sha256": source_before, "protected_tree_before": protected_before,
            "formal_output_reused": False, "retrospective_subsampling_used": False,
        }
        write_json(run_dir / "R1C2_PREFLIGHT.json", preflight)
        if os.environ.get("APD_R1C2_PREFLIGHT_ONLY", "0") == "1":
            status.update({"status": "R1C2_PREFLIGHT_PASS", "overall_ready": False, "preflight_only": True, "completed_utc": utc_now()})
            write_json(run_dir / "STATUS.json", status)
            print(f"R1C2_PREFLIGHT_PASS\n{run_dir}")
            return 0
        run_inference(run_dir, dataset, details)
        stats = analyze_results(run_dir)
        generate_figures(run_dir, dataset)
        generate_latex(run_dir, stats)
        export_validated_r1c3_tables(run_dir)
        audit = independent_audit(run_dir, dataset)
        tests = run_test_gate(run_dir)
        if implementation_snapshot(entry_path) != source_before:
            raise R1C2Blocked("R1C2_FORMAL_EVALUATION_INCOMPLETE", "implementation changed during run")
        protected_after = {
            "formal_r1c3": tree_snapshot(FORMAL_R1C3_RUN),
            "manuscript_candidate": tree_snapshot(MANUSCRIPT_CANDIDATE),
        }
        if protected_after != protected_before:
            raise R1C2Blocked("R1C2_PROTECTED_INPUT_MODIFIED", "formal run or manuscript changed")
        visual = visual_qa(run_dir)
        final_report = {
            "status": "R1C2_FRAME_BUDGET_30FOV_READY", "scope": "direct controller-defined DMD-3F/6F/9F evaluation",
            "n_fov": 30, "per_fov_rows": 180, "tests_passed": tests["passed"],
            "independent_audit": audit, "visual_qa": visual,
            "protected_formal_r1c3_unchanged": True, "protected_manuscript_unchanged": True,
            "limitations": [
                "Frame count and orientation support both change; this is not a frame-count-only causal experiment.",
                "Manifest-derived parent IDs are unique, but biological independence beyond file identity is unresolved.",
                "Censored/unresolved FRC values are not treated as exact spatial periods.",
            ],
        }
        write_json(run_dir / "FINAL_REPORT.json", final_report)
        manifest = artifact_manifest(run_dir)
        status.update({
            "status": "R1C2_FRAME_BUDGET_30FOV_READY", "overall_ready": True,
            "preflight_only": False, "completed_utc": utc_now(), "n_fov": 30,
            "per_fov_rows": 180, "tests_passed": tests["passed"],
            "protected_formal_r1c3_unchanged": True, "protected_manuscript_unchanged": True,
            "artifact_manifest_sha256": sha_file(run_dir / "R1C2_ARTIFACT_MANIFEST.json"),
            "artifact_aggregate_sha256": manifest["aggregate_sha256"],
        })
        write_json(run_dir / "STATUS.json", status)
        print(f"R1C2_FRAME_BUDGET_30FOV_READY\n{run_dir}")
        return 0
    except Exception as error:
        fail_status = error.status if isinstance(error, R1C2Blocked) else "R1C2_FORMAL_EVALUATION_INCOMPLETE"
        status.update({"status": fail_status, "overall_ready": False, "error_type": type(error).__name__, "detail": str(error), "completed_utc": utc_now()})
        try:
            if protected_before:
                after = {"formal_r1c3": tree_snapshot(FORMAL_R1C3_RUN), "manuscript_candidate": tree_snapshot(MANUSCRIPT_CANDIDATE)}
                status["protected_trees_unchanged_after_failure"] = after == protected_before
            if source_before:
                status["implementation_unchanged_after_failure"] = implementation_snapshot(entry_path) == source_before
            write_json(run_dir / "STATUS.json", status)
        finally:
            print(f"{fail_status}\n{run_dir}\n{error}", file=sys.stderr)
        return 1
    finally:
        lock_handle.close()
        if lock_path.exists():
            lock_path.unlink()

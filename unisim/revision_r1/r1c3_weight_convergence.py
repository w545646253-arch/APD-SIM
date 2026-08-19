"""Independent APD weight-branch and strict refinement convergence audit.

This module is deliberately outside the completed Reviewer #1 Comment 3 run.
It reads the protected run and checkpoint but writes only into the new audit
directory supplied by the caller.  Extended optimization is diagnostic-only:
the frozen 40-update configuration remains the formal result.
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
import statistics
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unisim.revision_r1 import physmap6_core as core
from unisim.revision_r1 import physmap6_experiment as experiment
from unisim.revision_r1 import physmap6_pipeline as pipeline
from unisim.sim_forward_2d import embed_raw_to_slots_2d, forward_protocol_clean_2d


ROOT = Path(__file__).resolve().parents[2]
FORMAL_RUN = ROOT / "outputs" / "reviewer1_physmap6_strict" / "20260813T183229Z"
CHECKPOINT = ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt"
CHECKPOINT_RECEIPT = CHECKPOINT.parent / "best_checkpoint_receipt.json"
VALIDATION_HISTORY = CHECKPOINT.parent / "validation_history.csv"
CONFIG = ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
TRAINING_SOURCE_SNAPSHOT = (
    ROOT / "audit" / "dmd3_nonfinite_recovery_20260814_010345"
    / "source_backups" / "unisim" / "formal_training_2d.py"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "10fb16662a8b71b877f2cab81bdc151dcded92f6efd1c4b006306b901a8adff7"
)
EXPECTED_TRAINING_SOURCE_SHA256 = (
    "ed67f3756c6bc2d04fbb81bba8e80096491d934abb0e30f7624968800ddc0bc1"
)
EXPECTED_RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
EXPECTED_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
FORMAL_UPDATE = 40
DIAGNOSTIC_ENDPOINTS = (80, 160, 320)
FORMAL_CORE_RECORDED_SHA256 = (
    "2e29bbc465e7e07c178c2c2c4e6d6e250b9a16fab884199e0c06b00777ab59e8"
)
FORMAL40_NUMERICAL_TOLERANCES = {
    "max_abs": float(32.0 * np.finfo(np.float32).eps),
    "rmse": float(4.0 * np.finfo(np.float32).eps),
    "psnr_abs": 1e-5,
    "ssim_abs": 1e-8,
    "objective_abs": 1e-6,
    "observed_nrmse_abs": 1e-7,
}
TRACE_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "method", "update",
    "initializer", "scope", "objective", "observed_nrmse", "psnr", "ssim", "gradient_norm",
    "finite", "image_min", "image_max", "image_mean", "image_std",
    "fraction_at_clip_min", "fraction_at_clip_max", "preclip_below_fraction",
    "preclip_above_fraction", "raw_stack_sha256", "initialization_sha256",
    "checkpoint_sha256", "protocol_id", "protocol_hash", "raw_frame_order",
    "learning_rate", "optimizer", "formal_update", "diagnostic_endpoint",
)
FORMAL40_REGRESSION_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "method",
    "old_authoritative_prediction_path", "old_prediction_file_sha256",
    "old_prediction_array_sha256", "current_replay_prediction_path",
    "current_prediction_file_sha256", "current_prediction_array_sha256",
    "bitwise_exact", "different_element_count", "max_abs_difference",
    "max_abs_tolerance", "max_abs_pass", "rmse_difference", "rmse_tolerance",
    "rmse_pass", "old_psnr", "current_psnr", "abs_psnr_difference",
    "psnr_abs_tolerance", "psnr_pass", "old_ssim", "current_ssim",
    "abs_ssim_difference", "ssim_abs_tolerance", "ssim_pass", "old_objective",
    "current_objective", "abs_objective_difference", "objective_abs_tolerance",
    "objective_pass", "old_observed_nrmse", "current_observed_nrmse",
    "abs_observed_nrmse_difference", "observed_nrmse_abs_tolerance",
    "observed_nrmse_pass", "numeric_equivalence_pass",
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        + b"\n",
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
    return stream.getvalue()


def _array_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left32 = np.ascontiguousarray(left, dtype=np.float32)
    right32 = np.ascontiguousarray(right, dtype=np.float32)
    if left32.shape != right32.shape:
        raise RuntimeError(
            f"Formal update-40 replay shape mismatch: {left32.shape} != {right32.shape}"
        )
    difference = left32.astype(np.float64) - right32.astype(np.float64)
    left_bits = left32.view(np.uint32)
    right_bits = right32.view(np.uint32)
    return {
        "bitwise_exact": bool(left32.tobytes(order="C") == right32.tobytes(order="C")),
        "different_element_count": int(np.count_nonzero(left_bits != right_bits)),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "rmse_difference": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def _state_value_sha(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state value is not a tensor: {name}")
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(header + b"\n" + array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_receipts(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state value is not a tensor: {name}")
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        receipts.append({
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "numel": int(array.size),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        })
    return receipts


def _source_line(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return number
    raise RuntimeError(f"Required source evidence not found in {path}: {needle}")


def _best_validation_row() -> dict[str, str]:
    with VALIDATION_HISTORY.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise RuntimeError("Validation history must contain exactly 50 rows")
    return min(
        rows,
        key=lambda row: (
            float(row["mean_val_total_loss"]),
            -float(row["mean_val_x0_psnr"]),
            -float(row["mean_val_x0_ssim"]),
            float(row["global_step"]),
        ),
    )


def audit_weight_branch(output_dir: Path) -> dict[str, Any]:
    """Write a CPU/read-only proof that validation and inference both use EMA."""
    output_dir = Path(output_dir).resolve()
    if sha_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("R1C3_WEIGHT_BRANCH_MISMATCH: checkpoint SHA changed")
    if sha_file(TRAINING_SOURCE_SNAPSHOT) != EXPECTED_TRAINING_SOURCE_SHA256:
        raise RuntimeError("R1C3_WEIGHT_BRANCH_MISMATCH: frozen training source snapshot changed")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("R1C3_WEIGHT_BRANCH_MISMATCH: checkpoint is not a mapping")
    branches: dict[str, Mapping[str, Any]] = {}
    for branch in ("model", "ema"):
        state = payload.get(branch)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"R1C3_WEIGHT_BRANCH_MISMATCH: {branch} branch absent")
        branches[branch] = state
    model_names = sorted(branches["model"])
    ema_names = sorted(branches["ema"])
    if model_names != ema_names or len(ema_names) != 270:
        raise RuntimeError("R1C3_WEIGHT_BRANCH_MISMATCH: model/EMA tensor schema mismatch")
    nonfinite = {
        branch: [
            name for name, value in state.items()
            if not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
        ]
        for branch, state in branches.items()
    }
    if any(nonfinite.values()):
        raise RuntimeError(f"R1C3_WEIGHT_BRANCH_MISMATCH: non-finite state: {nonfinite}")

    pipeline_source = Path(pipeline.__file__).resolve()
    inference_line = _source_line(pipeline_source, 'state = payload.get("ema")')
    inference_policy_line = _source_line(pipeline_source, '"weights": "ema"')
    validation_line = _source_line(
        TRAINING_SOURCE_SNAPSHOT, "components.model.load_state_dict(components.ema.shadow"
    )
    restore_line = _source_line(TRAINING_SOURCE_SNAPSHOT, "components.model.load_state_dict(original")
    best = _best_validation_row()
    source_receipt = json.loads(CHECKPOINT_RECEIPT.read_text(encoding="utf-8-sig"))
    metrics = source_receipt.get("metrics", {})
    metric_match = all(
        math.isclose(float(best[key]), float(metrics[key]), rel_tol=0.0, abs_tol=0.0)
        for key in (
            "global_step", "mean_val_total_loss", "mean_val_x0_psnr",
            "mean_val_x0_ssim", "validation_entry_count",
        )
    )
    consistent = (
        metric_match
        and 'state = payload.get("ema")' in inspect.getsource(pipeline.load_stage1)
        and "components.model.load_state_dict(components.ema.shadow" in TRAINING_SOURCE_SNAPSHOT.read_text(encoding="utf-8")
    )
    status = "PASS" if consistent else "R1C3_WEIGHT_BRANCH_MISMATCH"
    result = {
        "schema_version": 1,
        "status": status,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "CPU-only read-only model-versus-EMA branch audit",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha_file(CHECKPOINT),
        "checkpoint_model_tensor_count": len(model_names),
        "checkpoint_ema_tensor_count": len(ema_names),
        "tensor_names_identical_between_model_and_ema": model_names == ema_names,
        "loaded_branch": "ema",
        "validation_branch": "ema",
        "formal_inference_branch": "ema",
        "validation_and_inference_branch_consistent": consistent,
        "loaded_ema_tensor_names": ema_names,
        "loaded_ema_tensors": _tensor_receipts(branches["ema"]),
        "loaded_ema_state_value_sha256": _state_value_sha(branches["ema"]),
        "model_state_value_sha256": _state_value_sha(branches["model"]),
        "all_model_tensors_finite": not nonfinite["model"],
        "all_ema_tensors_finite": not nonfinite["ema"],
        "validation_selection_rule": pipeline.BEST_RULE_ID,
        "validation_best_row": {
            key: float(value) for key, value in best.items()
        },
        "selection_receipt_metrics_exact_match": metric_match,
        "test_data_used_for_selection": source_receipt.get("test_data_used_for_selection"),
        "source_evidence": {
            "inference_loader": {
                "path": str(pipeline_source),
                "sha256": sha_file(pipeline_source),
                "ema_policy_line": inference_policy_line,
                "ema_load_line": inference_line,
            },
            "validation_training_snapshot": {
                "path": str(TRAINING_SOURCE_SNAPSHOT),
                "sha256": sha_file(TRAINING_SOURCE_SNAPSHOT),
                "ema_swap_line": validation_line,
                "model_restore_line": restore_line,
                "note": "Validation temporarily loads EMA, evaluates, then restores model state.",
            },
            "selection_receipt": {
                "path": str(CHECKPOINT_RECEIPT),
                "sha256": sha_file(CHECKPOINT_RECEIPT),
            },
            "validation_history": {
                "path": str(VALIDATION_HISTORY),
                "sha256": sha_file(VALIDATION_HISTORY),
                "row_count": 50,
            },
        },
    }
    write_json(output_dir / "APD6_WEIGHT_BRANCH_AUDIT.json", result)
    markdown = f"""# APD-SIM-6 weight-branch audit

Status: `{status}`

- Formal inference branch: **EMA** (`payload[\"ema\"]`, strict load).
- Validation and checkpoint-selection branch: **EMA** (temporary EMA swap before metrics).
- Branch consistency: `{consistent}`.
- Checkpoint: `{CHECKPOINT}`
- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`
- Loaded tensor count: `270`; all finite: `{not nonfinite['ema']}`.
- Loaded EMA state-value SHA-256: `{result['loaded_ema_state_value_sha256']}`
- Selected validation event: `{int(float(best['global_step']))}`; PSNR/SSIM:
  `{float(best['mean_val_x0_psnr']):.12g}` / `{float(best['mean_val_x0_ssim']):.12g}`.

The archived training source is used because it is the source snapshot nearest the checkpoint
generation; the current training engine has since been hardened. The historical validation
implementation loads `ema.shadow`, evaluates all entries, and restores the model branch.
"""
    atomic_write(output_dir / "APD6_WEIGHT_BRANCH_AUDIT.md", markdown.encode("utf-8"))
    if status != "PASS":
        raise RuntimeError(status)
    return result


def _load_metrics() -> Any:
    spec = importlib.util.spec_from_file_location("r1c3_convergence_metrics", experiment.METRICS_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen official metrics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_metrics(gt: np.ndarray, image: torch.Tensor, metrics: Any) -> tuple[float, float]:
    value = np.ascontiguousarray(image[0, 0].detach().cpu().numpy(), dtype=np.float32)
    return float(metrics.psnr_native(gt, value)), float(metrics.ssim_native(gt, value))


def _image_summary(image: torch.Tensor, *, include_sha256: bool = False) -> dict[str, Any]:
    value = image.detach()
    result = {
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
        "mean": float(value.to(dtype=torch.float64).mean().cpu()),
        "std": float(value.to(dtype=torch.float64).std(unbiased=False).cpu()),
        "fraction_at_clip_min": float((value <= 0.0).to(dtype=torch.float64).mean().cpu()),
        "fraction_at_clip_max": float((value >= 1.0).to(dtype=torch.float64).mean().cpu()),
        "finite": bool(torch.isfinite(value).all()),
    }
    if include_sha256:
        result["sha256"] = pipeline.sha_array(
            np.ascontiguousarray(value[0, 0].cpu().numpy(), dtype=np.float32)
        )
    return result


def _observed_diagnostics(
    estimate: torch.Tensor,
    raw: torch.Tensor,
    sim_config: Any,
    theta: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    slotted, _mask = embed_raw_to_slots_2d(raw, pipeline.PROTOCOL_ID)
    predicted_raw, _ = forward_protocol_clean_2d(
        estimate, sim_config, pipeline.PROTOCOL_ID, theta=dict(theta), randomize=False
    )
    predicted_slotted, _ = embed_raw_to_slots_2d(predicted_raw, pipeline.PROTOCOL_ID)
    objective = core.masked_poisson_gaussian_likelihood_2d(
        slotted, predicted_slotted, pipeline.PROTOCOL_ID,
        photon_scale=theta["photon_scale"], read_noise_e=theta["read_noise_e"], reduce="mean",
    )
    nrmse = torch.mean((raw - predicted_raw).square()).sqrt() / torch.mean(
        raw.square()
    ).sqrt().clamp_min(1e-12)
    return objective, nrmse


def _baseline_hashes() -> dict[str, str]:
    if not FORMAL_RUN.is_dir():
        raise RuntimeError(f"Formal run absent: {FORMAL_RUN}")
    return {
        str(path.relative_to(FORMAL_RUN)).replace("\\", "/"): sha_file(path)
        for path in sorted(FORMAL_RUN.rglob("*")) if path.is_file()
    }


def _formal_nominal_rows() -> dict[tuple[int, str], dict[str, str]]:
    path = FORMAL_RUN / "R1C3_NOMINAL_PER_FOV.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120:
        raise RuntimeError("Formal nominal CSV is not 120 rows")
    return {(int(row["sample_order"]), row["method"]): row for row in rows}


def _formal_predictions_receipt() -> dict[str, Any]:
    nominal = _formal_nominal_rows()
    mismatches: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    for (order, method), row in sorted(nominal.items()):
        path = _prediction_path(order, row["sample_id"], method)
        if not path.is_file():
            mismatches.append({"sample_order": order, "method": method, "field": "missing"})
            continue
        array = np.load(path, allow_pickle=False)
        array_hash = pipeline.sha_array(np.ascontiguousarray(array, dtype=np.float32))
        file_hashes[str(path.relative_to(FORMAL_RUN)).replace("\\", "/")] = sha_file(path)
        if array_hash != row["prediction_sha256"]:
            mismatches.append({
                "sample_order": order, "method": method,
                "field": "prediction_array_sha256", "observed": array_hash,
                "expected": row["prediction_sha256"],
            })
    if mismatches:
        raise RuntimeError(f"Formal nominal prediction audit failed: {mismatches[:3]}")
    return {
        "count": len(file_hashes),
        "file_sha256_mapping": file_hashes,
        "mapping_sha256": canonical_sha(file_hashes),
    }


def _prediction_path(order: int, sample_id: str, method: str) -> Path:
    return FORMAL_RUN / "nominal_predictions" / f"{order:03d}_{sample_id}_{method.replace(' ', '_')}.npy"


def _formal40_replay_path(
    output_dir: Path, order: int, sample_id: str, method: str
) -> Path:
    return (
        output_dir / "formal40_replay_predictions"
        / f"{order:03d}_{sample_id}_{method.replace(' ', '_')}.npy"
    )


def _cuda_nondeterminism_evidence() -> dict[str, Any]:
    artifact = (
        ROOT / "outputs" / "reviewer1_physmap6_dataset_fig5_audit"
        / "20260814T012618Z" / "DEBUG_SAMPLE0_CUDA_DETERMINISM.json"
    )
    evidence: dict[str, Any] = {
        "confirmed": True,
        "nondeterministic_operator": "adaptive_avg_pool2d_backward_cuda",
        "source_operation": "F.interpolate(raw, mode='area') after exact 2x upsample",
        "deterministic_mode_error": (
            "adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation"
        ),
        "scientific_consequence": (
            "A CUDA formal-configuration replay can be float32-equivalent without having the "
            "same prediction-array SHA-256 as the authoritative old formal NPY."
        ),
        "debug_artifact_available": artifact.is_file(),
    }
    if not artifact.is_file():
        return evidence
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    default_gradients = payload.get("default_mode_gradients", {}).get("comparison", {})
    deterministic = payload.get("deterministic_mode", {})
    evidence.update({
        "debug_artifact_path": str(artifact),
        "debug_artifact_sha256": sha_file(artifact),
        "same_state_default_gradient_bitwise_exact": default_gradients.get(
            "exact_array_equal"
        ),
        "same_state_default_gradient_different_element_count": default_gradients.get(
            "different_element_count"
        ),
        "same_state_default_gradient_max_abs": default_gradients.get("max_abs"),
        "integer_avg_pool_forward_exact_to_area": deterministic.get(
            "integer_area_replacement", {}
        ).get("forward_comparison", {}).get("exact_array_equal"),
        "deterministic_replacement_gradient_exact": deterministic.get(
            "gradients", {}
        ).get("comparison", {}).get("exact_array_equal"),
        "deterministic_replacement_physmap40_exact_repeat": deterministic.get(
            "PhysMap-6", {}
        ).get("comparison", {}).get("exact_array_equal"),
        "deterministic_replacement_apd40_exact_repeat": deterministic.get(
            "APD-SIM-6", {}
        ).get("comparison", {}).get("exact_array_equal"),
    })
    return evidence


def _formal_core_provenance(current_core_sha256: str) -> dict[str, Any]:
    preflight = FORMAL_RUN / "R1C3_PREFLIGHT.json"
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    matches = {
        value for path, value in payload.get("implementation_sha256", {}).items()
        if str(path).replace("/", "\\").endswith("\\unisim\\revision_r1\\physmap6_core.py")
    }
    if matches != {FORMAL_CORE_RECORDED_SHA256}:
        raise RuntimeError(f"Formal core preflight hash drift: {sorted(matches)}")
    return {
        "formal_preflight_path": str(preflight),
        "formal_preflight_sha256": sha_file(preflight),
        "old_formal_core_recorded_sha256": FORMAL_CORE_RECORDED_SHA256,
        "old_formal_core_source_snapshot_available": False,
        "current_replay_core_sha256": current_core_sha256,
        "current_core_source_differs_from_recorded_formal_core": (
            current_core_sha256 != FORMAL_CORE_RECORDED_SHA256
        ),
        "provenance_gap": (
            "The old run records the formal core SHA-256 but contains no archived source copy; "
            "therefore algorithm identity is hash-bound but the old source cannot be diffed line-by-line."
        ),
    }


def _formal40_regression_row(
    *,
    sample_order: int,
    sample: Mapping[str, str],
    method: str,
    current_array: np.ndarray,
    current_trace_row: Mapping[str, Any],
    current_path: Path,
    old_row: Mapping[str, str],
) -> dict[str, Any]:
    """Compare a replayed update-40 array to the immutable formal result.

    The old NPY remains the authoritative formal identity.  The current array
    is a formal-configuration replay only; GPU bitwise equality is reported,
    never assumed.  Acceptance uses the frozen float32 numerical-equivalence
    tolerances declared at module import.
    """
    old_path = _prediction_path(sample_order, sample["sample_id"], method)
    old_array = np.ascontiguousarray(np.load(old_path, allow_pickle=False), dtype=np.float32)
    current_array = np.ascontiguousarray(current_array, dtype=np.float32)
    old_array_hash = pipeline.sha_array(old_array)
    current_array_hash = pipeline.sha_array(current_array)
    if old_array_hash != old_row["prediction_sha256"]:
        raise RuntimeError(
            f"Protected formal prediction identity changed: {sample['sample_id']}/{method}"
        )
    if not current_path.is_file():
        raise RuntimeError(f"Current update-40 replay NPY absent: {current_path}")
    difference = _array_difference(current_array, old_array)
    old_psnr = float(old_row["psnr"])
    old_ssim = float(old_row["ssim"])
    old_objective = float(old_row["poisson_gaussian_objective"])
    old_nrmse = float(old_row["observed_nrmse"])
    current_psnr = float(current_trace_row["psnr"])
    current_ssim = float(current_trace_row["ssim"])
    current_objective = float(current_trace_row["objective"])
    current_nrmse = float(current_trace_row["observed_nrmse"])
    metric_differences = {
        "abs_psnr_difference": abs(current_psnr - old_psnr),
        "abs_ssim_difference": abs(current_ssim - old_ssim),
        "abs_objective_difference": abs(current_objective - old_objective),
        "abs_observed_nrmse_difference": abs(current_nrmse - old_nrmse),
    }
    finite_values = [
        difference["max_abs_difference"], difference["rmse_difference"],
        *metric_differences.values(),
    ]
    if not all(math.isfinite(value) for value in finite_values):
        raise FloatingPointError(
            f"Non-finite formal update-40 regression: {sample['sample_id']}/{method}"
        )
    passes = {
        "max_abs_pass": difference["max_abs_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["max_abs"],
        "rmse_pass": difference["rmse_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["rmse"],
        "psnr_pass": metric_differences["abs_psnr_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["psnr_abs"],
        "ssim_pass": metric_differences["abs_ssim_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["ssim_abs"],
        "objective_pass": metric_differences["abs_objective_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["objective_abs"],
        "observed_nrmse_pass": metric_differences["abs_observed_nrmse_difference"]
        <= FORMAL40_NUMERICAL_TOLERANCES["observed_nrmse_abs"],
    }
    return {
        "sample_order": sample_order,
        "sample_id": sample["sample_id"],
        "parent_id": sample["parent_id"],
        "structure": sample["structure_class"],
        "method": method,
        "old_authoritative_prediction_path": str(old_path),
        "old_prediction_file_sha256": sha_file(old_path),
        "old_prediction_array_sha256": old_array_hash,
        "current_replay_prediction_path": str(current_path),
        "current_prediction_file_sha256": sha_file(current_path),
        "current_prediction_array_sha256": current_array_hash,
        **difference,
        "max_abs_tolerance": FORMAL40_NUMERICAL_TOLERANCES["max_abs"],
        "max_abs_pass": passes["max_abs_pass"],
        "rmse_tolerance": FORMAL40_NUMERICAL_TOLERANCES["rmse"],
        "rmse_pass": passes["rmse_pass"],
        "old_psnr": old_psnr,
        "current_psnr": current_psnr,
        **metric_differences,
        "psnr_abs_tolerance": FORMAL40_NUMERICAL_TOLERANCES["psnr_abs"],
        "psnr_pass": passes["psnr_pass"],
        "old_ssim": old_ssim,
        "current_ssim": current_ssim,
        "ssim_abs_tolerance": FORMAL40_NUMERICAL_TOLERANCES["ssim_abs"],
        "ssim_pass": passes["ssim_pass"],
        "old_objective": old_objective,
        "current_objective": current_objective,
        "objective_abs_tolerance": FORMAL40_NUMERICAL_TOLERANCES["objective_abs"],
        "objective_pass": passes["objective_pass"],
        "old_observed_nrmse": old_nrmse,
        "current_observed_nrmse": current_nrmse,
        "observed_nrmse_abs_tolerance": FORMAL40_NUMERICAL_TOLERANCES[
            "observed_nrmse_abs"
        ],
        "observed_nrmse_pass": passes["observed_nrmse_pass"],
        "numeric_equivalence_pass": all(passes.values()),
    }


def _verify_formal40_replay_artifacts(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Load back and hash every current and authoritative update-40 NPY."""
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 formal update-40 regression rows, got {len(rows)}")
    file_hashes: dict[str, str] = {}
    array_hashes: dict[str, str] = {}
    for row in rows:
        order = int(row["sample_order"])
        sample_id = str(row["sample_id"])
        method = str(row["method"])
        current_path = Path(str(row["current_replay_prediction_path"])).resolve()
        expected_current = _formal40_replay_path(
            output_dir, order, sample_id, method
        ).resolve()
        old_path = Path(str(row["old_authoritative_prediction_path"])).resolve()
        expected_old = _prediction_path(order, sample_id, method).resolve()
        if current_path != expected_current or old_path != expected_old:
            raise RuntimeError(
                f"Formal update-40 replay path drift: {sample_id}/{method}"
            )
        if not current_path.is_file() or not old_path.is_file():
            raise RuntimeError(f"Formal update-40 NPY absent: {sample_id}/{method}")
        current_file_hash = sha_file(current_path)
        old_file_hash = sha_file(old_path)
        if (
            current_file_hash != row["current_prediction_file_sha256"]
            or old_file_hash != row["old_prediction_file_sha256"]
        ):
            raise RuntimeError(
                f"Formal update-40 NPY file hash drift: {sample_id}/{method}"
            )
        current_raw = np.load(current_path, allow_pickle=False)
        old_raw = np.load(old_path, allow_pickle=False)
        if (
            current_raw.dtype != np.dtype(np.float32)
            or old_raw.dtype != np.dtype(np.float32)
            or current_raw.shape != old_raw.shape
        ):
            raise RuntimeError(
                f"Formal update-40 NPY dtype/shape drift: {sample_id}/{method}"
            )
        current_array = np.ascontiguousarray(current_raw, dtype=np.float32)
        old_array = np.ascontiguousarray(old_raw, dtype=np.float32)
        current_array_hash = pipeline.sha_array(current_array)
        old_array_hash = pipeline.sha_array(old_array)
        if (
            current_array_hash != row["current_prediction_array_sha256"]
            or old_array_hash != row["old_prediction_array_sha256"]
        ):
            raise RuntimeError(
                f"Formal update-40 NPY array hash drift: {sample_id}/{method}"
            )
        relative = str(current_path.relative_to(output_dir)).replace("\\", "/")
        file_hashes[relative] = current_file_hash
        array_hashes[relative] = current_array_hash
    if len(file_hashes) != 60 or len(array_hashes) != 60:
        raise RuntimeError("Formal update-40 replay mapping contains duplicate paths")
    return file_hashes, array_hashes


def _numerical_completion_disposition(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int = 60
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Numerical-equivalence disposition expected {expected_count} rows, got {len(rows)}"
        )
    failures = [row for row in rows if not bool(row["numeric_equivalence_pass"])]
    mismatch_count = len(failures)
    pass_count = expected_count - mismatch_count
    passed = mismatch_count == 0
    return {
        "expected_count": expected_count,
        "pass_count": pass_count,
        "mismatch_count": mismatch_count,
        "passed": passed,
        "summary_status": "PASS" if passed else "FAIL_NUMERICAL_EQUIVALENCE",
        "progress_status": (
            "COMPLETE" if passed
            else "COMPLETE_WITH_NUMERICAL_EQUIVALENCE_FAILURES"
        ),
        "failures": failures,
    }


def _row_from_observation(
    *, sample_order: int, sample: Mapping[str, str], method: str, update: int,
    scope: str, endpoint: int, observation: Mapping[str, Any], gt: np.ndarray,
    metrics: Any, raw_hash: str, init_hash: str,
) -> dict[str, Any]:
    image = observation["estimate"]
    psnr, ssim = _image_metrics(gt, image, metrics)
    summary = _image_summary(image)
    row = {
        "sample_order": sample_order,
        "sample_id": sample["sample_id"],
        "parent_id": sample["parent_id"],
        "structure": sample["structure_class"],
        "method": method,
        "update": update,
        "initializer": "x_init_six_frame_mean" if method == "PhysMap-6" else "x_ws",
        "scope": scope,
        "objective": float(observation["objective"]),
        "observed_nrmse": float(observation["observed_nrmse"]),
        "psnr": psnr,
        "ssim": ssim,
        "gradient_norm": float(observation["gradient_norm"]),
        "finite": bool(summary["finite"] and all(math.isfinite(float(observation[key])) for key in ("objective", "observed_nrmse", "gradient_norm"))),
        "image_min": summary["min"], "image_max": summary["max"],
        "image_mean": summary["mean"], "image_std": summary["std"],
        "fraction_at_clip_min": summary["fraction_at_clip_min"],
        "fraction_at_clip_max": summary["fraction_at_clip_max"],
        "preclip_below_fraction": "NOT_MEASURED",
        "preclip_above_fraction": "NOT_MEASURED",
        "raw_stack_sha256": raw_hash, "initialization_sha256": init_hash,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "protocol_id": pipeline.PROTOCOL_ID, "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_frame_order": "/".join(EXPECTED_RAW_ORDER),
        "learning_rate": experiment.REFINEMENT_CONFIG.learning_rate,
        "optimizer": experiment.REFINEMENT_CONFIG.optimizer,
        "formal_update": FORMAL_UPDATE, "diagnostic_endpoint": endpoint,
    }
    if not row["finite"]:
        raise FloatingPointError(f"Non-finite convergence row: {sample['sample_id']}/{method}/{update}")
    return row


def _run_trace(
    initial: torch.Tensor, raw: torch.Tensor, gt: np.ndarray, sample_order: int,
    sample: Mapping[str, str], method: str, sim_config: Any,
    theta: Mapping[str, torch.Tensor], geometry: Mapping[str, Any], metrics: Any,
) -> tuple[list[dict[str, Any]], torch.Tensor, dict[int, str], np.ndarray]:
    endpoint = DIAGNOSTIC_ENDPOINTS[-1]
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[int, str] = {}
    formal40_array: np.ndarray | None = None
    raw_hash = pipeline.sha_array(raw[0].detach().cpu().numpy().astype(np.float32))
    init_hash = _image_summary(initial, include_sha256=True)["sha256"]
    def observer(value: Mapping[str, Any]) -> None:
        nonlocal formal40_array
        update = int(value["update"])
        if update < 0:
            return
        rows.append(_row_from_observation(
            sample_order=sample_order, sample=sample, method=method, update=update,
            scope=(
                "formal_configuration_replay"
                if update <= FORMAL_UPDATE else "diagnostic_only"
            ),
            endpoint=endpoint, observation=value, gt=gt, metrics=metrics,
            raw_hash=raw_hash, init_hash=init_hash,
        ))
        if update in {40, 80, 160, 320}:
            checkpoint_hashes[update] = _image_summary(
                value["estimate"], include_sha256=True
            )["sha256"]
        if update == FORMAL_UPDATE:
            formal40_array = np.array(
                value["estimate"][0, 0].detach().cpu().numpy(),
                dtype=np.float32,
                order="C",
                copy=True,
            )

    result = core.masked_refine(
        initial, raw,
        torch.tensor(EXPECTED_MASK, device=raw.device, dtype=torch.float32),
        geometry, {"sim_config": sim_config, "theta": theta},
        experiment.REFINEMENT_CONFIG,
        diagnostic_updates=endpoint,
        diagnostic_observer=observer,
    )
    if [int(value["update"]) for value in rows] != list(range(endpoint + 1)):
        raise RuntimeError(f"Trace update sequence is not exact 0..{endpoint}")
    if formal40_array is None:
        raise RuntimeError("Trace did not capture the formal-configuration replay at update 40")
    return rows, result.final_reconstruction, checkpoint_hashes, formal40_array


def _summarize_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    endpoints = (0, 40, 80, 160, 320)
    result: dict[str, Any] = {}
    for method in ("PhysMap-6", "APD-SIM-6"):
        selected = [row for row in rows if row["method"] == method]
        result[method] = {}
        for endpoint in endpoints:
            values = [row for row in selected if int(row["update"]) == endpoint]
            if len(values) != 30:
                raise RuntimeError(f"Missing endpoint rows: {method}/{endpoint}")
            result[method][str(endpoint)] = {
                metric: {
                    "mean": float(np.mean([float(row[metric]) for row in values])),
                    "sample_sd": float(np.std([float(row[metric]) for row in values], ddof=1)),
                }
                for metric in ("objective", "observed_nrmse", "psnr", "ssim", "gradient_norm")
            }
    return result


def _convergence_flags(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case = {(int(row["sample_order"]), row["method"], int(row["update"])): row for row in rows}
    apd_relative_objective_change = []
    apd_relative_gradient_at_40 = []
    relative_objective_gaps = []
    for order in range(30):
        apd40 = by_case[(order, "APD-SIM-6", 40)]
        apd320 = by_case[(order, "APD-SIM-6", 320)]
        phys320 = by_case[(order, "PhysMap-6", 320)]
        apd_relative_objective_change.append(
            abs(float(apd40["objective"]) - float(apd320["objective"]))
            / max(abs(float(apd40["objective"])), 1e-12)
        )
        apd_relative_gradient_at_40.append(
            float(apd40["gradient_norm"]) / max(float(apd320["gradient_norm"]), 1e-12)
        )
        apd_objective = float(apd320["objective"])
        phys_objective = float(phys320["objective"])
        relative_objective_gaps.append(
            abs(apd_objective - phys_objective)
            / max(abs(apd_objective), abs(phys_objective), 1e-12)
        )
    not_converged_count = sum(value > 1e-3 for value in apd_relative_objective_change)
    basin_different_count = sum(value > 0.01 for value in relative_objective_gaps)
    return {
            "explicit_post_analysis_diagnostic_rule": {
            "apd_not_converged_at_40": "relative absolute objective change from update40 to update320 > 1e-3",
            "different_objective_basin": "absolute APD-minus-PhysMap objective gap divided by max absolute objective at update320 > 0.01",
            "rule_is_diagnostic_not_a_method_redefinition": True,
        },
        "apd_not_converged_at_40_fov_count": not_converged_count,
        "apd_not_converged_at_40": not_converged_count > 0,
        "apd_relative_objective_change_40_to_320_median": float(statistics.median(apd_relative_objective_change)),
        "apd_gradient_norm_ratio_40_to_320_median": float(statistics.median(apd_relative_gradient_at_40)),
        "different_objective_basin_at_320_fov_count": basin_different_count,
        "same_basin_for_all_fov": basin_different_count == 0,
        "relative_objective_gap_at_320_median": float(statistics.median(relative_objective_gaps)),
    }


def _clipping_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("PhysMap-6", "APD-SIM-6"):
        post_update = [
            row for row in rows
            if row["method"] == method and 1 <= int(row["update"]) <= 320
        ]
        if len(post_update) != 30 * 320:
            raise RuntimeError(
                f"Post-clamp clipping grid incomplete: {method}/{len(post_update)}"
            )
        result[method] = {
            "update_range": [1, 320],
            "update0_initializer_excluded": True,
            "max_postclip_at_min_fraction": max(
                float(row["fraction_at_clip_min"]) for row in post_update
            ),
            "max_postclip_at_max_fraction": max(
                float(row["fraction_at_clip_max"]) for row in post_update
            ),
            "preclip_below_fraction": "NOT_MEASURED",
            "preclip_above_fraction": "NOT_MEASURED",
        }
    return result


def _render_plot(rows: Sequence[Mapping[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = {"PhysMap-6": "#0065bd", "APD-SIM-6": "#d95f02"}
    for method in ("PhysMap-6", "APD-SIM-6"):
        method_rows = [row for row in rows if row["method"] == method]
        x = np.arange(321)
        values = np.asarray([
            [float(row[metric]) for row in method_rows if int(row["sample_order"]) == order]
            for order in range(30)
        ], dtype=np.float64)
        if values.shape != (30, 321):
            raise RuntimeError(f"Unexpected trajectory matrix: {method}/{metric}/{values.shape}")
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        ax.plot(x, mean, color=colors[method], linewidth=2.0, label=method)
        ax.fill_between(x, mean - sd, mean + sd, color=colors[method], alpha=0.14)
    for endpoint in (40, 80, 160, 320):
        ax.axvline(endpoint, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvspan(40, 320, color="#999999", alpha=0.06, label="diagnostic-only extension")
    ax.set_xlabel("Adam updates")
    ax.set_ylabel(ylabel)
    ax.set_title("Strict shared refinement convergence (30 FOVs, mean +/- sample SD)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_convergence_diagnostics(
    output_dir: Path,
    *,
    full: bool = True,
    immutable_old_run_hash_baseline: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run exact 0..320 traces with a fail-closed update-40 numerical gate.

    The immutable old NPY remains authoritative.  Current CUDA replay hashes
    are recorded honestly; acceptance requires every predeclared float32 array
    and metric tolerance to pass, not a false claim of bitwise reproducibility.

    ``immutable_old_run_hash_baseline`` is the direct ``relative/path ->
    SHA-256`` mapping for every protected file, not a containing snapshot or
    receipt object.  When omitted, this function captures the mapping itself.
    """
    if not full:
        raise ValueError("Only the full 30-FOV convergence diagnostic is supported")
    output_dir = Path(output_dir).resolve()
    formal_run_resolved = FORMAL_RUN.resolve()
    if output_dir == formal_run_resolved or formal_run_resolved in output_dir.parents:
        raise RuntimeError(
            "Convergence output_dir must not be the protected formal run or any descendant"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full convergence diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _baseline_hashes()
    if immutable_old_run_hash_baseline is not None and before != dict(immutable_old_run_hash_baseline):
        raise RuntimeError("Protected formal-run baseline differs before convergence diagnostic")

    formal_prediction_receipt = _formal_predictions_receipt()
    source_hashes = {
        str(Path(core.__file__).resolve()): sha_file(Path(core.__file__).resolve()),
        str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
        str(Path(pipeline.__file__).resolve()): sha_file(Path(pipeline.__file__).resolve()),
        str(Path(experiment.__file__).resolve()): sha_file(Path(experiment.__file__).resolve()),
        str(ROOT / "unisim" / "sim_forward_2d.py"): sha_file(ROOT / "unisim" / "sim_forward_2d.py"),
        str(ROOT / "unisim" / "model2d.py"): sha_file(ROOT / "unisim" / "model2d.py"),
        str(ROOT / "unisim" / "checkpoint_contract.py"): sha_file(ROOT / "unisim" / "checkpoint_contract.py"),
        str(ROOT / "unisim" / "formal_training_2d.py"): sha_file(ROOT / "unisim" / "formal_training_2d.py"),
        str(ROOT / "unisim" / "protocol_runtime.py"): sha_file(ROOT / "unisim" / "protocol_runtime.py"),
        str(ROOT / "unisim" / "protocols.py"): sha_file(ROOT / "unisim" / "protocols.py"),
        str(experiment.METRICS_SOURCE): sha_file(experiment.METRICS_SOURCE),
        str(CONFIG): sha_file(CONFIG), str(CHECKPOINT): sha_file(CHECKPOINT),
        str(experiment.BUNDLE_MANIFEST): sha_file(experiment.BUNDLE_MANIFEST),
        str(experiment.GT_MANIFEST): sha_file(experiment.GT_MANIFEST),
        str(FORMAL_RUN / "R1C3_NOMINAL_PER_FOV.csv"): sha_file(FORMAL_RUN / "R1C3_NOMINAL_PER_FOV.csv"),
    }
    fingerprint = canonical_sha({
        "source_hashes": source_hashes,
        "config": experiment.REFINEMENT_CONFIG.receipt(),
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_order": EXPECTED_RAW_ORDER,
        "formal_update": FORMAL_UPDATE,
        "diagnostic_endpoints": DIAGNOSTIC_ENDPOINTS,
        "formal40_numerical_tolerances": FORMAL40_NUMERICAL_TOLERANCES,
        "formal40_regression_fields": FORMAL40_REGRESSION_FIELDS,
        "scope_policy": {
            "updates_0_through_40": "formal_configuration_replay",
            "updates_41_through_320": "diagnostic_only",
        },
        "formal_prediction_mapping_sha256": formal_prediction_receipt["mapping_sha256"],
    })
    progress_path = output_dir / "APD_PHYSMAP_CONVERGENCE_PROGRESS.json"
    per_fov_dir = output_dir / "convergence_fov_parts"
    replay_dir = output_dir / "formal40_replay_predictions"
    per_fov_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    part_index: dict[int, dict[str, Any]] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint_sha256") != fingerprint:
            raise RuntimeError("Convergence resume fingerprint mismatch")
        for key, value in progress.get("parts", {}).items():
            part_index[int(key)] = dict(value)
        completed_orders = [int(value) for value in progress.get("completed_sample_orders", [])]
        if completed_orders != list(range(len(completed_orders))):
            raise RuntimeError("Convergence resume completed-sample sequence is not contiguous")
        if sorted(part_index) != completed_orders:
            raise RuntimeError("Convergence resume part index disagrees with completed samples")
    else:
        progress = {
            "schema_version": 2,
            "status": "IN_PROGRESS",
            "fingerprint_sha256": fingerprint,
            "source_hashes": source_hashes,
            "formal40_numerical_tolerances": FORMAL40_NUMERICAL_TOLERANCES,
            "completed_sample_orders": [],
            "parts": {},
        }
        write_json(progress_path, progress)

    config = experiment.read_json(CONFIG)
    device = torch.device("cuda:0")
    model, scheduler, _metadata = pipeline.load_stage1(
        config, CHECKPOINT, EXPECTED_CHECKPOINT_SHA256, device
    )
    sim_config = pipeline.make_sim_config(config)
    bundle_rows = experiment.load_bundle_rows(verify_payloads=True)
    gt_mapping = experiment.load_gt_mapping()
    protocol = experiment.protocol_receipt()
    geometry = experiment._geometry_receipt(protocol)
    if tuple(geometry["raw_frame_order"]) != EXPECTED_RAW_ORDER:
        raise RuntimeError("APD conditioning order is not DMD_6F_2O3P H/V 0/120/240")
    metrics = _load_metrics()
    formal_rows = _formal_nominal_rows()
    hard_mismatch_details: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []

    for order, sample in enumerate(bundle_rows):
        current_source_hashes = {path: sha_file(Path(path)) for path in source_hashes}
        if current_source_hashes != source_hashes:
            changed = sorted(
                path for path in source_hashes
                if source_hashes[path] != current_source_hashes.get(path)
            )
            raise RuntimeError(f"Convergence source/config fingerprint changed mid-run: {changed}")
        part = per_fov_dir / f"{order:03d}_{sample['sample_id']}.csv"
        if part.is_file():
            expected_part = part_index.get(order)
            if not expected_part or expected_part.get("sha256") != sha_file(part):
                raise RuntimeError(f"Convergence resume part is not hash-bound: {part}")
            with part.open("r", encoding="utf-8", newline="") as handle:
                cached = list(csv.DictReader(handle))
            if (
                len(cached) != 642
                or [int(row["update"]) for row in cached[:321]] != list(range(321))
                or [int(row["update"]) for row in cached[321:]] != list(range(321))
                or {row["method"] for row in cached[:321]} != {"PhysMap-6"}
                or {row["method"] for row in cached[321:]} != {"APD-SIM-6"}
                or any(int(row["sample_order"]) != order for row in cached)
                or any(row["sample_id"] != sample["sample_id"] for row in cached)
                or any(
                    row["scope"] != (
                        "formal_configuration_replay"
                        if int(row["update"]) <= FORMAL_UPDATE else "diagnostic_only"
                    )
                    for row in cached
                )
                or any(
                    row["preclip_below_fraction"] != "NOT_MEASURED"
                    or row["preclip_above_fraction"] != "NOT_MEASURED"
                    for row in cached
                )
            ):
                raise RuntimeError(f"Invalid convergence resume part: {part}")
            if any(row["raw_stack_sha256"] != sample["raw_stack_sha256"] for row in cached):
                raise RuntimeError(f"Convergence resume input fingerprint mismatch: {part}")
            replay_receipts = expected_part.get("formal40_replay_predictions")
            if not isinstance(replay_receipts, Mapping) or set(replay_receipts) != {
                "PhysMap-6", "APD-SIM-6"
            }:
                raise RuntimeError(f"Convergence resume replay receipt missing: {part}")
            for method in ("PhysMap-6", "APD-SIM-6"):
                replay_path = _formal40_replay_path(
                    output_dir, order, sample["sample_id"], method
                )
                receipt = replay_receipts[method]
                relative = str(replay_path.relative_to(output_dir)).replace("\\", "/")
                if (
                    not replay_path.is_file()
                    or receipt.get("path") != relative
                    or receipt.get("file_sha256") != sha_file(replay_path)
                ):
                    raise RuntimeError(
                        f"Convergence resume replay NPY is not file-hash-bound: {replay_path}"
                    )
                replay_array = np.ascontiguousarray(
                    np.load(replay_path, allow_pickle=False), dtype=np.float32
                )
                replay_array_hash = pipeline.sha_array(replay_array)
                if (
                    receipt.get("array_sha256") != replay_array_hash
                    or receipt.get("shape") != list(replay_array.shape)
                ):
                    raise RuntimeError(
                        f"Convergence resume replay NPY payload changed: {replay_path}"
                    )
                current40_matches = [
                    row for row in cached
                    if row["method"] == method and int(row["update"]) == FORMAL_UPDATE
                ]
                if len(current40_matches) != 1:
                    raise RuntimeError(f"Convergence resume update40 row missing: {part}/{method}")
                regression = _formal40_regression_row(
                    sample_order=order,
                    sample=sample,
                    method=method,
                    current_array=replay_array,
                    current_trace_row=current40_matches[0],
                    current_path=replay_path,
                    old_row=formal_rows[(order, method)],
                )
                if receipt.get("regression_row_sha256") != canonical_sha(regression):
                    raise RuntimeError(
                        f"Convergence resume regression receipt changed: {replay_path}"
                    )
                if receipt.get("numeric_equivalence_pass") is not bool(
                    regression["numeric_equivalence_pass"]
                ):
                    raise RuntimeError(
                        f"Convergence resume numerical status changed: {replay_path}"
                    )
                regression_rows.append(regression)
            continue
        if order in part_index:
            raise RuntimeError(f"Convergence resume progress references missing part: {part}")
        experiment.assert_no_external_cuda_compute()
        bundle_path = experiment.BUNDLE_MANIFEST.parent / sample["npz_path"]
        with np.load(bundle_path, allow_pickle=False) as archive:
            raw_np = np.asarray(archive["raw_stack"], dtype=np.float32)
            raw = torch.from_numpy(raw_np)[None].to(device)
            theta = experiment._theta_from_archive(archive, device)
        if pipeline.sha_array(raw_np) != sample["raw_stack_sha256"]:
            raise RuntimeError(f"Raw input hash mismatch: {sample['sample_id']}")
        gt = experiment._late_gt(gt_mapping[sample["sample_id"]])
        wide = raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        diffusion_seed = experiment._official_nominal_diffusion_seed(sample["raw_stack_sha256"])
        x_ws, _runtime, _peak = pipeline.stage1_reconstruct(raw, model, scheduler, seed=diffusion_seed)

        diff_summary = _image_summary(x_ws, include_sha256=True)
        old_diff = formal_rows[(order, "DiffWS-6")]
        old_diff_path = _prediction_path(order, sample["sample_id"], "DiffWS-6")
        old_diff_array = np.load(old_diff_path, allow_pickle=False)
        old_diff_file_array_hash = pipeline.sha_array(np.ascontiguousarray(old_diff_array, dtype=np.float32))
        if diff_summary["sha256"] != old_diff["prediction_sha256"] or old_diff_file_array_hash != old_diff["prediction_sha256"]:
            hard_mismatch_details.append({
                "sample_order": order,
                "sample_id": sample["sample_id"],
                "method": "DiffWS-6",
                "field": "prediction_sha256",
                "current": diff_summary["sha256"],
                "old_csv": old_diff["prediction_sha256"],
                "old_npy": old_diff_file_array_hash,
            })
            write_json(output_dir / "APD_PHYSMAP_CONVERGENCE_MISMATCH.json", {
                "status": "FAIL",
                "reason": "Stage-1 DiffWS identity is required to be bitwise exact",
                "mismatches": hard_mismatch_details,
            })
            raise RuntimeError(
                f"Formal DiffWS identity mismatch: {sample['sample_id']}"
            )

        phys_rows, _phys_final, phys_hashes, phys40_array = _run_trace(
            wide, raw, gt, order, sample, "PhysMap-6", sim_config, theta, geometry, metrics
        )
        apd_rows, _apd_final, apd_hashes, apd40_array = _run_trace(
            x_ws, raw, gt, order, sample, "APD-SIM-6", sim_config, theta, geometry, metrics
        )
        case_rows = phys_rows + apd_rows
        replay_receipts: dict[str, dict[str, Any]] = {}
        for method, method_rows, state_hashes, replay_array in (
            ("PhysMap-6", phys_rows, phys_hashes, phys40_array),
            ("APD-SIM-6", apd_rows, apd_hashes, apd40_array),
        ):
            current40 = method_rows[FORMAL_UPDATE]
            old = formal_rows[(order, method)]
            replay_path = _formal40_replay_path(
                output_dir, order, sample["sample_id"], method
            )
            atomic_write(replay_path, _npy_bytes(replay_array))
            replay_array_hash = pipeline.sha_array(replay_array)
            if replay_array_hash != state_hashes[FORMAL_UPDATE]:
                raise RuntimeError(
                    f"Captured update-40 replay hash changed before persistence: "
                    f"{sample['sample_id']}/{method}"
                )
            persisted_replay = np.load(replay_path, allow_pickle=False)
            if (
                persisted_replay.dtype != np.dtype(np.float32)
                or persisted_replay.shape != replay_array.shape
            ):
                raise RuntimeError(
                    f"Persisted update-40 replay dtype/shape mismatch: {replay_path}"
                )
            persisted_replay = np.ascontiguousarray(
                persisted_replay, dtype=np.float32
            )
            if pipeline.sha_array(persisted_replay) != replay_array_hash:
                raise RuntimeError(
                    f"Persisted update-40 replay payload mismatch: {replay_path}"
                )
            regression = _formal40_regression_row(
                sample_order=order,
                sample=sample,
                method=method,
                current_array=persisted_replay,
                current_trace_row=current40,
                current_path=replay_path,
                old_row=old,
            )
            regression_rows.append(regression)
            replay_receipts[method] = {
                "path": str(replay_path.relative_to(output_dir)).replace("\\", "/"),
                "file_sha256": regression["current_prediction_file_sha256"],
                "array_sha256": regression["current_prediction_array_sha256"],
                "shape": list(replay_array.shape),
                "regression_row_sha256": canonical_sha(regression),
                "numeric_equivalence_pass": bool(
                    regression["numeric_equivalence_pass"]
                ),
            }
        atomic_write(part, _csv_bytes(case_rows, TRACE_FIELDS))
        part_index[order] = {
            "path": str(part.relative_to(output_dir)).replace("\\", "/"),
            "sha256": sha_file(part),
            "row_count": len(case_rows),
            "raw_stack_sha256": sample["raw_stack_sha256"],
            "formal40_replay_predictions": replay_receipts,
        }
        progress["completed_sample_orders"] = list(range(order + 1))
        progress["parts"] = {str(key): value for key, value in sorted(part_index.items())}
        progress["formal40_numeric_equivalence_pass_count_so_far"] = sum(
            bool(row["numeric_equivalence_pass"]) for row in regression_rows
        )
        progress["formal40_numeric_equivalence_mismatch_count_so_far"] = sum(
            not bool(row["numeric_equivalence_pass"]) for row in regression_rows
        )
        write_json(progress_path, progress)
        del (
            raw, wide, x_ws, phys_rows, apd_rows, _phys_final, _apd_final,
            phys40_array, apd40_array,
        )
        torch.cuda.empty_cache()

    expected_part_names = {
        f"{order:03d}_{sample['sample_id']}.csv"
        for order, sample in enumerate(bundle_rows)
    }
    extra_parts = sorted(
        path.name for path in per_fov_dir.glob("*.csv")
        if path.name not in expected_part_names
    )
    if extra_parts:
        raise RuntimeError(f"Unexpected convergence part files: {extra_parts}")
    if sorted(part_index) != list(range(30)):
        raise RuntimeError(f"Convergence progress index incomplete: {sorted(part_index)}")
    expected_replay_names = {
        _formal40_replay_path(output_dir, order, sample["sample_id"], method).name
        for order, sample in enumerate(bundle_rows)
        for method in ("PhysMap-6", "APD-SIM-6")
    }
    observed_replay_names = {path.name for path in replay_dir.glob("*.npy")}
    if observed_replay_names != expected_replay_names:
        raise RuntimeError(
            "Formal update-40 replay NPY set mismatch: "
            f"missing={sorted(expected_replay_names - observed_replay_names)}, "
            f"extra={sorted(observed_replay_names - expected_replay_names)}"
        )
    if len(regression_rows) != 60:
        raise RuntimeError(
            f"Formal update-40 regression grid incomplete: {len(regression_rows)}"
        )
    regression_rows.sort(
        key=lambda row: (
            int(row["sample_order"]),
            0 if row["method"] == "PhysMap-6" else 1,
        )
    )
    numerical_disposition = _numerical_completion_disposition(regression_rows)
    numerical_failures = numerical_disposition["failures"]
    regression_path = output_dir / "APD_PHYSMAP_FORMAL40_NUMERICAL_REGRESSION.csv"
    atomic_write(
        regression_path,
        _csv_bytes(regression_rows, FORMAL40_REGRESSION_FIELDS),
    )
    regression_csv_sha256 = sha_file(regression_path)
    numerical_failure_receipt_path = (
        output_dir / "APD_PHYSMAP_CONVERGENCE_MISMATCH.json"
    )
    numerical_failure_receipt_sha256: str | None = None
    if numerical_failures:
        write_json(numerical_failure_receipt_path, {
            "status": "FAIL_NUMERICAL_EQUIVALENCE",
            "reason": (
                "One or more formal update-40 replays exceeded a predeclared "
                "float32 numerical-equivalence tolerance; thresholds were not changed."
            ),
            "formal40_numerical_tolerances": FORMAL40_NUMERICAL_TOLERANCES,
            "numeric_equivalence_expected_count": 60,
            "numeric_equivalence_pass_count": 60 - len(numerical_failures),
            "numeric_equivalence_mismatch_count": len(numerical_failures),
            "regression_csv": str(regression_path),
            "regression_csv_sha256": regression_csv_sha256,
            "regressions": numerical_failures,
        })
        numerical_failure_receipt_sha256 = sha_file(
            numerical_failure_receipt_path
        )
    elif numerical_failure_receipt_path.exists():
        raise RuntimeError(
            "Unexpected numerical-equivalence failure receipt in a passing replay"
        )

    rows: list[dict[str, Any]] = []
    for order, sample in enumerate(bundle_rows):
        with (per_fov_dir / f"{order:03d}_{sample['sample_id']}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 30 * 2 * 321:
        raise RuntimeError(f"Convergence grid incomplete: {len(rows)}")
    if any(str(row["finite"]).lower() != "true" for row in rows):
        raise RuntimeError("Convergence grid contains non-finite rows")
    if any(
        row["preclip_below_fraction"] != "NOT_MEASURED"
        or row["preclip_above_fraction"] != "NOT_MEASURED"
        for row in rows
    ):
        raise RuntimeError("Unavailable preclip fields must be uniformly NOT_MEASURED")
    atomic_write(output_dir / "APD_PHYSMAP_CONVERGENCE_PER_FOV.csv", _csv_bytes(rows, TRACE_FIELDS))

    endpoints = _summarize_trace(rows)
    flags = _convergence_flags(rows)
    init_rows = [row for row in rows if int(row["update"]) == 0]
    initializers: dict[str, Any] = {}
    for method in ("PhysMap-6", "APD-SIM-6"):
        selected = [row for row in init_rows if row["method"] == method]
        initializers[method] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in selected])),
                "sample_sd": float(np.std([float(row[metric]) for row in selected], ddof=1)),
            }
            for metric in ("image_min", "image_max", "image_mean", "image_std", "psnr", "ssim", "fraction_at_clip_min", "fraction_at_clip_max")
        }
    xws_below_x_init = sum(
        float(next(row for row in init_rows if int(row["sample_order"]) == order and row["method"] == "APD-SIM-6")["psnr"])
        < float(next(row for row in init_rows if int(row["sample_order"]) == order and row["method"] == "PhysMap-6")["psnr"])
        for order in range(30)
    )
    xws_below_formal_wf = sum(
        float(formal_rows[(order, "DiffWS-6")]["psnr"])
        < float(formal_rows[(order, "WF")]["psnr"])
        for order in range(30)
    )
    clipping = _clipping_summary(rows)
    after = _baseline_hashes()
    if after != before:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise RuntimeError(f"Protected formal run changed during diagnostic: {changed}")
    final_prediction_receipt = _formal_predictions_receipt()
    if final_prediction_receipt != formal_prediction_receipt:
        raise RuntimeError("Protected formal prediction files changed during diagnostic")
    bitwise_match_count = sum(
        bool(row["bitwise_exact"]) for row in regression_rows
    )
    numerical_pass_count = int(numerical_disposition["pass_count"])
    numerical_mismatch_count = int(numerical_disposition["mismatch_count"])
    summary_status = str(numerical_disposition["summary_status"])
    completion_status = str(numerical_disposition["progress_status"])
    replay_file_hashes, replay_array_hashes = _verify_formal40_replay_artifacts(
        regression_rows, output_dir
    )
    nondeterminism_evidence = _cuda_nondeterminism_evidence()
    core_provenance = _formal_core_provenance(
        source_hashes[str(Path(core.__file__).resolve())]
    )
    summary = {
        "schema_version": 2,
        "status": summary_status,
        "overall_ready": summary_status == "PASS",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "diagnostic only; does not replace formal 40-update results",
        "formal_method_configuration": experiment.REFINEMENT_CONFIG.receipt(),
        "formal_update": FORMAL_UPDATE,
        "diagnostic_endpoints": list(DIAGNOSTIC_ENDPOINTS),
        "trajectory_update_range": [0, 320],
        "trajectory_scope_policy": {
            "updates_0_through_40": "formal_configuration_replay",
            "updates_41_through_320": "diagnostic_only",
            "old_authoritative_formal_update40_identity": (
                "immutable old-run nominal_predictions NPY and nominal CSV row"
            ),
        },
        "fov_count": 30,
        "method_count": 2,
        "row_count": len(rows),
        "shared_core": f"{core.masked_refine.__module__}.{core.masked_refine.__qualname__}",
        "shared_optimizer": "torch.optim.Adam",
        "shared_learning_rate": 0.005,
        "shared_input_and_geometry": True,
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_frame_order": list(EXPECTED_RAW_ORDER),
        "conditioning_order_consistent": tuple(geometry["raw_frame_order"]) == EXPECTED_RAW_ORDER,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "resume_fingerprint_sha256": fingerprint,
        "source_hashes": source_hashes,
        "protected_formal_run_hash_baseline_sha256": canonical_sha(before),
        "protected_formal_run_unchanged": before == after,
        "formal_update40_hard_identity_mismatch_count": len(hard_mismatch_details),
        "formal_update40_prediction_bitwise_exact_match_count": bitwise_match_count,
        "formal_update40_prediction_bitwise_mismatch_count": 60 - bitwise_match_count,
        "formal_update40_numeric_equivalence_pass_count": numerical_pass_count,
        "formal_update40_numeric_equivalence_expected_count": 60,
        "formal_update40_numeric_equivalence_mismatch_count": numerical_mismatch_count,
        "formal_update40_numeric_equivalence_passed": numerical_mismatch_count == 0,
        "formal_update40_numerical_failure_policy": (
            "collect_all_30_fov_and_60_regression_rows_then_fail_overall"
        ),
        "formal_update40_thresholds_relaxed_after_observation": False,
        "formal_update40_retry_selected_to_obtain_pass": False,
        "formal_update40_numeric_equivalence_failure_cases": [
            {
                "sample_order": int(row["sample_order"]),
                "sample_id": row["sample_id"],
                "method": row["method"],
                "failed_checks": [
                    field for field in (
                        "max_abs_pass", "rmse_pass", "psnr_pass", "ssim_pass",
                        "objective_pass", "observed_nrmse_pass",
                    )
                    if not bool(row[field])
                ],
            }
            for row in numerical_failures
        ],
        "formal_update40_numeric_equivalence_failure_receipt": (
            str(numerical_failure_receipt_path) if numerical_failures else None
        ),
        "formal_update40_numeric_equivalence_failure_receipt_sha256": (
            numerical_failure_receipt_sha256
        ),
        "formal_update40_numerical_tolerances": FORMAL40_NUMERICAL_TOLERANCES,
        "formal_update40_regression_csv": str(regression_path),
        "formal_update40_regression_csv_sha256": regression_csv_sha256,
        "formal_update40_current_replay_prediction_count": len(replay_file_hashes),
        "formal_update40_current_replay_file_mapping_sha256": canonical_sha(
            replay_file_hashes
        ),
        "formal_update40_current_replay_array_mapping_sha256": canonical_sha(
            replay_array_hashes
        ),
        "formal_update40_old_npy_is_authoritative": True,
        "formal_configuration_replay_does_not_replace_formal_result": True,
        "x_ws_formal_prediction_hash_exact_match_count": 30,
        "formal_prediction_files_preflight_receipt": formal_prediction_receipt,
        "cuda_nondeterminism_evidence": nondeterminism_evidence,
        "formal_core_provenance": core_provenance,
        "endpoint_summaries": endpoints,
        "initialization_summaries": initializers,
        "x_ws_psnr_below_x_init_fov_count": xws_below_x_init,
        "x_ws_psnr_below_formal_wf_fov_count": xws_below_formal_wf,
        "formal_wf_is_clipped_six_frame_mean": True,
        "x_ws_intensity_domain_mismatch_evidence": {
            "interpretation_rule": "report observed initializer scale/clipping and quality; no post-hoc rescaling",
            "apd_initializer": initializers["APD-SIM-6"],
            "physmap_initializer": initializers["PhysMap-6"],
        },
        "clipping_diagnostics": clipping,
        "preclip_fraction_measurement_available": False,
        "preclip_fraction_note": (
            "Pre-step out-of-range fractions are not measured because inserting reductions "
            "between Adam.step and the frozen clamp could perturb the formal operation order; "
            "post-clamp saturation fractions are the authoritative clipping diagnostic."
        ),
        "convergence_assessment": flags,
        "all_rows_finite": True,
        "formal_configuration_changed": False,
    }
    write_json(output_dir / "APD_PHYSMAP_CONVERGENCE_SUMMARY.json", summary)
    if numerical_failures:
        failure_labels = ", ".join(
            f"order {int(row['sample_order'])} {row['method']}"
            for row in numerical_failures
        )
        numerical_disposition_md = (
            "**FAIL:** the predeclared update-40 numerical-equivalence gate failed for "
            f"{len(numerical_failures)}/60 replay-method cases ({failure_labels}). "
            "Thresholds were not relaxed and no retry was selected to obtain a passing replay. "
            "Remaining trajectories, regression rows and figures were collected only to preserve "
            "complete diagnostic evidence; this run is not READY."
        )
    else:
        numerical_disposition_md = (
            "**PASS:** all 60 replay-method cases passed every predeclared update-40 "
            "float32 numerical-equivalence threshold."
        )
    diagnostic_md = f"""# APD-versus-PhysMap strict convergence diagnostic

Status: `{summary_status}`

## Overall numerical-equivalence disposition

{numerical_disposition_md}

This is a formal-configuration replay through update 40 followed by a diagnostic-only extension.
Both methods use the same `masked_refine` core, Adam, `lr=0.005`, raw six-frame tensor, mask,
geometry and forward parameters. Only initialization differs. Replay rows 0-40 and diagnostic
updates 41-320 do not replace the authoritative old-run formal result.

## Formal-boundary regression gate

- Authoritative formal identity: immutable old-run update-40 NPY and nominal CSV row.
- Bitwise-exact replay prediction matches: `{bitwise_match_count}/60`; bitwise mismatches:
  `{60 - bitwise_match_count}/60` (reported, not hidden or relabelled exact).
- Predeclared float32 numerical-equivalence passes: `{numerical_pass_count}/60`; failures:
  `{60 - numerical_pass_count}/60`.
- Tolerances: max abs `{FORMAL40_NUMERICAL_TOLERANCES['max_abs']:.12g}`, RMSE
  `{FORMAL40_NUMERICAL_TOLERANCES['rmse']:.12g}`, PSNR
  `{FORMAL40_NUMERICAL_TOLERANCES['psnr_abs']:.12g}` dB, SSIM
  `{FORMAL40_NUMERICAL_TOLERANCES['ssim_abs']:.12g}`, objective
  `{FORMAL40_NUMERICAL_TOLERANCES['objective_abs']:.12g}`, observed NRMSE
  `{FORMAL40_NUMERICAL_TOLERANCES['observed_nrmse_abs']:.12g}`.
- Exact `x_ws` hashes matching formal DiffWS outputs: `30/30`.
- Protected formal run unchanged: `{before == after}`.
- Per-FOV receipt: `APD_PHYSMAP_FORMAL40_NUMERICAL_REGRESSION.csv`; all 60 current replay
  arrays are persisted under `formal40_replay_predictions/` and hash-bound for resume.

## Reproducibility qualification

- CUDA debug isolated `adaptive_avg_pool2d_backward_cuda` from the area downsampling path as
  nondeterministic; repeated same-state gradients and 40-step replays are not bitwise stable.
- The old formal run records core SHA-256 `{FORMAL_CORE_RECORDED_SHA256}` but contains no old
  core source snapshot. The current core SHA-256 is
  `{source_hashes[str(Path(core.__file__).resolve())]}` and differs, so line-by-line old/current
  source comparison is unavailable. This provenance gap is disclosed rather than inferred away.

## Diagnostic findings

- APD not-converged-at-40 count under the stated objective-change rule:
  `{flags['apd_not_converged_at_40_fov_count']}/30`.
- Different APD/PhysMap update-320 objective-basin count under the stated 1% rule:
  `{flags['different_objective_basin_at_320_fov_count']}/30`.
- `x_ws` PSNR below `x_init` (the formal clipped six-frame-mean WF): `{xws_below_x_init}/30`.
- Formal DiffWS (`x_ws`) PSNR below formal WF row: `{xws_below_formal_wf}/30`.
- Maximum APD post-clip zero/one fractions: `{clipping['APD-SIM-6']['max_postclip_at_min_fraction']:.6g}` /
  `{clipping['APD-SIM-6']['max_postclip_at_max_fraction']:.6g}`.

These rules are descriptive diagnostic criteria, not new method-selection criteria. Preclip
fractions are `NOT_MEASURED`; only post-clamp saturation is used for clipping conclusions. No
formal 40-update configuration or authoritative formal result was changed; only the explicitly
diagnostic endpoint was extended through update 320.
"""
    atomic_write(output_dir / "APD_PHYSMAP_CONVERGENCE_DIAGNOSTIC.md", diagnostic_md.encode("utf-8"))
    _render_plot(rows, "objective", "Masked Poisson-Gaussian objective", output_dir / "APD_PHYSMAP_OBJECTIVE_TRAJECTORY.pdf")
    _render_plot(rows, "psnr", "PSNR (dB)", output_dir / "APD_PHYSMAP_PSNR_TRAJECTORY.pdf")
    final_replay_file_hashes, final_replay_array_hashes = (
        _verify_formal40_replay_artifacts(regression_rows, output_dir)
    )
    if (
        final_replay_file_hashes != replay_file_hashes
        or final_replay_array_hashes != replay_array_hashes
        or sha_file(regression_path) != regression_csv_sha256
        or (
            numerical_failures
            and sha_file(numerical_failure_receipt_path)
            != numerical_failure_receipt_sha256
        )
    ):
        raise RuntimeError(
            "Formal update-40 replay artifacts or regression CSV changed before completion"
        )
    progress["status"] = completion_status
    progress["summary_status"] = summary_status
    progress["overall_ready"] = summary_status == "PASS"
    progress["summary_sha256"] = sha_file(output_dir / "APD_PHYSMAP_CONVERGENCE_SUMMARY.json")
    progress["formal40_regression_csv_sha256"] = regression_csv_sha256
    progress["formal40_replay_file_mapping_sha256"] = canonical_sha(replay_file_hashes)
    progress["formal40_replay_array_mapping_sha256"] = canonical_sha(replay_array_hashes)
    progress["formal40_prediction_bitwise_exact_match_count"] = bitwise_match_count
    progress["formal40_numeric_equivalence_pass_count"] = numerical_pass_count
    progress["formal40_numeric_equivalence_mismatch_count"] = numerical_mismatch_count
    progress["formal40_numeric_equivalence_failure_receipt_sha256"] = (
        numerical_failure_receipt_sha256
    )
    write_json(progress_path, progress)
    return summary


__all__ = [
    "DIAGNOSTIC_ENDPOINTS", "EXPECTED_MASK", "EXPECTED_RAW_ORDER", "FORMAL_RUN",
    "FORMAL40_NUMERICAL_TOLERANCES", "FORMAL40_REGRESSION_FIELDS", "FORMAL_UPDATE",
    "TRACE_FIELDS", "audit_weight_branch", "run_convergence_diagnostics",
]

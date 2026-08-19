"""Prespecified three-GT Figure 5 production and independent CPU audit.

This Figure-only workflow intentionally does not read or modify the completed
R1C3 run.  It freezes sample/crop/profile/display/seed choices before any model
inference, then produces nine matched conditions and 27 method rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile

from .physmap6_pipeline import NORMALIZATION_HASH, PROTOCOL_HASH, PROTOCOL_ID, RAW_ORDER, sha_array
from .physmap6_core import RefinementConfig


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "outputs" / "reviewer1_physmap6_fig5_prespecified"
GT_MANIFEST = ROOT / "_REVISION_R1_20260812T082048Z" / "DATASET_MANIFEST.csv"
BUNDLE_MANIFEST = (
    ROOT / "outputs" / "OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
    / "01_shared_contract" / "test30_dmd6_manifest.tsv"
)
PROTECTED_RUN = ROOT / "outputs" / "reviewer1_physmap6_strict" / "20260813T183229Z"
CHECKPOINT = ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt"
CONFIG = ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
EXPECTED_CHECKPOINT_SHA256 = "10fb16662a8b71b877f2cab81bdc151dcded92f6efd1c4b006306b901a8adff7"
SELECTION_POLICY_ID = "R1C3_FIG5_PRESPECIFIED_THREE_GT_V1"
PROFILE_POLICY_ID = "GT_ROW_VARIANCE_ARGMAX_FLOAT64_FIRST_V1"
CROP_POLICY_ID = "FULL_SOURCE_P0P5_P99P5_THEN_CENTER_320_V1"
DISPLAY_POLICY_ID = "NATIVE_NORMALIZED_FIXED_ZERO_ONE_BEFORE_INFERENCE_V1"
METHOD_ORDER = ("APD-SIM-6", "PhysMap-6", "DiffWS-6")
FACTOR_ORDER = ("phase_jitter_rad", "psf_blur", "photon_scale_mul")
SEVERITIES = {
    "phase_jitter_rad": (0.1, 0.4, 0.6),
    "psf_blur": (0.1, 0.2, 0.3),
    "photon_scale_mul": (0.5, 0.25, 0.125),
}
LEGACY_ROBUSTNESS_SEED = 20260812
CROP_SIZE = 320
REFINEMENT_RECEIPT = RefinementConfig().receipt()
FLOAT32_EPS = float(np.finfo(np.float32).eps)
REPLAY_MAX_ABS_THRESHOLD = 32.0 * FLOAT32_EPS
REPLAY_RMSE_THRESHOLD = 4.0 * FLOAT32_EPS


@dataclass(frozen=True)
class Figure5Target:
    factor: str
    sample_order: int
    sample_id: str
    parent_id: str
    structure: str
    source_file_sha256: str


TARGETS = (
    Figure5Target(
        "phase_jitter_rad", 19, "ER_Cell_068_GTSIM_level_06", "ER:Cell_068", "ER",
        "42593cb2dec4bb6ca6f77c80e9888a0ddd95c773e0c4fc4a39b61b4c0806c40f",
    ),
    Figure5Target(
        "psf_blur", 9, "CCPs_Cell_054_SIM_gt", "CCP:Cell_054", "CCP",
        "cd99a6825246fcbfe583f4d782d68796e9c234cc5d0492fbd125cafa83335be8",
    ),
    Figure5Target(
        "photon_scale_mul", 29, "microtubules_Cell_055_SIM_gt", "MT:Cell_055", "MT",
        "426f575183112dfd643df1d75d3610bac9a6320770d4dc6a0d58bf2f7ed13b3a",
    ),
)

SHARED_FIELDS = (
    "selection_payload_sha256", "selection_receipt_file_sha256",
    "sample_id", "parent_id", "structure",
    "gt_patch_sha256", "raw_stack_sha256", "validity_mask_sha256",
    "geometry_sha256", "forward_parameters_sha256", "noise_identity_sha256",
    "normalization_sha256", "perturbation_direction_seed", "noise_seed", "diffusion_seed",
    "profile_row", "display_min", "display_max", "method_independent_before_inference",
    "condition_npz", "condition_npz_sha256",
)
CSV_FIELDS = (
    "factor_order", "factor", "severity_order", "severity", "condition_id",
    "sample_order", "sample_id", "parent_id", "structure", "method_order", "method",
    "source_file_sha256", "gt_patch_sha256", "raw_stack_sha256",
    "validity_mask_sha256", "geometry_sha256", "forward_parameters_sha256",
    "noise_identity_sha256", "normalization_sha256", "noise_seed", "diffusion_seed",
    "perturbation_direction_seed",
    "refinement_config_sha256", "profile_row", "display_min", "display_max",
    "method_independent_before_inference", "prediction_sha256", "replay_prediction_sha256",
    "replay_bitwise_match", "replay_numeric_equivalent", "replay_max_abs_error",
    "replay_rmse", "psnr", "ssim", "observed_nrmse",
    "poisson_gaussian_objective", "gradient_finite", "output_finite",
    "condition_npz", "condition_npz_sha256", "status",
    "selection_payload_sha256", "selection_receipt_file_sha256",
)


class Figure5Blocked(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Figure5Blocked(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise Figure5Blocked(f"stale temporary path {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link is an atomic create-if-absent publish on
        # NTFS.  Unlike os.replace, it cannot overwrite a concurrently created
        # target; unsupported filesystems fail closed.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_new(path: Path, value: Any) -> None:
    _atomic_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def _read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _normalize_full(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise Figure5Blocked(f"invalid GT shape/finite state: {value.shape}")
    low, high = np.percentile(value, (0.5, 99.5))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise Figure5Blocked("degenerate GT normalization range")
    return np.ascontiguousarray(np.clip((value - low) / (high - low + 1e-8), 0.0, 1.0), dtype=np.float32)


def select_gt_profile(gt: np.ndarray) -> dict[str, Any]:
    """Return a profile coordinate using GT only; ties select the first row."""
    value = np.asarray(gt)
    if value.ndim != 2 or value.dtype != np.float32 or not bool(np.isfinite(value).all()):
        raise Figure5Blocked("profile GT must be finite 2-D float32")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise Figure5Blocked("profile GT outside native [0,1]")
    scores = np.var(value.astype(np.float64), axis=1, dtype=np.float64)
    row = int(np.argmax(scores))
    return {
        "policy_id": PROFILE_POLICY_ID, "axis": "horizontal", "row_index": row,
        "column_start": 0, "column_stop_exclusive": int(value.shape[1]),
        "score": float(scores[row]), "score_vector_sha256": sha_array(scores),
        "gt_only": True, "frozen_before_inference": True,
    }


def display_range_for_condition(gt: np.ndarray) -> tuple[float, float]:
    value = np.asarray(gt)
    if value.ndim != 2 or not bool(np.isfinite(value).all()):
        raise Figure5Blocked("display GT invalid")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise Figure5Blocked("display GT outside normalized range")
    return 0.0, 1.0


def _measurement_seed(factor: str, sample_order: int) -> int:
    return (LEGACY_ROBUSTNESS_SEED
            + int(hashlib.sha256(factor.encode("utf-8")).hexdigest()[:8], 16)
            + int(sample_order) * 131) % 2147483647


def _diffusion_seed(sample_order: int) -> int:
    return (LEGACY_ROBUSTNESS_SEED + int(sample_order) * 17) % 2147483647


def _perturbation_seed(factor: str, sample_order: int) -> int:
    return (LEGACY_ROBUSTNESS_SEED + int(sample_order) * 1009
            + int(hashlib.sha256(factor.encode("utf-8")).hexdigest()[:8], 16) % 100000)


def validate_runtime_refinement_receipt(runtime: Mapping[str, Any]) -> None:
    """Validate frozen config keys plus the audited RefineResult extensions."""
    if not isinstance(runtime, Mapping):
        raise Figure5Blocked("runtime refinement receipt is not a mapping")
    for key, expected in REFINEMENT_RECEIPT.items():
        if runtime.get(key) != expected:
            raise Figure5Blocked(f"runtime refinement config drift: {key}")
    extensions = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "raw_frame_order": list(RAW_ORDER),
        "observed_frame_count": 6,
        "invalid_slots_excluded": True,
        "history_includes_initial_and_each_post_update_state": True,
        "executed_updates": 40,
        "formal_result_update": 40,
        "diagnostic_only_extension": False,
        "diagnostic_extension_does_not_replace_formal_result": False,
    }
    for key, expected in extensions.items():
        if runtime.get(key) != expected:
            raise Figure5Blocked(f"runtime refinement extension drift: {key}")
    validity = runtime.get("validity_mask")
    if validity != [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
        raise Figure5Blocked("runtime refinement validity mask drift")
    if not isinstance(runtime.get("forward_theta"), Mapping):
        raise Figure5Blocked("runtime refinement theta receipt missing")


def _resolve_source(manifest_path: Path, expected_name: str, expected_sha: str,
                    search_roots: Sequence[Path] | None) -> tuple[Path, dict[str, str]]:
    rows = _read_rows(manifest_path)
    matches = [row for row in rows if row.get("sample_id") == expected_name
               and row.get("dataset") == "30-FOV GT benchmark"]
    if len(matches) != 1 or matches[0].get("sha256") != expected_sha:
        raise Figure5Blocked(f"dataset manifest identity mismatch for {expected_name}")
    manifest_path_value = Path(matches[0]["absolute_path"])
    candidates: list[Path] = []
    if manifest_path_value.is_file():
        candidates.append(manifest_path_value.resolve())
    if search_roots is None:
        data_root = Path("data/external_input")
        roots = tuple(
            path for path in (
                manifest_path_value.parent,
                *((path for path in data_root.glob("GT*") if path.is_dir())
                  if data_root.is_dir() else ()),
            ) if path.is_dir()
        )
    else:
        roots = tuple(search_roots)
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in root.rglob(manifest_path_value.name):
            if candidate.is_file() and candidate.resolve() not in candidates:
                candidates.append(candidate.resolve())
    hashes = {path: sha_file(path) for path in candidates}
    bad = {str(path): digest for path, digest in hashes.items() if digest != expected_sha}
    if bad:
        raise Figure5Blocked(f"conflicting same-name GT source(s) for {expected_name}: {bad}")
    good = [path for path, digest in hashes.items() if digest == expected_sha]
    if not good:
        raise Figure5Blocked(f"no SHA-matched GT source for {expected_name}")
    selected = sorted(good, key=lambda item: str(item).casefold())[0]
    return selected, matches[0]


def _load_bundle_identity(target: Figure5Target) -> dict[str, str]:
    rows = _read_rows(BUNDLE_MANIFEST, delimiter="\t")
    matches = [row for row in rows if row.get("sample_id") == target.sample_id]
    if len(matches) != 1:
        raise Figure5Blocked(f"sealed bundle identity missing: {target.sample_id}")
    row = matches[0]
    checks = (
        int(row["order"]) == target.sample_order,
        row["parent_id"] == target.parent_id,
        row["structure_class"] == target.structure,
        row["source_file_sha256"] == target.source_file_sha256,
        row["protocol_id"] == PROTOCOL_ID,
        row["protocol_hash"] == PROTOCOL_HASH,
        row["frame_order"].split("/") == list(RAW_ORDER),
        row["test_gt_embedded_in_npz"] == "False",
    )
    if not all(checks):
        raise Figure5Blocked(f"sealed bundle contract mismatch: {target.sample_id}")
    return row


def build_selection_receipt(*, search_roots: Sequence[Path] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Resolve and freeze all GT-dependent choices before any inference."""
    if not GT_MANIFEST.is_file() or not BUNDLE_MANIFEST.is_file():
        raise Figure5Blocked("required sealed manifests unavailable")
    samples: list[dict[str, Any]] = []
    patches: dict[str, np.ndarray] = {}
    for target in TARGETS:
        bundle = _load_bundle_identity(target)
        source_path, manifest_row = _resolve_source(
            GT_MANIFEST, target.sample_id, target.source_file_sha256, search_roots
        )
        source_bytes_sha = sha_file(source_path)
        original = tifffile.imread(source_path)
        normalized = _normalize_full(original)
        if normalized.shape != (1004, 1004):
            raise Figure5Blocked(f"unexpected source shape {normalized.shape}: {target.sample_id}")
        y = (normalized.shape[0] - CROP_SIZE) // 2
        x = (normalized.shape[1] - CROP_SIZE) // 2
        gt = np.ascontiguousarray(normalized[y:y + CROP_SIZE, x:x + CROP_SIZE], dtype=np.float32)
        normalized_full_sha = sha_array(normalized)
        if normalized_full_sha != bundle["gt_normalized_array_sha256"]:
            raise Figure5Blocked(f"sealed normalized GT SHA mismatch: {target.sample_id}")
        profile = select_gt_profile(gt)
        display_min, display_max = display_range_for_condition(gt)
        patches[target.factor] = gt
        conditions = []
        for severity_index, severity in enumerate(SEVERITIES[target.factor]):
            conditions.append({
                "severity_order": severity_index, "severity": severity,
                "condition_id": f"{target.factor}|{severity:.12g}|{target.sample_id}",
                "perturbation_direction_seed": _perturbation_seed(
                    target.factor, target.sample_order
                ),
                "noise_seed": _measurement_seed(target.factor, target.sample_order),
                "diffusion_seed": _diffusion_seed(target.sample_order),
                "display_min": display_min, "display_max": display_max,
            })
        samples.append({
            "factor_order": FACTOR_ORDER.index(target.factor), "factor": target.factor,
            "sample_order": target.sample_order, "sample_id": target.sample_id,
            "parent_id": target.parent_id, "structure": target.structure,
            "dataset_manifest_parent_id": manifest_row["parent_id"],
            "source_path_recorded_in_manifest": manifest_row["absolute_path"],
            "resolved_source_path": str(source_path), "source_file_sha256": source_bytes_sha,
            "source_size_bytes": source_path.stat().st_size,
            "source_shape": list(np.asarray(original).shape), "source_dtype": str(np.asarray(original).dtype),
            "sealed_gt_normalized_array_sha256": bundle["gt_normalized_array_sha256"],
            "normalized_full_sha256": normalized_full_sha,
            "crop": {"y": y, "x": x, "height": CROP_SIZE, "width": CROP_SIZE,
                     "policy_id": CROP_POLICY_ID},
            "gt_patch_sha256": sha_array(gt), "profile": profile,
            "conditions": conditions,
        })
    receipt = {
        "schema_version": 2, "status": "PRESPECIFIED_BEFORE_INFERENCE",
        "selection_policy_id": SELECTION_POLICY_ID,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "method_independent_before_inference": True,
        "first_inference_started": False,
        "source_scope": "sealed 30-FOV identities; newly generated qualitative Figure-only cohort",
        "not_table2_statistical_cohort": True,
        "factor_order": list(FACTOR_ORDER), "method_order": list(METHOD_ORDER),
        "severity_order": {key: list(value) for key, value in SEVERITIES.items()},
        "display": {"policy_id": DISPLAY_POLICY_ID, "global_native_range": [0.0, 1.0],
                    "method_output_consulted": False, "frozen_before_inference": True},
        "normalization": {"policy": "full-source percentiles 0.5/99.5 then clip [0,1]",
                          "normalization_sha256": NORMALIZATION_HASH},
        "protocol": {"id": PROTOCOL_ID, "hash": PROTOCOL_HASH, "raw_order": list(RAW_ORDER)},
        "refinement_config": REFINEMENT_RECEIPT,
        "replay_equivalence": {
            "policy_id": "FLOAT32_MAXABS32EPS_RMSE4EPS_V1",
            "dtype": "float32",
            "float32_epsilon": FLOAT32_EPS,
            "max_abs_error_threshold": REPLAY_MAX_ABS_THRESHOLD,
            "rmse_threshold": REPLAY_RMSE_THRESHOLD,
            "bitwise_match_required": False,
            "frozen_before_inference": True,
        },
        "legacy_robustness_seed_base": LEGACY_ROBUSTNESS_SEED,
        "dataset_manifest": str(GT_MANIFEST), "dataset_manifest_sha256": sha_file(GT_MANIFEST),
        "sealed_bundle_manifest": str(BUNDLE_MANIFEST),
        "sealed_bundle_manifest_sha256": sha_file(BUNDLE_MANIFEST),
        "config": str(CONFIG), "config_sha256": sha_file(CONFIG),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "samples": samples,
    }
    receipt["selection_payload_sha256"] = _json_hash(receipt)
    return receipt, patches


def audit_selection_sources(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read manifests and GT bytes and recompute every GT-derived choice."""
    validate_selection_receipt(receipt)
    fixed_files = (
        (GT_MANIFEST, receipt.get("dataset_manifest_sha256"), "dataset manifest"),
        (BUNDLE_MANIFEST, receipt.get("sealed_bundle_manifest_sha256"), "bundle manifest"),
        (CONFIG, receipt.get("config_sha256"), "training config"),
        (CHECKPOINT, receipt.get("checkpoint_sha256"), "checkpoint"),
    )
    for path, expected, label in fixed_files:
        if not path.is_file() or sha_file(path) != expected:
            raise Figure5Blocked(f"{label} SHA changed after selection")
    recomputed: dict[str, str] = {}
    for target, sample in zip(TARGETS, receipt["samples"]):
        bundle = _load_bundle_identity(target)
        source = Path(sample["resolved_source_path"])
        if (not source.is_file() or source.name != Path(sample["source_path_recorded_in_manifest"]).name
                or sha_file(source) != target.source_file_sha256):
            raise Figure5Blocked(f"selected GT source changed: {target.sample_id}")
        original = tifffile.imread(source)
        normalized = _normalize_full(original)
        normalized_sha = sha_array(normalized)
        if (normalized.shape != (1004, 1004)
                or normalized_sha != bundle["gt_normalized_array_sha256"]
                or normalized_sha != sample["normalized_full_sha256"]):
            raise Figure5Blocked(f"normalized full GT changed: {target.sample_id}")
        crop = sample["crop"]
        gt = np.ascontiguousarray(
            normalized[int(crop["y"]):int(crop["y"]) + int(crop["height"]),
                       int(crop["x"]):int(crop["x"]) + int(crop["width"])],
            dtype=np.float32,
        )
        if (sha_array(gt) != sample["gt_patch_sha256"]
                or select_gt_profile(gt) != sample["profile"]
                or display_range_for_condition(gt) != (0.0, 1.0)):
            raise Figure5Blocked(f"GT crop/profile/display changed: {target.sample_id}")
        recomputed[target.factor] = sha_array(gt)
    return {
        "status": "PASS", "sample_count": 3,
        "recomputed_gt_patch_sha256": recomputed,
        "manifest_and_source_bytes_recomputed": True,
        "method_outputs_consulted": False,
    }


def build_condition_plan(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_selection_receipt(receipt)
    plan: list[dict[str, Any]] = []
    for sample in receipt["samples"]:
        for condition in sample["conditions"]:
            plan.append({**condition, **{key: sample[key] for key in (
                "factor_order", "factor", "sample_order", "sample_id", "parent_id", "structure",
                "source_file_sha256", "gt_patch_sha256",
            )}, "profile_row": int(sample["profile"]["row_index"])})
    if len(plan) != 9:
        raise Figure5Blocked("condition plan is not exactly nine conditions")
    return plan


def validate_selection_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != 2 or receipt.get("selection_policy_id") != SELECTION_POLICY_ID:
        raise Figure5Blocked("selection receipt schema/policy mismatch")
    if receipt.get("method_independent_before_inference") is not True:
        raise Figure5Blocked("selection was not method independent")
    payload_hash = receipt.get("selection_payload_sha256")
    if payload_hash != _json_hash({key: value for key, value in receipt.items()
                                   if key != "selection_payload_sha256"}):
        raise Figure5Blocked("selection payload SHA mismatch")
    if receipt.get("factor_order") != list(FACTOR_ORDER) or receipt.get("method_order") != list(METHOD_ORDER):
        raise Figure5Blocked("Figure order drift")
    if receipt.get("severity_order") != {key: list(value) for key, value in SEVERITIES.items()}:
        raise Figure5Blocked("severity order drift")
    refinement = receipt.get("refinement_config")
    if (not isinstance(refinement, Mapping) or refinement != REFINEMENT_RECEIPT
            or int(refinement.get("updates", -1)) != 40):
        raise Figure5Blocked("40-update refinement contract drift")
    if receipt.get("replay_equivalence") != {
        "policy_id": "FLOAT32_MAXABS32EPS_RMSE4EPS_V1",
        "dtype": "float32",
        "float32_epsilon": FLOAT32_EPS,
        "max_abs_error_threshold": REPLAY_MAX_ABS_THRESHOLD,
        "rmse_threshold": REPLAY_RMSE_THRESHOLD,
        "bitwise_match_required": False,
        "frozen_before_inference": True,
    }:
        raise Figure5Blocked("replay numeric-equivalence contract drift")
    display = receipt.get("display", {})
    if display.get("global_native_range") != [0.0, 1.0] or display.get("method_output_consulted") is not False:
        raise Figure5Blocked("display range not fixed independently")
    samples = receipt.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise Figure5Blocked("selection sample count mismatch")
    for target, sample in zip(TARGETS, samples):
        if any(sample.get(key) != expected for key, expected in (
            ("factor", target.factor), ("sample_order", target.sample_order),
            ("sample_id", target.sample_id), ("parent_id", target.parent_id),
            ("structure", target.structure), ("source_file_sha256", target.source_file_sha256),
        )):
            raise Figure5Blocked("selection target drift")
        if sample.get("crop") != {"y": 342, "x": 342, "height": 320, "width": 320,
                                  "policy_id": CROP_POLICY_ID}:
            raise Figure5Blocked("crop contract drift")
        profile = sample.get("profile", {})
        if profile.get("policy_id") != PROFILE_POLICY_ID or profile.get("gt_only") is not True:
            raise Figure5Blocked("profile contract drift")
        if not (0 <= int(profile.get("row_index", -1)) < 320):
            raise Figure5Blocked("profile coordinate invalid")
        expected_conditions = []
        for severity_order, severity in enumerate(SEVERITIES[target.factor]):
            expected_conditions.append({
                "severity_order": severity_order,
                "severity": severity,
                "condition_id": f"{target.factor}|{severity:.12g}|{target.sample_id}",
                "perturbation_direction_seed": _perturbation_seed(
                    target.factor, target.sample_order
                ),
                "noise_seed": _measurement_seed(target.factor, target.sample_order),
                "diffusion_seed": _diffusion_seed(target.sample_order),
                "display_min": 0.0,
                "display_max": 1.0,
            })
        if sample.get("conditions") != expected_conditions:
            raise Figure5Blocked("condition seed/severity contract drift")


def validate_condition_rows(rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]) -> None:
    plan = build_condition_plan(receipt)
    if len(rows) != 27:
        raise Figure5Blocked(f"expected 27 rows, got {len(rows)}")
    expected_keys = []
    for condition in plan:
        for method_index, method in enumerate(METHOD_ORDER):
            expected_keys.append((condition["factor"], float(condition["severity"]), method_index, method))
    actual_keys = [(str(row["factor"]), float(row["severity"]), int(row["method_order"]), str(row["method"]))
                   for row in rows]
    if actual_keys != expected_keys:
        raise Figure5Blocked("condition/method Cartesian order mismatch")
    payload_hash = str(receipt["selection_payload_sha256"])
    for condition_index, condition in enumerate(plan):
        group = rows[condition_index * 3:(condition_index + 1) * 3]
        for field in SHARED_FIELDS:
            if len({str(row[field]) for row in group}) != 1:
                raise Figure5Blocked(f"condition methods do not share {field}")
        if str(group[0]["selection_payload_sha256"]) != payload_hash:
            raise Figure5Blocked("selection payload hash binding mismatch")
        if len(str(group[0]["selection_receipt_file_sha256"])) != 64:
            raise Figure5Blocked("selection receipt file SHA malformed")
        exact_fields = {
            "factor_order": condition["factor_order"],
            "factor": condition["factor"],
            "severity_order": condition["severity_order"],
            "severity": condition["severity"],
            "condition_id": condition["condition_id"],
            "sample_order": condition["sample_order"],
            "sample_id": condition["sample_id"],
            "parent_id": condition["parent_id"],
            "structure": condition["structure"],
            "source_file_sha256": condition["source_file_sha256"],
            "gt_patch_sha256": condition["gt_patch_sha256"],
            "perturbation_direction_seed": condition["perturbation_direction_seed"],
            "noise_seed": condition["noise_seed"],
            "diffusion_seed": condition["diffusion_seed"],
            "profile_row": condition["profile_row"],
        }
        for row in group:
            for field, expected in exact_fields.items():
                actual = float(row[field]) if field == "severity" else str(row[field])
                reference = float(expected) if field == "severity" else str(expected)
                if actual != reference:
                    raise Figure5Blocked(f"condition row {field} does not match selection")
        if any(str(row["status"]) != "PASS" for row in group):
            raise Figure5Blocked("condition row not PASS")
        if any(str(row["method"]) == "WF" for row in group):
            raise Figure5Blocked("WF forbidden in Figure columns")
        for row in group:
            numeric = (
                float(row["psnr"]), float(row["ssim"]), float(row["observed_nrmse"]),
                float(row["poisson_gaussian_objective"]), float(row["display_min"]),
                float(row["display_max"]), float(row["replay_max_abs_error"]),
                float(row["replay_rmse"]),
            )
            if not all(math.isfinite(value) for value in numeric):
                raise Figure5Blocked("non-finite condition row")
            if float(row["display_min"]) != 0.0 or float(row["display_max"]) != 1.0:
                raise Figure5Blocked("condition display range drift")
            if str(row["method_independent_before_inference"]).lower() != "true":
                raise Figure5Blocked("method-independent display flag missing")
            bitwise_text = str(row["replay_bitwise_match"]).lower()
            numeric_text = str(row["replay_numeric_equivalent"]).lower()
            if bitwise_text not in {"true", "false"} or numeric_text != "true":
                raise Figure5Blocked("replay equivalence flags invalid")
            max_abs_error = float(row["replay_max_abs_error"])
            replay_rmse = float(row["replay_rmse"])
            if (max_abs_error < 0.0 or replay_rmse < 0.0
                    or max_abs_error > REPLAY_MAX_ABS_THRESHOLD
                    or replay_rmse > REPLAY_RMSE_THRESHOLD):
                raise Figure5Blocked("replay exceeds frozen float32 numeric-equivalence gate")
            hashes_equal = row["prediction_sha256"] == row["replay_prediction_sha256"]
            if (bitwise_text == "true") != hashes_equal:
                raise Figure5Blocked("replay bitwise flag/hash inconsistency")
            if (str(row["gradient_finite"]).lower() != "true"
                    or str(row["output_finite"]).lower() != "true"):
                raise Figure5Blocked("finite output/gradient gate failed")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _severity_label(factor: str, severity: float) -> str:
    if factor == "phase_jitter_rad":
        return f"phase jitter {severity:g} rad"
    if factor == "psf_blur":
        return f"PSF scale x{1.0 + severity:g}"
    return f"photon scale x{severity:g}"


def _factor_label(factor: str) -> str:
    return {
        "phase_jitter_rad": "Phase jitter",
        "psf_blur": "PSF blur",
        "photon_scale_mul": "Photon-scale reduction",
    }[factor]


def figure5_caption() -> str:
    return (
        "\\textbf{Figure 5 (prespecified qualitative examples).} "
        "The user-prespecified representative fields were fixed before inference: "
        "ER (ER068) under phase jitter; CCPs (CCP054) under PSF blur; and microtubules "
        "(MT055) under photon-scale reduction. Columns are "
        "APD-SIM-6, PhysMap-6, and DiffWS-6 in that fixed order; all methods use the same "
        "six-frame input in each condition. The horizontal profile row "
        "was selected from normalized ground truth alone (maximum row variance, first tie) "
        "and frozen before inference. Every image and profile uses the method-independent "
        "native normalized range [0,1], also frozen before inference. These three examples "
        "form a Figure-only qualitative cohort; Table 2 is computed from the complete fixed "
        "20-patch robustness grid. This qualitative figure does not claim that APD-SIM-6 "
        "outperforms PhysMap-6 in every condition.\n"
    )


def render_figure5(bundles: Mapping[str, Mapping[str, np.ndarray]],
                   receipt: Mapping[str, Any]) -> tuple[bytes, bytes, str]:
    validate_selection_receipt(receipt)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(12, 3, figsize=(10.5, 24), constrained_layout=True,
                                gridspec_kw={"height_ratios": [1, 1, 1, 0.65] * 3})
    colors = {"GT": "black", "APD-SIM-6": "#d62728", "PhysMap-6": "#1f77b4",
              "DiffWS-6": "#2ca02c"}
    for factor_index, sample in enumerate(receipt["samples"]):
        factor = sample["factor"]
        row_profile = int(sample["profile"]["row_index"])
        for severity_index, severity in enumerate(SEVERITIES[factor]):
            key = f"{factor}|{severity:.12g}|{sample['sample_id']}"
            bundle = bundles[key]
            image_row = factor_index * 4 + severity_index
            for method_index, method in enumerate(METHOD_ORDER):
                array = np.asarray(bundle[method], dtype=np.float32)
                if array.shape != (320, 320) or not bool(np.isfinite(array).all()):
                    raise Figure5Blocked("invalid rendered prediction")
                if float(array.min()) < 0.0 or float(array.max()) > 1.0:
                    raise Figure5Blocked("prediction outside native display range")
                axis = axes[image_row, method_index]
                axis.imshow(array, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                axis.axhline(row_profile, color="#ffcc00", linewidth=0.7, alpha=0.85)
                axis.set_xticks([]); axis.set_yticks([])
                if severity_index == 0:
                    axis.set_title(method, fontsize=11, weight="bold")
                if method_index == 0:
                    axis.set_ylabel(f"{_factor_label(factor)}\n{sample['sample_id']}\n{_severity_label(factor, severity)}",
                                    fontsize=8)
        profile_key = f"{factor}|{SEVERITIES[factor][-1]:.12g}|{sample['sample_id']}"
        profile_bundle = bundles[profile_key]
        gt = np.asarray(profile_bundle["gt"], dtype=np.float32)
        for method_index, method in enumerate(METHOD_ORDER):
            axis = axes[factor_index * 4 + 3, method_index]
            axis.plot(gt[row_profile], color=colors["GT"], linestyle="--", linewidth=1.0, label="GT")
            axis.plot(np.asarray(profile_bundle[method])[row_profile], color=colors[method],
                      linewidth=1.0, label=method)
            axis.set_xlim(0, 319); axis.set_ylim(0.0, 1.0); axis.grid(alpha=0.15)
            axis.set_xlabel("pixel"); axis.set_ylabel("native intensity")
            axis.legend(loc="upper right", fontsize=7, frameon=False)
    figure.suptitle("Prespecified three-GT strict DMD six-frame robustness", fontsize=14, weight="bold")
    png = io.BytesIO(); pdf = io.BytesIO()
    figure.savefig(png, format="png", dpi=220, metadata={"Software": "matplotlib"})
    fixed_pdf_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    figure.savefig(
        pdf, format="pdf",
        metadata={
            "Title": "Prespecified three-GT Figure 5",
            "Creator": "R1C3_FIG5_PRESPECIFIED_THREE_GT_V1",
            "CreationDate": fixed_pdf_date,
            "ModDate": fixed_pdf_date,
        },
    )
    plt.close(figure)
    caption = figure5_caption()
    return png.getvalue(), pdf.getvalue(), caption


def _tree_snapshot(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise Figure5Blocked(f"protected run unavailable: {path}")
    return {item.relative_to(path).as_posix(): sha_file(item)
            for item in sorted(path.rglob("*")) if item.is_file()}


def _snapshot_hash(snapshot: Mapping[str, str]) -> str:
    return _json_hash(dict(snapshot))


def _safe_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    outputs = (ROOT / "outputs").resolve()
    if resolved != outputs and outputs not in resolved.parents:
        raise Figure5Blocked("output directory must be under project outputs")
    protected = PROTECTED_RUN.resolve()
    if resolved == protected or protected in resolved.parents:
        raise Figure5Blocked("new Figure output cannot be inside protected formal run")
    resolved.mkdir(parents=True, exist_ok=True)
    owned = (
        "FIG5_SELECTION_RECEIPT.json",
        "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_DATA.csv",
        "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.png",
        "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.pdf",
        "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_CAPTION.tex",
        "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_AUDIT.json",
        "FIG5_STATUS.json",
        "condition_arrays",
    )
    conflicts = [name for name in owned if (resolved / name).exists()]
    if conflicts:
        raise Figure5Blocked(f"refusing to overwrite Figure-owned targets: {conflicts}")
    return resolved


def generate_prespecified_figure5(output_dir: str | Path, *, device: str = "cuda:0",
                                  search_roots: Sequence[Path] | None = None,
                                  verify_replay: bool = True) -> dict[str, Any]:
    """Generate all new Figure-only arrays/reports without touching the old run."""
    import torch
    from unisim.sim_forward_2d import forward_protocol_sim_2d, nominal_theta_2d
    from . import physmap6_experiment as experiment
    from .physmap6_pipeline import load_stage1, make_sim_config, run_four_methods

    if not __debug__:
        raise Figure5Blocked("python -O forbidden")
    if verify_replay is not True:
        raise Figure5Blocked("formal publication requires exact same-raw replay")
    run_dir = _safe_output_dir(Path(output_dir))
    protected_before = _tree_snapshot(PROTECTED_RUN)
    source_hashes = {
        str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()),
        str((ROOT / "unisim/revision_r1/physmap6_pipeline.py").resolve()):
            sha_file(ROOT / "unisim/revision_r1/physmap6_pipeline.py"),
        str((ROOT / "unisim/revision_r1/physmap6_core.py").resolve()):
            sha_file(ROOT / "unisim/revision_r1/physmap6_core.py"),
        str((ROOT / "unisim/revision_r1/physmap6_experiment.py").resolve()):
            sha_file(ROOT / "unisim/revision_r1/physmap6_experiment.py"),
    }
    receipt, patches = build_selection_receipt(search_roots=search_roots)
    if sha_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise Figure5Blocked("checkpoint SHA mismatch")
    receipt["protected_formal_run"] = {
        "path": str(PROTECTED_RUN), "file_count": len(protected_before),
        "tree_manifest_sha256_before": _snapshot_hash(protected_before),
    }
    receipt["implementation_sha256_before"] = source_hashes
    receipt["selection_payload_sha256"] = _json_hash(
        {key: value for key, value in receipt.items() if key != "selection_payload_sha256"}
    )
    selection_path = run_dir / "FIG5_SELECTION_RECEIPT.json"
    _write_json_new(selection_path, receipt)
    selection_file_hash = sha_file(selection_path)
    gpu_before = experiment.assert_no_external_cuda_compute()
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise Figure5Blocked("formal Figure generation requires CUDA")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol = experiment.protocol_receipt()
    geometry = experiment._geometry_receipt(protocol)
    geometry_hash = _json_hash(geometry)
    mask_hash = sha_array(np.asarray(experiment.VALIDITY_MASK, dtype=np.float32))
    model, scheduler, metadata = load_stage1(config, CHECKPOINT, EXPECTED_CHECKPOINT_SHA256,
                                             torch.device(device))
    if experiment.REFINEMENT_CONFIG.receipt() != receipt["refinement_config"]:
        raise Figure5Blocked("runtime refinement config differs from frozen receipt")
    sim_config = make_sim_config(config)
    nominal = nominal_theta_2d(sim_config, torch.device(device))
    metrics = experiment._metrics_module()
    rows: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, np.ndarray]] = {}
    condition_dir = run_dir / "condition_arrays"
    condition_dir.mkdir(exist_ok=False)
    plan = build_condition_plan(receipt)
    for condition in plan:
        factor = condition["factor"]; severity = float(condition["severity"])
        gt = patches[factor]
        gt_tensor = torch.from_numpy(gt)[None, None].to(torch.device(device))
        theta_true = experiment.perturb_theta(
            nominal, factor, severity, sample_order=int(condition["sample_order"]),
            device=torch.device(device),
        )
        theta_inverse = experiment.inverse_theta_for_robustness(theta_true, nominal)
        noise_seed = int(condition["noise_seed"])
        generator = torch.Generator(device=torch.device(device)).manual_seed(noise_seed)
        raw, _ = forward_protocol_sim_2d(
            gt_tensor, sim_config, PROTOCOL_ID, theta=theta_true, randomize=False,
            noise_generator=generator,
        )
        raw_pointer = int(raw.data_ptr())
        diffusion_seed = int(condition["diffusion_seed"])
        result = run_four_methods(
            raw, model, scheduler, sim_config, theta_inverse,
            diffusion_seed=diffusion_seed, refinement_config=experiment.REFINEMENT_CONFIG,
            geometry_receipt=geometry,
        )
        replay = run_four_methods(
            raw, model, scheduler, sim_config, theta_inverse,
            diffusion_seed=diffusion_seed, refinement_config=experiment.REFINEMENT_CONFIG,
            geometry_receipt=geometry,
        ) if verify_replay else result
        for payload in (result, replay):
            for refined_method in ("APD-SIM-6", "PhysMap-6"):
                refinement = payload[refined_method]["refinement"]
                if (len(refinement.objective_history) != 41
                        or len(refinement.observed_nrmse_history) != 41):
                    raise Figure5Blocked("40-update refinement history/config gate failed")
                validate_runtime_refinement_receipt(refinement.configuration_receipt)
        if int(raw.data_ptr()) != raw_pointer or result["raw_stack_sha256"] != replay["raw_stack_sha256"]:
            raise Figure5Blocked("raw identity changed during shared/replay inference")
        theta_true_json = experiment._tensor_theta_json(theta_true)
        theta_inverse_json = experiment._tensor_theta_json(theta_inverse)
        forward_hash = _json_hash({"true": theta_true_json, "inverse": theta_inverse_json})
        noise_hash = _json_hash({"noise_seed": noise_seed, "generator_device": device,
                                 "randomize": False, "raw_generated_once": True})
        bundle: dict[str, np.ndarray] = {"gt": gt}
        method_data: dict[str, dict[str, Any]] = {}
        for method in METHOD_ORDER:
            image, _runtime, _peak, objective, nrmse, grad_finite, output_finite = experiment._method_payload(method, result)
            replay_image = replay[method]["image"]
            prediction = np.ascontiguousarray(image[0, 0].detach().cpu().numpy(), dtype=np.float32)
            replay_prediction = np.ascontiguousarray(replay_image[0, 0].detach().cpu().numpy(), dtype=np.float32)
            if prediction.shape != (320, 320) or not bool(np.isfinite(prediction).all()):
                raise Figure5Blocked("non-finite/shape prediction")
            if float(prediction.min()) < 0.0 or float(prediction.max()) > 1.0:
                raise Figure5Blocked("prediction outside native [0,1]")
            pred_hash = sha_array(prediction); replay_hash = sha_array(replay_prediction)
            delta = prediction.astype(np.float64) - replay_prediction.astype(np.float64)
            replay_max_abs = float(np.max(np.abs(delta)))
            replay_rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            replay_bitwise = pred_hash == replay_hash
            replay_numeric = (
                replay_max_abs <= REPLAY_MAX_ABS_THRESHOLD
                and replay_rmse <= REPLAY_RMSE_THRESHOLD
            )
            if not replay_numeric:
                raise Figure5Blocked(
                    f"replay exceeds numeric gate: {condition['condition_id']}/{method}: "
                    f"maxabs={replay_max_abs}, rmse={replay_rmse}"
                )
            psnr = float(metrics.psnr_native(gt, prediction)); ssim = float(metrics.ssim_native(gt, prediction))
            if not math.isfinite(psnr) or not math.isfinite(ssim):
                raise Figure5Blocked("non-finite metric")
            bundle[method] = prediction
            method_data[method] = {
                "prediction_sha256": pred_hash, "replay_prediction_sha256": replay_hash,
                "replay_bitwise_match": replay_bitwise,
                "replay_numeric_equivalent": replay_numeric,
                "replay_max_abs_error": replay_max_abs,
                "replay_rmse": replay_rmse,
                "psnr": psnr, "ssim": ssim,
                "gradient_finite": bool(grad_finite), "output_finite": bool(output_finite),
                "observed_nrmse": nrmse,
                "poisson_gaussian_objective": objective,
            }
        bundles[condition["condition_id"]] = bundle
        npz_stream = io.BytesIO()
        np.savez_compressed(npz_stream, gt=gt, raw=raw[0].detach().cpu().numpy().astype(np.float32),
                            APD_SIM_6=bundle["APD-SIM-6"], PhysMap_6=bundle["PhysMap-6"],
                            DiffWS_6=bundle["DiffWS-6"])
        slug = f"{condition['factor_order']}_{factor}_{condition['severity_order']}_{severity:.12g}_{condition['sample_order']:03d}"
        npz_path = condition_dir / f"{slug}.npz"
        _atomic_new(npz_path, npz_stream.getvalue())
        npz_sha = sha_file(npz_path)
        for method_index, method in enumerate(METHOD_ORDER):
            data = method_data[method]
            rows.append({
                "factor_order": condition["factor_order"], "factor": factor,
                "severity_order": condition["severity_order"], "severity": severity,
                "condition_id": condition["condition_id"], "sample_order": condition["sample_order"],
                "sample_id": condition["sample_id"], "parent_id": condition["parent_id"],
                "structure": condition["structure"], "method_order": method_index, "method": method,
                "source_file_sha256": condition["source_file_sha256"],
                "gt_patch_sha256": condition["gt_patch_sha256"],
                "raw_stack_sha256": result["raw_stack_sha256"], "validity_mask_sha256": mask_hash,
                "geometry_sha256": geometry_hash, "forward_parameters_sha256": forward_hash,
                "noise_identity_sha256": noise_hash, "normalization_sha256": NORMALIZATION_HASH,
                "perturbation_direction_seed": condition["perturbation_direction_seed"],
                "noise_seed": noise_seed, "diffusion_seed": diffusion_seed,
                "refinement_config_sha256": experiment.REFINEMENT_CONFIG.receipt()["config_sha256"]
                    if method in {"APD-SIM-6", "PhysMap-6"} else "NA",
                "profile_row": condition["profile_row"], "display_min": 0.0, "display_max": 1.0,
                "method_independent_before_inference": True, **data,
                "condition_npz": npz_path.relative_to(run_dir).as_posix(),
                "condition_npz_sha256": npz_sha, "status": "PASS",
                "selection_payload_sha256": receipt["selection_payload_sha256"],
                "selection_receipt_file_sha256": selection_file_hash,
            })
    validate_condition_rows(rows, receipt)
    data_path = run_dir / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_DATA.csv"
    _atomic_new(data_path, _csv_bytes(rows))
    png, pdf, caption = render_figure5(bundles, receipt)
    png_path = run_dir / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.png"
    pdf_path = run_dir / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.pdf"
    caption_path = run_dir / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_CAPTION.tex"
    _atomic_new(png_path, png); _atomic_new(pdf_path, pdf); _atomic_new(caption_path, caption.encode("utf-8"))
    protected_after = _tree_snapshot(PROTECTED_RUN)
    source_after = {path: sha_file(Path(path)) for path in source_hashes}
    if protected_after != protected_before:
        raise Figure5Blocked("protected formal run changed")
    if source_after != source_hashes:
        raise Figure5Blocked("implementation changed during generation")
    gpu_after = experiment.assert_no_external_cuda_compute()
    audit = independent_audit(run_dir)
    audit.update({
        "gpu_gate_before": gpu_before, "gpu_gate_after": gpu_after,
        "checkpoint_metadata_global_step": int(metadata["global_step"]),
        "protected_formal_run_tree_manifest_sha256_before": _snapshot_hash(protected_before),
        "protected_formal_run_tree_manifest_sha256_after": _snapshot_hash(protected_after),
        "protected_formal_run_unchanged": True,
    })
    audit_path = run_dir / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_AUDIT.json"
    _write_json_new(audit_path, audit)
    bitwise_count = sum(
        str(row["replay_bitwise_match"]).lower() == "true" for row in rows
    )
    numeric_count = sum(
        str(row["replay_numeric_equivalent"]).lower() == "true" for row in rows
    )
    status = {
        "schema_version": 1, "status": "R1C3_FIG5_PRESPECIFIED_READY",
        "scope": "component_only", "overall_ready": False,
        "run_dir": str(run_dir), "condition_count": 9, "method_row_count": 27,
        "raw_generated_condition_count": 9,
        "replay_numeric_equivalent_for_all_methods": numeric_count == 27,
        "replay_numeric_equivalent_count": numeric_count,
        "replay_bitwise_match_count": bitwise_count,
        "replay_total_method_count": 27,
        "selection_receipt": str(selection_path), "selection_receipt_sha256": selection_file_hash,
        "data_csv": str(data_path), "data_csv_sha256": sha_file(data_path),
        "figure_png_sha256": sha_file(png_path), "figure_pdf_sha256": sha_file(pdf_path),
        "audit_sha256": sha_file(audit_path),
    }
    _write_json_new(run_dir / "FIG5_STATUS.json", status)
    return status


def independent_audit(run_dir: str | Path) -> dict[str, Any]:
    """Independently recompute selection/CSV/NPZ bindings using only CPU."""
    path = Path(run_dir).resolve()
    receipt_path = path / "FIG5_SELECTION_RECEIPT.json"
    data_path = path / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_DATA.csv"
    if not receipt_path.is_file() or not data_path.is_file():
        raise Figure5Blocked("selection/data artifact missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_selection_receipt(receipt)
    source_audit = audit_selection_sources(receipt)
    protected = receipt.get("protected_formal_run")
    if not isinstance(protected, Mapping):
        raise Figure5Blocked("protected formal-run receipt missing")
    protected_now = _snapshot_hash(_tree_snapshot(PROTECTED_RUN))
    if protected_now != protected.get("tree_manifest_sha256_before"):
        raise Figure5Blocked("protected formal run changed after selection")
    implementation = receipt.get("implementation_sha256_before")
    if (not isinstance(implementation, Mapping)
            or any(not Path(source).is_file() or sha_file(Path(source)) != digest
                   for source, digest in implementation.items())):
        raise Figure5Blocked("implementation source changed after selection")
    rows = _read_rows(data_path)
    validate_condition_rows(rows, receipt)
    replay_bitwise_count = sum(
        row["replay_bitwise_match"].lower() == "true" for row in rows
    )
    replay_numeric_count = sum(
        row["replay_numeric_equivalent"].lower() == "true" for row in rows
    )
    if replay_numeric_count != 27:
        raise Figure5Blocked("not all replay rows satisfy numeric equivalence")
    receipt_file_sha = sha_file(receipt_path)
    if any(row["selection_receipt_file_sha256"] != receipt_file_sha for row in rows):
        raise Figure5Blocked("selection receipt file SHA binding mismatch")
    seen_npz: set[str] = set()
    bundles: dict[str, dict[str, np.ndarray]] = {}
    for group_start in range(0, len(rows), 3):
        group = rows[group_start:group_start + 3]
        relative = str(group[0]["condition_npz"])
        candidate = (path / relative).resolve()
        if path not in candidate.parents or candidate.suffix != ".npz":
            raise Figure5Blocked("unsafe condition NPZ path")
        if relative in seen_npz or not candidate.is_file():
            raise Figure5Blocked("duplicate/missing condition NPZ")
        seen_npz.add(relative)
        if sha_file(candidate) != group[0]["condition_npz_sha256"]:
            raise Figure5Blocked("condition NPZ SHA mismatch")
        with np.load(candidate, allow_pickle=False) as archive:
            expected_keys = {"gt", "raw", "APD_SIM_6", "PhysMap_6", "DiffWS_6"}
            if set(archive.files) != expected_keys:
                raise Figure5Blocked("condition NPZ key drift")
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        gt = arrays["gt"]
        raw = arrays["raw"]
        if (gt.dtype != np.float32 or gt.shape != (320, 320)
                or not bool(np.isfinite(gt).all()) or float(gt.min()) < 0.0 or float(gt.max()) > 1.0):
            raise Figure5Blocked("GT array invalid")
        if (raw.dtype != np.float32 or raw.shape != (6, 320, 320)
                or not bool(np.isfinite(raw).all())):
            raise Figure5Blocked("raw array invalid")
        if sha_array(gt) != group[0]["gt_patch_sha256"]:
            raise Figure5Blocked("GT patch hash mismatch")
        if sha_array(raw) != group[0]["raw_stack_sha256"]:
            raise Figure5Blocked("raw stack hash mismatch")
        bundle: dict[str, np.ndarray] = {"gt": gt}
        for row, key in zip(group, ("APD_SIM_6", "PhysMap_6", "DiffWS_6")):
            array = np.asarray(arrays[key])
            if array.dtype != np.float32 or array.shape != (320, 320) or not bool(np.isfinite(array).all()):
                raise Figure5Blocked("prediction array invalid")
            if float(array.min()) < 0.0 or float(array.max()) > 1.0:
                raise Figure5Blocked("prediction outside frozen native range")
            if sha_array(array) != row["prediction_sha256"]:
                raise Figure5Blocked("prediction hash mismatch")
            bundle[str(row["method"])] = array
        bundles[str(group[0]["condition_id"])] = bundle
    actual_npz = {item.relative_to(path).as_posix() for item in (path / "condition_arrays").glob("*.npz")}
    if actual_npz != seen_npz or len(seen_npz) != 9:
        raise Figure5Blocked("condition NPZ file set mismatch")
    expected_png, expected_pdf, expected_caption = render_figure5(bundles, receipt)
    figure_paths = {
        "png": path / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.png",
        "pdf": path / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT.pdf",
        "caption": path / "FIG5_PHYSMAP6_STRICT_PRESPECIFIED_GT_CAPTION.tex",
    }
    expected_bytes = {
        "png": expected_png, "pdf": expected_pdf,
        "caption": expected_caption.encode("utf-8"),
    }
    for key, artifact in figure_paths.items():
        if not artifact.is_file() or artifact.read_bytes() != expected_bytes[key]:
            raise Figure5Blocked(f"independent Figure {key} semantic reproduction mismatch")
    return {
        "schema_version": 1, "status": "PASS", "condition_count": 9,
        "method_row_count": 27, "method_order": list(METHOD_ORDER),
        "factor_order": list(FACTOR_ORDER), "selection_receipt_sha256": sha_file(receipt_path),
        "data_csv_sha256": sha_file(data_path), "condition_npz_count": 9,
        "figure_png_sha256": sha_file(figure_paths["png"]),
        "figure_pdf_sha256": sha_file(figure_paths["pdf"]),
        "caption_sha256": sha_file(figure_paths["caption"]),
        "selection_source_audit": source_audit,
        "protected_formal_run_tree_manifest_sha256": protected_now,
        "figure_semantically_reproduced_from_npz": True,
        "replay_equivalence_policy": receipt["replay_equivalence"],
        "replay_numeric_equivalent_count": replay_numeric_count,
        "replay_bitwise_match_count": replay_bitwise_count,
        "replay_total_method_count": 27,
        "shared_identity_fields": list(SHARED_FIELDS), "primary_physmap9_values_used": False,
    }


def self_test() -> dict[str, Any]:
    synthetic = np.zeros((32, 40), dtype=np.float32)
    synthetic[7, 1::2] = 1.0
    profile = select_gt_profile(synthetic)
    checks = {
        "profile_row_known": profile["row_index"] == 7,
        "display_fixed": display_range_for_condition(synthetic) == (0.0, 1.0),
        "factor_order": FACTOR_ORDER == ("phase_jitter_rad", "psf_blur", "photon_scale_mul"),
        "method_order": METHOD_ORDER == ("APD-SIM-6", "PhysMap-6", "DiffWS-6"),
        "condition_count": sum(len(SEVERITIES[factor]) for factor in FACTOR_ORDER) == 9,
        "row_count": 9 * len(METHOD_ORDER) == 27,
        "replay_float32_gate": (
            REPLAY_MAX_ABS_THRESHOLD == 32.0 * FLOAT32_EPS
            and REPLAY_RMSE_THRESHOLD == 4.0 * FLOAT32_EPS
        ),
    }
    if not all(checks.values()):
        raise Figure5Blocked(f"self-test failed: {checks}")
    return {"status": "PASS", "checks": checks}


def _allocate_cli_dir() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = OUTPUT_BASE / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
        if not candidate.exists():
            return candidate
    raise Figure5Blocked("cannot allocate output directory")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    if args.audit is not None:
        print(json.dumps(independent_audit(args.audit), indent=2)); return 0
    output = args.output_dir or _allocate_cli_dir()
    print(json.dumps(generate_prespecified_figure5(output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FACTOR_ORDER", "METHOD_ORDER", "SEVERITIES", "TARGETS", "Figure5Blocked",
    "Figure5Target", "audit_selection_sources", "build_condition_plan", "build_selection_receipt",
    "display_range_for_condition", "generate_prespecified_figure5", "independent_audit",
    "figure5_caption", "render_figure5", "select_gt_profile", "self_test", "validate_condition_rows",
    "validate_runtime_refinement_receipt", "validate_selection_receipt",
]

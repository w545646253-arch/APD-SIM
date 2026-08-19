"""One-run sealed APD-SIM-3/6/9 evaluation for the DMD9 repair.

This entry binds the frozen R2 DMD9 validation-best EMA checkpoint to the
diagnosed full-size tiled Stage-1 path.  A persistent run UUID permits resuming
an interrupted invocation without creating a second formal test run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "DMD9_REPAIR_R4_20260817_104841"
OLD_FORMAL_ROOT = ROOT / "outputs" / "APD369_FINAL_REEVALUATION_20260817_090027"
ACTIVE_DMD9 = ROOT / "checkpoints" / "apd_dmd_geometry_r4" / "dmd9"
MANIFEST_SHA256 = "3df5625841af40893a23bc9f26cd43764128d8ae310128a4b7fccdd7f64859c9"
SELECTION_RULE = "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _begin_or_resume_run(output_root: Path) -> dict[str, Any]:
    final_dir = output_root / "08_final_test"
    receipt = final_dir / "formal_test_receipt.json"
    lock = final_dir / "formal_test_lock.json"
    if receipt.exists():
        raise RuntimeError("formal sealed test already completed; a second run is forbidden")
    if lock.exists():
        payload = _read_json(lock)
        if payload.get("formal_test_run_count") != 1 or payload.get("status") != "IN_PROGRESS":
            raise RuntimeError("invalid formal-test lock")
        return payload
    gate = _read_json(output_root / "07_validation_gate" / "validation_gate.json")
    if gate.get("status") != "DMD9_VALIDATION_GATE_PASS" or gate.get("test_access_count") != 0:
        raise RuntimeError("sealed test remains locked: validation gate did not pass cleanly")
    selection = _read_json(ACTIVE_DMD9 / "checkpoint_selection.json")
    if selection.get("status") != "DMD9_EXISTING_CHECKPOINT_FROZEN":
        raise RuntimeError("active DMD9 checkpoint was not independently frozen")
    payload = {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "run_uuid": str(uuid.uuid4()),
        "formal_test_run_count": 1,
        "sealed_test_premature_access_count": 0,
        "validation_gate_status": gate["status"],
        "selected_dmd9_checkpoint_sha256": selection["selected_checkpoint_sha256"],
    }
    atomic_json(lock, payload)
    return payload


def _prepare_frozen_contract(output_root: Path) -> dict[str, Any]:
    manifest_source = OLD_FORMAL_ROOT / "01_test_manifest" / "test30_manifest.json"
    if sha256_file(manifest_source) != MANIFEST_SHA256:
        raise RuntimeError("frozen 30-FOV manifest identity drift")
    manifest_dir = output_root / "01_test_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_target = manifest_dir / "test30_manifest.json"
    if manifest_target.exists() and sha256_file(manifest_target) != MANIFEST_SHA256:
        raise RuntimeError("local sealed manifest copy differs")
    if not manifest_target.exists():
        shutil.copy2(manifest_source, manifest_target)
    for name in ("test30_manifest.tsv", "test30_manifest_sha256.txt"):
        source = OLD_FORMAL_ROOT / "01_test_manifest" / name
        target = manifest_dir / name
        if not target.exists():
            shutil.copy2(source, target)

    old_contract = _read_json(
        OLD_FORMAL_ROOT / "00_checkpoint_contract" / "FINAL_APD369_CHECKPOINTS.json"
    )
    selection = _read_json(ACTIVE_DMD9 / "checkpoint_selection.json")
    dmd9 = dict(old_contract["checkpoints"]["APD-SIM-9"])
    dmd9.update({
        "checkpoint_path": str((ACTIVE_DMD9 / "best.pt").resolve()),
        "checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "config_path": str((ROOT / "configs" / "apd_dmd_r2" / "train9_formal.json").resolve()),
        "selected_validation_iteration": selection["selected_iteration"],
        "selection_rule": SELECTION_RULE,
        "selection_metric": "mean_val_total_loss",
        "selection_metric_value": "frozen R2 validation-best receipt",
        "receipt_path": str((ACTIVE_DMD9 / "completion_receipt.json").resolve()),
        "receipt_sha256": sha256_file(ACTIVE_DMD9 / "completion_receipt.json"),
        "history_path": str((ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd9" / "validation_history.csv").resolve()),
        "inference_mode": selection["inference_mode"],
        "test_access_count": 0,
        "status": "PASS",
    })
    contract = dict(old_contract)
    contract["checkpoints"] = dict(old_contract["checkpoints"])
    contract["checkpoints"]["APD-SIM-9"] = dmd9
    contract["dmd9_training_completion"] = {
        "status": "NOT_RUN_NOT_REQUIRED",
        "training_executed": False,
        "repair_type": "inference_only",
    }
    contract["test_used_for_checkpoint_selection"] = False
    contract["training_executed"] = False
    target = output_root / "00_checkpoint_contract" / "FINAL_APD369_CHECKPOINTS.json"
    atomic_json(target, contract)
    return contract


def _copy_final_artifacts(output_root: Path, run_lock: dict[str, Any], numerical: dict[str, Any]) -> dict[str, Any]:
    source = output_root / "04_metrics"
    target = output_root / "08_final_test"
    renames = {
        "per_fov_metrics.csv": "per_fov_metrics.csv",
        "summary_mean_sd.csv": "summary_mean_sd.csv",
        "summary_median_iqr.csv": "summary_median_iqr.csv",
        "summary_by_structure.csv": "summary_by_structure.csv",
        "paired_protocol_differences.csv": "paired_differences.csv",
        "frc_censoring_summary.csv": "frc_censoring.csv",
        "Table_APD369_final.tex": "Table_APD369_R4.tex",
        "Table_APD369_final.csv": "Table_APD369_R4.csv",
    }
    for source_name, target_name in renames.items():
        shutil.copy2(source / source_name, target / target_name)
    with (target / "summary_mean_sd.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    means = {
        (row["method"], row["metric"]): float(row["mean"])
        for row in summary_rows
    }
    psnr_9_minus_6 = means[("APD-SIM-9", "psnr")] - means[("APD-SIM-6", "psnr")]
    ssim_9_minus_6 = means[("APD-SIM-9", "ssim")] - means[("APD-SIM-6", "ssim")]
    relation_holds = (
        psnr_9_minus_6 >= -1.0
        and ssim_9_minus_6 >= -0.020
        and means[("APD-SIM-9", "psnr")] > means[("APD-SIM-3", "psnr")]
        and means[("APD-SIM-9", "ssim")] > means[("APD-SIM-3", "ssim")]
    )
    final_status = (
        "DMD9_REPAIRED_APD369_READY"
        if relation_holds
        else "DMD9_VALIDATION_READY_TEST_NONMONOTONIC"
    )
    per_fov = target / "per_fov_metrics.csv"
    with per_fov.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise RuntimeError("formal 3x30 metric grid is incomplete")
    if len({row["generation_call_uuid"] for row in rows}) != 90:
        raise RuntimeError("protocol raw generation calls were not independent")
    receipt = {
        "schema_version": 1,
        "status": final_status,
        "run_uuid": run_lock["run_uuid"],
        "formal_test_run_count": 1,
        "sealed_manifest_sha256": MANIFEST_SHA256,
        "sealed_test_premature_access_count": 0,
        "validation_gate_status": "DMD9_VALIDATION_GATE_PASS",
        "checkpoint_selection_used_test": False,
        "training_execution_count": 0,
        "protocol_forward_call_count": 90,
        "independent_protocol_raw_generation_count": 90,
        "common_nine_frame_subsampling_count": 0,
        "best_of_n_count": 0,
        "principal_diffusion_seed_policy": "20260812 + FOV index",
        "inference_weight_branch": "ema",
        "dmd9_inference_mode": "tiled_320_core_160_single_spatial_noise_field",
        "means": {
            method: {
                metric: means[(method, metric)]
                for metric in ("psnr", "ssim", "frc_auc")
            }
            for method in ("APD-SIM-3", "APD-SIM-6", "APD-SIM-9")
        },
        "paired_mean_9_minus_6": {
            "psnr": psnr_9_minus_6,
            "ssim": ssim_9_minus_6,
        },
        "practical_relation_holds": relation_holds,
        "numerical_evaluation_receipt": numerical,
        "per_fov_metrics_sha256": sha256_file(per_fov),
    }
    atomic_json(target / "formal_test_receipt.json", receipt)
    completed_lock = dict(run_lock)
    completed_lock["status"] = "COMPLETE"
    completed_lock["final_status"] = final_status
    atomic_json(target / "formal_test_lock.json", completed_lock)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    run_lock = _begin_or_resume_run(output_root)
    _prepare_frozen_contract(output_root)

    import evaluate_apd369_protocols_final as evaluator
    from unisim.revision_r1.physmap6_pipeline import stage1_reconstruct_registered_tiled

    monolithic = evaluator.stage1_reconstruct_registered

    def stage1_dispatch(raw_frames: Any, model: Any, scheduler: Any, *, protocol_id: str, seed: int):
        if protocol_id == "DMD_9F_3O3P":
            return stage1_reconstruct_registered_tiled(
                raw_frames, model, scheduler, protocol_id=protocol_id, seed=seed,
                tile_size=320, core_size=160, tile_batch_size=4,
            )
        return monolithic(raw_frames, model, scheduler, protocol_id=protocol_id, seed=seed)

    evaluator.stage1_reconstruct_registered = stage1_dispatch
    numerical = evaluator.run(output_root)
    receipt = _copy_final_artifacts(output_root, run_lock, numerical)
    print(json.dumps({"status": receipt["status"], "formal_test_run_count": 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

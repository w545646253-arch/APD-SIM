"""Five prespecified APD-SIM-6 DDIM-seed trajectories on fixed DMD-6F raw data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from revision_dmd6_common import (  # noqa: E402
    BASELINE_ROOT, PIXEL_SIZE_UM, PROTOCOL_HASH, PROTOCOL_ID, REVISION_ROOT,
    array_sha256, atomic_json, gt_frc, metrics_module, normalize_gt, write_csv,
)
from unisim.revision_r1.physmap6_core import RefinementConfig, masked_refine  # noqa: E402
from unisim.revision_r1.physmap6_pipeline import load_stage1_registered, make_sim_config, stage1_reconstruct_registered  # noqa: E402
from unisim.sim_forward_2d import embed_raw_to_slots_2d, nominal_theta_2d  # noqa: E402


OUTPUT = REVISION_ROOT / "03_seed_sensitivity"
MANIFEST = BASELINE_ROOT / "01_shared_contract/test30_dmd6_manifest.tsv"
CESHIJI = Path(r"data/sealed_test_gt")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/apd_dmd_r2/train6_formal.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "checkpoints/apd_dmd_geometry_r2/dmd6/best_checkpoint_receipt.json").read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    model, scheduler, _ = load_stage1_registered(config, Path(receipt["checkpoint_path"]), receipt["checkpoint_sha256"], device, protocol_id=PROTOCOL_ID)
    sim_config = make_sim_config(config); theta = nominal_theta_2d(sim_config, device)
    metric_api = metrics_module()
    manifest = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    matched_metrics_path = REVISION_ROOT / "01_matched_baselines/per_fov_metrics.csv"
    matched_rows = list(csv.DictReader(matched_metrics_path.open("r", encoding="utf-8", newline="")))
    matched_principal = {
        int(row["fov_index"]): row
        for row in matched_rows
        if row["method"] == "APD-SIM-6"
    }
    if len(matched_principal) != 30:
        raise RuntimeError("the frozen principal APD-SIM-6 trajectory is incomplete")
    rows: list[dict[str, object]] = []
    geometry = {"protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH, "raw_frame_order": ["H0", "H120", "H240", "V0", "V120", "V240"]}
    for item in manifest:
        order = int(item["order"]); sample_id = item["sample_id"]
        gt_path = next(path for path in (CESHIJI / f"{sample_id}.tif", CESHIJI / f"{sample_id}.tiff") if path.is_file())
        gt = normalize_gt(tifffile.imread(gt_path))
        bundle_path = MANIFEST.parent / item["npz_path"]
        with np.load(bundle_path, allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_stack"], dtype=np.float32)
        if array_sha256(raw) != item["raw_stack_sha256"]:
            raise RuntimeError(f"fixed raw hash mismatch: {sample_id}")
        raw_tensor = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
        _, mask = embed_raw_to_slots_2d(raw_tensor, PROTOCOL_ID)
        for repeat in range(5):
            seed = 20260812 + 1000 * repeat + order
            if repeat == 0:
                principal_row = matched_principal[order]
                if (
                    int(principal_row["principal_diffusion_seed"]) != seed
                    or principal_row["raw_stack_sha256"] != item["raw_stack_sha256"]
                    or principal_row["protocol_hash"] != PROTOCOL_HASH
                ):
                    raise RuntimeError(f"principal trajectory binding mismatch: {sample_id}")
                prediction = np.ascontiguousarray(
                    np.load(principal_row["native_path"], allow_pickle=False), dtype=np.float32
                )
                diffws_source = (
                    REVISION_ROOT / "01_matched_baselines/internal_component_outputs/DiffWS-6"
                    / f"{order:03d}_{sample_id}.npy"
                )
                diffws = np.ascontiguousarray(np.load(diffws_source, allow_pickle=False), dtype=np.float32)
                principal_dir = OUTPUT / "principal_outputs"
                principal_dir.mkdir(parents=True, exist_ok=True)
                np.save(principal_dir / f"{order:03d}_{sample_id}_APD-SIM-6.npy", prediction, allow_pickle=False)
                np.save(principal_dir / f"{order:03d}_{sample_id}_DiffWS-6.npy", diffws, allow_pickle=False)
            else:
                x_ws, _, _ = stage1_reconstruct_registered(raw_tensor, model, scheduler, protocol_id=PROTOCOL_ID, seed=seed)
                result = masked_refine(x_ws, raw_tensor, mask[0, :, 0, 0], geometry, {"sim_config": sim_config, "theta": theta}, RefinementConfig())
                prediction = np.ascontiguousarray(result.final_reconstruction[0, 0].detach().cpu().numpy(), dtype=np.float32)
            frc, _ = gt_frc(gt, prediction)
            period = frc["cutoff_derived_spatial_period_um"]
            if period is None and frc["right_censored_at_nyquist"]:
                period = 2.0 * PIXEL_SIZE_UM
            rows.append({
                "fov_index": order, "sample_id": sample_id, "parent_id": item["parent_id"],
                "structure_class": item["structure_class"], "repeat_index": repeat, "seed": seed,
                "principal_seed": repeat == 0, "raw_stack_sha256": item["raw_stack_sha256"],
                "checkpoint_sha256": receipt["checkpoint_sha256"], "psnr": metric_api.psnr_native(gt, prediction),
                "ssim": metric_api.ssim_native(gt, prediction), "frc_period_um_for_sensitivity": period,
                "frc_right_censored": frc["right_censored_at_nyquist"], "output_sha256": array_sha256(prediction),
                "finite": bool(np.isfinite(prediction).all()),
            })
            print(f"seed sensitivity FOV {order + 1}/30 repeat {repeat + 1}/5", flush=True)
    write_csv(OUTPUT / "per_fov_seed_metrics.csv", rows)
    per_fov_summary: list[dict[str, object]] = []
    for order in range(30):
        selected = [row for row in rows if int(row["fov_index"]) == order]
        result: dict[str, object] = {"fov_index": order, "sample_id": selected[0]["sample_id"], "principal_repeat": 0}
        for metric in ("psnr", "ssim", "frc_period_um_for_sensitivity"):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            result[f"{metric}_sd"] = float(values.std(ddof=1)); result[f"{metric}_range"] = float(values.max() - values.min())
            rank = int(np.argsort(np.argsort(values))[0]) + 1
            result[f"{metric}_principal_rank_of_5"] = rank
            result[f"{metric}_principal_within_five_seed_range"] = bool(values.min() <= values[0] <= values.max())
        per_fov_summary.append(result)
    aggregate: list[dict[str, object]] = []
    for metric in ("psnr", "ssim", "frc_period_um_for_sensitivity"):
        sds = np.asarray([float(row[f"{metric}_sd"]) for row in per_fov_summary])
        ranges = np.asarray([float(row[f"{metric}_range"]) for row in per_fov_summary])
        aggregate.append({"metric": metric, "n_fov": 30, "median_within_fov_sd": float(np.median(sds)), "p95_within_fov_sd": float(np.quantile(sds, 0.95)), "maximum_range": float(ranges.max()), "principal_seed_all_within_observed_range": True})
    write_csv(OUTPUT / "summary.csv", aggregate)
    write_csv(OUTPUT / "per_fov_summary.csv", per_fov_summary)
    atomic_json(OUTPUT / "seed_policy_receipt.json", {
        "status": "APD6_FIVE_SEED_SENSITIVITY_COMPLETE", "principal_seed_policy": "20260812 + FOV index",
        "sensitivity_seed_policy": "20260812 + 1000*repeat + FOV index; repeat=0..4",
        "raw_stack_fixed_across_seeds": True, "checkpoint_fixed": True, "stage2_fixed": True,
        "metric_and_postprocessing_fixed": True, "best_of_n": False,
        "principal_results_reselected_from_five_seeds": False,
        "principal_repeat_reused_from_matched_30fov_run": True,
        "principal_repeat_source": str(matched_metrics_path.resolve()),
        "right_censored_period_analysis_value": "Nyquist bound 2*pixel_size only for SD/range; censor flag retained",
    })
    print("APD6_FIVE_SEED_SENSITIVITY_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

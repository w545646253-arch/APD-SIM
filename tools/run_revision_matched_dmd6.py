"""Build the formal 30-FOV matched DMD-6F comparison from frozen inputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from revision_dmd6_common import (  # noqa: E402
    BASELINE_ROOT,
    MCSIM_CALIBRATION,
    PROTOCOL_HASH,
    REVISION_ROOT,
    array_sha256,
    atomic_json,
    gt_frc,
    harmonize,
    metrics_module,
    normalize_gt,
    sha256_file,
    write_csv,
)


TEST_MANIFEST = BASELINE_ROOT / "01_shared_contract/test30_dmd6_manifest.tsv"
MCSIM_OUTPUTS = BASELINE_ROOT / "09_baseline_only_results/mcsim_wiener6/formal_outputs"
CESHIJI = Path(r"data/sealed_test_gt")
OUTPUT = REVISION_ROOT / "01_matched_baselines"
METHODS = ("WF-6", "ML-SIM-6R", "mcSIM-Wiener-6", "APD-SIM-6")
METRIC_FIELDS = ("psnr", "ssim", "gt_frc_period_um", "frc_auc")


def _gt_path(sample_id: str) -> Path:
    candidates = [CESHIJI / f"{sample_id}.tif", CESHIJI / f"{sample_id}.tiff"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"cannot uniquely recover sealed GT for {sample_id}: {found}")
    return found[0]


def _summary(values: np.ndarray) -> dict[str, float | int]:
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "n": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "median": float(median), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(MCSIM_CALIBRATION.read_text(encoding="utf-8"))
    ml_receipt_path = BASELINE_ROOT / "08_training/mlsim_6r/formal_run/best_checkpoint_receipt.json"
    ml_receipt = json.loads(ml_receipt_path.read_text(encoding="utf-8"))
    ml_checkpoint = Path(ml_receipt["checkpoint"])
    if sha256_file(ml_checkpoint) != ml_receipt["checkpoint_sha256"] or ml_receipt.get("test_accessed") is not False:
        raise RuntimeError("ML-SIM-6R checkpoint receipt mismatch")
    import torch
    from official_r2_adapters import build_mlsim_model
    from revision_dmd6_common import MLSIM_SOURCE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ml_payload = torch.load(ml_checkpoint, map_location="cpu", weights_only=False)
    ml_model = build_mlsim_model(MLSIM_SOURCE)
    ml_model.load_state_dict(ml_payload["state_dict"], strict=True)
    ml_model.to(device).eval()
    from unisim.revision_r1.physmap6_core import RefinementConfig, masked_refine
    from unisim.revision_r1.physmap6_pipeline import load_stage1_registered, make_sim_config, stage1_reconstruct_registered
    from unisim.sim_forward_2d import embed_raw_to_slots_2d, nominal_theta_2d
    apd_config = json.loads((ROOT / "configs/apd_dmd_r2/train6_formal.json").read_text(encoding="utf-8"))
    apd_receipt = json.loads((ROOT / "checkpoints/apd_dmd_geometry_r2/dmd6/best_checkpoint_receipt.json").read_text(encoding="utf-8"))
    apd_model, apd_scheduler, _ = load_stage1_registered(apd_config, Path(apd_receipt["checkpoint_path"]), apd_receipt["checkpoint_sha256"], device, protocol_id="DMD_6F_2O3P")
    sim_config = make_sim_config(apd_config); theta = nominal_theta_2d(sim_config, device)
    geometry = {"protocol_id": "DMD_6F_2O3P", "protocol_hash": PROTOCOL_HASH, "raw_frame_order": ["H0", "H120", "H240", "V0", "V120", "V240"]}
    manifest = list(csv.DictReader(TEST_MANIFEST.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    if len(manifest) != 30 or any(row["protocol_hash"] != PROTOCOL_HASH for row in manifest):
        raise RuntimeError("frozen test manifest contract failed")
    metric_api = metrics_module()
    rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    native_root = OUTPUT / "native_outputs"
    harmonized_root = OUTPUT / "harmonized_outputs"
    for item in manifest:
        order = int(item["order"]); sample_id = item["sample_id"]
        gt_path = _gt_path(sample_id)
        if sha256_file(gt_path) != item["source_file_sha256"]:
            raise RuntimeError(f"sealed GT hash mismatch: {sample_id}")
        gt = normalize_gt(tifffile.imread(gt_path))
        method_native: dict[str, np.ndarray] = {}
        bundle = TEST_MANIFEST.parent / item["npz_path"]
        with np.load(bundle, allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_stack"], dtype=np.float32)
        if array_sha256(raw) != item["raw_stack_sha256"]:
            raise RuntimeError(f"raw hash mismatch: {sample_id}")
        method_native["WF-6"] = np.ascontiguousarray(raw.mean(axis=0), dtype=np.float32)
        raw_tensor = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
        x_ws, _, _ = stage1_reconstruct_registered(raw_tensor, apd_model, apd_scheduler, protocol_id="DMD_6F_2O3P", seed=20260812 + order)
        diffws = np.ascontiguousarray(x_ws[0, 0].detach().cpu().numpy(), dtype=np.float32)
        diffws_path = OUTPUT / "internal_component_outputs/DiffWS-6" / f"{order:03d}_{sample_id}.npy"
        diffws_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(diffws_path, diffws, allow_pickle=False)
        _, mask = embed_raw_to_slots_2d(raw_tensor, "DMD_6F_2O3P")
        apd_result = masked_refine(x_ws, raw_tensor, mask[0, :, 0, 0], geometry, {"sim_config": sim_config, "theta": theta}, RefinementConfig())
        method_native["APD-SIM-6"] = np.ascontiguousarray(apd_result.final_reconstruction[0, 0].detach().cpu().numpy(), dtype=np.float32)
        with torch.no_grad():
            ml_tensor = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
            ml_output = ml_model(ml_tensor)
        method_native["ML-SIM-6R"] = np.ascontiguousarray(ml_output[0, 0].detach().cpu().numpy(), dtype=np.float32)
        mc_path = MCSIM_OUTPUTS / f"{sample_id}.npz"
        with np.load(mc_path, allow_pickle=False) as archive:
            method_native["mcSIM-Wiener-6"] = np.asarray(archive["reconstruction"], dtype=np.float32)
        for method in METHODS:
            native = np.ascontiguousarray(method_native[method], dtype=np.float32)
            expected_shape = (2 * gt.shape[0], 2 * gt.shape[1]) if method == "mcSIM-Wiener-6" else gt.shape
            if native.shape != expected_shape or not np.isfinite(native).all():
                raise RuntimeError(f"{method} invalid for {sample_id}: {native.shape}")
            harm = harmonize(method, native, calibration["methods"].get(method))
            if harm.shape != gt.shape:
                raise RuntimeError(f"harmonized support mismatch for {method}/{sample_id}: {harm.shape}")
            native_path = native_root / method / f"{order:03d}_{sample_id}.npy"
            harmonized_path = harmonized_root / method / f"{order:03d}_{sample_id}.npy"
            native_path.parent.mkdir(parents=True, exist_ok=True); harmonized_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(native_path, native, allow_pickle=False); np.save(harmonized_path, harm, allow_pickle=False)
            frc, curve = gt_frc(gt, harm)
            period_um = frc["cutoff_derived_spatial_period_um"]
            if period_um is None and frc["right_censored_at_nyquist"]:
                period_um = 2.0 * (6.5 / 60.0)
            key = f"{order:03d}_{method.replace('-', '_') }"
            curves[key + "_frequency"] = curve["frequency_cycles_per_pixel"]
            curves[key + "_frc"] = curve["frc"]
            rows.append({
                "fov_index": order, "sample_id": sample_id, "parent_id": item["parent_id"],
                "structure_class": item["structure_class"], "method": method,
                "protocol_id": item["protocol_id"], "protocol_hash": item["protocol_hash"],
                "raw_stack_sha256": item["raw_stack_sha256"], "input_frame_count": 6,
                "raw_order": item["frame_order"], "native_array_sha256": array_sha256(native),
                "principal_diffusion_seed": 20260812 + order if method == "APD-SIM-6" else "NA",
                "harmonized_array_sha256": array_sha256(harm), "psnr": metric_api.psnr_native(gt, harm),
                "ssim": metric_api.ssim_native(gt, harm),
                "gt_frc_cutoff_cycles_per_pixel": frc["cutoff_cycles_per_pixel"],
                "gt_frc_period_px": frc["cutoff_derived_spatial_period_px"],
                "gt_frc_period_um": period_um,
                "frc_auc": frc["frc_auc_to_cutoff_or_nyquist"],
                "frc_right_censored": frc["right_censored_at_nyquist"],
                "native_path": str(native_path.resolve()), "harmonized_path": str(harmonized_path.resolve()),
                "finite": True,
            })
        print(f"matched {order + 1}/30", flush=True)
    write_csv(OUTPUT / "per_fov_metrics.csv", rows)
    np.savez_compressed(OUTPUT / "gt_frc_curves.npz", **curves)

    mean_rows: list[dict[str, object]] = []
    median_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        for metric in METRIC_FIELDS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            summary = _summary(values)
            mean_rows.append({"method": method, "metric": metric, "n_fov": summary["n"], "mean": summary["mean"], "sd": summary["sd"]})
            median_rows.append({"method": method, "metric": metric, "n_fov": summary["n"], "median": summary["median"], "q1": summary["q1"], "q3": summary["q3"], "iqr": summary["iqr"]})
            for class_name in ("CCP", "ER", "MT"):
                class_values = np.asarray([float(row[metric]) for row in selected if row["structure_class"] == class_name], dtype=np.float64)
                class_rows.append({"method": method, "metric": metric, "structure_class": class_name, **_summary(class_values)})
    write_csv(OUTPUT / "summary_mean_sd.csv", mean_rows)
    write_csv(OUTPUT / "summary_median_iqr.csv", median_rows)
    write_csv(OUTPUT / "class_stratified.csv", class_rows)

    effect_rows: list[dict[str, object]] = []
    raw_p: dict[str, float] = {}
    by_method_metric = {(method, metric): np.asarray([float(row[metric]) for row in rows if row["method"] == method]) for method in METHODS for metric in METRIC_FIELDS}
    parent_ids = [row["parent_id"] for row in rows if row["method"] == "APD-SIM-6"]
    classes = [row["structure_class"] for row in rows if row["method"] == "APD-SIM-6"]
    for comparator in ("WF-6", "ML-SIM-6R", "mcSIM-Wiener-6"):
        for metric in METRIC_FIELDS:
            apd = by_method_metric[("APD-SIM-6", metric)]; baseline = by_method_metric[(comparator, metric)]
            ci = metric_api.parent_image_bootstrap_ci(apd, baseline, parent_ids, class_labels=classes, n_resamples=10_000, seed=20260813)
            wilcoxon = metric_api.paired_wilcoxon(apd, baseline, parent_ids=parent_ids)
            label = f"APD-SIM-6_vs_{comparator}:{metric}"; raw_p[label] = float(wilcoxon["p_value"])
            effect_rows.append({
                "comparison": label, "metric": metric, "contrast": "APD-SIM-6 minus comparator",
                "mean_paired_difference": ci["estimate"], "bootstrap_ci_low": ci["confidence_interval"][0],
                "bootstrap_ci_high": ci["confidence_interval"][1], "bootstrap_resamples": 10000,
                "wilcoxon_statistic": wilcoxon["statistic"], "wilcoxon_p_value": wilcoxon["p_value"],
            })
    holm = metric_api.holm_correction(raw_p)
    adjusted = {item["label"]: item for item in holm["results"]}
    for row in effect_rows:
        row["holm_adjusted_p_value"] = adjusted[row["comparison"]]["holm_adjusted_p_value"]
        row["holm_reject_0_05"] = adjusted[row["comparison"]]["reject_at_alpha"]
    write_csv(OUTPUT / "paired_effects.csv", effect_rows)
    atomic_json(OUTPUT / "statistics.json", {
        "status": "MATCHED_DMD6_30FOV_COMPLETE", "n_fov": 30, "methods": list(METHODS),
        "protocol_hash": PROTOCOL_HASH, "paired_unit": "parent/FOV", "bootstrap_resamples": 10_000,
        "wilcoxon": "two-sided signed-rank", "multiple_testing": holm,
        "sealed_test_used_for_training_or_tuning": False, "ssr_sim_9f_in_matched_group": False,
        "raw_hash_equality_per_fov": all(len({row["raw_stack_sha256"] for row in rows if row["fov_index"] == order}) == 1 for order in range(30)),
        "mlsim_checkpoint": str(ml_checkpoint), "mlsim_checkpoint_sha256": sha256_file(ml_checkpoint),
        "mcsim_calibration_sha256": sha256_file(MCSIM_CALIBRATION),
    })
    print("MATCHED_DMD6_30FOV_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

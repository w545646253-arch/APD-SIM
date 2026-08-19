"""No-figure 30-FOV APD-SIM-3/6/9 controller-protocol comparison."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from unisim.revision_r1 import frame_budget_r1c2 as fb  # noqa: E402
from revision_dmd6_common import REVISION_ROOT, atomic_json, gt_frc, metrics_module, normalize_gt, write_csv  # noqa: E402


OUTPUT = REVISION_ROOT / "02_frame_budget"


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {"n_fov": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)), "median": float(np.median(values))}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    for directory in (ROOT / "outputs/reviewer1_frame_budget_30fov").glob("*"):
        status_path = directory / "STATUS.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "R1C2_FRAME_BUDGET_30FOV_READY":
                candidates.append(directory)
    if candidates:
        reused = sorted(candidates)[-1]
        raw_rows = list(csv.DictReader((reused / "R1C2_FRAME_BUDGET_PER_FOV.csv").open("r", encoding="utf-8", newline="")))
        atomic_json(OUTPUT / "source_run_receipt.json", {"status": "REUSED_COMPLETE_VALIDATED_NO_NEW_GPU_RUN", "source_run": str(reused.resolve()), "source_status_sha256": fb.sha_file(reused / "STATUS.json"), "source_per_fov_sha256": fb.sha_file(reused / "R1C2_FRAME_BUDGET_PER_FOV.csv")})
    else:
        _checkpoint_rows, details = fb.audit_checkpoints(OUTPUT)
        fb.audit_protocols(OUTPUT)
        dataset = fb.audit_dataset(OUTPUT)
        fb.audit_cross_protocol_config(details)
        raw_rows = fb.run_inference(OUTPUT, dataset, details)
    metric_api = metrics_module()
    selected = [row for row in raw_rows if row["method"] == "APD-SIM" or (row["method"] == "WF" and row["protocol_label"] == "DMD-6F")]
    rows: list[dict[str, object]] = []
    for row in selected:
        prediction = np.load(row["prediction_path"], allow_pickle=False)
        gt = normalize_gt(tifffile.imread(Path(row["gt_path"])))
        frc, _ = gt_frc(gt, prediction)
        period_um = frc["cutoff_derived_spatial_period_um"]
        if period_um is None and frc["right_censored_at_nyquist"]:
            period_um = 2.0 * (6.5 / 60.0)
        method = "WF reference" if row["method"] == "WF" else "APD-SIM-" + row["protocol_label"].split("-")[1].replace("F", "")
        rows.append({
            "fov_index": row["sample_order"], "sample_id": row["sample_id"], "parent_id": row["parent_id"],
            "structure_class": row["structure"], "method": method, "protocol": row["protocol_label"],
            "protocol_id": row["protocol_id"], "protocol_hash": row["protocol_hash"],
            "frame_count": row["frame_count"], "raw_order": row["raw_frame_order"],
            "measurement_seed": row["measurement_seed"], "diffusion_seed": row["diffusion_seed"],
            "psnr": metric_api.psnr_native(gt, prediction), "ssim": metric_api.ssim_native(gt, prediction),
            "gt_frc_period_um": period_um,
            "gt_frc_right_censored": frc["right_censored_at_nyquist"],
            "prediction_path": row["prediction_path"], "prediction_sha256": row["prediction_array_sha256"],
        })
    write_csv(OUTPUT / "per_fov.csv", rows)
    summary_rows: list[dict[str, object]] = []
    for method in ("APD-SIM-3", "APD-SIM-6", "APD-SIM-9", "WF reference"):
        subset = [row for row in rows if row["method"] == method]
        for metric in ("psnr", "ssim", "gt_frc_period_um"):
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            summary_rows.append({"method": method, "protocol": subset[0]["protocol"], "metric": metric, **describe(values)})
    write_csv(OUTPUT / "summary.csv", summary_rows)
    effects: list[dict[str, object]] = []
    parent_ids = [row["parent_id"] for row in rows if row["method"] == "APD-SIM-6"]
    classes = [row["structure_class"] for row in rows if row["method"] == "APD-SIM-6"]
    for comparator in ("APD-SIM-3", "APD-SIM-9"):
        for metric in ("psnr", "ssim", "gt_frc_period_um"):
            ref = np.asarray([float(row[metric]) for row in rows if row["method"] == "APD-SIM-6"])
            alt = np.asarray([float(row[metric]) for row in rows if row["method"] == comparator])
            ci = metric_api.parent_image_bootstrap_ci(alt, ref, parent_ids, class_labels=classes, n_resamples=10_000, seed=20260813)
            test = metric_api.paired_wilcoxon(alt, ref, parent_ids=parent_ids)
            effects.append({"comparison": f"{comparator}_minus_APD-SIM-6", "metric": metric, "mean_paired_difference": ci["estimate"], "bootstrap_ci_low": ci["confidence_interval"][0], "bootstrap_ci_high": ci["confidence_interval"][1], "wilcoxon_p_value": test["p_value"]})
    write_csv(OUTPUT / "paired_effects.csv", effects)
    atomic_json(OUTPUT / "protocol_comparison_receipt.json", {
        "status": "FRAME_BUDGET_369_COMPLETE", "n_fov": 30,
        "interpretation": "paired comparison across controller-defined protocols",
        "not_claimed": "frame count is the only changed variable",
        "non_geometry_policy": "shared forward family, common measurement and diffusion seed bases",
        "paper_figures_created": False,
    })
    print("FRAME_BUDGET_369_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

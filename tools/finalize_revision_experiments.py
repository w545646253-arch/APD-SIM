"""CPU-only postprocessing, tables, evidence text, and package validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from revision_dmd6_common import (  # noqa: E402
    BASELINE_ROOT, PIXEL_SIZE_UM, PROTOCOL_HASH, REVISION_ROOT, array_sha256,
    atomic_json, gt_frc, metrics_module, normalize_gt, sha256_file, write_csv,
)


STRICT_SOURCE = ROOT / "outputs/reviewer1_physmap6_strict/20260813T183229Z"
CESHIJI = Path(r"data/sealed_test_gt")


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def describe(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {"n": int(array.size), "mean": float(array.mean()), "sd": float(array.std(ddof=1)), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def gt_for(sample_id: str) -> np.ndarray:
    path = next(path for path in (CESHIJI / f"{sample_id}.tif", CESHIJI / f"{sample_id}.tiff") if path.is_file())
    return normalize_gt(tifffile.imread(path))


def finalize_strict() -> list[dict[str, Any]]:
    out = REVISION_ROOT / "05_strict_ablation"; out.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(STRICT_SOURCE / "R1C3_NOMINAL_PER_FOV.csv")
    manifest = read_csv(BASELINE_ROOT / "01_shared_contract/test30_dmd6_manifest.tsv", delimiter="\t")
    matched = REVISION_ROOT / "01_matched_baselines"
    rows: list[dict[str, Any]] = []
    equality: list[dict[str, Any]] = []
    metric_api = metrics_module()
    for item in manifest:
        order = int(item["order"]); sample_id = item["sample_id"]; gt = gt_for(sample_id)
        paths = {
            "WF-6": STRICT_SOURCE / "nominal_predictions" / f"{order:03d}_{sample_id}_WF.npy",
            "PhysMap-6": STRICT_SOURCE / "nominal_predictions" / f"{order:03d}_{sample_id}_PhysMap-6.npy",
            "DiffWS-6": matched / "internal_component_outputs/DiffWS-6" / f"{order:03d}_{sample_id}.npy",
            "APD-SIM-6": matched / "native_outputs/APD-SIM-6" / f"{order:03d}_{sample_id}.npy",
        }
        source_group = [row for row in source_rows if int(row["sample_order"]) == order]
        raw_hashes = {row["raw_stack_sha256"] for row in source_group}
        equality.append({"fov_index": order, "sample_id": sample_id, "manifest_raw_stack_sha256": item["raw_stack_sha256"], "source_method_raw_hash_count": len(raw_hashes), "source_hash_matches_manifest": raw_hashes == {item["raw_stack_sha256"]}, "protocol_hash": PROTOCOL_HASH, "identical_six_frame_input": raw_hashes == {item["raw_stack_sha256"]}})
        for method, path in paths.items():
            prediction = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            frc, _ = gt_frc(gt, prediction)
            period = frc["cutoff_derived_spatial_period_um"]
            if period is None and frc["right_censored_at_nyquist"]:
                period = 2 * PIXEL_SIZE_UM
            rows.append({"fov_index": order, "sample_id": sample_id, "parent_id": item["parent_id"], "structure_class": item["structure_class"], "method": method, "protocol_id": item["protocol_id"], "protocol_hash": item["protocol_hash"], "raw_stack_sha256": item["raw_stack_sha256"], "psnr": metric_api.psnr_native(gt, prediction), "ssim": metric_api.ssim_native(gt, prediction), "gt_frc_period_um": period, "gt_frc_right_censored": frc["right_censored_at_nyquist"], "prediction_path": str(path.resolve()), "prediction_sha256": array_sha256(prediction), "finite": bool(np.isfinite(prediction).all())})
    write_csv(out / "per_fov.csv", rows); write_csv(out / "input_hash_equality.tsv", equality, delimiter="\t")
    summary: list[dict[str, Any]] = []
    for method in ("WF-6", "DiffWS-6", "PhysMap-6", "APD-SIM-6"):
        for metric in ("psnr", "ssim", "gt_frc_period_um"):
            summary.append({"method": method, "metric": metric, **describe([float(row[metric]) for row in rows if row["method"] == method])})
    write_csv(out / "summary.csv", summary)
    effects: list[dict[str, Any]] = []
    parents = [row["parent_id"] for row in rows if row["method"] == "APD-SIM-6"]
    classes = [row["structure_class"] for row in rows if row["method"] == "APD-SIM-6"]
    for comparator in ("WF-6", "DiffWS-6", "PhysMap-6"):
        for metric in ("psnr", "ssim", "gt_frc_period_um"):
            apd = [float(row[metric]) for row in rows if row["method"] == "APD-SIM-6"]
            other = [float(row[metric]) for row in rows if row["method"] == comparator]
            ci = metric_api.parent_image_bootstrap_ci(apd, other, parents, class_labels=classes, n_resamples=10_000, seed=20260813)
            test = metric_api.paired_wilcoxon(apd, other, parent_ids=parents)
            effects.append({"comparison": f"APD-SIM-6_minus_{comparator}", "metric": metric, "mean_paired_difference": ci["estimate"], "bootstrap_ci_low": ci["confidence_interval"][0], "bootstrap_ci_high": ci["confidence_interval"][1], "wilcoxon_p_value": test["p_value"]})
    write_csv(out / "paired_effects.csv", effects)
    protocol_receipt = json.loads((STRICT_SOURCE / "DMD6_PROTOCOL_RECEIPT.json").read_text(encoding="utf-8"))
    robustness_ok = protocol_receipt.get("protocol_hash") == PROTOCOL_HASH and (STRICT_SOURCE / "R1C3_ROBUSTNESS_PER_SAMPLE.csv").is_file()
    if robustness_ok:
        (out / "robustness_reuse_receipt.md").write_text("Existing strict DMD-6F robustness is retained as previously completed evidence. It binds DMD_6F_2O3P and compares APD-SIM-6, DiffWS-6, and PhysMap-6 from matched six-frame inputs. No new robustness grid was run.\n", encoding="utf-8")
    else:
        (out / "robustness_section_removal_recommendation.md").write_text("Remove the legacy robustness values and Figure 5 generalization claim because current DMD-6F input hashes cannot be fully verified.\n", encoding="utf-8")
    return rows


def finalize_gt_frc() -> list[dict[str, Any]]:
    out = REVISION_ROOT / "04_gt_frc"; out.mkdir(parents=True, exist_ok=True)
    matched = read_csv(REVISION_ROOT / "01_matched_baselines/per_fov_metrics.csv")
    frame = read_csv(REVISION_ROOT / "02_frame_budget/per_fov.csv")
    selected: list[tuple[str, dict[str, str]]] = []
    for row in matched:
        if row["method"] in {"APD-SIM-6", "WF-6", "ML-SIM-6R", "mcSIM-Wiener-6"}:
            selected.append((row["method"], row))
    for row in frame:
        if row["method"] in {"APD-SIM-3", "APD-SIM-9"}:
            selected.append((row["method"], row))
    rows: list[dict[str, Any]] = []; curves: dict[str, np.ndarray] = {}
    for method, row in selected:
        prediction_path = row.get("harmonized_path") or row.get("prediction_path")
        if not prediction_path:
            raise KeyError(f"no evaluation-array path for {method}/{row['sample_id']}")
        gt = gt_for(row["sample_id"]); prediction = np.load(prediction_path, allow_pickle=False)
        frc, curve = gt_frc(gt, prediction)
        period = frc["cutoff_derived_spatial_period_um"]
        if period is None and frc["right_censored_at_nyquist"]:
            period = 2 * PIXEL_SIZE_UM
        index = int(row["fov_index"]); key = f"{index:03d}_{method.replace('-', '_')}"
        curves[key + "_frequency"] = curve["frequency_cycles_per_pixel"]; curves[key + "_frc"] = curve["frc"]
        rows.append({"fov_index": index, "sample_id": row["sample_id"], "parent_id": row["parent_id"], "structure_class": row["structure_class"], "method": method, "cutoff_cycles_per_pixel": frc["cutoff_cycles_per_pixel"], "cutoff_derived_spatial_period_px": frc["cutoff_derived_spatial_period_px"], "cutoff_derived_spatial_period_um": period, "right_censored_at_nyquist": frc["right_censored_at_nyquist"], "unresolved_no_crossing": frc["unresolved_no_crossing"], "frc_auc": frc["frc_auc_to_cutoff_or_nyquist"], "prediction_path": prediction_path})
    write_csv(out / "per_fov.csv", rows); np.savez_compressed(out / "curves.npz", **curves)
    summary: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        for metric in ("cutoff_derived_spatial_period_um", "frc_auc"):
            summary.append({"method": method, "metric": metric, **describe([float(row[metric]) for row in rows if row["method"] == method])})
    write_csv(out / "summary.csv", summary)
    atomic_json(out / "protocol.json", {"status": "GT_REFERENCED_FRC_COMPLETE", "terminology": "GT-referenced FRC cutoff-derived spatial period", "registration": "same pixel grid as PSNR/SSIM; no fitted test registration", "crop_each_edge_fraction": 0.05, "mean_centering": True, "window": "separable 2D Tukey", "tukey_alpha": 0.20, "radial_annuli": 100, "smoothing": "none", "threshold": "1/7", "crossing": "first downward crossing, adjacent-bin linear interpolation", "no_crossing": "right-censored at Nyquist", "period_formula": "period = pixel_size / cutoff = 2p/x when x is normalized to Nyquist", "independent_experimental_resolution_claimed": False})
    return rows


def finalize_real_audit() -> None:
    out = REVISION_ROOT / "06_real_data_audit"; out.mkdir(parents=True, exist_ok=True)
    source_root = ROOT / "outputs/DMD_GEOMETRY_MATCHED_REVISION_R1_20260812_233257/agent_loader_evidence"
    summary = json.loads((source_root / "loader_acquisition_summary.json").read_text(encoding="utf-8"))
    source_inventory = read_csv(source_root / "raw_stack_inventory.tsv", delimiter="\t")
    rows = []
    for row in source_inventory:
        rows.append({"evidence_class": row.get("evidence_class", row.get("record_type", "inventory")), "protocol_or_frame_count": row.get("protocol_id", row.get("frame_count", "")), "source_path": row.get("path", row.get("absolute_path", row.get("file_path", ""))), "raw_file_count": row.get("raw_file_count", ""), "raw_sha256": row.get("sha256", row.get("file_sha256", "")), "timestamp": row.get("timestamp", row.get("date_time", "")), "acquisition_order": row.get("raw_order", row.get("order", "")), "independent_trigger_supported": row.get("independent_trigger_supported", "UNRESOLVED")})
    write_csv(out / "real_data_manifest.tsv", rows, fields=list(rows[0]), delimiter="\t")
    atomic_json(out / "sampling_summary.json", {"status": "REAL_DATA_EVIDENCE_AUDIT_COMPLETE", "manifest_counts": summary["acquisition_manifest_counts"], "verified_3f_manifests": 18, "verified_6f_manifests": 0, "verified_9f_manifests": 19, "verified_15f_manifests": 13, "candidate_6f_folders": 3, "candidate_6f_binding": summary["K6_candidates"]["binding_status"], "prepared_slide_identity": "NOT_RECOVERABLE", "specimen_count": "NOT_RECOVERABLE", "fov_count": "NOT_RECOVERABLE_FROM_LOCAL_METADATA", "three_repeats_per_protocol_supported": False, "ten_fov_supported": False, "replication_interpretation": "technical spatial sampling only where file-level FOV identity exists; not independent biological replication"})
    (out / "evidence_supported_text.txt").write_text("Local acquisition receipts verify 18 three-frame and 19 nine-frame acquisition manifests, including independently triggered representative K3 and K9 acquisitions. Three candidate six-frame directories each contain six TIFFs, but no receipt binds those TIFFs to the H0/H120/H240/V0/V120/V240 controller sequence. Biological specimen and prepared-slide counts are not recoverable. Any file-level FOV sampling is technical spatial sampling, not independent biological replication.\n", encoding="utf-8")
    (out / "unsupported_claims.txt").write_text("Unsupported: 10 FOV per specimen; 3 independent exposure repeats per protocol; independently acquired matched K3/K6/K9 triplets; biological replication count; exact prepared-slide identity for all files. Re-running reconstruction on stored TIFFs is not an acquisition repeat.\n", encoding="utf-8")


def tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def tex_table(path: Path, caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], *, tiny: bool = False) -> None:
    align = "l" + "c" * (len(headers) - 1)
    body = [r"\begin{table*}[t]", r"\centering", r"\scriptsize" if tiny else r"\small", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + align + "}", r"\hline", " & ".join(tex_escape(x) for x in headers) + r" \\", r"\hline"]
    body.extend(" & ".join(tex_escape(x) for x in row) + r" \\" for row in rows)
    body.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(body), encoding="utf-8")


def pivot_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["method"], row["metric"]): row for row in read_csv(path)}


def maybe_bold(text: str, selected: bool) -> str:
    return rf"\textbf{{{text}}}" if selected else text


def plain_table_cell(value: Any) -> str:
    text = str(value)
    if text.startswith(r"\textbf{") and text.endswith("}"):
        text = text[len(r"\textbf{"):-1]
    return text.replace(r" $\pm$ ", " ± ")


def tex_s1(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Render the 15-field S1 as three readable panels across two table floats."""
    identity_indices = (0, 1, 2, 3, 14)
    acquisition_indices = (0, 4, 5, 6, 13)
    algorithm_indices = (0, 7, 8, 9)
    policy_indices = (0, 10, 11, 12)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Source, protocol, adaptation, and execution evidence for all reconstruction methods. Matched DMD-6F and native 9F groups are not pooled. Full commit identifiers are retained in the accompanying CSV and evidence TSV.}",
        r"\label{tab:supp_method_evidence}", r"\setlength{\tabcolsep}{2pt}",
        r"\textit{Panel A: method identity, source, environment, and evidence status}\par\smallskip",
        r"\begin{tabular}{p{1.3cm}p{1.6cm}p{3.7cm}p{3.5cm}p{2.5cm}}", r"\hline",
        " & ".join(tex_escape(headers[i]) for i in identity_indices) + r" \\", r"\hline",
    ]
    lines.extend(" & ".join(tex_escape(row[i]) for i in identity_indices) + r" \\" for row in rows)
    lines.extend([
        r"\hline", r"\end{tabular}", r"\par\medskip",
        r"\textit{Panel B: frame geometry and comparison group}\par\smallskip",
        r"\begin{tabular}{p{1.6cm}p{1.0cm}p{2.2cm}p{5.2cm}p{2.6cm}}", r"\hline",
        " & ".join(tex_escape(headers[i]) for i in acquisition_indices) + r" \\", r"\hline",
    ])
    lines.extend(" & ".join(tex_escape(row[i]) for i in acquisition_indices) + r" \\" for row in rows)
    lines.extend([
        r"\hline", r"\end{tabular}", r"\end{table*}", "",
        r"\addtocounter{table}{-1}", r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Supplementary Table S1 (continued): adaptation, initialization, parameters, and evaluation policy.}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\textit{Panel C: adaptation, initialization, and principal parameters}\par\smallskip",
        r"\begin{tabular}{p{1.6cm}p{3.0cm}p{3.6cm}p{4.4cm}}", r"\hline",
        " & ".join(tex_escape(headers[i]) for i in algorithm_indices) + r" \\", r"\hline",
    ])
    lines.extend(" & ".join(tex_escape(row[i]) for i in algorithm_indices) + r" \\" for row in rows)
    lines.extend([
        r"\hline", r"\end{tabular}", r"\par\medskip",
        r"\textit{Panel D: stopping, seed, and output harmonization}\par\smallskip",
        r"\begin{tabular}{p{1.6cm}p{3.8cm}p{2.2cm}p{5.0cm}}", r"\hline",
        " & ".join(tex_escape(headers[i]) for i in policy_indices) + r" \\", r"\hline",
    ])
    lines.extend(" & ".join(tex_escape(row[i]) for i in policy_indices) + r" \\" for row in rows)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_tables() -> None:
    out = REVISION_ROOT / "07_tables"; out.mkdir(parents=True, exist_ok=True)
    matched = pivot_summary(REVISION_ROOT / "01_matched_baselines/summary_mean_sd.csv")
    med = pivot_summary(REVISION_ROOT / "01_matched_baselines/summary_median_iqr.csv")
    methods = ("WF-6", "ML-SIM-6R", "mcSIM-Wiener-6", "APD-SIM-6")
    main_best = {
        "psnr": max(float(matched[(m, "psnr")]["mean"]) for m in methods),
        "ssim": max(float(matched[(m, "ssim")]["mean"]) for m in methods),
        "period": min(float(matched[(m, "gt_frc_period_um")]["mean"]) for m in methods),
        "median_psnr": max(float(med[(m, "psnr")]["median"]) for m in methods),
        "median_ssim": max(float(med[(m, "ssim")]["median"]) for m in methods),
    }
    main_rows = []
    for method in methods:
        p = float(matched[(method,'psnr')]['mean']); s = float(matched[(method,'ssim')]['mean']); g = float(matched[(method,'gt_frc_period_um')]['mean'])
        mp = float(med[(method,'psnr')]['median']); ms = float(med[(method,'ssim')]['median'])
        main_rows.append([method, "DMD-6F 2O3P", maybe_bold(f"{p:.3f} $\\pm$ {float(matched[(method,'psnr')]['sd']):.3f}", p == main_best["psnr"]), maybe_bold(f"{s:.4f} $\\pm$ {float(matched[(method,'ssim')]['sd']):.4f}", s == main_best["ssim"]), maybe_bold(f"{g:.4f} $\\pm$ {float(matched[(method,'gt_frc_period_um')]['sd']):.4f}", g == main_best["period"]), maybe_bold(f"{mp:.3f}", mp == main_best["median_psnr"]), maybe_bold(f"{ms:.4f}", ms == main_best["median_ssim"])])
    write_csv(out / "Table_main_matched_dmd6.csv", [dict(zip(("Method","Input protocol","PSNR mean SD","SSIM mean SD","GT-FRC period mean SD","Median PSNR","Median SSIM"), [plain_table_cell(cell) for cell in row])) for row in main_rows])
    tex_table(out / "Table_main_matched_dmd6.tex", "Matched six-frame comparison on 30 frozen FOVs. Lower GT-FRC period is better; all methods received byte-identical DMD-6F measurements.", "tab:matched_dmd6", ["Method","Input protocol","PSNR mean $\\pm$ SD","SSIM mean $\\pm$ SD","GT-FRC period ($\\mu$m) mean $\\pm$ SD","Median PSNR","Median SSIM"], main_rows)
    frame = pivot_summary(REVISION_ROOT / "02_frame_budget/summary.csv")
    frame_methods=("APD-SIM-3","APD-SIM-6","APD-SIM-9","WF reference")
    frame_best={"psnr":max(float(frame[(m,"psnr")]["mean"]) for m in frame_methods),"ssim":max(float(frame[(m,"ssim")]["mean"]) for m in frame_methods),"gt_frc_period_um":min(float(frame[(m,"gt_frc_period_um")]["mean"]) for m in frame_methods)}
    frame_rows=[]
    for method in frame_methods:
        cells=[]
        for metric, digits in (("psnr",3),("ssim",4),("gt_frc_period_um",4)):
            mean=float(frame[(method,metric)]["mean"]); sd=float(frame[(method,metric)]["sd"])
            cells.append(maybe_bold(f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}", mean == frame_best[metric]))
        frame_rows.append([method, frame[(method,"psnr")]["protocol"], *cells])
    write_csv(out / "Table_frame_budget_369.csv", [dict(zip(("Method","Protocol","PSNR","SSIM","GT-FRC period"), [plain_table_cell(cell) for cell in row])) for row in frame_rows])
    tex_table(out / "Table_frame_budget_369.tex", "Paired comparison across controller-defined DMD protocols. Frame number and orientation support change together, so this is not a frame-count-only causal experiment.", "tab:frame_budget_369", ["Method","Protocol","PSNR","SSIM","GT-FRC period ($\\mu$m)"], frame_rows)
    strict = pivot_summary(REVISION_ROOT / "05_strict_ablation/summary.csv")
    strict_methods=("WF-6","DiffWS-6","PhysMap-6","APD-SIM-6")
    strict_best={"psnr":max(float(strict[(m,"psnr")]["mean"]) for m in strict_methods),"ssim":max(float(strict[(m,"ssim")]["mean"]) for m in strict_methods),"gt_frc_period_um":min(float(strict[(m,"gt_frc_period_um")]["mean"]) for m in strict_methods)}
    strict_rows=[]
    for method in strict_methods:
        cells=[]
        for metric, digits in (("psnr",3),("ssim",4),("gt_frc_period_um",4)):
            mean=float(strict[(method,metric)]["mean"]); sd=float(strict[(method,metric)]["sd"])
            cells.append(maybe_bold(f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}", mean == strict_best[metric]))
        strict_rows.append([method,*cells])
    write_csv(out / "Table_strict_ablation_dmd6.csv", [dict(zip(("Method","PSNR","SSIM","GT-FRC period"), [plain_table_cell(cell) for cell in row])) for row in strict_rows])
    tex_table(out / "Table_strict_ablation_dmd6.tex", "Strict DMD-6F component comparison from identical measurements and matched 40-update refinement settings.", "tab:strict_ablation_dmd6", ["Method","PSNR","SSIM","GT-FRC period ($\\mu$m)"], strict_rows)
    s1_headers=("Method","Manuscript label","Source/repository/commit","Execution environment","Frame count","Orientation x phase geometry","Raw order","Adaptation","Training/initialization","Principal parameters","Stopping/checkpoint rule","Seed policy","Output harmonization","Comparison group","Evidence status")
    s1_rows=[
        ["APD-SIM-6","APD-SIM-6","local formal source snapshot","newenv PyTorch CUDA","6","2 x 3","H0,H120,H240,V0,V120,V240","none","validation-selected EMA","DDIM80; Stage2 Adam40 lr5e-3","R2 validation rule","20260812+i","identity clip [0,1]","Matched DMD-6F","PASS"],
        ["WF-6","WF-6","defined arithmetic mean","NumPy/PyTorch","6","2 x 3","H0,H120,H240,V0,V120,V240","none","none","mean of six frames","none","not stochastic","identity clip [0,1]","Matched DMD-6F","PASS"],
        ["ML-SIM-6R","ML-SIM-6R","fairSIM/ML-SIM commit 25e289eca8571621e85f2d32ae09174b4c841b70","apd_mlsim_official_r2","6","2 x 3","H0,H120,H240,V0,V120,V240","official RCAN retrained with 6 inputs","from scratch; 132 train","RCAN 2RG/5RB/48 features; MSE; Adam1e-4","100 epochs; minimum validation MSE","20260813 training","identity clip [0,1]","Matched DMD-6F","PASS"],
        ["mcSIM-Wiener-6","mcSIM-Wiener-6","QI2lab/mcSIM commit 43b8b54535c3f4af666fb711dd630e903f156805","apd_mcsim_official_r2","6","2 x 3","H0,H120,H240,V0,V120,V240","direct 2O3P configuration","six-frame mean initialization internal","Wicker phase; band correlation; Wiener0.1; fairSIM bands","direct deterministic configuration","not stochastic","2x2 area map + validation affine + clip","Matched DMD-6F","PASS"],
        ["Hessian-SIM-6","Hessian-SIM-6","publication-linked complete source not runnable locally","MATLAB unavailable/incomplete wrapper","6","2 x 3","H0,H120,H240,V0,V120,V240","none claimed","not executed","not recovered","not executed","not applicable","not applicable","Not executed","HESSIAN_SIM6_NOT_EXECUTED_NONFATAL"],
        ["SSR-SIM-9F","SSR-SIM-9F","native publication protocol only","not executed in this task","9","3 x 3","native nine-frame order","no 6F adaptation","native 9F only","not mixed with matched table","not executed","not applicable","separate native support","Native 9F supplementary","NATIVE_9F_ONLY"],
    ]
    write_csv(out / "Supplementary_Table_S1.csv", [dict(zip(s1_headers,row)) for row in s1_rows]); write_csv(out / "Supplementary_Table_S1_evidence.tsv", [{"method":r[0],"evidence_status":r[-1],"source":r[2],"comparison_group":r[-2]} for r in s1_rows], delimiter="\t")
    s1_tex_rows=[list(row) for row in s1_rows]
    s1_tex_rows[2][2]="ML-SIM commit 25e289eca857"
    s1_tex_rows[3][2]="mcSIM commit 43b8b54535c3"
    s1_tex_rows[4][2]="publication-linked source; complete wrapper not runnable"
    s1_tex_rows[4][3]="MATLAB unavailable; wrapper incomplete"
    for row in s1_tex_rows:
        row[6]=str(row[6]).replace(",", ", ")
    s1_tex_rows[4][14]="NOT EXECUTED"
    s1_tex_rows[5][14]="NATIVE 9F ONLY"
    tex_s1(out / "Supplementary_Table_S1.tex", s1_headers, s1_tex_rows)
    seed_rows=read_csv(REVISION_ROOT / "03_seed_sensitivity/summary.csv")
    write_csv(out / "Supplementary_Table_S2_seed_sensitivity.csv", seed_rows)
    seed_labels={"psnr":"PSNR","ssim":"SSIM","frc_period_um_for_sensitivity":"GT-FRC period ($\\mu$m)"}
    seed_tex_rows=[[seed_labels[row["metric"]],row["n_fov"],f"{float(row['median_within_fov_sd']):.5f}",f"{float(row['p95_within_fov_sd']):.5f}",f"{float(row['maximum_range']):.5f}",row["principal_seed_all_within_observed_range"]] for row in seed_rows]
    tex_table(out / "Supplementary_Table_S2_seed_sensitivity.tex", "Five-seed APD-SIM-6 sensitivity under fixed raw data, checkpoint, Stage 2, metrics, and post-processing.", "tab:supp_seed_sensitivity", ["Metric","FOVs","Median within-FOV SD","95th percentile SD","Maximum range","Principal seed within range"], seed_tex_rows)
    frc_rows=read_csv(REVISION_ROOT / "04_gt_frc/summary.csv")
    write_csv(out / "Supplementary_Table_S3_gt_frc.csv", frc_rows)
    frc_map={(row["method"],row["metric"]):row for row in frc_rows}; frc_methods=sorted({row["method"] for row in frc_rows})
    best_period=min(float(frc_map[(m,"cutoff_derived_spatial_period_um")]["mean"]) for m in frc_methods); best_auc=max(float(frc_map[(m,"frc_auc")]["mean"]) for m in frc_methods)
    frc_tex_rows=[]
    for method in frc_methods:
        period=frc_map[(method,"cutoff_derived_spatial_period_um")]; auc=frc_map[(method,"frc_auc")]
        pmean=float(period["mean"]); amean=float(auc["mean"])
        frc_tex_rows.append([method,maybe_bold(f"{pmean:.4f} $\\pm$ {float(period['sd']):.4f}",pmean==best_period),f"{float(period['median']):.4f} [{float(period['q1']):.4f}, {float(period['q3']):.4f}]",maybe_bold(f"{amean:.4f} $\\pm$ {float(auc['sd']):.4f}",amean==best_auc),f"{float(auc['median']):.4f} [{float(auc['q1']):.4f}, {float(auc['q3']):.4f}]"])
    tex_table(out / "Supplementary_Table_S3_gt_frc.tex", "GT-referenced FRC cutoff-derived spatial period and FRC AUC on the 30-FOV cohort. These values are not independent experimental resolution estimates.", "tab:supp_gt_frc", ["Method","Period ($\\mu$m), mean $\\pm$ SD","Period, median [IQR]","FRC AUC, mean $\\pm$ SD","FRC AUC, median [IQR]"], frc_tex_rows)
    driver = r"""\documentclass[preprint,12pt,review]{elsarticle}
\usepackage{graphicx}
\begin{document}
\input{Table_main_matched_dmd6.tex}
\input{Table_frame_budget_369.tex}
\input{Table_strict_ablation_dmd6.tex}
\input{Supplementary_Table_S1.tex}
\input{Supplementary_Table_S2_seed_sensitivity.tex}
\input{Supplementary_Table_S3_gt_frc.tex}
\end{document}
"""
    (out / "table_compile_driver.tex").write_text(driver, encoding="utf-8")


def write_handoff_and_facts() -> None:
    hand = REVISION_ROOT / "08_user_figure_handoff"; hand.mkdir(parents=True, exist_ok=True)
    single_root = ROOT / "outputs/single_input_dmd6_comparison"
    runs = sorted([path for path in single_root.glob("microtubules_Cell_046_SIM_gt_seed20260812_*") if path.is_dir()])
    single = runs[-1] if runs else Path("NOT_YET_RUN")
    display = {"vmin":0.0,"vmax":1.0,"method_specific_percentile_remapping":False,"gamma":False,"histogram_equalization":False}
    atomic_json(hand / "display_contract.json", display)
    atomic_json(hand / "scale_bar_contract.json", {"pixel_size_um":PIXEL_SIZE_UM,"one_micrometre_scale_bar_pixels":1.0/PIXEL_SIZE_UM,"rounding_note":"use 9.2308 pixels in vector layout; do not rescale image"})
    labels=["GT","WF-6","ML-SIM-6R","mcSIM-Wiener-6","Hessian-SIM-6 (only if formally available)","APD-SIM-6"]
    (hand / "figure_method_labels.txt").write_text("\n".join(labels)+"\n",encoding="utf-8")
    path_rows=[]
    for method in ("GT","WF-6","ML-SIM-6R","mcSIM-Wiener-6","APD-SIM-6"):
        path_rows.append({"method":method,"native":str(single/"native"/f"{method}.npy") if method!="GT" else str(single/"gt_native.npy"),"harmonized":str(single/"harmonized"/f"{method}.npy") if method!="GT" else str(single/"gt_native.npy"),"display16":str(single/"display16"/f"{method}.tif") if method!="GT" else str(single/"gt_display16.tif")})
    write_csv(hand / "figure_output_paths.tsv",path_rows,delimiter="\t")
    (hand / "README_FOR_MANUAL_FIGURES.md").write_text(f"""# Manual figure handoff

- Right-click script: `{ROOT / 'compare_single_dmd6.py'}`
- Frozen default run: `{single}`
- Use common display range [0,1] for every method.
- Pixel size: {PIXEL_SIZE_UM:.8f} um; 1 um corresponds to {1/PIXEL_SIZE_UM:.4f} pixels.
- Recommended order: GT, WF-6, ML-SIM-6R, mcSIM-Wiener-6, Hessian-SIM-6 only if available, APD-SIM-6.
- SSR-SIM-9F must not enter the matched DMD-6F panel.
- Extract profiles from native/harmonized float arrays, never display16.
- Do not tune brightness separately by method.
- The microtubules Cell 046 image selection was frozen before inference in selection_receipt.json.
""",encoding="utf-8")
    evidence=REVISION_ROOT/"09_manuscript_evidence"; evidence.mkdir(parents=True,exist_ok=True)
    matched=read_csv(REVISION_ROOT/"01_matched_baselines/summary_mean_sd.csv")
    strict=read_csv(REVISION_ROOT/"05_strict_ablation/summary.csv")
    seed=read_csv(REVISION_ROOT/"03_seed_sensitivity/summary.csv")
    real=json.loads((REVISION_ROOT/"06_real_data_audit/sampling_summary.json").read_text(encoding="utf-8"))
    facts={
        "reviewer1_experiment_facts.txt":"30 frozen FOVs were compared under matched DMD-6F measurements. PhysMap-6 and APD-SIM-6 use identical six-frame hashes. PhysMap-9 is not described as an upper bound. Nominal rankings must be read directly from Table_strict_ablation_dmd6.csv.\n",
        "reviewer2_experiment_facts.txt":"The matched comparison includes source-verified ML-SIM-6R and mcSIM-Wiener-6, plus WF-6 and APD-SIM-6. All display arrays use a common [0,1] range without method-specific percentile remapping.\n",
        "reviewer3_comment4_facts.txt":"Principal APD-SIM-6 inference uses seed 20260812+i, one trajectory per FOV, with no best-of-N. Five prespecified seeds quantify sensitivity without reselecting the main result.\n",
        "reviewer3_comment5_facts.txt":"GT-referenced FRC uses 5% edge crop, mean centering, separable Tukey alpha 0.20, 100 radial annuli, no smoothing, and the first downward 1/7 crossing. It is not an independent experimental resolution estimate.\n",
        "reviewer3_comment6_facts.txt":"Supplementary Table S1 records source commit, environment, frame geometry, parameters, seed policy, and stopping rule. Matched DMD-6F and native 9F methods are separated; no local proxy is presented as a published method. Runtime values from incomparable proxy paths are removed.\n",
        "reviewer3_comment7_facts.txt":"The strict APD-SIM-6/DiffWS-6/PhysMap-6/WF-6 comparison uses identical DMD-6F measurements. PhysMap-6 uses the six-frame mean initialization; APD-SIM-6 uses DiffWS-6 initialization; both use Adam, 40 updates, lr 5e-3, float32, clipping [0,1], and no early stopping.\n",
        "reviewer3_comment8_facts.txt":f"File evidence recovers {real['verified_3f_manifests']} K3 and {real['verified_9f_manifests']} K9 manifests, but zero verified K6 acquisition manifests. Claims of 10 FOV and three exposure repeats per protocol are unsupported. File-level sampling is technical spatial sampling, not independent biological replication.\n",
    }
    for name,text in facts.items():(evidence/name).write_text(text,encoding="utf-8")
    write_csv(evidence/"numeric_replacements.csv",matched)
    (evidence/"claims_allowed.md").write_text("- under the matched DMD-6F protocol\n- among the source-verified implementations evaluated from identical measurements\n- the advantage was condition dependent\n- the nominal ranking differed from the robustness ranking\n",encoding="utf-8")
    (evidence/"claims_not_supported.md").write_text("- state-of-the-art\n- universally superior\n- outperforms all existing methods\n- PhysMap-9 is an upper bound\n- independent experimental resolution\n- ten real-data FOVs or three acquisition repeats per protocol\n",encoding="utf-8")
    (evidence/"manuscript_blocks_to_remove.txt").write_text("Remove legacy proxy runtime table and speed claims; remove any PhysMap-9 upper-bound wording; remove SSR-SIM-9F from the matched DMD-6F table; remove unsupported 10-FOV/three-repeat real-data claims.\n",encoding="utf-8")
    (REVISION_ROOT/"runtime_removal_recommendation.md").write_text("Reviewer 3 Comment 6 requests stopping rules, not runtime. Existing proxy runtime is not a formally comparable source-verified benchmark; delete it rather than retain incomparable values. No new runtime benchmark was run.\n",encoding="utf-8")
    (REVISION_ROOT/"runtime_latex_block_to_delete.txt").write_text("Delete the legacy runtime table and all associated speed-comparison sentences.\n",encoding="utf-8")


def main() -> int:
    strict=finalize_strict(); frc=finalize_gt_frc(); finalize_real_audit(); build_tables(); write_handoff_and_facts()
    atomic_json(REVISION_ROOT/"postprocess_status.json",{"status":"POSTPROCESS_COMPLETE_TABLE_COMPILE_PENDING","strict_rows":len(strict),"gt_frc_rows":len(frc),"paper_figures_created":False,"main_tex_modified":False})
    print("POSTPROCESS_COMPLETE_TABLE_COMPILE_PENDING")
    return 0


if __name__=="__main__": raise SystemExit(main())

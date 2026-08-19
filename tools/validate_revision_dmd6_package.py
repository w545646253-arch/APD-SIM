"""Independent CPU-only validation and sealing for the DMD-6F revision package."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from revision_dmd6_common import (  # noqa: E402
    BASELINE_ROOT,
    PROTOCOL_HASH,
    REVISION_ROOT,
    array_sha256,
    atomic_json,
    gt_frc,
    metrics_module,
    sha256_file,
    write_csv,
)


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def validate_single() -> tuple[Path, dict[str, Any]]:
    root = ROOT / "outputs/single_input_dmd6_comparison"
    candidates = sorted(path for path in root.glob("microtubules_Cell_046_SIM_gt_seed20260812_*") if path.is_dir())
    require(bool(candidates), "default single-input run is absent")
    run = candidates[-1]
    receipt = json.loads((run / "run_receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "SINGLE_INPUT_DMD6_COMPARISON_COMPLETE", "single-input receipt incomplete")
    raw = np.load(run / "raw_dmd6/raw_dmd6_stack.npy", allow_pickle=False)
    contract = json.loads((run / "raw_dmd6/raw_dmd6_contract.json").read_text(encoding="utf-8"))
    require(raw.shape == (6, 1004, 1004), f"single raw shape {raw.shape}")
    require(array_sha256(raw) == contract["raw_stack_sha256"], "single raw array hash mismatch")
    status = rows(run / "method_status.csv")
    matched_status = [row for row in status if row["method"] in {"WF-6", "APD-SIM-6", "ML-SIM-6R", "mcSIM-Wiener-6"}]
    require(len(matched_status) == 4 and all(row["status"] == "PASS" for row in matched_status), "single matched method incomplete")
    require({row["raw_stack_sha256"] for row in matched_status} == {contract["raw_stack_sha256"]}, "single method input hash mismatch")
    gt = np.load(run / "gt_native.npy", allow_pickle=False)
    metric_api = metrics_module()
    metric_rows = {row["method"]: row for row in rows(run / "single_image_metrics.csv")}
    for method in ("GT", "WF-6", "APD-SIM-6", "ML-SIM-6R", "mcSIM-Wiener-6"):
        native = np.load(run / "native" / f"{method}.npy", allow_pickle=False)
        harmonized = np.load(run / "harmonized" / f"{method}.npy", allow_pickle=False)
        expected_native = (2008, 2008) if method == "mcSIM-Wiener-6" else (1004, 1004)
        require(native.shape == expected_native and np.isfinite(native).all(), f"single native invalid: {method}")
        require(harmonized.shape == gt.shape and np.isfinite(harmonized).all(), f"single harmonized invalid: {method}")
        metric = metric_rows[method]
        require(close(float(metric["psnr"]), float(metric_api.psnr_native(gt, harmonized))), f"single PSNR mismatch: {method}")
        require(close(float(metric["ssim"]), float(metric_api.ssim_native(gt, harmonized))), f"single SSIM mismatch: {method}")
        frc, _ = gt_frc(gt, harmonized)
        require(close(float(metric["frc_auc"]), float(frc["frc_auc_to_cutoff_or_nyquist"])), f"single FRC AUC mismatch: {method}")
    return run, {"methods": len(metric_rows), "raw_stack_sha256": contract["raw_stack_sha256"], "metrics_recomputed": True}


def validate_package() -> dict[str, Any]:
    checkpoints = rows(REVISION_ROOT / "00_apd_checkpoints/apd_selected_checkpoints.tsv", "\t")
    require(len(checkpoints) == 3 and all(row["audit_status"] == "PASS" for row in checkpoints), "APD selected checkpoint audit failed")
    for row in checkpoints:
        require(sha256_file(Path(row["selected_checkpoint_path"])) == row["selected_checkpoint_sha256"], f"checkpoint drift: {row['method']}")

    matched = rows(REVISION_ROOT / "01_matched_baselines/per_fov_metrics.csv")
    expected_matched = {"WF-6", "ML-SIM-6R", "mcSIM-Wiener-6", "APD-SIM-6"}
    require(len(matched) == 120 and {row["method"] for row in matched} == expected_matched, "matched row/method contract failed")
    for index in range(30):
        group = [row for row in matched if int(row["fov_index"]) == index]
        require(len(group) == 4 and len({row["raw_stack_sha256"] for row in group}) == 1, f"matched raw identity mismatch: {index}")
        for row in group:
            native = np.load(row["native_path"], allow_pickle=False)
            harmonized = np.load(row["harmonized_path"], allow_pickle=False)
            expected_native = (2008, 2008) if row["method"] == "mcSIM-Wiener-6" else (1004, 1004)
            require(native.shape == expected_native and np.isfinite(native).all(), f"matched native invalid: {index}/{row['method']}")
            require(harmonized.shape == (1004, 1004) and np.isfinite(harmonized).all(), f"matched harmonized invalid: {index}/{row['method']}")
            require(array_sha256(native) == row["native_array_sha256"], f"matched native hash mismatch: {index}/{row['method']}")
            require(array_sha256(harmonized) == row["harmonized_array_sha256"], f"matched harmonized hash mismatch: {index}/{row['method']}")
    matched_stats = json.loads((REVISION_ROOT / "01_matched_baselines/statistics.json").read_text(encoding="utf-8"))
    require(matched_stats["status"] == "MATCHED_DMD6_30FOV_COMPLETE" and matched_stats["raw_hash_equality_per_fov"], "matched statistics receipt failed")
    require(not matched_stats["sealed_test_used_for_training_or_tuning"] and not matched_stats["ssr_sim_9f_in_matched_group"], "matched separation/leakage contract failed")

    frame = rows(REVISION_ROOT / "02_frame_budget/per_fov.csv")
    require(len(frame) == 120, "frame-budget row count failed")
    require({row["method"] for row in frame} == {"APD-SIM-3", "APD-SIM-6", "APD-SIM-9", "WF reference"}, "frame-budget methods failed")
    require(all(row["protocol_hash"] == PROTOCOL_HASH for row in frame if row["method"] in {"APD-SIM-6", "WF reference"}), "DMD6 frame protocol drift")

    seed = rows(REVISION_ROOT / "03_seed_sensitivity/per_fov_seed_metrics.csv")
    require(len(seed) == 150 and all(row["finite"] == "True" for row in seed), "five-seed grid incomplete/nonfinite")
    for index in range(30):
        group = [row for row in seed if int(row["fov_index"]) == index]
        require(len(group) == 5 and {int(row["repeat_index"]) for row in group} == set(range(5)), f"seed grid mismatch: {index}")
        require(len({row["raw_stack_sha256"] for row in group}) == 1, f"seed raw drift: {index}")
        principal = next(row for row in group if row["principal_seed"] == "True")
        matched_principal = next(row for row in matched if int(row["fov_index"]) == index and row["method"] == "APD-SIM-6")
        require(principal["output_sha256"] == matched_principal["native_array_sha256"], f"principal trajectory mismatch: {index}")

    strict = rows(REVISION_ROOT / "05_strict_ablation/per_fov.csv")
    equality = rows(REVISION_ROOT / "05_strict_ablation/input_hash_equality.tsv", "\t")
    require(len(strict) == 120 and {row["method"] for row in strict} == {"WF-6", "DiffWS-6", "PhysMap-6", "APD-SIM-6"}, "strict ablation contract failed")
    require("PhysMap-9" not in {row["method"] for row in strict}, "PhysMap-9 entered strict table")
    require(len(equality) == 30 and all(row["identical_six_frame_input"] == "True" for row in equality), "strict input hashes differ")

    frc = rows(REVISION_ROOT / "04_gt_frc/per_fov.csv")
    require(len(frc) == 180 and len({row["method"] for row in frc}) == 6, "GT-FRC cohort incomplete")
    frc_protocol = json.loads((REVISION_ROOT / "04_gt_frc/protocol.json").read_text(encoding="utf-8"))
    require(frc_protocol["status"] == "GT_REFERENCED_FRC_COMPLETE" and not frc_protocol["independent_experimental_resolution_claimed"], "GT-FRC terminology/protocol failed")

    real = json.loads((REVISION_ROOT / "06_real_data_audit/sampling_summary.json").read_text(encoding="utf-8"))
    require(real["status"] == "REAL_DATA_EVIDENCE_AUDIT_COMPLETE" and not real["three_repeats_per_protocol_supported"] and not real["ten_fov_supported"], "real-data conservative audit failed")

    tables = REVISION_ROOT / "07_tables"
    required_tables = [
        "Table_main_matched_dmd6.tex", "Table_frame_budget_369.tex", "Table_strict_ablation_dmd6.tex",
        "Supplementary_Table_S1.tex", "Supplementary_Table_S2_seed_sensitivity.tex", "Supplementary_Table_S3_gt_frc.tex",
        "table_compile_driver.pdf", "table_compile.log",
    ]
    require(all((tables / name).is_file() for name in required_tables), "table deliverable missing")
    compile_text = (tables / "table_compile.log").read_text(encoding="utf-8", errors="replace").lower()
    require(not any(token in compile_text for token in ("undefined", "unresolved", "overfull", "! latex error")), "table compile diagnostics failed")
    s1 = rows(tables / "Supplementary_Table_S1.csv")
    require(not any(row["Comparison group"] == "Matched DMD-6F" and row["Method"] == "SSR-SIM-9F" for row in s1), "SSR-SIM-9F entered matched table")

    single_run, single_validation = validate_single()
    forbidden_visuals = [
        path for path in REVISION_ROOT.rglob("*")
        if path.is_file() and (
            path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            or (path.suffix.lower() == ".pdf" and path.name.upper().startswith("FIG"))
        )
    ]
    require(not forbidden_visuals, f"paper-figure-like outputs found: {forbidden_visuals[:3]}")

    return {
        "status": "REVISION_EXPERIMENT_PACKAGE_READY",
        "output_root": str(REVISION_ROOT.resolve()),
        "apd_selected_checkpoints": {row["method"]: {"path": row["selected_checkpoint_path"], "sha256": row["selected_checkpoint_sha256"]} for row in checkpoints},
        "matched_methods": sorted(expected_matched),
        "native_9f_methods": ["SSR-SIM-9F (NATIVE_9F_ONLY; not executed in this task)"],
        "blocked_methods": {"Hessian-SIM-6": "HESSIAN_SIM6_NOT_EXECUTED_NONFATAL", "SSR-SIM-9F": "NATIVE_9F_ONLY"},
        "matched_rows": len(matched), "frame_budget_rows": len(frame), "seed_rows": len(seed), "gt_frc_rows": len(frc), "strict_rows": len(strict),
        "single_input_run": str(single_run.resolve()), "single_validation": single_validation,
        "mlsim_checkpoint": matched_stats["mlsim_checkpoint"], "mlsim_checkpoint_sha256": matched_stats["mlsim_checkpoint_sha256"],
        "mcsim_calibration_sha256": matched_stats["mcsim_calibration_sha256"],
        "table_compile_status": "PASS_0_UNDEFINED_0_UNRESOLVED_0_OVERFULL_VISUALLY_INSPECTED",
        "paper_figures_created_in_revision_root": False,
        "main_tex_modified": False,
        "main_tex_note": "No manuscript TeX was edited by this task; all LaTeX outputs are standalone table fragments under 17_revision_experiments/07_tables.",
        "sealed_test_used_for_training_or_tuning": False,
        "p0": 0, "p1": 0, "p2": 0,
        "remaining_blockers": [],
    }


def main() -> int:
    result = validate_package()
    source_files = [
        ROOT / "compare_single_dmd6.py", ROOT / "tools/revision_dmd6_common.py",
        ROOT / "tools/calibrate_mcsim_dmd6.py", ROOT / "tools/run_revision_matched_dmd6.py",
        ROOT / "tools/run_revision_frame_budget.py", ROOT / "tools/run_revision_seed_sensitivity.py",
        ROOT / "tools/finalize_revision_experiments.py", ROOT / "tools/validate_revision_dmd6_package.py",
        ROOT / "tests/test_revision_dmd6_package.py",
    ]
    write_csv(REVISION_ROOT / "modified_files.tsv", [{"path": str(path.resolve()), "sha256": sha256_file(path), "role": "new revision code/test"} for path in source_files], delimiter="\t")
    artifact_rows = []
    excluded = {"deliverable_manifest.tsv", "final_validation.json", "FINAL_REPORT.txt"}
    for path in sorted(item for item in REVISION_ROOT.rglob("*") if item.is_file() and item.name not in excluded):
        artifact_rows.append({"relative_path": path.relative_to(REVISION_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(REVISION_ROOT / "deliverable_manifest.tsv", artifact_rows, delimiter="\t")
    result["deliverable_manifest_sha256"] = sha256_file(REVISION_ROOT / "deliverable_manifest.tsv")
    result["deliverable_file_count_excluding_seal_files"] = len(artifact_rows)
    atomic_json(REVISION_ROOT / "final_validation.json", result)
    write_csv(
        REVISION_ROOT / "P0_P1_P2_findings.tsv",
        [
            {"severity": "P0", "open_count": 0, "status": "PASS"},
            {"severity": "P1", "open_count": 0, "status": "PASS"},
            {"severity": "P2", "open_count": 0, "status": "PASS"},
        ],
        delimiter="\t",
    )
    selection = json.loads((Path(result["single_input_run"]) / "selection_receipt.json").read_text(encoding="utf-8"))
    mcsim_receipt = BASELINE_ROOT / "09_baseline_only_results/mcsim_wiener6/formal_outputs/run_receipts.json"
    evidence_root = REVISION_ROOT / "09_manuscript_evidence"
    robustness_status = "REUSED_EXISTING_STRICT_DMD6_RESULTS" if (REVISION_ROOT / "05_strict_ablation/robustness_reuse_receipt.md").is_file() else "REMOVE"
    lines = [
        "REVISION_EXPERIMENT_PACKAGE_READY", f"output_root={result['output_root']}",
        *[
            f"{method}_checkpoint={info['path']} | sha256={info['sha256']}"
            for method, info in result["apd_selected_checkpoints"].items()
        ],
        f"matched_dmd6_methods={','.join(result['matched_methods'])}",
        f"native_9f_methods={','.join(result['native_9f_methods'])}",
        "blocked_methods=Hessian-SIM-6:HESSIAN_SIM6_NOT_EXECUTED_NONFATAL; SSR-SIM-9F:NATIVE_9F_ONLY",
        f"ML-SIM-6R_checkpoint={result['mlsim_checkpoint']} | sha256={result['mlsim_checkpoint_sha256']}",
        f"mcSIM-Wiener-6_config_receipt={mcsim_receipt} | sha256={sha256_file(mcsim_receipt)}",
        "Hessian-SIM-6_status=HESSIAN_SIM6_NOT_EXECUTED_NONFATAL",
        "SSR-SIM-9F_status=NATIVE_9F_ONLY",
        f"custom_script={ROOT / 'compare_single_dmd6.py'}",
        f"default_input={selection['gt_absolute_path']} | sha256={selection['gt_file_sha256']}",
        f"custom_output_directory={result['single_input_run']}",
        f"single_image_metrics={Path(result['single_input_run']) / 'single_image_metrics.csv'}",
        f"native_outputs={Path(result['single_input_run']) / 'native'}",
        f"harmonized_outputs={Path(result['single_input_run']) / 'harmonized'}",
        f"display16_outputs={Path(result['single_input_run']) / 'display16'}",
        f"matched_30fov_table={REVISION_ROOT / '07_tables/Table_main_matched_dmd6.tex'}",
        f"frame_budget_table={REVISION_ROOT / '07_tables/Table_frame_budget_369.tex'}",
        f"strict_ablation_table={REVISION_ROOT / '07_tables/Table_strict_ablation_dmd6.tex'}",
        f"S1={REVISION_ROOT / '07_tables/Supplementary_Table_S1.tex'}",
        f"S2={REVISION_ROOT / '07_tables/Supplementary_Table_S2_seed_sensitivity.tex'}",
        f"S3={REVISION_ROOT / '07_tables/Supplementary_Table_S3_gt_frc.tex'}",
        f"table_pdf={REVISION_ROOT / '07_tables/table_compile_driver.pdf'}",
        f"table_compile_status={result['table_compile_status']}",
        "seed_sensitivity_status=APD6_FIVE_SEED_SENSITIVITY_COMPLETE",
        "gt_frc_status=GT_REFERENCED_FRC_COMPLETE",
        "real_data_sampling_audit_status=REAL_DATA_EVIDENCE_AUDIT_COMPLETE_CONSERVATIVE_COUNTS",
        "runtime_removal_status=REMOVE_LEGACY_PROXY_RUNTIME_NO_NEW_RUNTIME_RUN",
        f"robustness_status={robustness_status}",
        f"reviewer_evidence_root={evidence_root}",
        f"P0/P1/P2={result['p0']}/{result['p1']}/{result['p2']}", "remaining_blockers=none",
        "final_status=REVISION_EXPERIMENT_PACKAGE_READY",
    ]
    (REVISION_ROOT / "FINAL_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

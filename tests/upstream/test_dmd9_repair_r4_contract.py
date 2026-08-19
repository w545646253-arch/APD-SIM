from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from unisim.protocol_runtime import require_protocol


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "DMD9_REPAIR_R4_20260817_104841"


def read_json(relative: str) -> dict:
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str, delimiter: str = ",") -> list[dict[str, str]]:
    with (OUT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_dmd9_raw_order() -> None:
    expected = ("X0", "X120", "X240", "Y0", "Y120", "Y240", "Z0", "Z120", "Z240")
    assert tuple(require_protocol("DMD_9F_3O3P").raw_frame_order) == expected
    assert tuple(read_json("01_protocol_and_dose/protocol_contract.json")["protocols"]["DMD_9F_3O3P"]["raw_order"]) == expected


def test_dmd9_validity_mask() -> None:
    expected = (1,) * 9 + (0,) * 6
    assert tuple(require_protocol("DMD_9F_3O3P").validity_mask) == expected


def test_nine_channel_sensitivity() -> None:
    audit = read_json("02_conditioning_audit/conditioning_decision.json")
    assert audit["all_nine_sensitivity_pass"] is True
    assert audit["third_orientation_measurement_weights_active"] is True
    assert audit["third_orientation_gradients_active"] is True


def test_frame_permutation_rejection() -> None:
    assert read_json("02_conditioning_audit/conditioning_decision.json")["frame_permutation_rejected"] is True


def test_checkpoint_protocol_mismatch_rejection() -> None:
    audit = read_json("02_conditioning_audit/conditioning_decision.json")
    assert audit["dmd9_checkpoint_under_dmd6_protocol_rejected"] is True
    assert audit["checkpoint_protocol_contract_bound"] is False


def test_per_frame_total_dose_contract() -> None:
    audit = read_json("01_protocol_and_dose/protocol_contract.json")
    assert audit["dose_policy"] == "PER_FRAME_DOSE_FIXED"
    assert audit["per_stack_normalization"] is False
    assert audit["stack_division_by_frame_count"] is False
    assert audit["ratio_gate_1p47_to_1p53"] is True


def test_patch_full_diagnostic_and_tiled_repair() -> None:
    audit = read_json("03_inference_audit/inference_decision.json")
    assert audit["direct_patch_full_threshold_pass"] is False
    assert audit["root_cause_classification"] == "DMD9_FULL_SIZE_GROUPNORM_INFERENCE_MISMATCH"
    assert audit["repair"] == "DETERMINISTIC_320_TILE_160_CORE_SINGLE_SPATIAL_NOISE_FIELD"
    assert audit["retraining_required_from_inference_audit"] is False


def test_stage1_ema_raw_comparison() -> None:
    rows = read_csv("04_checkpoint_sweep/validation_sweep.csv")
    assert len(rows) == 90
    assert {row["weight_branch"] for row in rows} == {"model", "ema"}
    assert all(int(row["sealed_test_access_count"]) == 0 for row in rows)


def test_physmap_6f_9f_comparison() -> None:
    audit = read_json("03_inference_audit/inference_decision.json")
    assert audit["metrics"]["phys9"]["psnr"] > audit["metrics"]["phys6"]["psnr"]
    assert audit["forward_contract_failure"] is False


def test_stage2_trajectory() -> None:
    rows = read_csv("03_inference_audit/stage_trajectory.csv")
    assert len(rows) == 240
    stage2 = [row for row in rows if row["method"] == "Stage2"]
    assert {int(row["updates"]) for row in stage2} == {1, 5, 10, 20, 40, 80}
    gate = read_json("07_validation_gate/validation_gate.json")
    assert gate["summary"][2]["mean_psnr"] > gate["summary"][1]["mean_psnr"]


def test_dmd6_selection_rule_recovery() -> None:
    selection = read_json("04_checkpoint_sweep/selected_existing_checkpoint.json")
    assert selection["selection_rule"] == "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"
    assert selection["selected_iteration"] == 88000


def test_validation_only_checkpoint_sweep() -> None:
    selection = read_json("04_checkpoint_sweep/selected_existing_checkpoint.json")
    assert selection["logical_sweep_rows"] == 90
    assert selection["unique_state_evaluations"] == 4
    assert selection["selection_was_validation_only"] is True
    assert selection["sealed_test_access_count"] == 0


def test_sealed_test_no_premature_access_and_one_run() -> None:
    gate = read_json("07_validation_gate/validation_gate.json")
    receipt = read_json("08_final_test/formal_test_receipt.json")
    assert gate["test_access_count"] == 0
    assert receipt["sealed_test_premature_access_count"] == 0
    assert receipt["formal_test_run_count"] == 1
    assert receipt["protocol_forward_call_count"] == 90


def test_r4_committed_update_budget_is_not_applicable_without_retraining() -> None:
    decision = read_json("05_repair_decision/decision.json")
    training = read_json("06_r4_training/NOT_RUN.json")
    assert decision["decision"] == "DMD9_INFERENCE_FIX_ONLY"
    assert decision["retraining_required"] is False
    assert training["training_execution_count"] == 0
    assert training["committed_optimizer_updates"] == 0


def test_archive_sha_and_relocation() -> None:
    receipt = read_json("10_archive/archive_receipt.json")
    relocation = read_json("10_archive/relocation_map.json")
    rows = read_csv("10_archive/archive_manifest.tsv", delimiter="\t")
    assert receipt["loss_count"] == 0
    assert len(relocation["relocations"]) > 0
    by_root = {row["original_path"]: Path(row["archive_path"]) for row in relocation["relocations"]}
    for row in rows:
        destination = by_root[row["original_root"]]
        path = destination if destination.is_file() else destination / row["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["size_bytes"])
        assert sha256_file(path) == row["sha256"]


@pytest.mark.parametrize(
    "categories",
    [
        {"APD3_CHECKPOINT", "APD6_CHECKPOINT"},
        {"DMD6_EXTERNAL_BASELINES", "DMD6_EXTERNAL_FAIRSIM", "DMD6_RAW_BUNDLE"},
        {"DMD6_ABLATION"},
        {"DMD6_ROBUSTNESS_COMBINED", "DMD6_ROBUSTNESS_HOLM", "DMD6_ROBUSTNESS_LEVELWISE", "DMD6_ROBUSTNESS_SOURCE_OUT", "DMD6_ROBUSTNESS_STRICT"},
    ],
)
def test_protected_hashes_unchanged(categories: set[str]) -> None:
    rows = read_csv("00_preflight/protected_hashes.tsv", delimiter="\t")
    selected = [row for row in rows if row["category"] in categories]
    assert selected
    for row in selected:
        path = Path(row["path"])
        assert path.is_file()
        assert path.stat().st_size == int(row["size_bytes"])
        assert sha256_file(path) == row["sha256"]


def test_active_checkpoint_directory_contract() -> None:
    root = ROOT / "checkpoints" / "apd_dmd_geometry_r4" / "dmd9"
    assert {path.name for path in root.iterdir()} == {
        "best.pt", "latest.pt", "final.pt", "completion_receipt.json", "checkpoint_selection.json",
    }
    assert sha256_file(root / "best.pt") == "62831cc9798c9d005fdbf56b343928cc592646b6e70a16f58399b6da0d01b63e"

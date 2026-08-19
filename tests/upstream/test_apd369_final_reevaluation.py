from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools.apd369_final_contract import DEFAULT_OUTPUT_ROOT, PLANS, load_frozen_contract, sha256_file
from tools import revision_dmd6_common as common


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict[str, str]]:
    path = DEFAULT_OUTPUT_ROOT / "04_metrics" / "per_fov_metrics.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_checkpoint_and_manifest_contracts() -> None:
    checkpoints, manifest = load_frozen_contract(DEFAULT_OUTPUT_ROOT)
    assert checkpoints["status"] == "FINAL_APD369_CHECKPOINTS_FROZEN"
    assert set(checkpoints["checkpoints"]) == {plan.method for plan in PLANS}
    assert manifest["count"] == 30
    assert manifest["class_counts"] == {"CCP": 10, "ER": 10, "MT": 10}
    assert all(value == 0 for split in manifest["train_validation_overlap"].values() for value in split.values())


def test_checkpoint_protocol_binding_and_new_dmd9() -> None:
    checkpoints, _manifest = load_frozen_contract(DEFAULT_OUTPUT_ROOT)
    for plan in PLANS:
        row = checkpoints["checkpoints"][plan.method]
        assert row["protocol_id"] == plan.protocol_id
        assert row["protocol_hash"] == plan.protocol_hash
        assert tuple(row["raw_order"]) == plan.raw_order
        assert tuple(row["validity_mask"]) == plan.validity_mask
        assert sha256_file(Path(row["checkpoint_path"])) == row["checkpoint_sha256"]
    dmd9 = checkpoints["checkpoints"]["APD-SIM-9"]
    assert "apd_dmd_geometry_r3" in dmd9["checkpoint_path"]
    assert dmd9["selected_validation_iteration"] == 5000


def test_formal_entrypoint_has_no_forbidden_execution() -> None:
    source = (ROOT / "evaluate_apd369_protocols_final.py").read_text(encoding="utf-8")
    assert "import test_369" not in source and "from test_369" not in source
    assert "train3.py" not in source and "train6.py" not in source and "train9.py" not in source
    assert '"best_of_n_count": 0' in source
    assert "external baseline" in source.lower()
    assert "forward_protocol_sim_2d" in source


def test_completed_grid_raw_independence_and_seeds() -> None:
    rows = _rows()
    if len(rows) != 90:
        return
    assert {(row["method"], int(row["order"])) for row in rows} == {(plan.method, order) for plan in PLANS for order in range(30)}
    for order in range(30):
        group = [row for row in rows if int(row["order"]) == order]
        assert len({row["generation_call_uuid"] for row in group}) == 3
        assert len({row["diffusion_seed"] for row in group}) == 1
        assert int(group[0]["diffusion_seed"]) == 20260812 + order
        assert len({row["raw_npz_path"] for row in group}) == 3
    assert all(row["finite"].lower() == "true" for row in rows)


def test_dmd6_raw_ledger_and_metric_recompute() -> None:
    rows = _rows()
    if len(rows) != 90:
        return
    _checkpoints, manifest = load_frozen_contract(DEFAULT_OUTPUT_ROOT)
    for row in rows:
        if row["method"] == "APD-SIM-6":
            assert row["raw_stack_sha256"] == manifest["samples"][int(row["order"])]["formal_dmd6_raw_sha256"]
    row = rows[0]
    sample = manifest["samples"][int(row["order"])]
    import tifffile
    from unisim.revision_r1 import frame_budget_r1c2 as fb
    gt = fb.normalize_image(tifffile.imread(sample["absolute_path"]))
    prediction = np.load(row["prediction_harmonized_path"], allow_pickle=False)
    metrics = common.metrics_module()
    assert np.isclose(float(row["psnr"]), float(metrics.psnr_native(gt, prediction)), rtol=0, atol=1e-10)
    assert np.isclose(float(row["ssim"]), float(metrics.ssim_native(gt, prediction)), rtol=0, atol=1e-10)


def test_same_seed_raw_repeatability_for_all_protocols() -> None:
    import torch
    import tifffile
    import evaluate_apd369_protocols_final as evaluator
    from unisim.revision_r1 import frame_budget_r1c2 as fb

    _checkpoints, manifest = load_frozen_contract(DEFAULT_OUTPUT_ROOT)
    sample = manifest["samples"][0]
    gt = fb.normalize_image(tifffile.imread(sample["absolute_path"]))
    tensor = torch.from_numpy(gt)[None, None]
    module = evaluator._load_frozen_forward()
    for plan in PLANS:
        config = evaluator.read_json(plan.config_path)
        sim_config = evaluator._sim_config(module, config)
        first, _meta1, _theta1 = evaluator._generate_raw(module, tensor, sim_config, plan, int(sample["measurement_seed"]))
        second, _meta2, _theta2 = evaluator._generate_raw(module, tensor, sim_config, plan, int(sample["measurement_seed"]))
        assert torch.equal(first, second)
        stored = np.load(DEFAULT_OUTPUT_ROOT / "02_protocol_raw_bundles" / plan.method / f"000_{sample['sample_id']}.npz", allow_pickle=False)["raw_stack"]
        assert common.array_sha256(first[0].numpy()) == common.array_sha256(stored)


def test_frc_censoring_contract() -> None:
    reference = np.random.default_rng(20260817).normal(size=(256, 256)).astype(np.float32)
    meta, curve = common.gt_frc(reference, reference)
    assert meta["threshold"] == 1.0 / 7.0
    assert meta["right_censored_at_nyquist"] is True
    assert meta["unresolved_no_crossing"] is False
    assert curve["frc"].shape == (100,)


def test_real_data_pending_is_honest() -> None:
    receipt = json.loads((DEFAULT_OUTPUT_ROOT / "06_real_dmd369" / "real_dmd369_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "APD369_NUMERICAL_READY_REAL_DMD_PENDING"
    assert receipt["real_inference_execution_count"] == 0
    assert receipt["common_nine_frame_subsampling_count"] == 0
    assert receipt["experimental_psnr_ssim_claim_count"] == 0


def test_single_source_and_supplement_contracts_when_finalized() -> None:
    figure_receipt = DEFAULT_OUTPUT_ROOT / "05_figure3_source" / "figure_source_receipt.json"
    supplement = DEFAULT_OUTPUT_ROOT / "07_supplementary_minimal" / "APD_SIM_Supplementary_Minimal.tex"
    if not figure_receipt.is_file() or not supplement.is_file():
        return
    receipt = json.loads(figure_receipt.read_text(encoding="utf-8"))
    assert receipt["independently_generated_raw_bundles"] is True
    assert receipt["common_nine_frame_subsampling_count"] == 0
    text = supplement.read_text(encoding="utf-8")
    for required in ("Supplementary Note S1", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3"):
        assert required in text
    assert "fairSIM-6-native" in text
    assert "PhysMap-9" not in text

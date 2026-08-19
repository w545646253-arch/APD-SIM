from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from unisim.protocol_runtime import require_protocol
from unisim.revision_r1 import frame_budget_r1c2 as r1c2
from unisim.sim_forward_2d import embed_raw_to_slots_2d, masked_poisson_gaussian_likelihood_2d


def test_unique_right_click_entry_has_no_argument_parser() -> None:
    entry = r1c2.ROOT / "R1C2_run_frame_budget_30fov.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    assert "argparse" not in entry.read_text(encoding="utf-8")
    assert any(isinstance(node, ast.If) for node in ast.walk(tree))


@pytest.mark.parametrize("plan", r1c2.PLANS)
def test_checkpoint_receipt_and_validation_selected_best_identity(plan: r1c2.ProtocolPlan) -> None:
    digest = hashlib.sha256(plan.checkpoint.read_bytes()).hexdigest()
    receipt = json.loads(plan.receipt.read_text(encoding="utf-8"))
    best, count = r1c2._validation_best(plan.history)
    assert count == 50
    assert receipt["completion_status"] == "FORMAL_TRAINING_COMPLETE"
    assert receipt["test_data_used_for_selection"] is False
    assert receipt["checkpoint_sha256"] == digest
    assert int(float(best["global_step"])) == int(receipt["metrics"]["global_step"])


@pytest.mark.parametrize("plan", r1c2.PLANS)
def test_direct_protocol_order_mask_and_slot_mapping(plan: r1c2.ProtocolPlan) -> None:
    spec = require_protocol(plan.protocol_id)
    assert tuple(spec.raw_frame_order) == plan.raw_order
    assert tuple(spec.validity_mask) == plan.validity_mask
    assert tuple(spec.raw_to_slot_mapping) == tuple(range(spec.frame_count))
    assert spec.phases_per_orientation == 3


def test_refinement_contract_is_exactly_frozen() -> None:
    config = r1c2.REFINEMENT_CONFIG
    assert config.optimizer == "torch.optim.Adam"
    assert config.updates == 40
    assert config.learning_rate == 0.005
    assert config.lambda_prior == 0.0
    assert config.clip_min == 0.0 and config.clip_max == 1.0
    assert config.dtype == "float32"
    assert config.stopping_rule == "exactly_40_updates_no_early_stopping"


@pytest.mark.parametrize("plan", r1c2.PLANS)
def test_invalid_slots_do_not_enter_masked_likelihood(plan: r1c2.ProtocolPlan) -> None:
    raw = torch.full((1, len(plan.raw_order), 8, 8), 0.2, dtype=torch.float32)
    observed, mask = embed_raw_to_slots_2d(raw, plan.protocol_id)
    prediction = observed.clone()
    invalid = mask == 0
    prediction[invalid.expand_as(prediction)] = 1000.0
    value = masked_poisson_gaussian_likelihood_2d(
        observed, prediction, plan.protocol_id,
        photon_scale=torch.tensor([8000.0]), read_noise_e=torch.tensor([1.6]), reduce="mean",
    )
    baseline = masked_poisson_gaussian_likelihood_2d(
        observed, observed, plan.protocol_id,
        photon_scale=torch.tensor([8000.0]), read_noise_e=torch.tensor([1.6]), reduce="mean",
    )
    assert torch.equal(value, baseline)


def test_native_metric_and_frc_censor_contract() -> None:
    metrics = r1c2._metrics_module()
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:48, 16:48] = 1.0
    assert np.isinf(metrics.psnr_native(image, image))
    frc, _ = metrics.reference_frc_1over7(image, image, min_samples_per_bin=8)
    assert frc["dc_excluded"] is True
    assert frc["cutoff_cycles_per_pixel"] is None
    assert frc["right_censored_at_nyquist"] is True
    assert frc["cutoff_derived_spatial_period_px"] is None


def test_common_seeds_are_protocol_independent_and_not_retrospective() -> None:
    assert r1c2.COMMON_MEASUREMENT_SEED_BASE > 0
    assert r1c2.COMMON_DIFFUSION_SEED_BASE > 0
    source = Path(r1c2.__file__).read_text(encoding="utf-8")
    assert "forward_protocol_sim_2d" in source
    assert "retrospective" in source.lower()
    assert "subset" not in source[source.index("def run_inference"):source.index("def _read_csv")].lower()


def test_fixed_representative_roi_and_display_are_not_performance_selected() -> None:
    assert r1c2.REPRESENTATIVE_SAMPLE_ORDER == 19
    assert r1c2.REPRESENTATIVE_ROI == {"y": 342, "x": 342, "height": 320, "width": 320}


def test_shared_refinement_has_no_gt_input_and_is_used_by_all_protocols() -> None:
    from unisim.revision_r1.physmap6_core import masked_refine

    assert "gt" not in inspect.signature(masked_refine).parameters
    source = inspect.getsource(r1c2.refine_protocol)
    assert "masked_refine(" in source
    assert "torch.optim" not in source
    stage1_source = inspect.getsource(r1c2.stage1_reconstruct)
    loader_source = inspect.getsource(r1c2.load_ema_model)
    assert "stage1_reconstruct_registered" in stage1_source
    assert "load_stage1_registered" in loader_source


@pytest.mark.parametrize("protocol_id,frame_count", [
    ("DMD_3F_1O3P", 3), ("DMD_6F_2O3P", 6), ("DMD_9F_3O3P", 9),
])
def test_registered_protocols_reach_same_shared_refinement_validation(
    protocol_id: str, frame_count: int
) -> None:
    from unisim.revision_r1.physmap6_core import masked_refine

    signature = inspect.signature(masked_refine)
    assert "refinement_config" in signature.parameters
    spec = require_protocol(protocol_id)
    assert spec.frame_count == frame_count


@pytest.mark.parametrize("plan", r1c2.PLANS)
def test_shared_refinement_executes_registered_protocol_on_cpu(plan: r1c2.ProtocolPlan) -> None:
    config = r1c2.read_json(plan.config)
    sim_config = r1c2._config_for_sim(config)
    gt = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    theta = r1c2.nominal_theta_2d(sim_config, torch.device("cpu"))
    raw, _ = r1c2.forward_protocol_clean_2d(
        gt, sim_config, plan.protocol_id, theta=dict(theta), randomize=False
    )
    initial = raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    result, _runtime, objective, nrmse = r1c2.refine_protocol(
        initial, raw, plan.protocol_id, sim_config, theta
    )
    assert result.shape == initial.shape
    assert torch.isfinite(result).all()
    assert np.isfinite(objective) and np.isfinite(nrmse)


def test_existing_r1c3_tables_are_independently_validated_before_copy(tmp_path: Path) -> None:
    receipt = r1c2.export_validated_r1c3_tables(tmp_path)
    assert receipt["status"] == "PASS"
    assert receipt["robustness_rows"] == 4320
    assert receipt["table2_rows_independently_recomputed"] == 24
    assert receipt["runtime_expected_values_exact_match"] is True
    assert receipt["physmap9_present"] is False
    assert (tmp_path / "R1C3_ROBUSTNESS_TABLE_DIRECT.tex").is_file()
    assert (tmp_path / "R1C3_RUNTIME_TABLE_DIRECT.tex").is_file()


def test_contention_monitor_is_present_in_formal_inference_path() -> None:
    source = inspect.getsource(r1c2.run_inference)
    assert "CudaContentionMonitor" in source
    assert "monitor.checkpoint()" in source
    assert "monitor.stop_and_validate()" in source


def test_direct_latex_files_are_complete_and_do_not_use_input(tmp_path: Path) -> None:
    descriptive = {"n": 30, "mean": 1.0, "sd": 0.1, "median": 1.0, "q1": 0.9, "q3": 1.1, "iqr": 0.2, "min": 0.8, "max": 1.2}
    groups = {}
    for protocol in ("DMD-3F", "DMD-6F", "DMD-9F"):
        groups[f"{protocol}/APD-SIM"] = {
            "psnr": dict(descriptive), "ssim": dict(descriptive),
            "frc_spatial_period_px": {"status_counts": {"CUTOFF": 30, "RIGHT_CENSORED": 0, "UNRESOLVED": 0}},
        }
    paired = []
    for contrast in ("DMD-3F_minus_DMD-6F", "DMD-6F_minus_DMD-9F"):
        for metric in ("PSNR", "SSIM"):
            paired.append({"contrast": contrast, "metric": metric, "estimate": 0.2, "ci_low": 0.1, "ci_high": 0.3})
    hashes = r1c2.generate_latex(tmp_path, {"groups": groups, "paired_differences": paired})
    assert len(hashes) == 8
    figure = (tmp_path / "R1C2_FIG3_DIRECT.tex").read_text(encoding="utf-8")
    assert "\\begin{figure*}" in figure and "\\includegraphics" in figure
    caption = (tmp_path / "FIG3_FRAME_BUDGET_30FOV_CAPTION.tex").read_text(encoding="utf-8")
    assert "Direct DMD acquisition-budget comparison" in caption
    for path in tmp_path.glob("*.tex"):
        text = path.read_text(encoding="utf-8")
        assert "\\input" not in text
        assert not any(token in text for token in ("TODO", "TBD", "PLACEHOLDER", "[INSERT]"))

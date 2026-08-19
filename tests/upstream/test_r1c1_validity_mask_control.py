from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from unisim.revision_r1 import validity_mask_control as control


class _RecordingZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, condition: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
        self.inputs.append(condition.detach().clone())
        return torch.zeros_like(condition[:, :1])


class _SeedScheduler:
    def __init__(self) -> None:
        self.alpha_bar = torch.ones(601, dtype=torch.float32)

    def q_sample(
        self, image: torch.Tensor, _timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return image + noise * 0.01

    def predict_x0(
        self, image: torch.Tensor, _timestep: torch.Tensor, _epsilon: torch.Tensor
    ) -> torch.Tensor:
        return image


def test_masks_are_exact_and_only_logical_slots_6_to_8_differ() -> None:
    assert control.CORRECT_LOGICAL_MASK == (1, 1, 1, 1, 1, 1, 0, 0, 0)
    assert control.BLIND_LOGICAL_MASK == (1,) * 9
    assert control.CORRECT_MASK == (1, 1, 1, 1, 1, 1) + (0,) * 9
    assert control.BLIND_MASK == (1,) * 9 + (0,) * 6
    differing = tuple(
        index
        for index, values in enumerate(zip(control.CORRECT_MASK, control.BLIND_MASK))
        if values[0] != values[1]
    )
    assert differing == (6, 7, 8)
    assert control.CORRECT_MASK[9:] == control.BLIND_MASK[9:] == (0,) * 6


def test_stage1_pair_changes_only_validity_channels_and_reuses_noise() -> None:
    raw = torch.linspace(0.0, 1.0, 1 * 6 * 17 * 19, dtype=torch.float32).reshape(1, 6, 17, 19)
    model = _RecordingZeroModel()
    correct, blind, audit = control.stage1_mask_pair(raw, model, _SeedScheduler(), seed=718)
    assert correct.shape == blind.shape == (1, 1, 17, 19)
    assert len(model.inputs) == 160
    first_correct = model.inputs[0]
    first_blind = model.inputs[80]
    difference = first_correct != first_blind
    differing_channels = tuple(
        index for index in range(first_correct.shape[1]) if bool(difference[:, index].any())
    )
    # Model input is x (1), slotted conditioning (15), validity state (15).
    assert differing_channels == (22, 23, 24)
    assert torch.equal(first_correct[:, :22], first_blind[:, :22])
    assert torch.equal(first_correct[:, 25:], first_blind[:, 25:])
    assert audit["correct_slotted_conditioning_sha256"] == audit["maskblind_slotted_conditioning_sha256"]
    assert audit["correct_gaussian_sha256"] == audit["maskblind_gaussian_sha256"]
    assert audit["only_differing_mask_slots"] == [6, 7, 8]
    assert audit["padding_slots_9_14_invalid_both"] is True


def test_statistics_preserve_correct_minus_maskblind_sign_and_holm_family(tmp_path: Path) -> None:
    rows = []
    for order in range(30):
        offset = order / 1000.0
        rows.append({
            "Final_correct_PSNR": 31.0 + offset,
            "Final_maskblind_PSNR": 30.0 + offset,
            "Stage1_correct_PSNR": 29.5 + offset,
            "Stage1_maskblind_PSNR": 29.0 + offset,
            "Stage1_correct_SSIM": 0.85 + offset / 10,
            "Stage1_maskblind_SSIM": 0.84 + offset / 10,
            "Final_correct_SSIM": 0.90 + offset / 10,
            "Final_maskblind_SSIM": 0.88 + offset / 10,
            "Final_correct_observed_NRMSE": 0.10 + offset / 100,
            "Final_maskblind_observed_NRMSE": 0.12 + offset / 100,
        })
    _methods, effects, statistics = control.analyze_statistics(tmp_path, rows)
    by_endpoint = {row["endpoint"]: row for row in effects}
    assert by_endpoint["final_psnr"]["paired_mean_difference"] == pytest.approx(1.0)
    assert by_endpoint["stage1_psnr"]["paired_mean_difference"] == pytest.approx(0.5)
    assert by_endpoint["final_observed_nrmse"]["paired_mean_difference"] == pytest.approx(-0.02)
    assert statistics["holm"]["family_size"] == 5
    assert set(statistics["holm"]["adjusted_p_values"]) == {
        "final_psnr", "stage1_psnr", "stage1_ssim", "final_ssim", "final_observed_nrmse"
    }
    assert statistics["stage2_mask_effect_assessment"]["classification"] == "amplified"
    assert (tmp_path / "VALIDITY_MASK_CONTROL_METHOD_STATS.csv").is_file()
    assert (tmp_path / "VALIDITY_MASK_CONTROL_PAIRED_EFFECTS.csv").is_file()
    assert (tmp_path / "VALIDITY_MASK_CONTROL_STATISTICS.json").is_file()


def test_refinement_contract_is_one_frozen_object() -> None:
    config = control.REFINEMENT_CONFIG
    assert config.optimizer == "torch.optim.Adam"
    assert config.updates == 40
    assert config.learning_rate == pytest.approx(0.005)
    assert config.lambda_prior == 0.0
    assert config.dtype == "float32"
    assert (config.clip_min, config.clip_max) == (0.0, 1.0)
    assert config.stopping_rule == "exactly_40_updates_no_early_stopping"
    base = config.receipt()
    assert control.receipt_contains_config({**base, "protocol_id": "DMD_6F_2O3P"}, base)
    assert not control.receipt_contains_config({**base, "updates": 41}, base)


def test_reconstruction_path_has_no_gt_reader_or_cuda_exclusivity_gate() -> None:
    reconstruction_source = inspect.getsource(control.reconstruct_all)
    module_source = Path(control.__file__).read_text(encoding="utf-8")
    assert "load_gt_mapping" not in reconstruction_source
    assert "tifffile.imread" not in reconstruction_source
    assert "assert_no_external_cuda_compute" not in module_source
    assert "CudaContentionMonitor" not in module_source


def test_snapshot_comparison_detects_any_authority_change() -> None:
    before = {"formal": {"file_count": 1, "aggregate_sha256": "a", "entries": []}}
    assert control.compare_snapshots(before, before)["all_formal_directories_unmodified"] is True
    after = {"formal": {"file_count": 1, "aggregate_sha256": "b", "entries": []}}
    assert control.compare_snapshots(before, after)["all_formal_directories_unmodified"] is False


def test_json_boundary_serializes_numpy_and_torch_metadata(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    control.write_json(
        target,
        {
            "array": np.asarray([1, 2], dtype=np.int64),
            "scalar": np.float32(0.5),
            "tensor": torch.tensor([3.0], dtype=torch.float32),
        },
    )
    assert target.read_text(encoding="utf-8").strip().startswith("{")

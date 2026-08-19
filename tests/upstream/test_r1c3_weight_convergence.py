from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from unisim.revision_r1 import physmap6_core as core
from unisim.revision_r1 import physmap6_pipeline as pipeline
from unisim.revision_r1 import r1c3_weight_convergence as diagnostic
from unisim.sim_forward_2d import SIM2DConfig, forward_protocol_clean_2d, nominal_theta_2d


def _tiny_problem() -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    sim_config = SIM2DConfig(upsample=1, psf_size_xy=3, photon_scale=400.0, read_noise_e=1.0)
    gt = torch.linspace(0.05, 0.95, 36, dtype=torch.float32).reshape(1, 1, 6, 6)
    theta = nominal_theta_2d(sim_config, torch.device("cpu"))
    raw, _ = forward_protocol_clean_2d(
        gt, sim_config, pipeline.PROTOCOL_ID, theta=theta, randomize=False
    )
    geometry = {
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_frame_order": list(diagnostic.EXPECTED_RAW_ORDER),
    }
    forward = {"sim_config": sim_config, "theta": theta}
    return gt, raw, geometry, forward


def test_default_masked_refine_signature_and_result_remain_40_update_compatible() -> None:
    signature = inspect.signature(core.masked_refine)
    assert signature.parameters["diagnostic_updates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["diagnostic_observer"].kind is inspect.Parameter.KEYWORD_ONLY
    gt, raw, geometry, forward = _tiny_problem()
    result = core.masked_refine(
        raw.mean(dim=1, keepdim=True), raw,
        torch.tensor(diagnostic.EXPECTED_MASK, dtype=torch.float32),
        geometry, forward, core.RefinementConfig(),
    )
    assert result.final_reconstruction.shape == gt.shape
    assert len(result.objective_history) == 41
    assert result.configuration_receipt["diagnostic_only_extension"] is False
    assert result.configuration_receipt["executed_updates"] == 40


def test_diagnostic_extension_requires_frozen_allowed_endpoint_and_observer() -> None:
    _gt, raw, geometry, forward = _tiny_problem()
    arguments = (
        raw.mean(dim=1, keepdim=True), raw,
        torch.tensor(diagnostic.EXPECTED_MASK, dtype=torch.float32),
        geometry, forward, core.RefinementConfig(),
    )
    with pytest.raises(ValueError, match="80, 160, or 320"):
        core.masked_refine(*arguments, diagnostic_updates=41, diagnostic_observer=lambda _row: None)
    with pytest.raises(ValueError, match="requires an observer"):
        core.masked_refine(*arguments, diagnostic_updates=80)
    with pytest.raises(ValueError, match="explicitly diagnostic"):
        core.masked_refine(*arguments, diagnostic_observer=lambda _row: None)


def test_diagnostic_observer_records_exact_zero_to_endpoint_sequence() -> None:
    gt, raw, geometry, forward = _tiny_problem()
    initial = raw.mean(dim=1, keepdim=True)
    observations: list[dict] = []
    initial_probe = initial.detach().clone().requires_grad_(True)
    objective, nrmse = diagnostic._observed_diagnostics(
        initial_probe, raw, forward["sim_config"], forward["theta"]
    )
    objective.backward()
    observations.append({
        "update": 0,
        "objective": float(objective.detach()),
        "nrmse": float(nrmse.detach()),
    })
    result = core.masked_refine(
        initial, raw, torch.tensor(diagnostic.EXPECTED_MASK, dtype=torch.float32),
        geometry, forward, core.RefinementConfig(), diagnostic_updates=80,
        diagnostic_observer=lambda row: observations.append(dict(row)) if int(row["update"]) > 0 else None,
    )
    assert [int(row["update"]) for row in observations] == list(range(81))
    assert result.configuration_receipt["diagnostic_only_extension"] is True
    assert result.configuration_receipt["executed_updates"] == 80
    assert result.configuration_receipt["formal_result_update"] == 40
    assert torch.isfinite(result.final_reconstruction).all()
    assert np.isfinite([float(row["objective"]) for row in observations]).all()
    assert result.final_reconstruction.shape == gt.shape


def test_weight_branch_audit_reports_ema_and_270_tensors(tmp_path: Path) -> None:
    result = diagnostic.audit_weight_branch(tmp_path)
    assert result["status"] == "PASS"
    assert result["loaded_branch"] == "ema"
    assert result["validation_branch"] == "ema"
    assert result["formal_inference_branch"] == "ema"
    assert result["validation_and_inference_branch_consistent"] is True
    assert result["checkpoint_ema_tensor_count"] == 270
    assert len(result["loaded_ema_tensor_names"]) == 270
    assert len(result["loaded_ema_state_value_sha256"]) == 64
    assert (tmp_path / "APD6_WEIGHT_BRANCH_AUDIT.json").is_file()
    assert (tmp_path / "APD6_WEIGHT_BRANCH_AUDIT.md").is_file()


def test_trace_schema_has_all_required_diagnostics() -> None:
    assert {
        "initializer", "objective", "observed_nrmse", "psnr", "ssim", "gradient_norm", "finite",
        "image_min", "image_max", "image_mean", "image_std",
        "fraction_at_clip_min", "fraction_at_clip_max",
        "preclip_below_fraction", "preclip_above_fraction",
    }.issubset(diagnostic.TRACE_FIELDS)
    assert diagnostic.FORMAL_UPDATE == 40
    assert diagnostic.DIAGNOSTIC_ENDPOINTS == (80, 160, 320)
    assert diagnostic.EXPECTED_RAW_ORDER == ("H0", "H120", "H240", "V0", "V120", "V240")


def test_basin_gap_is_sign_safe_for_negative_objectives() -> None:
    rows = []
    for order in range(30):
        rows.extend(
            [
                {"sample_order": order, "method": "APD-SIM-6", "update": 40,
                 "objective": -2.0, "gradient_norm": 2.0},
                {"sample_order": order, "method": "APD-SIM-6", "update": 320,
                 "objective": -1.8, "gradient_norm": 1.0},
                {"sample_order": order, "method": "PhysMap-6", "update": 320,
                 "objective": -2.0, "gradient_norm": 1.0},
            ]
        )
    result = diagnostic._convergence_flags(rows)
    assert result["different_objective_basin_at_320_fov_count"] == 30
    assert result["relative_objective_gap_at_320_median"] == pytest.approx(0.1)


def test_image_summary_hash_is_explicit_and_matches_pipeline_hash() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16.0
    without_hash = diagnostic._image_summary(image)
    with_hash = diagnostic._image_summary(image, include_sha256=True)
    assert "sha256" not in without_hash
    assert with_hash["sha256"] == pipeline.sha_array(
        np.ascontiguousarray(image[0, 0].numpy(), dtype=np.float32)
    )


def test_formal40_numerical_regression_reports_hash_mismatch_but_passes_float32_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_array = np.zeros((8, 8), dtype=np.float32)
    current_array = old_array.copy()
    current_array[2, 3] = np.float32(2.0 * np.finfo(np.float32).eps)
    old_path = tmp_path / "old.npy"
    current_path = tmp_path / "current.npy"
    np.save(old_path, old_array, allow_pickle=False)
    np.save(current_path, current_array, allow_pickle=False)
    monkeypatch.setattr(diagnostic, "_prediction_path", lambda *_args: old_path)
    sample = {
        "sample_id": "sample",
        "parent_id": "parent",
        "structure_class": "microtubules",
    }
    old_row = {
        "prediction_sha256": pipeline.sha_array(old_array),
        "psnr": "40.0",
        "ssim": "0.95",
        "poisson_gaussian_objective": "-5.0",
        "observed_nrmse": "0.02",
    }
    current_trace = {
        "psnr": 40.0 + 1e-7,
        "ssim": 0.95 + 1e-10,
        "objective": -5.0 + 1e-8,
        "observed_nrmse": 0.02 + 1e-9,
    }
    row = diagnostic._formal40_regression_row(
        sample_order=0,
        sample=sample,
        method="PhysMap-6",
        current_array=current_array,
        current_trace_row=current_trace,
        current_path=current_path,
        old_row=old_row,
    )
    assert row["bitwise_exact"] is False
    assert row["different_element_count"] == 1
    assert row["current_prediction_array_sha256"] != row["old_prediction_array_sha256"]
    assert row["numeric_equivalence_pass"] is True
    assert set(row) == set(diagnostic.FORMAL40_REGRESSION_FIELDS)


def test_formal40_numerical_regression_fails_closed_above_predeclared_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_array = np.zeros((4, 4), dtype=np.float32)
    current_array = old_array.copy()
    current_array[0, 0] = np.float32(1e-3)
    old_path = tmp_path / "old.npy"
    current_path = tmp_path / "current.npy"
    np.save(old_path, old_array, allow_pickle=False)
    np.save(current_path, current_array, allow_pickle=False)
    monkeypatch.setattr(diagnostic, "_prediction_path", lambda *_args: old_path)
    row = diagnostic._formal40_regression_row(
        sample_order=0,
        sample={
            "sample_id": "sample",
            "parent_id": "parent",
            "structure_class": "microtubules",
        },
        method="APD-SIM-6",
        current_array=current_array,
        current_trace_row={
            "psnr": 40.0,
            "ssim": 0.95,
            "objective": -5.0,
            "observed_nrmse": 0.02,
        },
        current_path=current_path,
        old_row={
            "prediction_sha256": pipeline.sha_array(old_array),
            "psnr": "40.0",
            "ssim": "0.95",
            "poisson_gaussian_objective": "-5.0",
            "observed_nrmse": "0.02",
        },
    )
    assert row["max_abs_pass"] is False
    assert row["numeric_equivalence_pass"] is False


def test_trace_labels_replay_scope_and_unavailable_preclip_without_zero_sentinel() -> None:
    gt, raw, geometry, forward = _tiny_problem()
    rows, _final, hashes, formal40 = diagnostic._run_trace(
        raw.mean(dim=1, keepdim=True),
        raw,
        gt[0, 0].numpy(),
        0,
        {
            "sample_id": "sample",
            "parent_id": "parent",
            "structure_class": "microtubules",
        },
        "PhysMap-6",
        forward["sim_config"],
        forward["theta"],
        geometry,
        diagnostic._load_metrics(),
    )
    assert len(rows) == 321
    assert {row["scope"] for row in rows[:41]} == {"formal_configuration_replay"}
    assert {row["scope"] for row in rows[41:]} == {"diagnostic_only"}
    assert {row["preclip_below_fraction"] for row in rows} == {"NOT_MEASURED"}
    assert {row["preclip_above_fraction"] for row in rows} == {"NOT_MEASURED"}
    assert hashes[40] == pipeline.sha_array(formal40)


def test_formal40_tolerances_are_explicit_float32_units() -> None:
    eps = np.finfo(np.float32).eps
    assert diagnostic.FORMAL40_NUMERICAL_TOLERANCES == {
        "max_abs": 32.0 * eps,
        "rmse": 4.0 * eps,
        "psnr_abs": 1e-5,
        "ssim_abs": 1e-8,
        "objective_abs": 1e-6,
        "observed_nrmse_abs": 1e-7,
    }


def test_formal_core_provenance_and_cuda_nondeterminism_are_explicit() -> None:
    current_hash = diagnostic.sha_file(Path(core.__file__).resolve())
    provenance = diagnostic._formal_core_provenance(current_hash)
    assert provenance["old_formal_core_recorded_sha256"] == (
        "2e29bbc465e7e07c178c2c2c4e6d6e250b9a16fab884199e0c06b00777ab59e8"
    )
    assert provenance["old_formal_core_source_snapshot_available"] is False
    assert provenance["current_core_source_differs_from_recorded_formal_core"] is True
    evidence = diagnostic._cuda_nondeterminism_evidence()
    assert evidence["confirmed"] is True
    assert evidence["nondeterministic_operator"] == "adaptive_avg_pool2d_backward_cuda"


def test_bitwise_comparison_distinguishes_signed_zero() -> None:
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    result = diagnostic._array_difference(positive_zero, negative_zero)
    assert result["bitwise_exact"] is False
    assert result["different_element_count"] == 1
    assert result["max_abs_difference"] == 0.0
    assert result["rmse_difference"] == 0.0


def test_full_replay_verifier_loads_all_60_arrays_and_rejects_dtype_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    old_dir = tmp_path / "old"
    old_dir.mkdir(parents=True)

    def old_path(order: int, _sample_id: str, method: str) -> Path:
        return old_dir / f"{order:03d}_{method}.npy"

    monkeypatch.setattr(diagnostic, "_prediction_path", old_path)
    regression_rows = []
    for order in range(30):
        for method in ("PhysMap-6", "APD-SIM-6"):
            sample = {
                "sample_id": f"sample_{order}",
                "parent_id": f"parent_{order}",
                "structure_class": "microtubules",
            }
            old_array = np.full((2, 2), order / 100.0, dtype=np.float32)
            current_array = old_array.copy()
            authoritative = old_path(order, sample["sample_id"], method)
            replay = diagnostic._formal40_replay_path(
                output_dir, order, sample["sample_id"], method
            )
            replay.parent.mkdir(parents=True, exist_ok=True)
            np.save(authoritative, old_array, allow_pickle=False)
            np.save(replay, current_array, allow_pickle=False)
            regression_rows.append(diagnostic._formal40_regression_row(
                sample_order=order,
                sample=sample,
                method=method,
                current_array=current_array,
                current_trace_row={
                    "psnr": 40.0,
                    "ssim": 0.95,
                    "objective": -5.0,
                    "observed_nrmse": 0.02,
                },
                current_path=replay,
                old_row={
                    "prediction_sha256": pipeline.sha_array(old_array),
                    "psnr": "40.0",
                    "ssim": "0.95",
                    "poisson_gaussian_objective": "-5.0",
                    "observed_nrmse": "0.02",
                },
            ))
    file_hashes, array_hashes = diagnostic._verify_formal40_replay_artifacts(
        regression_rows, output_dir.resolve()
    )
    assert len(file_hashes) == len(array_hashes) == 60

    first_path = Path(regression_rows[0]["current_replay_prediction_path"])
    np.save(first_path, np.zeros((2, 2), dtype=np.float64), allow_pickle=False)
    regression_rows[0]["current_prediction_file_sha256"] = diagnostic.sha_file(first_path)
    with pytest.raises(RuntimeError, match="dtype/shape drift"):
        diagnostic._verify_formal40_replay_artifacts(
            regression_rows, output_dir.resolve()
        )


def test_clipping_summary_excludes_unclamped_initializer_update_zero() -> None:
    rows = []
    for method in ("PhysMap-6", "APD-SIM-6"):
        for order in range(30):
            for update in range(321):
                rows.append({
                    "sample_order": order,
                    "method": method,
                    "update": update,
                    "fraction_at_clip_min": 1.0 if update == 0 else 0.25,
                    "fraction_at_clip_max": 0.9 if update == 0 else 0.125,
                })
    summary = diagnostic._clipping_summary(rows)
    for method in ("PhysMap-6", "APD-SIM-6"):
        assert summary[method]["update0_initializer_excluded"] is True
        assert summary[method]["update_range"] == [1, 320]
        assert summary[method]["max_postclip_at_min_fraction"] == 0.25
        assert summary[method]["max_postclip_at_max_fraction"] == 0.125


def test_convergence_output_may_not_be_inside_protected_formal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    monkeypatch.setattr(diagnostic, "FORMAL_RUN", formal)
    with pytest.raises(RuntimeError, match="must not be the protected formal run"):
        diagnostic.run_convergence_diagnostics(formal / "forbidden")


def test_numerical_completion_collects_all_rows_then_marks_overall_failure() -> None:
    rows = [
        {
            "sample_order": index // 2,
            "method": "PhysMap-6" if index % 2 == 0 else "APD-SIM-6",
            "numeric_equivalence_pass": index != 55,
        }
        for index in range(60)
    ]
    result = diagnostic._numerical_completion_disposition(rows)
    assert result["expected_count"] == 60
    assert result["pass_count"] == 59
    assert result["mismatch_count"] == 1
    assert result["passed"] is False
    assert result["summary_status"] == "FAIL_NUMERICAL_EQUIVALENCE"
    assert result["progress_status"] == (
        "COMPLETE_WITH_NUMERICAL_EQUIVALENCE_FAILURES"
    )
    assert result["failures"] == [rows[55]]


def test_numerical_completion_pass_and_incomplete_grid_contract() -> None:
    passing = [{"numeric_equivalence_pass": True} for _ in range(60)]
    result = diagnostic._numerical_completion_disposition(passing)
    assert result["summary_status"] == "PASS"
    assert result["progress_status"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="expected 60 rows"):
        diagnostic._numerical_completion_disposition(passing[:-1])


def test_generator_has_no_per_case_numerical_equivalence_raise() -> None:
    source = inspect.getsource(diagnostic.run_convergence_diagnostics)
    assert 'if not regression["numeric_equivalence_pass"]' not in source
    assert '"FAIL_NUMERICAL_EQUIVALENCE"' not in source or (
        "summary_status" in source and "completion_status" in source
    )
    assert source.count("_render_plot(") == 2

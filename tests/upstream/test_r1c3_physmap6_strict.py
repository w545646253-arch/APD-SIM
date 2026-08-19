from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from unisim.protocol_runtime import require_protocol
from unisim.revision_r1 import physmap6_core as core
from unisim.revision_r1 import physmap6_experiment as experiment
from unisim.revision_r1 import physmap6_pipeline as pipeline
from unisim.sim_forward_2d import (
    SIM2DConfig,
    embed_raw_to_slots_2d,
    forward_protocol_clean_2d,
    masked_poisson_gaussian_likelihood_2d,
    nominal_theta_2d,
)


EXPECTED_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
EXPECTED_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")


@pytest.fixture()
def captured_four_method_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Execute the fairness wiring on CPU while replacing expensive solvers."""
    calls: list[dict[str, Any]] = []
    raw = torch.linspace(0.0, 1.0, 1 * 6 * 9 * 11, dtype=torch.float32).reshape(
        1, 6, 9, 11
    )
    x_ws = raw.mean(dim=1, keepdim=True).mul(0.75).add(0.1).clamp(0.0, 1.0)
    geometry = {
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_frame_order": list(EXPECTED_ORDER),
        "validity_mask": list(EXPECTED_MASK),
    }
    sim_config = SIM2DConfig(upsample=1, psf_size_xy=3)
    theta = nominal_theta_2d(sim_config, torch.device("cpu"))
    refinement_config = core.RefinementConfig()

    def fake_stage1(
        observed: torch.Tensor, _model: Any, _scheduler: Any, *, seed: int
    ) -> tuple[torch.Tensor, float, int]:
        assert observed is raw
        assert seed == 2468
        return x_ws, 0.25, 0

    def fake_refine(
        initial_image: torch.Tensor,
        observed_frames: torch.Tensor,
        validity_mask: torch.Tensor,
        acquisition_geometry: dict[str, Any],
        forward_operator: dict[str, Any],
        config: core.RefinementConfig,
    ) -> SimpleNamespace:
        calls.append(
            {
                "initial_image": initial_image,
                "observed_frames": observed_frames,
                "validity_mask": validity_mask,
                "geometry": acquisition_geometry,
                "forward_operator": forward_operator,
                "config": config,
            }
        )
        return SimpleNamespace(
            final_reconstruction=initial_image.detach().clone(),
            objective_history=[1.0, 0.5],
            observed_nrmse_history=[0.2, 0.1],
            gradient_finite=True,
            output_finite=True,
            runtime_seconds=0.5,
            peak_gpu_memory_bytes=0,
            configuration_receipt=config.receipt(),
        )

    monkeypatch.setattr(pipeline, "stage1_reconstruct", fake_stage1)
    monkeypatch.setattr(pipeline, "masked_refine", fake_refine)
    monkeypatch.setattr(
        pipeline,
        "evaluate_observed_fit",
        lambda *_args, **_kwargs: core.ObservedFit(1.0, 0.1, True),
    )
    result = pipeline.run_four_methods(
        raw,
        object(),
        object(),
        sim_config,
        theta,
        diffusion_seed=2468,
        refinement_config=refinement_config,
        geometry_receipt=geometry,
    )
    return {
        "calls": calls,
        "raw": raw,
        "x_ws": x_ws,
        "geometry": geometry,
        "theta": theta,
        "sim_config": sim_config,
        "config": refinement_config,
        "result": result,
    }


@pytest.fixture(scope="module")
def tiny_refinement() -> dict[str, Any]:
    """One real 40-update refinement on a deliberately tiny CPU image."""
    device = torch.device("cpu")
    config = SIM2DConfig(
        upsample=1,
        psf_size_xy=3,
        photon_scale=400.0,
        read_noise_e=1.0,
    )
    gt = torch.linspace(0.05, 0.95, 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    theta = nominal_theta_2d(config, device)
    raw, _ = forward_protocol_clean_2d(
        gt, config, pipeline.PROTOCOL_ID, theta=theta, randomize=False
    )
    geometry = {
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "raw_frame_order": list(EXPECTED_ORDER),
    }
    result = core.masked_refine(
        raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0),
        raw,
        torch.tensor(EXPECTED_MASK, dtype=torch.float32),
        geometry,
        {"sim_config": config, "theta": theta},
        core.RefinementConfig(),
    )
    return {"gt": gt, "raw": raw, "result": result}


def test_registered_dmd6_contract_is_exact() -> None:
    spec = require_protocol("DMD_6F_2O3P")
    assert spec.frame_count == 6
    assert spec.orientation_count == 2
    assert spec.phases_per_orientation == 3
    assert tuple(spec.raw_frame_order) == EXPECTED_ORDER
    assert tuple(spec.raw_to_slot_mapping) == tuple(range(6))
    assert tuple(spec.validity_mask) == EXPECTED_MASK


def test_pmon_parser_rejects_fragments_and_detects_external_compute() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        experiment._parse_pmon_output(b"R\r\n")
    with pytest.raises(ValueError, match="unparseable"):
        experiment._parse_pmon_output(b"python.exe\r\n")
    with pytest.raises(ValueError, match="invalid"):
        experiment._parse_pmon_output(b"garbage row C\r\n")
    offenders, count = experiment._parse_pmon_output(b"0 987654 C 10 10 - - - - python.exe\r\n")
    assert count == 1
    assert offenders == [{"pid": 987654, "command": "python.exe"}]
    clean, count = experiment._parse_pmon_output(b"0 123 C+G - - - - - - app.exe\r\n")
    assert clean == []
    assert count == 1


def test_implementation_hash_closure_includes_stage1_dependencies() -> None:
    relative = {path.relative_to(experiment.ROOT).as_posix() for path in experiment.IMPLEMENTATION_FILES}
    assert {
        "unisim/formal_training_2d.py",
        "unisim/model2d.py",
        "unisim/checkpoint_contract.py",
        "unisim/protocol_runtime.py",
        "unisim/protocols.py",
    }.issubset(relative)


def test_failed_case_schema_has_one_owner_and_is_byte_compatible() -> None:
    reporting = pytest.importorskip("unisim.revision_r1.physmap6_reporting")
    assert reporting._FAILED_FIELDS == ("factor", "severity", "sample_id", "error")
    expected = experiment._csv_bytes([], ("factor", "severity", "sample_id", "error"))
    assert reporting._csv_bytes([], reporting._FAILED_FIELDS) == expected


def test_checkpoint_receipt_distinguishes_event_from_optimizer_commits() -> None:
    reporting = pytest.importorskip("unisim.revision_r1.physmap6_reporting")
    receipt = {
        "status": "PASS",
        "checkpoint_sha256": "a" * 64,
        "protocol_id": pipeline.PROTOCOL_ID,
        "protocol_hash": pipeline.PROTOCOL_HASH,
        "test_data_used_for_selection": False,
        "all_model_parameters_finite": True,
        "all_ema_parameters_finite": True,
        "selected_step": 96000,
        "selected_step_semantics": "loop/data event step; not the number of committed optimizer updates",
        "optimizer_committed_updates_at_selected_checkpoint": 95956,
        "global_minus_optimizer_commits": 44,
        "validation_metric_value": 0.1,
    }
    assert reporting._validate_checkpoint_receipt(receipt)["global_minus_optimizer_commits"] == 44
    with pytest.raises(reporting.ReportingValidationError, match="event/commit"):
        reporting._validate_checkpoint_receipt({**receipt, "global_minus_optimizer_commits": 0})


def test_physmap_and_apd_call_the_same_refinement_core(
    captured_four_method_run: dict[str, Any],
) -> None:
    calls = captured_four_method_run["calls"]
    raw = captured_four_method_run["raw"]
    assert len(calls) == 2
    assert torch.equal(calls[0]["initial_image"], raw.mean(dim=1, keepdim=True))
    assert calls[1]["initial_image"] is captured_four_method_run["x_ws"]
    assert captured_four_method_run["result"]["only_allowed_difference"].startswith(
        "initial_image"
    )


def test_physmap_and_apd_share_one_optimizer_config_object(
    captured_four_method_run: dict[str, Any],
) -> None:
    calls = captured_four_method_run["calls"]
    assert calls[0]["config"] is calls[1]["config"] is captured_four_method_run["config"]
    assert calls[0]["config"].receipt() == calls[1]["config"].receipt()
    assert calls[0]["config"].learning_rate == pytest.approx(0.005)
    assert calls[0]["config"].updates == 40
    assert calls[0]["config"].lambda_prior == 0.0


def test_physmap_and_apd_share_the_exact_raw_tensor_and_hash(
    captured_four_method_run: dict[str, Any],
) -> None:
    calls = captured_four_method_run["calls"]
    raw = captured_four_method_run["raw"]
    assert calls[0]["observed_frames"] is calls[1]["observed_frames"] is raw
    expected_hash = pipeline.sha_array(raw[0].numpy())
    assert captured_four_method_run["result"]["raw_stack_sha256"] == expected_hash


def test_physmap_and_apd_share_the_exact_validity_mask_object(
    captured_four_method_run: dict[str, Any],
) -> None:
    calls = captured_four_method_run["calls"]
    assert calls[0]["validity_mask"] is calls[1]["validity_mask"]
    assert tuple(int(value) for value in calls[0]["validity_mask"].tolist()) == EXPECTED_MASK


def test_physmap_and_apd_share_geometry_and_forward_operator_objects(
    captured_four_method_run: dict[str, Any],
) -> None:
    calls = captured_four_method_run["calls"]
    assert calls[0]["geometry"] is calls[1]["geometry"] is captured_four_method_run["geometry"]
    assert calls[0]["forward_operator"] is calls[1]["forward_operator"]
    assert calls[0]["forward_operator"]["theta"] is captured_four_method_run["theta"]
    assert calls[0]["forward_operator"]["sim_config"] is captured_four_method_run["sim_config"]


def _function_tree(function: Any) -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(function))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef))
    return node


def test_physmap_core_has_no_checkpoint_dependency_or_parameter() -> None:
    signature = inspect.signature(core.masked_refine)
    assert "checkpoint" not in signature.parameters
    module_tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module or ""
        for node in ast.walk(module_tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported.update(
        alias.name
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any("checkpoint" in name or "diffusion" in name for name in imported)
    calls = {
        node.func.id
        for node in ast.walk(_function_tree(core.masked_refine))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any("checkpoint" in name.lower() or "model" in name.lower() for name in calls)


def test_physmap_core_has_no_gt_input_or_read() -> None:
    signature = inspect.signature(core.masked_refine)
    assert "gt" not in signature.parameters
    assert "ground_truth" not in signature.parameters
    identifiers = {
        node.id.lower()
        for node in ast.walk(_function_tree(core.masked_refine))
        if isinstance(node, ast.Name)
    }
    assert "gt" not in identifiers
    assert "ground_truth" not in identifiers


def test_invalid_slots_are_excluded_from_likelihood() -> None:
    generator = torch.Generator(device="cpu").manual_seed(718)
    observed_raw = torch.rand((1, 6, 7, 9), generator=generator)
    predicted_raw = torch.rand((1, 6, 7, 9), generator=generator)
    observed, mask = embed_raw_to_slots_2d(observed_raw, pipeline.PROTOCOL_ID)
    predicted, _ = embed_raw_to_slots_2d(predicted_raw, pipeline.PROTOCOL_ID)
    baseline = masked_poisson_gaussian_likelihood_2d(
        observed,
        predicted,
        pipeline.PROTOCOL_ID,
        photon_scale=800.0,
        read_noise_e=1.6,
    )
    observed_changed = observed.clone()
    predicted_changed = predicted.clone()
    observed_changed[mask == 0] = 1.0e6
    predicted_changed[mask == 0] = -1.0e6
    changed = masked_poisson_gaussian_likelihood_2d(
        observed_changed,
        predicted_changed,
        pipeline.PROTOCOL_ID,
        photon_scale=800.0,
        read_noise_e=1.6,
    )
    assert torch.equal(baseline, changed)


def test_refinement_rejects_any_noncanonical_mask_before_optimization() -> None:
    config = SIM2DConfig(upsample=1, psf_size_xy=3)
    theta = nominal_theta_2d(config, torch.device("cpu"))
    initial = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    raw = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    bad_mask = torch.tensor((1, 1, 1, 1, 1, 0) + (0,) * 9, dtype=torch.float32)
    with pytest.raises(ValueError, match="Validity mask drift"):
        core.masked_refine(
            initial,
            raw,
            bad_mask,
            {"protocol_id": pipeline.PROTOCOL_ID},
            {"sim_config": config, "theta": theta},
            core.RefinementConfig(),
        )


def test_nine_frame_physmap_is_excluded_from_primary_methods() -> None:
    assert experiment.METHODS == ("WF", "DiffWS-6", "PhysMap-6", "APD-SIM-6")
    assert "PhysMap-9" not in experiment.METHODS
    with pytest.raises(ValueError, match="Exactly six observed raw frames"):
        core.masked_refine(
            torch.zeros((1, 1, 5, 5), dtype=torch.float32),
            torch.zeros((1, 9, 5, 5), dtype=torch.float32),
            torch.tensor(EXPECTED_MASK, dtype=torch.float32),
            {"protocol_id": pipeline.PROTOCOL_ID},
            {},
            core.RefinementConfig(),
        )


def test_real_refinement_output_and_histories_are_finite(
    tiny_refinement: dict[str, Any],
) -> None:
    result = tiny_refinement["result"]
    assert result.gradient_finite is True
    assert result.output_finite is True
    assert torch.isfinite(result.final_reconstruction).all()
    assert np.isfinite(result.objective_history).all()
    assert np.isfinite(result.observed_nrmse_history).all()
    assert len(result.objective_history) == 41
    assert len(result.observed_nrmse_history) == 41


def test_real_refinement_output_shape_matches_gt(
    tiny_refinement: dict[str, Any],
) -> None:
    assert tiny_refinement["result"].final_reconstruction.shape == tiny_refinement["gt"].shape


class _RecordingZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_shapes: list[tuple[int, int]] = []

    def forward(self, condition: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
        self.spatial_shapes.append(tuple(condition.shape[-2:]))
        return torch.zeros_like(condition[:, :1])


class _SeedSensitiveScheduler:
    def __init__(self) -> None:
        self.alpha_bar = torch.ones(601, dtype=torch.float32)

    def q_sample(
        self, image: torch.Tensor, _timestep: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return image + 0.01 * noise

    def predict_x0(
        self, image: torch.Tensor, _timestep: torch.Tensor, _epsilon: torch.Tensor
    ) -> torch.Tensor:
        return image


def test_stage1_padding_and_cropping_are_exactly_reversible() -> None:
    raw = torch.linspace(0.0, 1.0, 1 * 6 * 17 * 19, dtype=torch.float32).reshape(
        1, 6, 17, 19
    )
    model = _RecordingZeroModel()
    output, _runtime, peak = pipeline.stage1_reconstruct(
        raw, model, _SeedSensitiveScheduler(), seed=123
    )
    assert output.shape == (1, 1, 17, 19)
    assert set(model.spatial_shapes) == {(32, 32)}
    assert len(model.spatial_shapes) == 80
    assert peak == 0
    assert torch.isfinite(output).all()


def test_fixed_diffusion_seed_repeats_bitwise_on_cpu() -> None:
    raw = torch.linspace(0.0, 1.0, 1 * 6 * 17 * 19, dtype=torch.float32).reshape(
        1, 6, 17, 19
    )
    first, _runtime, _peak = pipeline.stage1_reconstruct(
        raw, _RecordingZeroModel(), _SeedSensitiveScheduler(), seed=20260814
    )
    second, _runtime, _peak = pipeline.stage1_reconstruct(
        raw, _RecordingZeroModel(), _SeedSensitiveScheduler(), seed=20260814
    )
    different, _runtime, _peak = pipeline.stage1_reconstruct(
        raw, _RecordingZeroModel(), _SeedSensitiveScheduler(), seed=20260815
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_raw_hash_is_shape_dtype_and_value_sensitive() -> None:
    array = np.arange(24, dtype=np.float32).reshape(1, 6, 2, 2)
    assert pipeline.sha_array(array) == pipeline.sha_array(array.copy())
    assert pipeline.sha_array(array) != pipeline.sha_array(array.astype(np.float64))
    changed = array.copy()
    changed[0, 0, 0, 0] += 1.0
    assert pipeline.sha_array(array) != pipeline.sha_array(changed)


def _synthetic_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def reporting_robustness_grid() -> tuple[Any, list[dict[str, Any]]]:
    reporting = pytest.importorskip("unisim.revision_r1.physmap6_reporting")
    metric_values = {
        "WF": (10.0, 0.50),
        "DiffWS-6": (20.0, 0.70),
        "PhysMap-6": (25.0, 0.80),
        "APD-SIM-6": (28.0, 0.86),
    }
    rows: list[dict[str, Any]] = []
    for factor, severities in reporting.DEFAULT_FACTOR_LEVELS.items():
        for severity in severities:
            for order in range(20):
                case = f"{factor}/{severity}/{order}"
                identity = {
                    field: _synthetic_sha(f"{case}/{field}")
                    for field in (
                        "raw_stack_sha256",
                        "validity_mask_sha256",
                        "geometry_sha256",
                        "forward_parameters_sha256",
                        "normalization_sha256",
                        "gt_identity_sha256",
                    )
                }
                for method in reporting.METHODS:
                    psnr, ssim = metric_values[method]
                    rows.append(
                        {
                            "factor": factor,
                            "severity": severity,
                            "sample_order": order,
                            "sample_id": f"S{order:02d}",
                            "parent_id": f"P{order:02d}",
                            "structure": ("CCP", "ER", "MT")[order % 3],
                            "method": method,
                            **identity,
                            "noise_seed": 1000 + order,
                            "diffusion_seed": 2000 + order,
                            "refinement_config_sha256": (
                                _synthetic_sha("shared-refinement")
                                if method in {"PhysMap-6", "APD-SIM-6"}
                                else "NA"
                            ),
                            "theta_true_json": "{}",
                            "theta_inverse_json": "{}",
                            "psnr": psnr,
                            "ssim": ssim,
                            "observed_nrmse": 0.1,
                            "poisson_gaussian_objective": 1.0,
                            "runtime_seconds": 0.1,
                            "peak_gpu_memory_bytes": 0,
                            "gradient_finite": True,
                            "output_finite": True,
                            "prediction_sha256": _synthetic_sha(f"{case}/{method}/prediction"),
                            "status": "PASS",
                        }
                    )
    assert len(rows) == 4320
    return reporting, rows


def test_figure5_arrays_are_hash_bound_to_robustness_csv_rows(
    tmp_path: Path, reporting_robustness_grid: tuple[Any, list[dict[str, Any]]]
) -> None:
    reporting, rows = reporting_robustness_grid
    visual_dir = tmp_path / "robustness_visual_arrays"
    visual_dir.mkdir()
    gt = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    selected = {
        (factor, severity)
        for factor, severities in zip(
            reporting.DEFAULT_FIGURE5_SPEC.factors,
            reporting.DEFAULT_FIGURE5_SPEC.severities,
        )
        for severity in severities
    }
    for factor, severity in selected:
        for method_index, method in enumerate(reporting.FIGURE_METHODS):
            prediction = np.ascontiguousarray(
                np.clip(gt + 0.01 * method_index, 0.0, 1.0), dtype=np.float32
            )
            np.savez_compressed(
                visual_dir / reporting._figure_npz_name(factor, severity, method),
                prediction=prediction,
                gt=gt,
            )
            matching = next(
                row
                for row in rows
                if row["factor"] == factor
                and float(row["severity"]) == severity
                and row["sample_order"] == 0
                and row["method"] == method
            )
            matching["prediction_sha256"] = reporting.sha256_array(prediction)

    arrays, receipt = reporting._load_figure_arrays(
        visual_dir, rows, reporting.DEFAULT_FIGURE5_SPEC
    )
    assert len(arrays) == 27
    assert receipt["source_count"] == 27
    assert receipt["column_order"] == ["APD-SIM-6", "PhysMap-6", "DiffWS-6"]
    assert all(
        source["prediction_sha256"] == source["csv_prediction_sha256"]
        for source in receipt["sources"]
    )

    changed = [dict(row) for row in rows]
    target = next(
        row
        for row in changed
        if row["factor"] == "phase_jitter_rad"
        and float(row["severity"]) == 0.1
        and row["sample_order"] == 0
        and row["method"] == "APD-SIM-6"
    )
    target["prediction_sha256"] = _synthetic_sha("csv-tampered")
    with pytest.raises(reporting.ReportingValidationError, match="NPZ/CSV prediction hash mismatch"):
        reporting._load_figure_arrays(
            visual_dir, changed, reporting.DEFAULT_FIGURE5_SPEC
        )


def test_table2_is_independently_recomputed_from_sample_rows(
    reporting_robustness_grid: tuple[Any, list[dict[str, Any]]]
) -> None:
    reporting, rows = reporting_robustness_grid
    table_rows, _model = reporting.build_table2(rows, bootstrap_seed=7000)
    assert len(table_rows) == 24
    for row in table_rows:
        assert row["n_paired_samples"] == 20
        assert row["best_matched_six_frame_baseline"] == "PhysMap-6"
        if row["metric"] == "psnr":
            assert row["physmap6_mean"] == pytest.approx(25.0)
            assert row["apd6_mean"] == pytest.approx(28.0)
            assert row["apd_minus_best_baseline_mean"] == pytest.approx(3.0)
        else:
            assert row["physmap6_mean"] == pytest.approx(0.80)
            assert row["apd6_mean"] == pytest.approx(0.86)
            assert row["apd_minus_best_baseline_mean"] == pytest.approx(0.06)

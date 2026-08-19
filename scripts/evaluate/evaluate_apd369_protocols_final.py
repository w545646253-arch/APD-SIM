"""Final, inference-only APD-SIM 3F/6F/9F protocol evaluation.

Right-click execution is the formal interface.  The script consumes frozen
checkpoint and test contracts, generates each protocol measurement in a
separate forward call, and evaluates one principal diffusion trajectory.
It never imports or invokes an external baseline or a training entry point.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.apd369_final_contract import (  # noqa: E402
    APD369ContractError,
    DEFAULT_OUTPUT_ROOT,
    PLANS,
    atomic_json,
    load_frozen_contract,
    read_json,
    sha256_file,
)
from tools import revision_dmd6_common as d6common  # noqa: E402
from unisim.formal_training_2d import DiffusionScheduler2D  # noqa: E402
from unisim.protocols import protocol_registry  # noqa: E402
from unisim.revision_r1 import frame_budget_r1c2 as fb  # noqa: E402
from unisim.revision_r1.physmap6_pipeline import stage1_reconstruct_registered  # noqa: E402


FROZEN_FORWARD_SOURCE = (
    ROOT / "audit" / "dmd3_nonfinite_recovery_20260814_010345"
    / "source_backups" / "unisim" / "sim_forward_2d.py"
)
FROZEN_FORWARD_SHA256 = "f067a832c2dbac2da32fb4c3a73ac39047a754985f27ecafa31071786321fdd8"
EXPECTED_COUNT = 90
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPLICATES = 10000
METRIC_FIELDS = (
    "order", "sample_id", "parent_id", "structure", "method", "protocol_id",
    "protocol_hash", "raw_frame_order", "validity_mask", "checkpoint_path",
    "checkpoint_sha256", "selected_validation_iteration", "inference_weight_branch",
    "measurement_seed", "diffusion_seed", "generation_call_uuid", "raw_stack_sha256",
    "raw_npz_path", "raw_npz_sha256", "prediction_native_path", "prediction_native_sha256",
    "prediction_harmonized_path", "prediction_harmonized_sha256", "psnr", "ssim",
    "frc_cutoff_period_px", "frc_auc", "frc_right_censored", "frc_unresolved",
    "stage1_runtime_seconds", "stage2_runtime_seconds", "runtime_seconds",
    "stage2_final_objective", "observed_frame_nrmse", "finite", "status",
)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.float32)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    _atomic_bytes(path, stream.getvalue())
    check = np.load(path, allow_pickle=False)
    if check.dtype != np.float32 or not np.array_equal(check, array):
        raise APD369ContractError(f"NPY roundtrip failure: {path}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_frozen_forward() -> Any:
    if not FROZEN_FORWARD_SOURCE.is_file() or sha256_file(FROZEN_FORWARD_SOURCE) != FROZEN_FORWARD_SHA256:
        raise APD369ContractError("frozen formal forward source identity unavailable")
    name = "unisim._apd369_final_forward_f067a832"
    spec = importlib.util.spec_from_file_location(name, FROZEN_FORWARD_SOURCE)
    if spec is None or spec.loader is None:
        raise APD369ContractError("cannot import frozen formal forward source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sim_config(module: Any, config: Mapping[str, Any]) -> Any:
    allowed = set(module.SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in config["forward"].items() if key in allowed}
    for key, value in tuple(values.items()):
        if key.startswith("rand_") and isinstance(value, list):
            values[key] = tuple(float(item) for item in value)
    return module.SIM2DConfig(**values)


def _load_ema(plan: Any, frozen: Mapping[str, Any], device: torch.device) -> tuple[Any, DiffusionScheduler2D, dict[str, Any]]:
    row = frozen["checkpoints"][plan.method]
    if sha256_file(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]:
        raise APD369ContractError(f"checkpoint changed after freeze: {plan.method}")
    config = read_json(Path(row["config_path"]))
    model = fb._model_from_config(config)
    payload = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=False)
    state = payload.get("ema")
    if not isinstance(state, Mapping):
        raise APD369ContractError(f"EMA state absent: {plan.method}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise APD369ContractError(f"strict EMA load failed: {plan.method}")
    if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise APD369ContractError(f"non-finite EMA state: {plan.method}")
    model.to(device=device, dtype=torch.float32).eval()
    scheduler = DiffusionScheduler2D(
        int(config["training"]["diffusion_steps"]), device,
        str(config["training"]["beta_schedule"]),
    )
    return model, scheduler, config


def _raw_bundle_bytes(raw: np.ndarray, metadata: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        raw_stack=np.ascontiguousarray(raw, dtype=np.float32),
        protocol_id=np.asarray(metadata["protocol_id"]),
        protocol_hash=np.asarray(metadata["protocol_hash"]),
        raw_frame_order=np.asarray(metadata["raw_frame_order"]),
        measurement_seed=np.asarray(metadata["measurement_seed"], dtype=np.int64),
        generation_call_uuid=np.asarray(metadata["generation_call_uuid"]),
        forward_source_sha256=np.asarray(FROZEN_FORWARD_SHA256),
    )
    return stream.getvalue()


def _generate_raw(
    module: Any,
    gt_tensor: torch.Tensor,
    sim_config: Any,
    plan: Any,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any], Mapping[str, torch.Tensor]]:
    spec = protocol_registry.require(plan.protocol_id)
    if tuple(spec.raw_frame_order) != plan.raw_order:
        raise APD369ContractError(f"registry order drift: {plan.method}")
    theta = module.nominal_theta_2d(sim_config, gt_tensor.device)
    generator = torch.Generator(device=gt_tensor.device).manual_seed(int(seed))
    call_id = str(uuid.uuid4())
    with torch.no_grad():
        raw, _ = module.forward_protocol_sim_2d(
            gt_tensor, sim_config, plan.protocol_id, theta=dict(theta),
            randomize=False, noise_generator=generator,
        )
    expected = (1, len(plan.raw_order), 1004, 1004)
    if tuple(raw.shape) != expected or not bool(torch.isfinite(raw).all()):
        raise APD369ContractError(f"invalid independently generated raw {plan.method}: {tuple(raw.shape)}")
    return raw, {
        "protocol_id": plan.protocol_id,
        "protocol_hash": plan.protocol_hash,
        "raw_frame_order": list(plan.raw_order),
        "measurement_seed": int(seed),
        "generation_call_uuid": call_id,
    }, theta


def _metric_row(gt: np.ndarray, prediction: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    metrics = d6common.metrics_module()
    psnr = float(metrics.psnr_native(gt, prediction))
    ssim = float(metrics.ssim_native(gt, prediction))
    frc, _curves = d6common.gt_frc(gt, prediction)
    if not math.isfinite(psnr) or not math.isfinite(ssim) or not math.isfinite(float(frc["frc_auc_to_cutoff_or_nyquist"])):
        raise APD369ContractError("non-finite metric")
    return psnr, ssim, frc


def _evaluate(output_root: Path) -> list[dict[str, Any]]:
    frozen, manifest = load_frozen_contract(output_root)
    from unisim.revision_r1.physmap6_experiment import assert_no_external_cuda_compute

    gate = assert_no_external_cuda_compute()
    if not torch.cuda.is_available():
        raise APD369ContractError("CUDA is required for final 30-FOV evaluation")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    monitor = fb.CudaContentionMonitor(interval_seconds=1.0)
    monitor.start()
    forward = _load_frozen_forward()
    per_fov = output_root / "04_metrics" / "per_fov_metrics.csv"
    prior = _load_rows(per_fov)
    completed = {(row["method"], int(row["order"])) for row in prior if row.get("status") == "PASS"}
    rows: list[dict[str, Any]] = [dict(row) for row in prior]
    for plan in PLANS:
        model, scheduler, config = _load_ema(plan, frozen, device)
        sim_config = _sim_config(forward, config)
        current_sim_config = fb._config_for_sim(config)
        current_theta = __import__("unisim.sim_forward_2d", fromlist=["nominal_theta_2d"]).nominal_theta_2d(current_sim_config, device)
        for sample in manifest["samples"]:
            order = int(sample["order"])
            if (plan.method, order) in completed:
                continue
            monitor.checkpoint()
            gt = fb.normalize_image(tifffile.imread(Path(sample["absolute_path"])))
            if fb.sha_array(gt) != sample["normalized_array_sha256"]:
                raise APD369ContractError(f"GT identity drift: {sample['sample_id']}")
            # The frozen formal acquisition ledger was generated with the CPU
            # torch.Generator.  Torch CPU and CUDA generators intentionally do
            # not share a bitstream, so measurements are produced on CPU and
            # only then transferred to CUDA for reconstruction.
            gt_tensor = torch.from_numpy(gt)[None, None].to(dtype=torch.float32)
            raw, raw_meta, _frozen_theta = _generate_raw(
                forward, gt_tensor, sim_config, plan, int(sample["measurement_seed"])
            )
            raw_np = np.ascontiguousarray(raw[0].detach().cpu().numpy(), dtype=np.float32)
            raw_sha = d6common.array_sha256(raw_np)
            if plan.method == "APD-SIM-6" and raw_sha != sample["formal_dmd6_raw_sha256"]:
                raise APD369ContractError(
                    f"DMD6 formal raw identity mismatch {sample['sample_id']}: {raw_sha}"
                )
            raw_path = output_root / "02_protocol_raw_bundles" / plan.method / f"{order:03d}_{sample['sample_id']}.npz"
            _atomic_bytes(raw_path, _raw_bundle_bytes(raw_np, raw_meta))
            raw = raw.to(device=device, dtype=torch.float32)
            x0, stage1_seconds, _peak = stage1_reconstruct_registered(
                raw, model, scheduler, protocol_id=plan.protocol_id,
                seed=int(sample["diffusion_seed"]),
            )
            apd, stage2_seconds, final_objective, observed_nrmse = fb.refine_protocol(
                x0, raw, plan.protocol_id, current_sim_config, current_theta
            )
            prediction = np.ascontiguousarray(apd[0, 0].detach().cpu().numpy(), dtype=np.float32)
            if prediction.shape != gt.shape or not np.isfinite(prediction).all():
                raise APD369ContractError(f"non-finite output: {plan.method}/{sample['sample_id']}")
            harmonized = np.ascontiguousarray(np.clip(prediction, 0.0, 1.0), dtype=np.float32)
            native_path = output_root / "03_harmonized_outputs" / "native" / plan.method / f"{order:03d}_{sample['sample_id']}.npy"
            harmonized_path = output_root / "03_harmonized_outputs" / "harmonized" / plan.method / f"{order:03d}_{sample['sample_id']}.npy"
            _save_npy(native_path, prediction)
            _save_npy(harmonized_path, harmonized)
            psnr, ssim, frc = _metric_row(gt, harmonized)
            checkpoint = frozen["checkpoints"][plan.method]
            row = {
                "order": order, "sample_id": sample["sample_id"], "parent_id": sample["parent_id"],
                "structure": sample["class"], "method": plan.method, "protocol_id": plan.protocol_id,
                "protocol_hash": plan.protocol_hash, "raw_frame_order": "/".join(plan.raw_order),
                "validity_mask": "/".join(str(value) for value in plan.validity_mask),
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_validation_iteration": checkpoint["selected_validation_iteration"],
                "inference_weight_branch": "ema", "measurement_seed": sample["measurement_seed"],
                "diffusion_seed": sample["diffusion_seed"], "generation_call_uuid": raw_meta["generation_call_uuid"],
                "raw_stack_sha256": raw_sha, "raw_npz_path": str(raw_path.resolve()),
                "raw_npz_sha256": sha256_file(raw_path), "prediction_native_path": str(native_path.resolve()),
                "prediction_native_sha256": sha256_file(native_path),
                "prediction_harmonized_path": str(harmonized_path.resolve()),
                "prediction_harmonized_sha256": sha256_file(harmonized_path),
                "psnr": psnr, "ssim": ssim,
                "frc_cutoff_period_px": "" if frc["cutoff_derived_spatial_period_px"] is None else frc["cutoff_derived_spatial_period_px"],
                "frc_auc": frc["frc_auc_to_cutoff_or_nyquist"],
                "frc_right_censored": bool(frc["right_censored_at_nyquist"]),
                "frc_unresolved": bool(frc["unresolved_no_crossing"]),
                "stage1_runtime_seconds": stage1_seconds, "stage2_runtime_seconds": stage2_seconds,
                "runtime_seconds": stage1_seconds + stage2_seconds,
                "stage2_final_objective": final_objective, "observed_frame_nrmse": observed_nrmse,
                "finite": True, "status": "PASS",
            }
            rows.append(row)
            rows.sort(key=lambda item: (int(item["order"]), item["method"]))
            _write_csv(per_fov, rows, METRIC_FIELDS)
            print(f"completed {len(rows)}/{EXPECTED_COUNT}: {plan.method} {sample['sample_id']}", flush=True)
            del gt_tensor, raw, x0, apd
            torch.cuda.empty_cache()
        del model, scheduler
        torch.cuda.empty_cache()
    observed = {(row["method"], int(row["order"])) for row in rows}
    expected = {(plan.method, order) for plan in PLANS for order in range(30)}
    if len(rows) != EXPECTED_COUNT or observed != expected:
        raise APD369ContractError(f"incomplete APD369 grid: {len(rows)}/{EXPECTED_COUNT}")
    monitor_receipt = monitor.stop_and_validate()
    atomic_json(output_root / "09_final" / "gpu_contention_monitor.json", {
        "initial_gate": gate, "continuous_monitor": monitor_receipt,
    })
    return rows


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise APD369ContractError("invalid descriptive input")
    q1, median, q3 = np.quantile(array, (0.25, 0.5, 0.75), method="linear")
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)), "median": float(median),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975), method="linear")
    return float(low), float(high)


def _finalize_metrics(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = output_root / "04_metrics"
    mean_rows, median_rows, structure_rows, censor_rows = [], [], [], []
    index = {(str(row["method"]), int(row["order"])): row for row in rows}
    for plan in PLANS:
        subset = [row for row in rows if row["method"] == plan.method]
        for metric in ("psnr", "ssim", "frc_cutoff_period_px", "frc_auc"):
            values = [float(row[metric]) for row in subset if str(row[metric]) != ""]
            stats = _descriptive(values)
            mean_rows.append({"method": plan.method, "metric": metric, **{key: stats[key] for key in ("n", "mean", "sd")}})
            median_rows.append({"method": plan.method, "metric": metric, **{key: stats[key] for key in ("n", "median", "q1", "q3", "iqr")}})
        for structure in ("CCP", "ER", "MT"):
            group = [row for row in subset if row["structure"] == structure]
            for metric in ("psnr", "ssim", "frc_cutoff_period_px", "frc_auc"):
                values = [float(row[metric]) for row in group if str(row[metric]) != ""]
                stats = _descriptive(values) if values else {
                    "n": 0, "mean": "", "sd": "", "median": "",
                    "q1": "", "q3": "", "iqr": "",
                }
                structure_rows.append({"method": plan.method, "structure": structure, "metric": metric, **stats})
        censor_rows.append({
            "method": plan.method, "n": len(subset),
            "right_censored": sum(str(row["frc_right_censored"]).lower() == "true" for row in subset),
            "unresolved": sum(str(row["frc_unresolved"]).lower() == "true" for row in subset),
            "cutoff_observed": sum(str(row["frc_cutoff_period_px"]) != "" for row in subset),
        })
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired_rows = []
    for minuend, subtrahend in (("APD-SIM-6", "APD-SIM-3"), ("APD-SIM-6", "APD-SIM-9"), ("APD-SIM-9", "APD-SIM-3")):
        for metric in ("psnr", "ssim"):
            values = np.asarray([
                float(index[(minuend, order)][metric]) - float(index[(subtrahend, order)][metric])
                for order in range(30)
            ], dtype=np.float64)
            low, high = _bootstrap_ci(values, rng)
            paired_rows.append({
                "comparison": f"{minuend} minus {subtrahend}", "metric": metric, "n": 30,
                "mean_difference": float(values.mean()), "sd_difference": float(values.std(ddof=1)),
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            })
    _write_csv(target / "summary_mean_sd.csv", mean_rows, tuple(mean_rows[0]))
    _write_csv(target / "summary_median_iqr.csv", median_rows, tuple(median_rows[0]))
    _write_csv(target / "summary_by_structure.csv", structure_rows, tuple(structure_rows[0]))
    _write_csv(target / "paired_protocol_differences.csv", paired_rows, tuple(paired_rows[0]))
    _write_csv(target / "frc_censoring_summary.csv", censor_rows, tuple(censor_rows[0]))
    table = []
    for plan in PLANS:
        method_stats = {row["metric"]: row for row in mean_rows if row["method"] == plan.method}
        method_median = {row["metric"]: row for row in median_rows if row["method"] == plan.method}
        censor = next(row for row in censor_rows if row["method"] == plan.method)
        table.append({
            "method": plan.method,
            "psnr_mean_sd": f"{method_stats['psnr']['mean']:.4f} +/- {method_stats['psnr']['sd']:.4f}",
            "psnr_median_iqr": f"{method_median['psnr']['median']:.4f} [{method_median['psnr']['q1']:.4f}, {method_median['psnr']['q3']:.4f}]",
            "ssim_mean_sd": f"{method_stats['ssim']['mean']:.6f} +/- {method_stats['ssim']['sd']:.6f}",
            "ssim_median_iqr": f"{method_median['ssim']['median']:.6f} [{method_median['ssim']['q1']:.6f}, {method_median['ssim']['q3']:.6f}]",
            "frc_period_px_mean_sd": f"{method_stats['frc_cutoff_period_px']['mean']:.4f} +/- {method_stats['frc_cutoff_period_px']['sd']:.4f}",
            "frc_auc_mean_sd": f"{method_stats['frc_auc']['mean']:.6f} +/- {method_stats['frc_auc']['sd']:.6f}",
            "frc_right_censored": censor["right_censored"],
        })
    _write_csv(target / "Table_APD369_final.csv", table, tuple(table[0]))
    latex = [
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Method & PSNR (dB) & SSIM & FRC period (px) & Right-censored \\\\", "\\midrule",
    ]
    for row in table:
        latex.append(
            f"{row['method']} & {row['psnr_mean_sd']} & {row['ssim_mean_sd']} & "
            f"{row['frc_period_px_mean_sd']} & {row['frc_right_censored']}/30 \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (target / "Table_APD369_final.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return {"table": table, "paired": paired_rows}


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    rows = _evaluate(output_root)
    metrics = _finalize_metrics(output_root, rows)
    receipt = {
        "schema_version": 1, "status": "APD369_NUMERICAL_EVALUATION_COMPLETE",
        "output_root": str(output_root.resolve()), "per_fov_rows": len(rows),
        "protocol_forward_call_count": len(rows), "common_nine_frame_subsampling_count": 0,
        "best_of_n_count": 0, "principal_trajectory_count_per_fov_protocol": 1,
        "old_test_369_access_count": 0, "external_baseline_execution_count": 0,
        "training_execution_count": 0, "all_outputs_finite": True,
        "frozen_forward_source": str(FROZEN_FORWARD_SOURCE.resolve()),
        "frozen_forward_source_sha256": FROZEN_FORWARD_SHA256,
        "measurement_generation_device": "cpu (frozen formal acquisition RNG contract)",
        "metrics": metrics,
    }
    atomic_json(output_root / "09_final" / "numerical_evaluation_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "output_root": result["output_root"]}, indent=2))

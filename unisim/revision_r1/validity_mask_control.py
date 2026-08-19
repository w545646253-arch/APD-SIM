"""Reviewer 1 Comment 1: controlled Stage-1 validity-mask experiment.

The reconstruction phase is deliberately completed for all 30 sealed fields
before any GT file is opened.  The two conditions share one immutable raw
tensor, one slotted conditioning tensor, one Gaussian realization, one EMA
model, and (in Stage 2) one physical mask/operator/configuration object.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon
import tifffile
import torch
import torch.nn.functional as F

from unisim.protocol_runtime import require_protocol
from unisim.sim_forward_2d import embed_raw_to_slots_2d
from .physmap6_core import RefinementConfig, masked_refine
from . import physmap6_experiment as experiment
from .physmap6_pipeline import (
    NORMALIZATION_HASH,
    PROTOCOL_HASH,
    PROTOCOL_ID,
    RAW_ORDER,
    STAGE1_POLICY,
    load_stage1,
    make_sim_config,
    sha_array,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "reviewer1_validity_mask_control"
FORMAL_RESULT_ROOT = (
    ROOT / "outputs" / "reviewer1_physmap6_strict" / "20260813T183229Z"
)
CHECKPOINT = ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt"
CONFIG = ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
SEALED_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "sealed_test_manifest.json"
BUNDLE_MANIFEST = (
    ROOT / "outputs" / "OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
    / "01_shared_contract" / "test30_dmd6_manifest.tsv"
)
EXPECTED_CHECKPOINT_SHA256 = "10fb16662a8b71b877f2cab81bdc151dcded92f6efd1c4b006306b901a8adff7"
EXPECTED_SEALED_MANIFEST_SHA256 = "495b554a19596b299f1bc5192ee3b1eb071414fde361c2f2eae0c17f2878d794"
EXPECTED_BUNDLE_MANIFEST_SHA256 = "91a7a8d4f7c264d0909bace489823e31fe5f09148a0632d476b4141abb17a526"
CORRECT_LOGICAL_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0)
BLIND_LOGICAL_MASK = (1, 1, 1, 1, 1, 1, 1, 1, 1)
CORRECT_MASK = CORRECT_LOGICAL_MASK + (0,) * 6
BLIND_MASK = BLIND_LOGICAL_MASK + (0,) * 6
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_RESAMPLES = 10_000
REFINEMENT_CONFIG = RefinementConfig()
REPRESENTATIVE_SELECTION_RULE = (
    "lowest sealed test-manifest order within each structure class, fixed before inference"
)
FORMAL_SNAPSHOT_ROOTS = (
    FORMAL_RESULT_ROOT,
    CHECKPOINT.parent,
    BUNDLE_MANIFEST.parent,
    SEALED_MANIFEST.parent,
)

PER_FOV_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure_class",
    "raw_stack_sha256", "conditioning_tensor_sha256", "gaussian_sha256",
    "diffusion_seed", "checkpoint_sha256", "ema_branch",
    "Stage1_correct_PSNR", "Stage1_maskblind_PSNR",
    "Stage1_correct_SSIM", "Stage1_maskblind_SSIM",
    "Final_correct_PSNR", "Final_maskblind_PSNR",
    "Final_correct_SSIM", "Final_maskblind_SSIM",
    "Final_correct_observed_NRMSE", "Final_maskblind_observed_NRMSE",
    "Stage1_PSNR_effect", "Stage1_SSIM_effect", "Final_PSNR_effect",
    "Final_SSIM_effect", "Final_observed_NRMSE_effect",
)

ENDPOINTS = (
    ("final_psnr", "Final PSNR", "Final_correct_PSNR", "Final_maskblind_PSNR", "dB", "primary"),
    ("stage1_psnr", "Stage 1 PSNR", "Stage1_correct_PSNR", "Stage1_maskblind_PSNR", "dB", "secondary"),
    ("stage1_ssim", "Stage 1 SSIM", "Stage1_correct_SSIM", "Stage1_maskblind_SSIM", "unitless", "secondary"),
    ("final_ssim", "Final SSIM", "Final_correct_SSIM", "Final_maskblind_SSIM", "unitless", "secondary"),
    (
        "final_observed_nrmse", "Final observed-slot NRMSE",
        "Final_correct_observed_NRMSE", "Final_maskblind_observed_NRMSE",
        "unitless", "secondary",
    ),
)


class ValidityMaskControlBlocked(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(
            value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default
        ).encode("utf-8") + b"\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue().encode("utf-8"))


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def allocate_run_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = OUTPUT_ROOT / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique UTC output directory")


def tree_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValidityMaskControlBlocked("R1C1_FORMAL_AUTHORITY_MISSING", str(root))
    entries = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix().lower()):
        stat = path.stat()
        entries.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": sha_file(path),
        })
    return {
        "root": str(root.resolve()),
        "file_count": len(entries),
        "aggregate_sha256": hashlib.sha256(canonical_bytes(entries)).hexdigest(),
        "entries": entries,
    }


def formal_snapshots() -> dict[str, dict[str, Any]]:
    return {str(root.resolve()): tree_snapshot(root) for root in FORMAL_SNAPSHOT_ROOTS}


def compare_snapshots(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    roots = sorted(set(before) | set(after))
    comparisons = []
    for root in roots:
        left = before.get(root)
        right = after.get(root)
        equal = left == right
        comparisons.append({
            "root": root,
            "unchanged": equal,
            "before_file_count": None if left is None else left["file_count"],
            "after_file_count": None if right is None else right["file_count"],
            "before_aggregate_sha256": None if left is None else left["aggregate_sha256"],
            "after_aggregate_sha256": None if right is None else right["aggregate_sha256"],
        })
    return {"all_formal_directories_unmodified": all(item["unchanged"] for item in comparisons), "roots": comparisons}


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(canonical_bytes({"dtype": value.dtype.str, "shape": list(value.shape)}))
        digest.update(b"\n" + value.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    return sha_array(tensor.detach().cpu().numpy().astype(np.float32, copy=False))


def _stage1_schedule() -> list[int]:
    timesteps = np.rint(np.linspace(600, 0, 80)).astype(np.int64).tolist()
    if len(set(timesteps)) != 80 or timesteps[0] != 600 or timesteps[-1] != 0:
        raise RuntimeError("Frozen Stage-1 timestep schedule changed")
    return timesteps


@torch.no_grad()
def stage1_mask_pair(
    raw_frames: torch.Tensor,
    model: torch.nn.Module,
    scheduler: Any,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run the two Stage-1 masks with a byte-identical non-mask condition/noise."""
    if raw_frames.ndim != 4 or tuple(raw_frames.shape[:2]) != (1, 6):
        raise ValueError(f"Stage 1 requires (1,6,H,W), got {tuple(raw_frames.shape)}")
    if raw_frames.dtype != torch.float32:
        raise ValueError("Stage 1 requires float32 raw frames")
    device = raw_frames.device
    height, width = raw_frames.shape[-2:]
    wide = raw_frames.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    slotted, correct_mask = embed_raw_to_slots_2d(raw_frames, PROTOCOL_ID)
    correct_vector = tuple(int(value) for value in correct_mask[0, :, 0, 0].tolist())
    if correct_vector != CORRECT_MASK:
        raise RuntimeError(f"Protocol-correct mask drift: {correct_vector}")
    if not bool(torch.equal(slotted[:, 6:15], torch.zeros_like(slotted[:, 6:15]))):
        raise RuntimeError("Zero-filled conditioning slots 6-14 are not exactly zero")
    blind_mask = correct_mask.clone()
    blind_mask[:, 6:9] = 1.0
    blind_vector = tuple(int(value) for value in blind_mask[0, :, 0, 0].tolist())
    if blind_vector != BLIND_MASK:
        raise RuntimeError(f"Mask-blind mask drift: {blind_vector}")
    differing = tuple(index for index, pair in enumerate(zip(correct_vector, blind_vector)) if pair[0] != pair[1])
    if differing != (6, 7, 8) or any(blind_vector[index] for index in range(9, 15)):
        raise RuntimeError("Only logical validity slots 6-8 may differ")

    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        wide = F.pad(wide, (0, pad_w, 0, pad_h), mode="reflect")
        slotted = F.pad(slotted, (0, pad_w, 0, pad_h), mode="reflect")
        correct_mask = F.pad(correct_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        blind_mask = F.pad(blind_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    gaussian = torch.randn(wide.shape, generator=generator, device=device, dtype=torch.float32)
    current_t = torch.full((1,), 600, device=device, dtype=torch.long)
    x_start = scheduler.q_sample(wide, current_t, gaussian)
    timesteps = _stage1_schedule()

    def run_one(mask: torch.Tensor) -> tuple[torch.Tensor, float, int]:
        x = x_start.clone()
        x0 = wide
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for index, current in enumerate(timesteps):
            timestep = torch.full((1,), current, device=device, dtype=torch.long)
            epsilon = model(torch.cat((x, slotted, mask), dim=1), timestep).float()
            x0 = scheduler.predict_x0(x, timestep, epsilon).clamp(0.0, 1.0)
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            if previous < 0:
                x = x0
            else:
                alpha = scheduler.alpha_bar[previous]
                x = alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * epsilon
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        output = x0[..., :height, :width]
        if tuple(output.shape) != (1, 1, height, width) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("R1C1_NONFINITE_RESULT: Stage 1")
        return output, elapsed, int(peak)

    correct, correct_runtime, correct_peak = run_one(correct_mask)
    blind, blind_runtime, blind_peak = run_one(blind_mask)
    audit = {
        "correct_logical_mask": list(CORRECT_LOGICAL_MASK),
        "maskblind_logical_mask": list(BLIND_LOGICAL_MASK),
        "correct_implementation_mask": list(CORRECT_MASK),
        "maskblind_implementation_mask": list(BLIND_MASK),
        "only_differing_mask_slots": list(differing),
        "padding_slots_9_14_invalid_both": True,
        "zero_filled_conditioning_slots_6_14": True,
        "slotted_conditioning_sha256": _tensor_sha(slotted[..., :height, :width]),
        "correct_slotted_conditioning_sha256": _tensor_sha(slotted[..., :height, :width]),
        "maskblind_slotted_conditioning_sha256": _tensor_sha(slotted[..., :height, :width]),
        "wide_initializer_sha256": _tensor_sha(wide[..., :height, :width]),
        "gaussian_sha256": _tensor_sha(gaussian),
        "correct_gaussian_sha256": _tensor_sha(gaussian),
        "maskblind_gaussian_sha256": _tensor_sha(gaussian),
        "x_start_sha256": _tensor_sha(x_start),
        "ddim_schedule": timesteps,
        "correct_runtime_seconds": correct_runtime,
        "maskblind_runtime_seconds": blind_runtime,
        "correct_peak_gpu_memory_bytes": correct_peak,
        "maskblind_peak_gpu_memory_bytes": blind_peak,
    }
    return correct, blind, audit


def _official_diffusion_seed(raw_hash: str) -> int:
    label = (
        f"APD6_OFFICIAL_R2_WARMSTART_MAP_V1|{EXPECTED_CHECKPOINT_SHA256}|{raw_hash}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") & ((1 << 63) - 1)


def _load_metrics_module() -> Any:
    source = ROOT / "tools" / "official_r2_common_metrics.py"
    expected = "9efd7efcc6ecf126816887a710478f592ecc3b29562003a2ea452e1b93deec9a"
    if sha_file(source) != expected:
        raise ValidityMaskControlBlocked("R1C1_METRIC_AUTHORITY_MISMATCH", str(source))
    spec = importlib.util.spec_from_file_location("r1c1_official_metrics", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load official metrics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _theta_json(theta: Mapping[str, torch.Tensor]) -> str:
    value = {
        key: [float(item) for item in tensor.detach().cpu().reshape(-1).tolist()]
        for key, tensor in sorted(theta.items())
    }
    return canonical_bytes(value).decode("utf-8")


def receipt_contains_config(
    executed_receipt: Mapping[str, Any], base_receipt: Mapping[str, Any]
) -> bool:
    return all(executed_receipt.get(key) == value for key, value in base_receipt.items())


def _save_reconstruction(
    run_dir: Path, order: int, sample_id: str, label: str, tensor: torch.Tensor
) -> dict[str, Any]:
    array = np.ascontiguousarray(tensor[0, 0].detach().cpu().numpy(), dtype=np.float32)
    if array.shape != (1004, 1004) or not np.isfinite(array).all():
        raise ValidityMaskControlBlocked("R1C1_NONFINITE_RESULT", f"{sample_id}/{label}")
    path = run_dir / "native_reconstructions" / f"{order:03d}_{sample_id}_{label}.npy"
    save_npy(path, array)
    stored = np.load(path, allow_pickle=False)
    if not np.array_equal(stored, array):
        raise ValidityMaskControlBlocked("R1C1_OUTPUT_ROUNDTRIP_FAILED", str(path))
    return {
        "path": str(path.resolve()),
        "file_sha256": sha_file(path),
        "array_sha256": sha_array(array),
    }


def reconstruct_all(
    run_dir: Path,
    bundle_rows: Sequence[Mapping[str, str]],
    model: torch.nn.Module,
    scheduler: Any,
    sim_config: Any,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    physical_mask = torch.tensor(CORRECT_MASK, device=device, dtype=torch.float32)
    forward_config_receipt = REFINEMENT_CONFIG.receipt()
    records: list[dict[str, Any]] = []
    sample_audits: list[dict[str, Any]] = []
    for sample_order, row in enumerate(bundle_rows):
        bundle_path = BUNDLE_MANIFEST.parent / row["npz_path"]
        with np.load(bundle_path, allow_pickle=False) as archive:
            if any("gt" in key.lower() for key in archive.files):
                raise ValidityMaskControlBlocked("R1C1_GT_LEAK_IN_RECONSTRUCTION_BUNDLE", str(bundle_path))
            raw_np = np.ascontiguousarray(archive["raw_stack"])
            if raw_np.dtype != np.float32 or raw_np.shape != (6, 1004, 1004):
                raise ValidityMaskControlBlocked("R1C1_TEST_BUNDLE_IDENTITY_MISMATCH", row["sample_id"])
            if sha_array(raw_np) != row["raw_stack_sha256"]:
                raise ValidityMaskControlBlocked("R1C1_RAW_HASH_MISMATCH", row["sample_id"])
            raw = torch.from_numpy(raw_np)[None].to(device=device, dtype=torch.float32)
            theta = experiment._theta_from_archive(archive, device)
            acquisition_noise_seed = int(np.asarray(archive["acquisition_noise_seed"]).reshape(-1)[0])
        raw_pointer = int(raw.data_ptr())
        raw_hash_before = sha_array(raw[0].detach().cpu().numpy())
        diffusion_seed = _official_diffusion_seed(row["raw_stack_sha256"])
        stage1_correct, stage1_blind, stage1_audit = stage1_mask_pair(
            raw, model, scheduler, seed=diffusion_seed
        )

        forward_operator = {"sim_config": sim_config, "theta": theta}
        correct_stage1_hash_before = _tensor_sha(stage1_correct)
        blind_stage1_hash_before = _tensor_sha(stage1_blind)
        correct_refined = masked_refine(
            stage1_correct, raw, physical_mask, geometry, forward_operator, REFINEMENT_CONFIG
        )
        blind_refined = masked_refine(
            stage1_blind, raw, physical_mask, geometry, forward_operator, REFINEMENT_CONFIG
        )
        if correct_refined.configuration_receipt != blind_refined.configuration_receipt:
            raise ValidityMaskControlBlocked("R1C1_STAGE2_CONFIG_MISMATCH", row["sample_id"])
        if not (
            receipt_contains_config(correct_refined.configuration_receipt, forward_config_receipt)
            and receipt_contains_config(blind_refined.configuration_receipt, forward_config_receipt)
        ):
            raise ValidityMaskControlBlocked("R1C1_STAGE2_CONFIG_DRIFT", row["sample_id"])
        if len(correct_refined.objective_history) != 41 or len(blind_refined.objective_history) != 41:
            raise ValidityMaskControlBlocked("R1C1_STAGE2_UPDATE_COUNT_MISMATCH", row["sample_id"])
        if raw_pointer != int(raw.data_ptr()) or raw_hash_before != sha_array(raw[0].detach().cpu().numpy()):
            raise ValidityMaskControlBlocked("R1C1_RAW_TENSOR_MUTATED", row["sample_id"])
        if correct_stage1_hash_before != _tensor_sha(stage1_correct) or blind_stage1_hash_before != _tensor_sha(stage1_blind):
            raise ValidityMaskControlBlocked("R1C1_STAGE1_OUTPUT_MUTATED_BY_STAGE2", row["sample_id"])

        outputs = {
            "stage1_correct": _save_reconstruction(run_dir, sample_order, row["sample_id"], "stage1_correct", stage1_correct),
            "stage1_maskblind": _save_reconstruction(run_dir, sample_order, row["sample_id"], "stage1_maskblind", stage1_blind),
            "final_correct": _save_reconstruction(run_dir, sample_order, row["sample_id"], "final_correct", correct_refined.final_reconstruction),
            "final_maskblind": _save_reconstruction(run_dir, sample_order, row["sample_id"], "final_maskblind", blind_refined.final_reconstruction),
        }
        all_histories = (
            correct_refined.objective_history + blind_refined.objective_history
            + correct_refined.observed_nrmse_history + blind_refined.observed_nrmse_history
        )
        if not np.isfinite(np.asarray(all_histories, dtype=np.float64)).all():
            raise ValidityMaskControlBlocked("R1C1_NONFINITE_RESULT", f"Stage 2 history {row['sample_id']}")
        record = {
            "sample_order": sample_order,
            "sample_id": row["sample_id"],
            "parent_id": row["parent_id"],
            "structure_class": row["structure_class"],
            "sealed_identity_digest": row["sealed_identity_digest"],
            "gt_normalized_array_sha256": row["gt_normalized_array_sha256"],
            "raw_stack_sha256": row["raw_stack_sha256"],
            "raw_npz_sha256": row["npz_sha256"],
            "acquisition_noise_seed": acquisition_noise_seed,
            "diffusion_seed": diffusion_seed,
            "conditioning_tensor_sha256": stage1_audit["slotted_conditioning_sha256"],
            "gaussian_sha256": stage1_audit["gaussian_sha256"],
            "theta_json": _theta_json(theta),
            "outputs": outputs,
            "final_correct_observed_nrmse": float(correct_refined.observed_nrmse_history[-1]),
            "final_maskblind_observed_nrmse": float(blind_refined.observed_nrmse_history[-1]),
        }
        records.append(record)
        sample_audits.append({
            "sample_order": sample_order,
            "sample_id": row["sample_id"],
            "raw_stack_sha256": row["raw_stack_sha256"],
            "raw_hash_equal_across_conditions": True,
            "raw_tensor_data_pointer_equal_across_stage2_conditions": True,
            "conditioning_tensor_byte_equal": (
                stage1_audit["correct_slotted_conditioning_sha256"]
                == stage1_audit["maskblind_slotted_conditioning_sha256"]
            ),
            "same_gaussian_realization": (
                stage1_audit["correct_gaussian_sha256"]
                == stage1_audit["maskblind_gaussian_sha256"]
            ),
            "only_stage1_mask_slots_6_8_differ": stage1_audit["only_differing_mask_slots"] == [6, 7, 8],
            "padding_slots_9_14_invalid_both": stage1_audit["padding_slots_9_14_invalid_both"],
            "stage2_physical_input_identical": True,
            "stage2_physical_mask": list(CORRECT_MASK),
            "stage2_config_same_object_id": id(REFINEMENT_CONFIG),
            "stage2_config_receipts_equal": True,
            "stage2_forward_operator_same_object": True,
            "stage2_updates_correct": len(correct_refined.objective_history) - 1,
            "stage2_updates_maskblind": len(blind_refined.objective_history) - 1,
            "outputs": outputs,
        })
        print(f"[{sample_order + 1:02d}/30] reconstructed {row['sample_id']}", flush=True)
        del raw, theta, stage1_correct, stage1_blind, correct_refined, blind_refined, forward_operator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    phase_receipt = {
        "schema_version": 1,
        "status": "PASS",
        "completed_utc": utc_now(),
        "gt_files_opened_during_reconstruction": 0,
        "n_fields": len(records),
        "correct_logical_mask": list(CORRECT_LOGICAL_MASK),
        "maskblind_logical_mask": list(BLIND_LOGICAL_MASK),
        "correct_implementation_mask": list(CORRECT_MASK),
        "maskblind_implementation_mask": list(BLIND_MASK),
        "stage1_policy": STAGE1_POLICY,
        "stage2_config": forward_config_receipt,
        "samples": sample_audits,
    }
    write_json(run_dir / "VALIDITY_MASK_CONTROL_RECONSTRUCTION_RECEIPT.json", phase_receipt)
    return records, phase_receipt


def _load_saved(path: str, expected_hash: str) -> np.ndarray:
    target = Path(path)
    array = np.ascontiguousarray(np.load(target, allow_pickle=False), dtype=np.float32)
    if sha_array(array) != expected_hash or not np.isfinite(array).all():
        raise ValidityMaskControlBlocked("R1C1_OUTPUT_IDENTITY_MISMATCH", str(target))
    return array


def compute_metrics(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
    representative_orders: set[int],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], str]:
    metrics = _load_metrics_module()
    gt_first_read_utc = utc_now()
    gt_mapping = experiment.load_gt_mapping()
    rows: list[dict[str, Any]] = []
    representative_gt: dict[str, np.ndarray] = {}
    for record in records:
        gt_row = gt_mapping[record["sample_id"]]
        gt = experiment.normalization(tifffile.imread(Path(gt_row["absolute_path"])))
        if sha_array(gt) != record["gt_normalized_array_sha256"]:
            raise ValidityMaskControlBlocked("R1C1_GT_NORMALIZED_IDENTITY_MISMATCH", record["sample_id"])
        expected_gt_hash = gt_row.get("normalized_array_sha256")
        if expected_gt_hash and experiment.legacy_patch_sha_array(gt) != expected_gt_hash:
            # DATASET_MANIFEST uses the legacy plain-array hash contract.
            raise ValidityMaskControlBlocked("R1C1_GT_IDENTITY_MISMATCH", record["sample_id"])
        arrays = {
            label: _load_saved(item["path"], item["array_sha256"])
            for label, item in record["outputs"].items()
        }
        values = {
            "Stage1_correct_PSNR": float(metrics.psnr_native(gt, arrays["stage1_correct"])),
            "Stage1_maskblind_PSNR": float(metrics.psnr_native(gt, arrays["stage1_maskblind"])),
            "Stage1_correct_SSIM": float(metrics.ssim_native(gt, arrays["stage1_correct"])),
            "Stage1_maskblind_SSIM": float(metrics.ssim_native(gt, arrays["stage1_maskblind"])),
            "Final_correct_PSNR": float(metrics.psnr_native(gt, arrays["final_correct"])),
            "Final_maskblind_PSNR": float(metrics.psnr_native(gt, arrays["final_maskblind"])),
            "Final_correct_SSIM": float(metrics.ssim_native(gt, arrays["final_correct"])),
            "Final_maskblind_SSIM": float(metrics.ssim_native(gt, arrays["final_maskblind"])),
            "Final_correct_observed_NRMSE": float(record["final_correct_observed_nrmse"]),
            "Final_maskblind_observed_NRMSE": float(record["final_maskblind_observed_nrmse"]),
        }
        if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
            raise ValidityMaskControlBlocked("R1C1_NONFINITE_RESULT", f"metrics {record['sample_id']}")
        effects = {
            "Stage1_PSNR_effect": values["Stage1_correct_PSNR"] - values["Stage1_maskblind_PSNR"],
            "Stage1_SSIM_effect": values["Stage1_correct_SSIM"] - values["Stage1_maskblind_SSIM"],
            "Final_PSNR_effect": values["Final_correct_PSNR"] - values["Final_maskblind_PSNR"],
            "Final_SSIM_effect": values["Final_correct_SSIM"] - values["Final_maskblind_SSIM"],
            "Final_observed_NRMSE_effect": (
                values["Final_correct_observed_NRMSE"] - values["Final_maskblind_observed_NRMSE"]
            ),
        }
        rows.append({
            "sample_order": record["sample_order"],
            "sample_id": record["sample_id"],
            "parent_id": record["parent_id"],
            "structure_class": record["structure_class"],
            "raw_stack_sha256": record["raw_stack_sha256"],
            "conditioning_tensor_sha256": record["conditioning_tensor_sha256"],
            "gaussian_sha256": record["gaussian_sha256"],
            "diffusion_seed": record["diffusion_seed"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "ema_branch": "ema",
            **values,
            **effects,
        })
        representative_gt[record["sample_id"]] = (
            gt if int(record["sample_order"]) in representative_orders else np.empty(0)
        )
        print(f"[{int(record['sample_order']) + 1:02d}/30] measured {record['sample_id']}", flush=True)
    write_csv(run_dir / "VALIDITY_MASK_CONTROL_PER_FOV.csv", rows, PER_FOV_FIELDS)
    return rows, {key: value for key, value in representative_gt.items() if value.size}, gt_first_read_utc


def descriptive(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size != 30 or not np.isfinite(vector).all():
        raise ValidityMaskControlBlocked("R1C1_STATISTICS_INVALID", "descriptive vector")
    return {
        "n": int(vector.size),
        "mean": float(np.mean(vector)),
        "sd": float(np.std(vector, ddof=1)),
        "median": float(np.median(vector)),
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
    }


def paired_bootstrap_ci(differences: np.ndarray, seed: int) -> tuple[float, float]:
    vector = np.asarray(differences, dtype=np.float64)
    if vector.shape != (30,) or not np.isfinite(vector).all():
        raise ValidityMaskControlBlocked("R1C1_STATISTICS_INVALID", "paired bootstrap vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 2000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(start + chunk, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, vector.size, size=(stop - start, vector.size), endpoint=False)
        estimates[start:stop] = np.mean(vector[indices], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975], method="linear")
    return float(low), float(high)


def paired_wilcoxon(differences: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(differences, dtype=np.float64)
    nonzero = vector != 0.0
    if not np.any(nonzero):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            vector, alternative="two-sided", zero_method="wilcox",
            correction=False, method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        raise ValidityMaskControlBlocked("R1C1_STATISTICS_INVALID", "Wilcoxon")
    return {
        "test": "two-sided Wilcoxon signed-rank",
        "zero_method": "wilcox",
        "continuity_correction": False,
        "method_requested": "auto",
        "statistic": statistic,
        "p_value": p_value,
        "n_nonzero": int(np.count_nonzero(nonzero)),
        "n_zero": int(vector.size - np.count_nonzero(nonzero)),
    }


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    labels = list(raw_p_values)
    values = np.asarray([raw_p_values[label] for label in labels], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))),
    )
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return {label: float(adjusted[index]) for index, label in enumerate(labels)}


def analyze_statistics(
    run_dir: Path, per_fov: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    method_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    endpoint_details: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for index, (key, label, correct_field, blind_field, unit, role) in enumerate(ENDPOINTS):
        correct = np.asarray([float(row[correct_field]) for row in per_fov], dtype=np.float64)
        blind = np.asarray([float(row[blind_field]) for row in per_fov], dtype=np.float64)
        difference = correct - blind
        correct_stats = descriptive(correct)
        blind_stats = descriptive(blind)
        seed = BOOTSTRAP_SEED + index
        ci_low, ci_high = paired_bootstrap_ci(difference, seed)
        test = paired_wilcoxon(difference)
        raw_p_values[key] = test["p_value"]
        for condition, stats in (("protocol_correct", correct_stats), ("mask_blind", blind_stats)):
            method_rows.append({
                "endpoint": key, "endpoint_label": label, "role": role, "condition": condition,
                "unit": unit, "n": stats["n"], "mean": stats["mean"], "sd": stats["sd"],
                "median": stats["median"], "min": stats["min"], "max": stats["max"],
            })
        endpoint_details[key] = {
            "endpoint": key,
            "endpoint_label": label,
            "role": role,
            "unit": unit,
            "effect_definition": "protocol-correct mask minus mask-blind control",
            "correct": correct_stats,
            "maskblind": blind_stats,
            "paired_mean_difference": float(np.mean(difference)),
            "paired_median_difference": float(np.median(difference)),
            "paired_bootstrap_95_ci_for_mean_difference": [ci_low, ci_high],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": seed,
            "bootstrap_unit": "paired technical image field",
            "bootstrap_interval": "percentile",
            "wilcoxon": test,
        }
    adjusted = holm_adjust(raw_p_values)
    for key, label, _correct_field, _blind_field, unit, role in ENDPOINTS:
        detail = endpoint_details[key]
        detail["holm_adjusted_p_value"] = adjusted[key]
        detail["holm_reject_alpha_0_05"] = adjusted[key] <= 0.05
        effect_rows.append({
            "endpoint": key,
            "endpoint_label": label,
            "role": role,
            "unit": unit,
            "effect_definition": "correct_minus_maskblind",
            "n_paired_technical_image_units": 30,
            "paired_mean_difference": detail["paired_mean_difference"],
            "paired_median_difference": detail["paired_median_difference"],
            "bootstrap_ci_low": detail["paired_bootstrap_95_ci_for_mean_difference"][0],
            "bootstrap_ci_high": detail["paired_bootstrap_95_ci_for_mean_difference"][1],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": detail["bootstrap_seed"],
            "wilcoxon_statistic": detail["wilcoxon"]["statistic"],
            "wilcoxon_p_value": detail["wilcoxon"]["p_value"],
            "holm_adjusted_p_value": detail["holm_adjusted_p_value"],
            "holm_reject_alpha_0_05": detail["holm_reject_alpha_0_05"],
        })
    method_fields = (
        "endpoint", "endpoint_label", "role", "condition", "unit", "n",
        "mean", "sd", "median", "min", "max",
    )
    effect_fields = (
        "endpoint", "endpoint_label", "role", "unit", "effect_definition",
        "n_paired_technical_image_units", "paired_mean_difference", "paired_median_difference",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples", "bootstrap_seed",
        "wilcoxon_statistic", "wilcoxon_p_value", "holm_adjusted_p_value",
        "holm_reject_alpha_0_05",
    )
    write_csv(run_dir / "VALIDITY_MASK_CONTROL_METHOD_STATS.csv", method_rows, method_fields)
    write_csv(run_dir / "VALIDITY_MASK_CONTROL_PAIRED_EFFECTS.csv", effect_rows, effect_fields)
    stage1_effect = endpoint_details["stage1_psnr"]["paired_mean_difference"]
    final_effect = endpoint_details["final_psnr"]["paired_mean_difference"]
    tolerance = 1.0e-12
    if abs(final_effect) < abs(stage1_effect) - tolerance:
        stage2_effect = "attenuated"
    elif abs(final_effect) > abs(stage1_effect) + tolerance:
        stage2_effect = "amplified"
    else:
        stage2_effect = "unchanged"
    attenuation_ratio = None if abs(stage1_effect) <= tolerance else abs(final_effect) / abs(stage1_effect)
    statistics = {
        "schema_version": 1,
        "status": "PASS",
        "n_fields": 30,
        "unit": "paired technical image field; not an independent biological replicate",
        "primary_endpoint": "final_psnr",
        "secondary_endpoints": ["stage1_psnr", "stage1_ssim", "final_ssim", "final_observed_nrmse"],
        "effect_definition": "protocol-correct mask minus mask-blind control",
        "endpoints": endpoint_details,
        "holm": {
            "method": "Holm family-wise correction",
            "alpha": 0.05,
            "family_size": 5,
            "raw_p_values": raw_p_values,
            "adjusted_p_values": adjusted,
            "reject": {key: value <= 0.05 for key, value in adjusted.items()},
        },
        "stage2_mask_effect_assessment": {
            "basis": "absolute paired mean PSNR effect",
            "stage1_mean_effect_db": stage1_effect,
            "final_mean_effect_db": final_effect,
            "classification": stage2_effect,
            "absolute_final_to_stage1_ratio": attenuation_ratio,
        },
    }
    write_json(run_dir / "VALIDITY_MASK_CONTROL_STATISTICS.json", statistics)
    return method_rows, effect_rows, statistics


def _fixed_representatives(bundle_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    representatives = []
    for structure in ("CCP", "ER", "MT"):
        candidates = [row for row in bundle_rows if row["structure_class"] == structure]
        if not candidates:
            raise ValidityMaskControlBlocked("R1C1_REPRESENTATIVE_SELECTION_FAILED", structure)
        selected = min(candidates, key=lambda row: int(row["order"]))
        representatives.append({
            "structure_class": structure,
            "sample_order": int(selected["order"]),
            "sample_id": selected["sample_id"],
            "rule": REPRESENTATIVE_SELECTION_RULE,
        })
    return representatives


def export_representatives(
    run_dir: Path,
    representatives: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    gt_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    by_sample = {record["sample_id"]: record for record in records}
    exported = []
    labels = (
        ("GT", None),
        ("correct_mask_stage1", "stage1_correct"),
        ("maskblind_stage1", "stage1_maskblind"),
        ("correct_mask_final", "final_correct"),
        ("maskblind_final", "final_maskblind"),
    )
    target = run_dir / "representative_cases"
    for selected in representatives:
        sample_id = selected["sample_id"]
        record = by_sample[sample_id]
        class_dir = target / selected["structure_class"]
        for label, output_key in labels:
            if output_key is None:
                array = np.ascontiguousarray(gt_arrays[sample_id], dtype=np.float32)
            else:
                item = record["outputs"][output_key]
                array = _load_saved(item["path"], item["array_sha256"])
            npy_path = class_dir / f"{sample_id}_{label}.npy"
            tiff_path = class_dir / f"{sample_id}_{label}_common_range_0_1.tif"
            save_npy(npy_path, array)
            display = np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)
            class_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(tiff_path, display, photometric="minisblack")
            exported.append({
                "structure_class": selected["structure_class"],
                "sample_order": selected["sample_order"],
                "sample_id": sample_id,
                "image": label,
                "native_npy": str(npy_path.resolve()),
                "native_npy_sha256": sha_file(npy_path),
                "common_range_tiff": str(tiff_path.resolve()),
                "common_range_tiff_sha256": sha_file(tiff_path),
                "common_display_range": [0.0, 1.0],
                "tiff_encoding": "uint16 round(clip(x,0,1)*65535)",
            })
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "selection_rule": REPRESENTATIVE_SELECTION_RULE,
        "selection_fixed_before_method_specific_inspection": True,
        "descriptive_only": True,
        "does_not_replace_30_field_statistics": True,
        "representatives": list(representatives),
        "exports": exported,
    }
    write_json(run_dir / "VALIDITY_MASK_CONTROL_REPRESENTATIVE_RECEIPT.json", receipt)
    return receipt


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def generate_direct_text(run_dir: Path, statistics: Mapping[str, Any]) -> dict[str, str]:
    details = statistics["endpoints"]
    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direct validity-mask control on 30 paired technical image fields. Effects are protocol-correct minus mask-blind.}",
        r"\label{tab:s5_validity_mask_control}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Endpoint & Correct mean $\pm$ SD & Blind mean $\pm$ SD & Mean effect (95\% CI) & Median effect & Holm $p$ \\",
        r"\midrule",
    ]
    table_order = ("stage1_psnr", "stage1_ssim", "final_psnr", "final_ssim", "final_observed_nrmse")
    digits = {"stage1_psnr": 3, "final_psnr": 3, "stage1_ssim": 4, "final_ssim": 4, "final_observed_nrmse": 5}
    latex_names = {
        "stage1_psnr": r"Stage 1 PSNR (dB)",
        "stage1_ssim": r"Stage 1 SSIM",
        "final_psnr": r"Final PSNR (dB)",
        "final_ssim": r"Final SSIM",
        "final_observed_nrmse": r"Final observed-slot NRMSE",
    }
    for key in table_order:
        item = details[key]
        d = digits[key]
        low, high = item["paired_bootstrap_95_ci_for_mean_difference"]
        table_lines.append(
            f"{latex_names[key]} & "
            f"{_fmt(item['correct']['mean'], d)} $\\pm$ {_fmt(item['correct']['sd'], d)} & "
            f"{_fmt(item['maskblind']['mean'], d)} $\\pm$ {_fmt(item['maskblind']['sd'], d)} & "
            f"{_fmt(item['paired_mean_difference'], d)} ({_fmt(low, d)}, {_fmt(high, d)}) & "
            f"{_fmt(item['paired_median_difference'], d)} & "
            f"{item['holm_adjusted_p_value']:.3g} \\\\"
        )
    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize",
        r"Values are means $\pm$ sample SD. Confidence intervals are paired percentile bootstrap intervals with 10,000 resamples. Two-sided Wilcoxon signed-rank tests were adjusted jointly over the five endpoints by Holm's method. Fields are paired technical image units, not independent biological replicates.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    table_path = run_dir / "TABLE_S5_VALIDITY_MASK_CONTROL_DIRECT.tex"
    atomic_write(table_path, "\n".join(table_lines).encode("utf-8"))

    s1p = details["stage1_psnr"]
    s1s = details["stage1_ssim"]
    fp = details["final_psnr"]
    fs = details["final_ssim"]
    fn = details["final_observed_nrmse"]
    assessment = statistics["stage2_mask_effect_assessment"]
    paragraph = (
        "In a prespecified paired validity-mask control across 30 technical image fields, the conditioning values "
        "and Gaussian realization were held fixed while only logical Stage-1 validity slots 6--8 were changed. "
        f"The protocol-correct minus mask-blind Stage-1 effects were {s1p['paired_mean_difference']:+.3f} dB "
        f"(95\\% CI {s1p['paired_bootstrap_95_ci_for_mean_difference'][0]:+.3f} to "
        f"{s1p['paired_bootstrap_95_ci_for_mean_difference'][1]:+.3f}) for PSNR and "
        f"{s1s['paired_mean_difference']:+.4f} (95\\% CI {s1s['paired_bootstrap_95_ci_for_mean_difference'][0]:+.4f} to "
        f"{s1s['paired_bootstrap_95_ci_for_mean_difference'][1]:+.4f}) for SSIM. After identical 40-update "
        f"six-measurement physical refinement, the effects were {fp['paired_mean_difference']:+.3f} dB "
        f"(95\\% CI {fp['paired_bootstrap_95_ci_for_mean_difference'][0]:+.3f} to "
        f"{fp['paired_bootstrap_95_ci_for_mean_difference'][1]:+.3f}) for PSNR, "
        f"{fs['paired_mean_difference']:+.4f} (95\\% CI {fs['paired_bootstrap_95_ci_for_mean_difference'][0]:+.4f} to "
        f"{fs['paired_bootstrap_95_ci_for_mean_difference'][1]:+.4f}) for SSIM, and "
        f"{fn['paired_mean_difference']:+.5f} (95\\% CI {fn['paired_bootstrap_95_ci_for_mean_difference'][0]:+.5f} to "
        f"{fn['paired_bootstrap_95_ci_for_mean_difference'][1]:+.5f}) for observed-slot NRMSE. "
        f"On the absolute mean PSNR-effect scale, Stage 2 {assessment['classification']} the mask effect."
    )
    main_path = run_dir / "MAIN_TEXT_VALIDITY_MASK_RESULT_DIRECT.tex"
    atomic_write(main_path, (paragraph + "\n").encode("utf-8"))

    response = (
        "Response to Reviewer 1, Comment 1\n\n"
        "We thank the reviewer for requesting a direct control of validity-mask handling. We evaluated the frozen "
        "validation-selected APD-SIM-6 EMA checkpoint on the same sealed 30-FOV DMD-6F bundle used in the formal "
        "analysis. The two Stage-1 conditions used byte-identical six-frame raw data, zero-filled 15-slot conditioning "
        "values, six-frame mean initializer, warm-start timestep, Gaussian realization, DDIM schedule, normalization, "
        "and output support. Their only difference was the logical validity state of slots 6-8: invalid under the "
        "registered two-orientation protocol and valid in the mask-blind control. Implementation-only padding slots "
        "9-14 remained invalid in both conditions.\n\n"
        "For Stage 2, each corresponding Stage-1 result was refined with the same physical six-frame tensor, registered "
        "two-orientation forward operator, protocol-correct mask, and one shared RefinementConfig object (Adam, 40 "
        "updates, lr=5e-3, lambda_prior=0, float32, clipping to [0,1], no early stopping). Zero-filled logical slots "
        "6-8 were not treated as physical observations because no third DMD direction exists in DMD_6F_2O3P.\n\n"
        f"Across the 30 paired technical image fields, the correct-minus-mask-blind Stage-1 effect was "
        f"{s1p['paired_mean_difference']:+.3f} dB in PSNR (95% CI "
        f"{s1p['paired_bootstrap_95_ci_for_mean_difference'][0]:+.3f} to {s1p['paired_bootstrap_95_ci_for_mean_difference'][1]:+.3f}) "
        f"and {s1s['paired_mean_difference']:+.4f} in SSIM (95% CI "
        f"{s1s['paired_bootstrap_95_ci_for_mean_difference'][0]:+.4f} to {s1s['paired_bootstrap_95_ci_for_mean_difference'][1]:+.4f}). "
        f"The final effects were {fp['paired_mean_difference']:+.3f} dB in PSNR (95% CI "
        f"{fp['paired_bootstrap_95_ci_for_mean_difference'][0]:+.3f} to {fp['paired_bootstrap_95_ci_for_mean_difference'][1]:+.3f}), "
        f"{fs['paired_mean_difference']:+.4f} in SSIM (95% CI "
        f"{fs['paired_bootstrap_95_ci_for_mean_difference'][0]:+.4f} to {fs['paired_bootstrap_95_ci_for_mean_difference'][1]:+.4f}), "
        f"and {fn['paired_mean_difference']:+.5f} in observed-slot NRMSE (95% CI "
        f"{fn['paired_bootstrap_95_ci_for_mean_difference'][0]:+.5f} to {fn['paired_bootstrap_95_ci_for_mean_difference'][1]:+.5f}). "
        f"Holm-adjusted p values across all five endpoints are reported in Table S5. Stage 2 "
        f"{assessment['classification']} the mask effect on the absolute paired mean PSNR scale. The 30 fields are "
        "technical image units and are not claimed as independent biological replicates.\n"
    )
    response_path = run_dir / "REVIEWER1_COMMENT1_RESPONSE_DIRECT.txt"
    atomic_write(response_path, response.encode("utf-8"))
    return {"table_s5": str(table_path.resolve()), "main_text": str(main_path.resolve()), "reviewer_response": str(response_path.resolve())}


def run_test_gate(run_dir: Path) -> dict[str, Any]:
    targets = [ROOT / "tests" / "test_r1c1_validity_mask_control.py"]
    targets.extend(sorted((ROOT / "tests").glob("test_r1c3_*.py")))
    command = [sys.executable, "-m", "pytest", "-q", *[str(path) for path in targets]]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    log = completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
    log_path = run_dir / "VALIDITY_MASK_CONTROL_TEST_LOG.txt"
    atomic_write(log_path, log.encode("utf-8"))
    receipt = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": command,
        "targets": [str(path.resolve()) for path in targets],
        "returncode": completed.returncode,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha_file(log_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    write_json(run_dir / "VALIDITY_MASK_CONTROL_TEST_RECEIPT.json", receipt)
    if completed.returncode != 0:
        raise ValidityMaskControlBlocked("R1C1_TEST_GATE_FAILED", f"return code {completed.returncode}")
    return receipt


def artifact_manifest(run_dir: Path) -> list[dict[str, Any]]:
    excluded = {"VALIDITY_MASK_CONTROL_RECEIPT.json", "STATUS.json"}
    result = []
    for path in sorted((item for item in run_dir.rglob("*") if item.is_file() and item.name not in excluded), key=lambda item: item.as_posix().lower()):
        result.append({
            "relative_path": path.relative_to(run_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha_file(path),
        })
    return result


def main(entry_path: Path) -> int:
    run_dir = allocate_run_dir()
    started_utc = utc_now()
    formal_before: dict[str, dict[str, Any]] | None = None
    status_path = run_dir / "STATUS.json"
    write_json(status_path, {
        "status": "R1C1_VALIDITY_MASK_CONTROL_RUNNING",
        "started_utc": started_utc,
        "entry_script": str(entry_path.resolve()),
        "output_directory": str(run_dir.resolve()),
    })
    try:
        if len(sys.argv) != 1:
            raise ValidityMaskControlBlocked("R1C1_ARGUMENT_CONTRACT_VIOLATION", "entry script accepts no arguments")
        if ROOT.resolve() != entry_path.resolve().parent:
            raise ValidityMaskControlBlocked("R1C1_PROJECT_ROOT_MISMATCH", str(entry_path))
        if sha_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
            raise ValidityMaskControlBlocked("R1C1_CHECKPOINT_IDENTITY_MISMATCH", str(CHECKPOINT))
        if sha_file(SEALED_MANIFEST) != EXPECTED_SEALED_MANIFEST_SHA256:
            raise ValidityMaskControlBlocked("R1C1_SEALED_MANIFEST_IDENTITY_MISMATCH", str(SEALED_MANIFEST))
        if sha_file(BUNDLE_MANIFEST) != EXPECTED_BUNDLE_MANIFEST_SHA256:
            raise ValidityMaskControlBlocked("R1C1_TEST_BUNDLE_MANIFEST_IDENTITY_MISMATCH", str(BUNDLE_MANIFEST))
        spec = require_protocol(PROTOCOL_ID)
        if (
            spec.protocol_hash != PROTOCOL_HASH
            or tuple(spec.raw_frame_order) != RAW_ORDER
            or tuple(spec.validity_mask) != CORRECT_MASK
        ):
            raise ValidityMaskControlBlocked("R1C1_PROTOCOL_IDENTITY_MISMATCH", PROTOCOL_ID)
        if not torch.cuda.is_available():
            raise ValidityMaskControlBlocked("R1C1_CUDA_UNAVAILABLE", "formal full-support control requires CUDA")

        print(f"Output: {run_dir}", flush=True)
        print("[preflight] hashing formal authorities for zero-modification audit", flush=True)
        formal_before = formal_snapshots()
        bundle_rows = experiment.load_bundle_rows(verify_payloads=True)
        representatives = _fixed_representatives(bundle_rows)
        representative_orders = {int(item["sample_order"]) for item in representatives}
        write_json(run_dir / "VALIDITY_MASK_CONTROL_REPRESENTATIVE_SELECTION.json", {
            "status": "LOCKED_BEFORE_INFERENCE",
            "rule": REPRESENTATIVE_SELECTION_RULE,
            "representatives": representatives,
        })

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        device = torch.device("cuda:0")
        model, scheduler, checkpoint_metadata = load_stage1(
            config, CHECKPOINT, EXPECTED_CHECKPOINT_SHA256, device
        )
        model.float().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValidityMaskControlBlocked("R1C1_NETWORK_NOT_FROZEN", "requires_grad remained enabled")
        model_hash_before = model_state_sha256(model)
        sim_config = make_sim_config(config)
        protocol_receipt = experiment.protocol_receipt()
        geometry = {
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": PROTOCOL_HASH,
            "raw_frame_order": list(RAW_ORDER),
            "validity_mask": list(CORRECT_MASK),
            "orientation_angles": protocol_receipt["orientation_angles_degree_mod_180"],
            "nominal_phase_values": protocol_receipt["nominal_phase_values_radian"],
        }

        print("[reconstruction] beginning GT-blind paired Stage 1 and Stage 2", flush=True)
        records, reconstruction_receipt = reconstruct_all(
            run_dir, bundle_rows, model, scheduler, sim_config, device, geometry
        )
        reconstruction_completed_utc = reconstruction_receipt["completed_utc"]
        model_hash_after = model_state_sha256(model)
        if model_hash_before != model_hash_after:
            raise ValidityMaskControlBlocked("R1C1_NETWORK_PARAMETERS_MUTATED", "state hash changed")

        print("[metrics] reconstruction complete; GT access begins only now", flush=True)
        per_fov, representative_gt, gt_first_read_utc = compute_metrics(
            run_dir, records, representative_orders
        )
        method_rows, effect_rows, statistics = analyze_statistics(run_dir, per_fov)
        representative_receipt = export_representatives(
            run_dir, representatives, records, representative_gt
        )
        direct_paths = generate_direct_text(run_dir, statistics)

        print("[tests] running new R1C1 and relevant existing R1C3 tests", flush=True)
        test_receipt = run_test_gate(run_dir)
        print("[audit] re-hashing formal authorities", flush=True)
        formal_after = formal_snapshots()
        formal_comparison = compare_snapshots(formal_before, formal_after)
        if not formal_comparison["all_formal_directories_unmodified"]:
            raise ValidityMaskControlBlocked("R1C1_FORMAL_DIRECTORY_MODIFIED", "authority snapshot changed")

        sample_audits = reconstruction_receipt["samples"]
        checks = {
            "checkpoint_hash_verified": sha_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256,
            "ema_branch_used": True,
            "test_manifest_identity_verified": (
                sha_file(SEALED_MANIFEST) == EXPECTED_SEALED_MANIFEST_SHA256
                and sha_file(BUNDLE_MANIFEST) == EXPECTED_BUNDLE_MANIFEST_SHA256
            ),
            "raw_stack_hash_equality_across_conditions": all(item["raw_hash_equal_across_conditions"] for item in sample_audits),
            "conditioning_tensor_byte_equality": all(item["conditioning_tensor_byte_equal"] for item in sample_audits),
            "same_gaussian_realization": all(item["same_gaussian_realization"] for item in sample_audits),
            "only_mask_slots_6_8_differ_in_stage1": all(item["only_stage1_mask_slots_6_8_differ"] for item in sample_audits),
            "padding_slots_9_14_zero_both_masks": all(item["padding_slots_9_14_invalid_both"] for item in sample_audits),
            "stage2_physical_input_identical": all(item["stage2_physical_input_identical"] for item in sample_audits),
            "stage2_config_identical": all(item["stage2_config_receipts_equal"] for item in sample_audits),
            "stage2_forward_operator_identical": all(item["stage2_forward_operator_same_object"] for item in sample_audits),
            "network_parameters_frozen": model_hash_before == model_hash_after and not any(parameter.requires_grad for parameter in model.parameters()),
            "gt_not_read_during_reconstruction": (
                reconstruction_receipt["gt_files_opened_during_reconstruction"] == 0
                and gt_first_read_utc >= reconstruction_completed_utc
            ),
            "all_metrics_finite": all(
                math.isfinite(float(row[field]))
                for row in per_fov
                for field in PER_FOV_FIELDS
                if field.startswith(("Stage1_", "Final_"))
            ),
            "formal_directories_unmodified": formal_comparison["all_formal_directories_unmodified"],
            "no_external_cuda_exclusivity_gate_called": True,
            "no_cuda_wait_performed": True,
            "no_training_or_finetuning": True,
            "representatives_prespecified_and_descriptive_only": (
                representative_receipt["selection_fixed_before_method_specific_inspection"]
                and representative_receipt["descriptive_only"]
            ),
            "test_gate_passed": test_receipt["status"] == "PASS",
        }
        if not all(checks.values()):
            failed = [key for key, value in checks.items() if not value]
            raise ValidityMaskControlBlocked("R1C1_IDENTITY_CHECK_FAILED", ", ".join(failed))

        audit = {
            "schema_version": 1,
            "status": "PASS",
            "completed_utc": utc_now(),
            "checks": checks,
            "checkpoint": {
                "path": str(CHECKPOINT.resolve()),
                "sha256": EXPECTED_CHECKPOINT_SHA256,
                "branch": "ema",
                "metadata": checkpoint_metadata,
                "model_state_sha256_before": model_hash_before,
                "model_state_sha256_after": model_hash_after,
                "all_parameters_require_grad_false": True,
            },
            "protocol": {
                "protocol_id": PROTOCOL_ID,
                "protocol_hash": PROTOCOL_HASH,
                "raw_order": list(RAW_ORDER),
                "correct_logical_mask": list(CORRECT_LOGICAL_MASK),
                "maskblind_logical_mask": list(BLIND_LOGICAL_MASK),
                "correct_implementation_mask": list(CORRECT_MASK),
                "maskblind_implementation_mask": list(BLIND_MASK),
            },
            "test_bundle": {
                "sealed_manifest": str(SEALED_MANIFEST.resolve()),
                "sealed_manifest_sha256": EXPECTED_SEALED_MANIFEST_SHA256,
                "bundle_manifest": str(BUNDLE_MANIFEST.resolve()),
                "bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST_SHA256,
                "field_count": len(bundle_rows),
                "sample_ids": [row["sample_id"] for row in bundle_rows],
                "raw_stack_sha256": [row["raw_stack_sha256"] for row in bundle_rows],
            },
            "reconstruction_phase_completed_utc": reconstruction_completed_utc,
            "gt_first_read_utc": gt_first_read_utc,
            "refinement_config_object_id": id(REFINEMENT_CONFIG),
            "refinement_config": REFINEMENT_CONFIG.receipt(),
            "sample_identity_checks": sample_audits,
            "formal_directory_zero_modification": formal_comparison,
            "test_receipt": test_receipt,
        }
        write_json(run_dir / "VALIDITY_MASK_CONTROL_AUDIT.json", audit)

        artifacts = artifact_manifest(run_dir)
        receipt = {
            "schema_version": 1,
            "status": "R1C1_VALIDITY_MASK_CONTROL_READY",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "entry_script": str(entry_path.resolve()),
            "output_directory": str(run_dir.resolve()),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "inference_weight_branch": "ema",
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": PROTOCOL_HASH,
            "sealed_test_manifest_sha256": EXPECTED_SEALED_MANIFEST_SHA256,
            "test30_bundle_manifest_sha256": EXPECTED_BUNDLE_MANIFEST_SHA256,
            "n_paired_technical_image_units": 30,
            "statistics": statistics,
            "direct_artifacts": direct_paths,
            "formal_directories_unmodified": True,
            "tests": test_receipt,
            "artifact_count_excluding_receipt_and_status": len(artifacts),
            "artifact_aggregate_sha256": hashlib.sha256(canonical_bytes(artifacts)).hexdigest(),
            "artifacts": artifacts,
        }
        write_json(run_dir / "VALIDITY_MASK_CONTROL_RECEIPT.json", receipt)
        write_json(status_path, {
            "status": "R1C1_VALIDITY_MASK_CONTROL_READY",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "entry_script": str(entry_path.resolve()),
            "output_directory": str(run_dir.resolve()),
            "receipt": str((run_dir / "VALIDITY_MASK_CONTROL_RECEIPT.json").resolve()),
        })
        print("R1C1_VALIDITY_MASK_CONTROL_READY", flush=True)
        print(str(run_dir.resolve()), flush=True)
        return 0
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            blocker = "R1C1_INTERRUPTED"
        elif isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            blocker = "R1C1_CUDA_OOM_NO_POLICY_CHANGE"
        elif isinstance(error, ValidityMaskControlBlocked):
            blocker = error.status
        else:
            blocker = "R1C1_UNEXPECTED_SCIENTIFIC_BLOCKER"
        zero_modification: dict[str, Any] | None = None
        if formal_before is not None:
            try:
                zero_modification = compare_snapshots(formal_before, formal_snapshots())
            except BaseException as snapshot_error:
                zero_modification = {"status": "UNRESOLVED", "error": repr(snapshot_error)}
        failure = {
            "status": blocker,
            "started_utc": started_utc,
            "failed_utc": utc_now(),
            "entry_script": str(entry_path.resolve()),
            "output_directory": str(run_dir.resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_directory_zero_modification": zero_modification,
            "cuda_oom_policy": "fail explicitly; no image/model/precision/batching change",
        }
        write_json(status_path, failure)
        write_json(run_dir / "VALIDITY_MASK_CONTROL_FAILURE.json", failure)
        print(f"{blocker}: {error}", file=sys.stderr, flush=True)
        print(str(run_dir.resolve()), file=sys.stderr, flush=True)
        return 1


__all__ = [
    "BLIND_LOGICAL_MASK", "BLIND_MASK", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED",
    "CORRECT_LOGICAL_MASK", "CORRECT_MASK", "ENDPOINTS", "REFINEMENT_CONFIG",
    "ValidityMaskControlBlocked", "analyze_statistics", "compare_snapshots",
    "main", "paired_bootstrap_ci", "paired_wilcoxon", "receipt_contains_config",
    "stage1_mask_pair",
]

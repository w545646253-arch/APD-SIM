"""Direct repaired APD-SIM-3/6/9 inference for one arbitrary GT image.

This module is deliberately independent of the sealed 30-FOV completion path.
It resolves immutable production checkpoints through the repair and relocation
receipts, generates each protocol measurement in a separate forward call, and
reuses the registered EMA/DDIM/physics/FRC implementations.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal.windows import tukey
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]
TARGET_SHAPE = (1004, 1004)
EXPECTED_REPAIR_STATUS = "DMD9_REPAIRED_APD369_READY"
REPAIR_POINTER = ROOT / "checkpoints" / "provenance" / "DMD9_REPAIR_R4_CURRENT.json"
FROZEN_FORWARD_SOURCE = (
    ROOT / "audit" / "dmd3_nonfinite_recovery_20260814_010345"
    / "source_backups" / "unisim" / "sim_forward_2d.py"
)
FROZEN_FORWARD_SHA256 = "f067a832c2dbac2da32fb4c3a73ac39047a754985f27ecafa31071786321fdd8"
FORBIDDEN_DMD9_SHA256 = "e4eb12c32041ba99a44ceb479aae431c3892f35f1408269d8d976d55ddb97c47"
FRC_THRESHOLD = 1.0 / 7.0
FRC_SMOOTH_WINDOW = 7
METHODS = ("APD-SIM-3", "APD-SIM-6", "APD-SIM-9")


class SingleGTContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtocolContract:
    method: str
    protocol_id: str
    protocol_hash: str
    raw_order: tuple[str, ...]
    validity_mask: tuple[int, ...]
    valid_slots: tuple[int, ...]
    checkpoint_path: Path
    checkpoint_sha256: str
    config_path: Path
    selected_iteration: int
    inference_mode: str


@dataclass(frozen=True)
class FrozenContract:
    repair_pointer: Path
    formal_output_root: Path
    final_status_path: Path
    selection_receipt_path: Path
    relocation_map_path: Path
    archive_checkpoint_root: Path
    plans: tuple[ProtocolContract, ...]


@dataclass(frozen=True)
class PreparedSingleGT:
    source_path: Path
    source_file_sha256: str
    source_array_sha256: str
    source_shape: tuple[int, int]
    source_dtype: str
    crop_xywh: tuple[int, int, int, int]
    native_crop_array_sha256: str
    normalized_array_sha256: str
    normalization: str
    resize: bool
    interpolation: bool
    transformation: str
    normalized: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode("utf-8"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SingleGTContractError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(stream.getvalue())
    os.replace(temporary, path)


def _plan_by_protocol(contract: FrozenContract, protocol_id: str) -> ProtocolContract:
    matches = [plan for plan in contract.plans if plan.protocol_id == protocol_id]
    if len(matches) != 1:
        raise SingleGTContractError(f"unresolved protocol plan: {protocol_id}")
    return matches[0]


def _validate_plan(plan: ProtocolContract) -> None:
    from unisim.protocol_runtime import require_protocol

    spec = require_protocol(plan.protocol_id)
    checks = {
        "protocol_hash": spec.protocol_hash == plan.protocol_hash,
        "raw_order": tuple(spec.raw_frame_order) == plan.raw_order,
        "validity_mask": tuple(spec.validity_mask) == plan.validity_mask,
        "valid_slots": tuple(spec.valid_slots) == plan.valid_slots,
        "checkpoint_exists": plan.checkpoint_path.is_file(),
        "config_exists": plan.config_path.is_file(),
        "checkpoint_hash": plan.checkpoint_path.is_file() and sha256_file(plan.checkpoint_path) == plan.checkpoint_sha256,
        "forbidden_r3_rejected": plan.checkpoint_sha256 != FORBIDDEN_DMD9_SHA256,
    }
    if not all(checks.values()):
        raise SingleGTContractError(f"protocol/checkpoint contract failed for {plan.method}: {checks}")


def load_apd369_frozen_contract() -> FrozenContract:
    """Resolve production identities only through frozen repair/relocation receipts."""
    if not REPAIR_POINTER.is_file():
        raise SingleGTContractError(f"repair pointer absent: {REPAIR_POINTER}")
    pointer = read_json(REPAIR_POINTER)
    formal_value = Path(pointer["formal_output_root"]); formal_output_root = (ROOT / formal_value).resolve() if not formal_value.is_absolute() else formal_value.resolve()
    final_status_path = formal_output_root / "11_final" / "final_status.json"
    final = read_json(final_status_path)
    if final.get("status") != EXPECTED_REPAIR_STATUS:
        raise SingleGTContractError(f"repair status mismatch: {final_status_path}")
    if int(final.get("formal_test_run_count", -1)) != 1:
        raise SingleGTContractError("repair provenance does not report the frozen one-run result")

    dmd9_value = Path(final["selected_checkpoint_path"]); dmd9_path = (ROOT / dmd9_value).resolve() if not dmd9_value.is_absolute() else dmd9_value.resolve()
    selection_receipt_path = formal_output_root / "checkpoint_selection.json"
    selection = read_json(selection_receipt_path)
    if (
        selection.get("status") != "DMD9_EXISTING_CHECKPOINT_FROZEN"
        or selection.get("selected_checkpoint_sha256") != "62831cc9798c9d005fdbf56b343928cc592646b6e70a16f58399b6da0d01b63e"
        or int(selection.get("selected_iteration", -1)) != 88000
        or selection.get("inference_weight_branch") != "ema"
        or selection.get("inference_mode") != "tiled_320_core_160_single_spatial_noise_field"
    ):
        raise SingleGTContractError("DMD9 repair selection receipt mismatch")

    relocation_map_path = formal_output_root / "10_archive" / "relocation_map.json"
    relocation = read_json(relocation_map_path)
    relocation_rows = [row for row in relocation.get("relocations", []) if row.get("kind") == "checkpoint"]
    if len(relocation_rows) != 1 or relocation_rows[0].get("selected_checkpoint_sha256") != selection["selected_checkpoint_sha256"]:
        raise SingleGTContractError("public relocation identity proof is absent")
    roots = [ROOT / "checkpoints" / "provenance" / "archive_not_redistributed"]

    plans = (
        ProtocolContract(
            "APD-SIM-3", "DMD_3F_1O3P",
            "e1e70fcfab3b97359fb0b9a44dfcace166922eaf8585927de5b9c9091fdc79e9",
            ("X0", "X120", "X240"), (1, 1, 1) + (0,) * 12, (0, 1, 2),
            ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "best.pt",
            "c749e25f625c88dc5f1fbb84ebb1854f760cad21f087c58ed4bf7e734e3919b5",
            ROOT / "configs" / "apd_dmd_r2" / "train3_formal_restart_simple_r1.json",
            96000, "monolithic_registered",
        ),
        ProtocolContract(
            "APD-SIM-6", "DMD_6F_2O3P",
            "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83",
            ("H0", "H120", "H240", "V0", "V120", "V240"), (1,) * 6 + (0,) * 9, tuple(range(6)),
            ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt",
            "10fb16662a8b71b877f2cab81bdc151dcded92f6efd1c4b006306b901a8adff7",
            ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json",
            96000, "monolithic_registered",
        ),
        ProtocolContract(
            "APD-SIM-9", "DMD_9F_3O3P",
            "449670667c6ecb043fc55a303872a9e47cddeceb9ef97204b087ca3d45b095e3",
            ("X0", "X120", "X240", "Y0", "Y120", "Y240", "Z0", "Z120", "Z240"),
            (1,) * 9 + (0,) * 6, tuple(range(9)), dmd9_path,
            "62831cc9798c9d005fdbf56b343928cc592646b6e70a16f58399b6da0d01b63e",
            ROOT / "configs" / "apd_dmd_r2" / "train9_formal.json",
            88000, "tiled_320_core_160_single_spatial_noise_field",
        ),
    )
    contract = FrozenContract(
        REPAIR_POINTER.resolve(), formal_output_root, final_status_path,
        selection_receipt_path, relocation_map_path, roots[0], plans,
    )
    for plan in contract.plans:
        _validate_plan(plan)
    return contract


def _read_native_2d(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SingleGTContractError(f"GT file absent: {path}")
    if path.suffix.lower() in {".tif", ".tiff"}:
        value = tifffile.imread(path)
    elif path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        raise SingleGTContractError("GT must be a single-channel TIFF/TIFF or NPY")
    array = np.squeeze(np.asarray(value))
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise SingleGTContractError(f"GT must be one finite 2-D channel, got {np.asarray(value).shape}")
    if not float(array.max()) > float(array.min()):
        raise SingleGTContractError("GT has no intensity range")
    return np.ascontiguousarray(array)


def prepare_single_gt(
    gt_path: str | Path,
    crop_xy: tuple[int, int] | None = None,
    output_dir: Path | None = None,
) -> PreparedSingleGT:
    """Crop without resize/interpolation, then apply the frozen GT normalization."""
    from unisim.revision_r1.dataset_fig5_audit_identity import normalize_image

    source_path = Path(gt_path).expanduser().resolve()
    source = _read_native_2d(source_path)
    height, width = source.shape
    if height < TARGET_SHAPE[0] or width < TARGET_SHAPE[1]:
        raise SingleGTContractError(f"GT is smaller than 1004x1004: {source.shape}")
    if crop_xy is None:
        x0 = (width - TARGET_SHAPE[1]) // 2
        y0 = (height - TARGET_SHAPE[0]) // 2
    else:
        x0, y0 = map(int, crop_xy)
    if x0 < 0 or y0 < 0 or x0 + 1004 > width or y0 + 1004 > height:
        raise SingleGTContractError(f"crop ({x0},{y0},1004,1004) lies outside {source.shape}")
    crop = np.ascontiguousarray(source[y0:y0 + 1004, x0:x0 + 1004])
    normalized = normalize_image(crop)
    if normalized.shape != TARGET_SHAPE or normalized.dtype != np.float32 or not np.isfinite(normalized).all():
        raise SingleGTContractError("formal GT normalization contract failed")
    result = PreparedSingleGT(
        source_path=source_path,
        source_file_sha256=sha256_file(source_path),
        source_array_sha256=array_sha256(source),
        source_shape=(height, width),
        source_dtype=str(source.dtype),
        crop_xywh=(x0, y0, 1004, 1004),
        native_crop_array_sha256=array_sha256(crop),
        normalized_array_sha256=array_sha256(normalized),
        normalization="percentile_0.5_99.5_clip_0_1_once_shared_by_all_protocols",
        resize=False,
        interpolation=False,
        transformation="IDENTITY_NO_RESIZE" if source.shape == TARGET_SHAPE and (x0, y0) == (0, 0) else "CROP_1004_NO_RESIZE",
        normalized=normalized,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_npy(output_dir / "prepared_gt_native.npy", crop)
        tifffile.imwrite(output_dir / "prepared_gt_native.tif", crop)
        _atomic_npy(output_dir / "prepared_gt_normalized.npy", normalized)
        tifffile.imwrite(output_dir / "prepared_gt_normalized.tif", normalized)
        receipt = asdict(result)
        receipt.pop("normalized")
        receipt.update({
            "source_path": str(source_path),
            "original_path": str(source_path), "original_shape": [height, width],
            "original_dtype": str(source.dtype), "crop_coordinates_xywh": [x0, y0, 1004, 1004],
            "cropped_array_sha256": result.native_crop_array_sha256,
        })
        atomic_json(output_dir / "input_preparation_receipt.json", receipt)
    return result


def _load_frozen_forward() -> Any:
    if not FROZEN_FORWARD_SOURCE.is_file() or sha256_file(FROZEN_FORWARD_SOURCE) != FROZEN_FORWARD_SHA256:
        raise SingleGTContractError("frozen production forward source identity mismatch")
    name = "unisim._single_gt_forward_f067a832"
    spec = importlib.util.spec_from_file_location(name, FROZEN_FORWARD_SOURCE)
    if spec is None or spec.loader is None:
        raise SingleGTContractError("cannot import frozen production forward")
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


def generate_protocol_measurement(
    gt: np.ndarray,
    protocol_id: str,
    seed: int,
    contract: FrozenContract | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one protocol stack in one independent production forward call."""
    from unisim.protocol_runtime import require_protocol

    frozen = contract or load_apd369_frozen_contract()
    plan = _plan_by_protocol(frozen, protocol_id)
    spec = require_protocol(protocol_id)
    config = read_json(plan.config_path)
    forward = _load_frozen_forward()
    sim_config = _sim_config(forward, config)
    gt_array = np.ascontiguousarray(gt, dtype=np.float32)
    if gt_array.shape != TARGET_SHAPE or not np.isfinite(gt_array).all():
        raise SingleGTContractError("measurement GT contract failed")
    gt_tensor = torch.from_numpy(gt_array)[None, None].to(dtype=torch.float32)
    theta = forward.nominal_theta_2d(sim_config, gt_tensor.device)
    generator = torch.Generator(device=gt_tensor.device).manual_seed(int(seed))
    call_uuid = str(uuid.uuid4())
    with torch.no_grad():
        raw, _ = forward.forward_protocol_sim_2d(
            gt_tensor, sim_config, protocol_id, theta=dict(theta),
            randomize=False, noise_generator=generator,
        )
    array = np.ascontiguousarray(raw[0].cpu().numpy(), dtype=np.float32)
    if array.shape != (spec.frame_count, 1004, 1004) or not np.isfinite(array).all():
        raise SingleGTContractError(f"raw stack contract failed: {protocol_id} {array.shape}")
    receipt = {
        "protocol_id": protocol_id,
        "protocol_hash": plan.protocol_hash,
        "raw_order": list(plan.raw_order),
        "valid_slots": list(plan.valid_slots),
        "validity_mask": list(plan.validity_mask),
        "measurement_seed": int(seed),
        "generation_call_uuid": call_uuid,
        "generation_call_kind": "independent_protocol_forward",
        "source_nine_frame_subsampling": False,
        "raw_stack_shape": list(array.shape),
        "raw_stack_dtype": str(array.dtype),
        "raw_stack_sha256": array_sha256(array),
        "forward_source": str(FROZEN_FORWARD_SOURCE.resolve()),
        "forward_source_sha256": FROZEN_FORWARD_SHA256,
    }
    return array, receipt


def _checkpoint_metadata_check(plan: ProtocolContract, metadata: Mapping[str, Any]) -> None:
    checks = {
        "protocol_id": metadata.get("training_protocol_id") == plan.protocol_id,
        "protocol_hash": metadata.get("training_protocol_hash") == plan.protocol_hash,
        "raw_order": tuple(metadata.get("raw_frame_order", ())) == plan.raw_order,
        "validity_mask": tuple(metadata.get("validity_mask", ())) == plan.validity_mask,
        "valid_slots": tuple(metadata.get("valid_slots", ())) == plan.valid_slots,
        "frame_count": int(metadata.get("frame_count", -1)) == len(plan.raw_order),
        "selected_iteration": int(metadata.get("global_step", -1)) == plan.selected_iteration,
        "architecture_contract": metadata.get("architecture_contract") == "APD_DMD_R2_STRICT_2D_CONV_V1",
        "selection_rule": metadata.get("checkpoint_selection_rule") == "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1",
        "completion": metadata.get("completion_status") == "FORMAL_TRAINING_COMPLETE",
    }
    if not all(checks.values()):
        raise SingleGTContractError(f"checkpoint metadata failed for {plan.method}: {checks}")


def reconstruct_apd_protocol(
    gt: np.ndarray,
    raw: np.ndarray,
    protocol_id: str,
    checkpoint: str | Path | None,
    seed: int,
    contract: FrozenContract | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run registered EMA DDIM Stage-1 and exactly-40-update Stage-2."""
    from unisim.revision_r1 import frame_budget_r1c2 as fb
    from unisim.revision_r1.physmap6_pipeline import (
        load_stage1_registered,
        stage1_reconstruct_registered,
        stage1_reconstruct_registered_tiled,
    )
    from unisim.sim_forward_2d import nominal_theta_2d

    frozen = contract or load_apd369_frozen_contract()
    plan = _plan_by_protocol(frozen, protocol_id)
    if checkpoint is not None and Path(checkpoint).resolve() != plan.checkpoint_path.resolve():
        raise SingleGTContractError(f"caller checkpoint differs from frozen plan: {checkpoint}")
    if not torch.cuda.is_available():
        raise SingleGTContractError("CUDA is required for production APD reconstruction")
    device = torch.device("cuda:0")
    config = read_json(plan.config_path)
    checkpoint_hash_before = sha256_file(plan.checkpoint_path)
    if checkpoint_hash_before != plan.checkpoint_sha256:
        raise SingleGTContractError(f"checkpoint changed before execution: {plan.method}")
    model, scheduler, metadata = load_stage1_registered(
        config, plan.checkpoint_path, plan.checkpoint_sha256, device,
        protocol_id=protocol_id,
    )
    _checkpoint_metadata_check(plan, metadata)
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))[None].to(device=device)
    if protocol_id == "DMD_9F_3O3P":
        stage1, stage1_seconds, peak_bytes = stage1_reconstruct_registered_tiled(
            raw_tensor, model, scheduler, protocol_id=protocol_id, seed=int(seed),
            tile_size=320, core_size=160, tile_batch_size=4,
        )
        stage1_contract = {
            "mode": "tiled", "tile_size": 320, "core_size": 160,
            "shared_single_spatial_noise_field": True,
            "deterministic_stitching": True,
            "single_block_1004_inference": False,
        }
    else:
        stage1, stage1_seconds, peak_bytes = stage1_reconstruct_registered(
            raw_tensor, model, scheduler, protocol_id=protocol_id, seed=int(seed)
        )
        stage1_contract = {"mode": "monolithic_registered"}
    sim_config = fb._config_for_sim(config)
    theta = nominal_theta_2d(sim_config, device)
    refined, stage2_seconds, final_objective, observed_nrmse = fb.refine_protocol(
        stage1, raw_tensor, protocol_id, sim_config, theta
    )
    output = np.ascontiguousarray(
        np.clip(refined[0, 0].detach().cpu().numpy(), 0.0, 1.0), dtype=np.float32
    )
    if output.shape != TARGET_SHAPE or output.dtype != np.float32 or not np.isfinite(output).all():
        raise SingleGTContractError(f"harmonized reconstruction contract failed: {plan.method}")
    checkpoint_hash_after = sha256_file(plan.checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise SingleGTContractError(f"checkpoint changed during inference: {plan.method}")
    receipt = {
        "method": plan.method,
        "protocol_id": protocol_id,
        "protocol_hash": plan.protocol_hash,
        "checkpoint_path": str(plan.checkpoint_path.resolve()),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_modified": False,
        "selected_iteration": plan.selected_iteration,
        "weight_branch": "ema",
        "ema_finite": True,
        "architecture_hash": metadata.get("architecture_hash"),
        "architecture_contract": metadata.get("architecture_contract"),
        "raw_order": list(plan.raw_order),
        "validity_mask": list(plan.validity_mask),
        "diffusion_seed": int(seed),
        "ddim_steps": 80,
        "best_of_n": False,
        "principal_trajectory_count": 1,
        "stage1": stage1_contract,
        "stage1_runtime_seconds": stage1_seconds,
        "stage1_peak_bytes": peak_bytes,
        "stage2": {
            "updates": 40, "optimizer": "Adam", "learning_rate": 0.005,
            "lambda_prior": 0.0, "clip": [0.0, 1.0],
            "runtime_seconds": stage2_seconds, "final_objective": final_objective,
            "observed_frame_nrmse": observed_nrmse,
        },
        "output_sha256": array_sha256(output),
    }
    del raw_tensor, stage1, refined, model, scheduler
    torch.cuda.empty_cache()
    return output, receipt


def _save_raw(run_dir: Path, plan: ProtocolContract, raw: np.ndarray, receipt: dict[str, Any]) -> None:
    target = run_dir / "01_raw_measurements" / plan.protocol_id
    target.mkdir(parents=True, exist_ok=True)
    _atomic_npy(target / "raw_stack.npy", raw)
    for index, label in enumerate(plan.raw_order):
        tifffile.imwrite(target / f"{index:02d}_{label}.tif", raw[index])
    atomic_json(target / "raw_stack_receipt.json", receipt)


def _save_float_pair(directory: Path, name: str, value: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(value, dtype=np.float32)
    _atomic_npy(directory / f"{name}.npy", array)
    tifffile.imwrite(directory / f"{name}.tif", array)


def run_single_gt_apd369(
    gt_path: str | Path,
    seed: int = 20260812,
    crop_xy: tuple[int, int] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute three independent measurements and three frozen reconstructions."""
    from unisim.revision_r1.physmap6_experiment import assert_no_external_cuda_compute

    frozen = load_apd369_frozen_contract()
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        root = ROOT / "outputs" / "single_gt_apd369_final_assets"
        run_dir = root / f"{Path(gt_path).stem}_seed{int(seed)}_{timestamp}"
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    prepared = prepare_single_gt(gt_path, crop_xy, run_dir / "00_input")
    gate = assert_no_external_cuda_compute()
    arrays: dict[str, np.ndarray] = {"GT": prepared.normalized}
    raw_receipts: dict[str, Any] = {}
    reconstruction_receipts: dict[str, Any] = {}
    call_ids: set[str] = set()
    for plan in frozen.plans:
        raw, raw_receipt = generate_protocol_measurement(
            prepared.normalized, plan.protocol_id, int(seed), frozen
        )
        if raw_receipt["generation_call_uuid"] in call_ids:
            raise SingleGTContractError("protocol measurements were not independent")
        call_ids.add(raw_receipt["generation_call_uuid"])
        _save_raw(run_dir, plan, raw, raw_receipt)
        prediction, reconstruction_receipt = reconstruct_apd_protocol(
            prepared.normalized, raw, plan.protocol_id, plan.checkpoint_path,
            int(seed), frozen,
        )
        arrays[plan.method] = prediction
        raw_receipts[plan.method] = raw_receipt
        reconstruction_receipts[plan.method] = reconstruction_receipt
        atomic_json(
            run_dir / "08_receipts" / f"reconstruction_{plan.protocol_id}.json",
            reconstruction_receipt,
        )
    for name, array in arrays.items():
        _save_float_pair(run_dir / "02_harmonized_float", name, array)
    return {
        "run_dir": run_dir,
        "contract": frozen,
        "prepared": prepared,
        "arrays": arrays,
        "raw_receipts": raw_receipts,
        "reconstruction_receipts": reconstruction_receipts,
        "initial_gpu_gate": gate,
        "measurement_seed_policy": "measurement_seed = base seed; no protocol offset",
        "diffusion_seed_policy": "diffusion_seed = base seed; one principal trajectory; no protocol offset",
        "independent_protocol_forward_call_count": 3,
        "common_nine_frame_subsampling_count": 0,
        "formal_completion_gate_access_count": 0,
        "training_execution_count": 0,
    }


def _realspace_rgb(value: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.float32)
    first = x <= (1.0 / 3.0)
    second = (x > (1.0 / 3.0)) & (x <= (2.0 / 3.0))
    third = x > (2.0 / 3.0)
    rgb[..., 2][first] = x[first] * 3.0
    t = (x[second] - 1.0 / 3.0) * 3.0
    rgb[..., 1][second] = t
    rgb[..., 2][second] = 1.0
    t = (x[third] - 2.0 / 3.0) * 3.0
    rgb[..., 0][third] = t
    rgb[..., 1][third] = 1.0
    rgb[..., 2][third] = 1.0
    return np.rint(rgb * 255.0).astype(np.uint8)


def _magma_rgb(value: np.ndarray) -> np.ndarray:
    try:
        import matplotlib
    except ImportError as exc:
        raise SingleGTContractError("matplotlib magma LUT is required") from exc
    rgba = matplotlib.colormaps["magma"](np.clip(value, 0.0, 1.0), bytes=True)
    return np.ascontiguousarray(rgba[..., :3], dtype=np.uint8)


def _save_rgb_pair(directory: Path, name: str, rgb: np.ndarray, dpi: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(directory / f"{name}.png", dpi=(dpi, dpi))
    tifffile.imwrite(directory / f"{name}.tif", rgb, photometric="rgb")


def _save_colorbar(path_stem: Path, mapper: Any, labels: Sequence[str], dpi: int) -> None:
    height, width, bar_width = 512, 180, 82
    values = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    field = np.repeat(values, bar_width, axis=1)
    bar = mapper(field)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(Image.fromarray(bar, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    positions = (4, height // 2 - 6, height - 14)
    for label, y in zip(labels, positions, strict=True):
        draw.text((bar_width + 12, y), label, fill="black", font=font)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path_stem.with_suffix(".png"), dpi=(dpi, dpi))
    tifffile.imwrite(path_stem.with_suffix(".tif"), np.asarray(canvas), photometric="rgb")


def _smooth_frc(values: np.ndarray, window: int = FRC_SMOOTH_WINDOW) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    radius = window // 2
    for index in range(values.size):
        segment = values[max(0, index - radius): min(values.size, index + radius + 1)]
        finite = segment[np.isfinite(segment)]
        if finite.size >= 3:
            result[index] = float(finite.mean())
    return result


def _half_bit_threshold(counts: np.ndarray) -> np.ndarray:
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    valid = counts > 0
    root = np.sqrt(counts[valid].astype(np.float64))
    result[valid] = (0.2071 + 1.9102 / root) / (1.2071 + 0.9102 / root)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_single_gt_assets(result: Mapping[str, Any], pixel_size_um: float, dpi: int) -> dict[str, Any]:
    """Create shared-scale images, spectra, exact FRC sources, and metrics."""
    from tools import revision_dmd6_common as common

    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise SingleGTContractError("pixel size must be positive and finite")
    run_dir = Path(result["run_dir"])
    (run_dir / "03_display16").mkdir(parents=True, exist_ok=True)
    arrays = result["arrays"]
    for name, value in arrays.items():
        if value.shape != TARGET_SHAPE or value.dtype != np.float32 or not np.isfinite(value).all():
            raise SingleGTContractError(f"invalid harmonized array: {name}")
        if float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise SingleGTContractError(f"harmonized array outside [0,1]: {name}")

    for name, value in arrays.items():
        tifffile.imwrite(run_dir / "03_display16" / f"{name}.tif", np.rint(value * 65535.0).astype(np.uint16))
        _save_rgb_pair(run_dir / "04_display_rgb", name, _realspace_rgb(value), dpi)
    _save_colorbar(
        run_dir / "04_display_rgb" / "intensity_colorbar_255_128_0",
        _realspace_rgb, ("255", "128", "0"), dpi,
    )

    window_1d = tukey(TARGET_SHAPE[0], alpha=0.20).astype(np.float64)
    window = np.outer(window_1d, window_1d)
    log_spectra: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        centered = value.astype(np.float64) - float(value.mean())
        amplitude = np.abs(np.fft.fftshift(np.fft.fft2(centered * window)))
        log_spectra[name] = np.ascontiguousarray(np.log1p(amplitude), dtype=np.float32)
        _save_float_pair(run_dir / "05_spectra" / "log_amplitude", name, log_spectra[name])
    global_low = min(float(value.min()) for value in log_spectra.values())
    global_high = max(float(value.max()) for value in log_spectra.values())
    if not global_high > global_low:
        raise SingleGTContractError("shared spectrum display range is degenerate")
    spectrum_norm: dict[str, np.ndarray] = {}
    for name, value in log_spectra.items():
        normalized = np.ascontiguousarray((value - global_low) / (global_high - global_low), dtype=np.float32)
        spectrum_norm[name] = normalized
        _save_float_pair(run_dir / "05_spectra" / "normalized_float", name, normalized)
        _save_rgb_pair(run_dir / "05_spectra" / "display_rgb", name, _magma_rgb(normalized), dpi)
    _save_colorbar(
        run_dir / "05_spectra" / "spectrum_energy_colorbar_1_0p5_0",
        _magma_rgb, ("1", "0.5", "0"), dpi,
    )

    metric_api = common.metrics_module()
    frc_results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    gt = arrays["GT"]
    contract: FrozenContract = result["contract"]
    for method in METHODS:
        prediction = arrays[method]
        meta, curve = common.gt_frc(gt, prediction, pixel_size_um=float(pixel_size_um))
        frequency = np.asarray(curve["frequency_cycles_per_pixel"], dtype=np.float64)
        frc_raw = np.asarray(curve["frc"], dtype=np.float64)
        counts = np.asarray(curve["count"], dtype=np.int64)
        smooth = _smooth_frc(frc_raw)
        halfbit = _half_bit_threshold(counts)
        f_norm = frequency * 2.0
        cutoff_cycles = meta["cutoff_cycles_per_pixel"]
        cutoff_norm = None if cutoff_cycles is None else float(cutoff_cycles) * 2.0
        plan = next(plan for plan in contract.plans if plan.method == method)
        raw_receipt = result["raw_receipts"][method]
        metric_rows.append({
            "method": method,
            "psnr_db": float(metric_api.psnr_native(gt, prediction)),
            "ssim": float(metric_api.ssim_native(gt, prediction)),
            "gt_frc_period_px": meta["cutoff_derived_spatial_period_px"],
            "gt_frc_period_um": meta["cutoff_derived_spatial_period_um"],
            "frc_auc": meta["frc_auc_to_cutoff_or_nyquist"],
            "frc_right_censored": bool(meta["right_censored_at_nyquist"]),
            "cutoff_f_norm_nyq1": cutoff_norm,
            "checkpoint_sha256": plan.checkpoint_sha256,
            "protocol_id": plan.protocol_id,
            "raw_stack_sha256": raw_receipt["raw_stack_sha256"],
        })
        frc_results[method] = {
            "frequency_cycles_per_pixel": frequency,
            "f_norm": f_norm,
            "frc_raw": frc_raw,
            "frc_smooth": smooth,
            "counts": counts,
            "halfbit": halfbit,
            "cutoff_f_norm": cutoff_norm,
            "meta": meta,
        }
        for index in range(frequency.size):
            long_rows.append({
                "method": method, "bin": index,
                "frequency_cycles_per_pixel": float(frequency[index]),
                "f_norm_nyquist_1": float(f_norm[index]),
                "frc_raw": None if not np.isfinite(frc_raw[index]) else float(frc_raw[index]),
                "frc_smooth_plot_only": None if not np.isfinite(smooth[index]) else float(smooth[index]),
                "Ni": int(counts[index]),
                "threshold_1over7": FRC_THRESHOLD,
                "halfbit_threshold": None if not np.isfinite(halfbit[index]) else float(halfbit[index]),
            })
    metrics_path = run_dir / "07_metrics" / "single_image_metrics.csv"
    _write_csv(metrics_path, metric_rows, tuple(metric_rows[0]))
    frc_dir = run_dir / "06_frc"
    _write_csv(frc_dir / "frc_curves_long.csv", long_rows, tuple(long_rows[0]))
    wide_rows = []
    for index in range(100):
        row: dict[str, Any] = {"f_norm_nyquist_1": float(frc_results[METHODS[0]]["f_norm"][index])}
        for method in METHODS:
            key = method.replace("APD-SIM-", "") + "F"
            raw_value = frc_results[method]["frc_raw"][index]
            smooth_value = frc_results[method]["frc_smooth"][index]
            row[f"{key}_frc_raw"] = None if not np.isfinite(raw_value) else float(raw_value)
            row[f"{key}_frc_smooth"] = None if not np.isfinite(smooth_value) else float(smooth_value)
        row["threshold_1over7"] = FRC_THRESHOLD
        wide_rows.append(row)
    _write_csv(frc_dir / "frc_curves_wide.csv", wide_rows, tuple(wide_rows[0]))
    np.savez_compressed(
        frc_dir / "frc_curves.npz",
        f_norm_nyquist_1=frc_results[METHODS[0]]["f_norm"],
        threshold_1over7=np.asarray(FRC_THRESHOLD),
        **{
            f"{method.replace('APD-SIM-', '')}F_{kind}": np.asarray(frc_results[method][source])
            for method in METHODS
            for kind, source in (("FRC_raw", "frc_raw"), ("FRC_smooth", "frc_smooth"), ("Ni", "counts"), ("thr_halfbit", "halfbit"))
        },
    )

    contracts_rows = [
        ["GT path", str(result["prepared"].source_path)],
        ["GT file SHA-256", result["prepared"].source_file_sha256],
        ["crop xywh", json.dumps(result["prepared"].crop_xywh)],
        ["pixel size um", float(pixel_size_um)],
        ["base seed", int(result["raw_receipts"]["APD-SIM-3"]["measurement_seed"])],
        ["repair pointer", str(contract.repair_pointer)],
        ["final status", str(contract.final_status_path)],
        ["selection receipt", str(contract.selection_receipt_path)],
        ["archive relocation map", str(contract.relocation_map_path)],
    ]
    for plan in contract.plans:
        raw_receipt = result["raw_receipts"][plan.method]
        contracts_rows.extend([
            [f"{plan.method} checkpoint path", str(plan.checkpoint_path)],
            [f"{plan.method} checkpoint SHA-256", plan.checkpoint_sha256],
            [f"{plan.method} protocol", plan.protocol_id],
            [f"{plan.method} protocol SHA-256", plan.protocol_hash],
            [f"{plan.method} raw order", ",".join(plan.raw_order)],
            [f"{plan.method} raw stack SHA-256", raw_receipt["raw_stack_sha256"]],
        ])
    contracts_rows.extend([
        ["DMD9 inference contract", "tile=320; core=160; shared single-spatial noise=true; deterministic stitching=true; single-block 1004=false"],
        ["formal FRC code path", str(Path(common.__file__).resolve())],
        ["formal FRC code SHA-256", sha256_file(Path(common.__file__).resolve())],
        ["formal FRC threshold", FRC_THRESHOLD],
        ["FRC smooth restriction", "fixed 7-bin plotting only; excluded from cutoff and AUC"],
        ["real-space display contract", "shared linear [0,1]; black-blue-cyan-white; no auto contrast/gamma/per-image stretch"],
        ["spectrum contract", f"mean subtract; full-image Tukey alpha=0.20; FFT2; fftshift; amplitude; log1p; shared range [{global_low},{global_high}]; magma"],
    ])
    workbook_payload = {
        "schema_version": 1,
        "methods": {
            method: {
                "f_norm": frc_results[method]["f_norm"].tolist(),
                "frc_raw": [None if not np.isfinite(v) else float(v) for v in frc_results[method]["frc_raw"]],
                "frc_smooth": [None if not np.isfinite(v) else float(v) for v in frc_results[method]["frc_smooth"]],
                "cutoff_f_norm": frc_results[method]["cutoff_f_norm"],
                "counts": frc_results[method]["counts"].tolist(),
                "halfbit": [None if not np.isfinite(v) else float(v) for v in frc_results[method]["halfbit"]],
            }
            for method in METHODS
        },
        "summary": metric_rows,
        "contracts": contracts_rows,
    }
    atomic_json(frc_dir / "workbook_payload.json", workbook_payload)
    return {
        "metrics": metric_rows,
        "frc_results": frc_results,
        "metrics_path": metrics_path,
        "workbook_payload": frc_dir / "workbook_payload.json",
        "workbook_path": frc_dir / "APD369_FRC_curves.xlsx",
        "realspace_contract": {"range": [0.0, 1.0], "lut": "black-blue-cyan-white", "shared": True},
        "spectrum_contract": {"global_low": global_low, "global_high": global_high, "lut": "magma", "shared": True, "tukey_alpha": 0.20},
    }


__all__ = [
    "FrozenContract", "PreparedSingleGT", "ProtocolContract", "SingleGTContractError",
    "array_sha256", "atomic_json", "build_single_gt_assets",
    "generate_protocol_measurement", "load_apd369_frozen_contract",
    "prepare_single_gt", "reconstruct_apd_protocol", "run_single_gt_apd369",
    "sha256_file",
]

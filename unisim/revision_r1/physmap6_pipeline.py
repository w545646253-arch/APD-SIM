"""Checkpoint-bound Stage 1 and strict DMD6 reconstruction pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from unisim.checkpoint_contract import architecture_hash, load_checkpoint_bound
from unisim.formal_training_2d import DiffusionScheduler2D
from unisim.model2d import APDConditionedUNet2D, assert_strictly_2d_model
from unisim.protocol_runtime import require_protocol
from unisim.sim_forward_2d import SIM2DConfig, embed_raw_to_slots_2d
from .physmap6_core import RefinementConfig, evaluate_observed_fit, masked_refine


PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"
RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
NORMALIZATION_HASH = "a148bcb41ab149285435bc0e0bd57526c6346fd905a6abece6721f204e1cd2d3"
BEST_RULE_ID = "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"
STAGE1_POLICY = {
    "weights": "ema",
    "diffusion_init_t": 600,
    "diffusion_steps_including_endpoints": 80,
    "ddim_eta": 0.0,
    "padding": "reflect_bottom_right_to_multiple_16_then_exact_unpad",
    "precision": "float32_model_and_scheduler_no_autocast",
}


def sha_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def _make_model(config: Mapping[str, Any]) -> APDConditionedUNet2D:
    model_cfg = config["model"]
    model = APDConditionedUNet2D(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(int(value) for value in model_cfg["channel_mults"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        dropout=float(model_cfg["dropout"]),
        time_dim=int(model_cfg["time_dim"]),
        groups=int(model_cfg["groups"]),
    )
    assert_strictly_2d_model(model)
    return model


def make_sim_config(config: Mapping[str, Any]) -> SIM2DConfig:
    forward = dict(config["forward"])
    allowed = set(SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in forward.items() if key in allowed}
    for key in (
        "rand_k_ratio_xy",
        "rand_mod_depth",
        "rand_psf_sigma_scale",
        "rand_background",
        "rand_photon_scale",
        "rand_read_noise_e",
    ):
        if key in values:
            values[key] = tuple(float(item) for item in values[key])
    return SIM2DConfig(**values)


def load_stage1_registered(
    config: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    device: torch.device,
    *,
    protocol_id: str,
) -> tuple[APDConditionedUNet2D, DiffusionScheduler2D, Mapping[str, Any]]:
    spec = require_protocol(protocol_id)
    model = _make_model(config)
    expected = {
        "training_protocol_id": protocol_id,
        "training_protocol_hash": spec.protocol_hash,
        "architecture_hash": architecture_hash(model),
        "architecture_contract": model.architecture_contract,
        "input_tensor_dimensionality": "4D_BCHW",
        "normalization_contract_hash": NORMALIZATION_HASH,
        "source_snapshot_id": config["source_snapshot_id"],
        "train_manifest_hash": config["train_manifest_hash"],
        "validation_manifest_hash": config["validation_manifest_hash"],
        "sealed_test_no_access_hash": config["sealed_test_manifest_hash"],
        "validation_bundle_hash": config["validation_bundle_hash"],
        "training_config_hash": config["config_payload_hash"],
        "training_seed": int(config["training"]["seed"]),
        "checkpoint_selection_rule": BEST_RULE_ID,
        "completion_status": "FORMAL_TRAINING_COMPLETE",
    }
    payload = load_checkpoint_bound(
        checkpoint,
        protocol=protocol_id,
        expected_sha256=checkpoint_sha256,
        expected_identities=expected,
    )
    state = payload.get("ema")
    if not isinstance(state, Mapping):
        raise RuntimeError("R1C3_APD6_CHECKPOINT_UNRESOLVED: EMA absent")
    if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("R1C3_APD6_CHECKPOINT_UNRESOLVED: EMA non-finite")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    scheduler = DiffusionScheduler2D(
        int(config["training"]["diffusion_steps"]),
        device,
        str(config["training"]["beta_schedule"]),
    )
    return model, scheduler, payload["metadata"]


def load_stage1(
    config: Mapping[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
    device: torch.device,
) -> tuple[APDConditionedUNet2D, DiffusionScheduler2D, Mapping[str, Any]]:
    """Backward-compatible audited DMD-6F EMA loader.

    This explicit branch is intentionally retained because sealed R1C3 audit
    receipts inspect the EMA load statement itself.
    """
    model = _make_model(config)
    expected = {
        "training_protocol_id": PROTOCOL_ID,
        "training_protocol_hash": PROTOCOL_HASH,
        "architecture_hash": architecture_hash(model),
        "architecture_contract": model.architecture_contract,
        "input_tensor_dimensionality": "4D_BCHW",
        "normalization_contract_hash": NORMALIZATION_HASH,
        "source_snapshot_id": config["source_snapshot_id"],
        "train_manifest_hash": config["train_manifest_hash"],
        "validation_manifest_hash": config["validation_manifest_hash"],
        "sealed_test_no_access_hash": config["sealed_test_manifest_hash"],
        "validation_bundle_hash": config["validation_bundle_hash"],
        "training_config_hash": config["config_payload_hash"],
        "training_seed": int(config["training"]["seed"]),
        "checkpoint_selection_rule": BEST_RULE_ID,
        "completion_status": "FORMAL_TRAINING_COMPLETE",
    }
    payload = load_checkpoint_bound(
        checkpoint,
        protocol=PROTOCOL_ID,
        expected_sha256=checkpoint_sha256,
        expected_identities=expected,
    )
    state = payload.get("ema")
    if not isinstance(state, Mapping):
        raise RuntimeError("R1C3_APD6_CHECKPOINT_UNRESOLVED: EMA absent")
    if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("R1C3_APD6_CHECKPOINT_UNRESOLVED: EMA non-finite")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    scheduler = DiffusionScheduler2D(
        int(config["training"]["diffusion_steps"]),
        device,
        str(config["training"]["beta_schedule"]),
    )
    return model, scheduler, payload["metadata"]


@torch.no_grad()
def stage1_reconstruct_registered(
    raw_frames: torch.Tensor,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    *,
    protocol_id: str,
    seed: int,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, int]:
    spec = require_protocol(protocol_id)
    if raw_frames.ndim != 4 or raw_frames.shape[1] != spec.frame_count:
        if protocol_id == PROTOCOL_ID:
            raise ValueError("Stage 1 requires exactly six raw frames")
        raise ValueError(
            f"Stage 1 frame count does not match registered protocol {protocol_id}"
        )
    device = raw_frames.device
    wide = raw_frames.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    slotted, mask = embed_raw_to_slots_2d(raw_frames, protocol_id)
    height, width = raw_frames.shape[-2:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        wide = F.pad(wide, (0, pad_w, 0, pad_h), mode="reflect")
        slotted = F.pad(slotted, (0, pad_w, 0, pad_h), mode="reflect")
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    if initial_noise is None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(wide.shape, generator=generator, device=device, dtype=torch.float32)
    else:
        noise = initial_noise.to(device=device, dtype=torch.float32)
        if tuple(noise.shape) != tuple(wide.shape) or not bool(torch.isfinite(noise).all()):
            raise ValueError(
                f"initial_noise must be finite with padded Stage-1 shape {tuple(wide.shape)}, "
                f"got {tuple(noise.shape)}"
            )
    timesteps = np.rint(np.linspace(600, 0, 80)).astype(np.int64).tolist()
    if len(set(timesteps)) != 80 or timesteps[0] != 600 or timesteps[-1] != 0:
        raise RuntimeError("Frozen Stage 1 timestep schedule changed")
    current_t = torch.full((raw_frames.shape[0],), 600, device=device, dtype=torch.long)
    x = scheduler.q_sample(wide, current_t, noise)
    x0 = wide
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, current in enumerate(timesteps):
        timestep = torch.full((raw_frames.shape[0],), current, device=device, dtype=torch.long)
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
    if output.shape != wide[..., :height, :width].shape or not bool(torch.isfinite(output).all()):
        raise RuntimeError("R1C3_NONFINITE_RESULT: Stage 1")
    return output, elapsed, int(peak)


@torch.no_grad()
def stage1_reconstruct_registered_tiled(
    raw_frames: torch.Tensor,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    *,
    protocol_id: str,
    seed: int,
    tile_size: int = 320,
    core_size: int = 160,
    tile_batch_size: int = 1,
) -> tuple[torch.Tensor, float, int]:
    """Training-support-matched full-size Stage 1 with one spatial noise field.

    Each 320x320 inference tile contributes only its central 160x160 core.  A
    single reflect-padded raw canvas and a single principal Gaussian field are
    shared by every tile, so this is deterministic tiling rather than multiple
    sampling or output selection.  The helper is protocol-generic; callers
    choose explicitly whether a diagnosed model requires the tiled path.
    """
    spec = require_protocol(protocol_id)
    if raw_frames.ndim != 4 or raw_frames.shape[1] != spec.frame_count:
        raise ValueError(f"Tiled Stage 1 frame count does not match {protocol_id}")
    if tile_size != 320 or core_size != 160 or tile_size != 2 * core_size:
        raise ValueError("Frozen tiled Stage-1 geometry requires tile_size=320 and core_size=160")
    if raw_frames.shape[0] != 1:
        raise ValueError("Tiled Stage 1 currently requires batch size one")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be positive")
    height, width = raw_frames.shape[-2:]
    canvas_h = int(math.ceil(height / core_size) * core_size)
    canvas_w = int(math.ceil(width / core_size) * core_size)
    halo = (tile_size - core_size) // 2
    padded = F.pad(
        raw_frames,
        (halo, halo + canvas_w - width, halo, halo + canvas_h - height),
        mode="reflect",
    )
    device = raw_frames.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise_field = torch.randn(
        (1, 1, padded.shape[-2], padded.shape[-1]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    output = torch.empty((1, 1, canvas_h, canvas_w), device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    coordinates = [
        (top, left)
        for top in range(0, canvas_h, core_size)
        for left in range(0, canvas_w, core_size)
    ]
    for start in range(0, len(coordinates), tile_batch_size):
        current = coordinates[start : start + tile_batch_size]
        raw_batch = torch.cat(
            [padded[..., top : top + tile_size, left : left + tile_size] for top, left in current],
            dim=0,
        )
        noise_batch = torch.cat(
            [noise_field[..., top : top + tile_size, left : left + tile_size] for top, left in current],
            dim=0,
        )
        tiles, _elapsed, _peak = stage1_reconstruct_registered(
                raw_batch,
                model,
                scheduler,
                protocol_id=protocol_id,
                seed=seed,
                initial_noise=noise_batch,
            )
        for batch_index, (top, left) in enumerate(current):
            output[..., top : top + core_size, left : left + core_size] = tiles[
                batch_index : batch_index + 1,
                ...,
                halo : halo + core_size,
                halo : halo + core_size,
            ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    result = output[..., :height, :width]
    if tuple(result.shape) != (1, 1, height, width) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("R1C3_NONFINITE_RESULT: tiled Stage 1")
    return result, elapsed, int(peak)


@torch.no_grad()
def stage1_reconstruct(
    raw_frames: torch.Tensor,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    *,
    seed: int,
) -> tuple[torch.Tensor, float, int]:
    """Backward-compatible audited DiffWS-6 Stage 1."""
    return stage1_reconstruct_registered(
        raw_frames, model, scheduler, protocol_id=PROTOCOL_ID, seed=seed
    )


def run_four_methods(
    raw_frames: torch.Tensor,
    model: APDConditionedUNet2D,
    scheduler: DiffusionScheduler2D,
    sim_config: SIM2DConfig,
    theta_inverse: Mapping[str, torch.Tensor],
    *,
    diffusion_seed: int,
    refinement_config: RefinementConfig,
    geometry_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Run WF, DiffWS-6, PhysMap-6 and APD-SIM-6 on one shared tensor."""
    raw_pointer = int(raw_frames.data_ptr())
    if raw_frames.shape[0] != 1:
        raise ValueError("Strict R1C3 evaluation requires batch size one")
    # Official DMD6 bundle receipts hash the native (K,H,W) raw stack; the
    # leading singleton batch is a runtime-only tensor dimension.
    raw_hash = sha_array(raw_frames[0].detach().cpu().numpy().astype(np.float32))
    device = raw_frames.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    wide = raw_frames.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wf_runtime = time.perf_counter() - started
    wf_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    x_ws, stage1_runtime, stage1_peak = stage1_reconstruct(
        raw_frames, model, scheduler, seed=diffusion_seed
    )
    validity = torch.tensor(geometry_receipt["validity_mask"], device=device, dtype=torch.float32)
    forward = {"sim_config": sim_config, "theta": theta_inverse}
    phys = masked_refine(
        wide, raw_frames, validity, geometry_receipt, forward, refinement_config
    )
    apd = masked_refine(
        x_ws, raw_frames, validity, geometry_receipt, forward, refinement_config
    )
    if raw_pointer != int(raw_frames.data_ptr()) or raw_hash != sha_array(
        raw_frames[0].detach().cpu().numpy().astype(np.float32)
    ):
        raise RuntimeError("R1C3_INPUT_IDENTITY_MISMATCH")
    if phys.configuration_receipt != apd.configuration_receipt:
        raise RuntimeError("R1C3_REFINEMENT_CONFIG_MISMATCH")
    wf_fit = evaluate_observed_fit(wide, raw_frames, geometry_receipt, forward)
    diff_fit = evaluate_observed_fit(x_ws, raw_frames, geometry_receipt, forward)
    return {
        "WF": {
            "image": wide,
            "runtime_seconds": wf_runtime,
            "peak_gpu_memory_bytes": int(wf_peak),
            "observed_fit": wf_fit,
        },
        "DiffWS-6": {
            "image": x_ws,
            "runtime_seconds": stage1_runtime,
            "peak_gpu_memory_bytes": stage1_peak,
            "observed_fit": diff_fit,
        },
        "PhysMap-6": {"image": phys.final_reconstruction, "refinement": phys},
        "APD-SIM-6": {"image": apd.final_reconstruction, "refinement": apd},
        "raw_stack_sha256": raw_hash,
        "raw_data_ptr": raw_pointer,
        "shared_refinement_function": f"{masked_refine.__module__}.{masked_refine.__qualname__}",
        "shared_refinement_config_object_id": id(refinement_config),
        "only_allowed_difference": "initial_image: widefield mean versus x_ws",
    }


__all__ = [
    "BEST_RULE_ID",
    "NORMALIZATION_HASH",
    "PROTOCOL_HASH",
    "PROTOCOL_ID",
    "RAW_ORDER",
    "STAGE1_POLICY",
    "load_stage1",
    "load_stage1_registered",
    "make_sim_config",
    "run_four_methods",
    "sha_array",
    "stage1_reconstruct",
    "stage1_reconstruct_registered",
    "stage1_reconstruct_registered_tiled",
]

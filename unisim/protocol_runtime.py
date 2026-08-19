"""Protocol-bound runtime operations for revised APD-SIM DMD models.

This module is the single bridge between an immutable :class:`ProtocolSpec`
and the existing differentiable SIM forward model.  It deliberately does not
infer physical geometry from ``K``.  Every public operation resolves an
explicit ``protocol_id`` (or an already-validated ``ProtocolSpec``).

The fixed network tensor has ``KMAX=15`` measurement channels and 15 mask
channels.  Physical row semantics are protocol-specific; raw frames are
therefore embedded through the registry's explicit raw-to-slot bijection.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .protocols import KMAX, ProtocolSpec, protocol_registry
from .sim_forward import (
    SIMConfig,
    forward_clean,
    forward_sim,
    nll_poisson_gaussian_cam,
)
from .utils import read_tiff


ProtocolLike = Union[str, ProtocolSpec]


class ProtocolRuntimeError(RuntimeError):
    """Base class for fail-closed protocol/runtime contract violations."""


class RawFrameOrderError(ProtocolRuntimeError):
    """Raised when raw frames are not in the controller-defined order."""


class CheckpointProtocolError(ProtocolRuntimeError):
    """Raised when a checkpoint is missing or violates protocol metadata."""


def require_protocol(protocol: ProtocolLike) -> ProtocolSpec:
    """Resolve and return an already hash-validated immutable protocol."""
    if isinstance(protocol, str):
        return protocol_registry.require(protocol)
    if not isinstance(protocol, ProtocolSpec):
        raise TypeError(f"Expected protocol ID or ProtocolSpec, got {type(protocol)!r}")
    # A registry round-trip also rejects an object from an untrusted source.
    registered = protocol_registry.require(protocol.protocol_id)
    if registered.protocol_hash != protocol.protocol_hash:
        raise ProtocolRuntimeError(
            f"Unregistered protocol payload for {protocol.protocol_id}: "
            f"{protocol.protocol_hash} != {registered.protocol_hash}"
        )
    return registered


def sim_config_for_protocol(cfg: SIMConfig, protocol: ProtocolLike) -> SIMConfig:
    """Return a copy of ``cfg`` bound to one controller-defined 2-D protocol.

    DMD-space carrier magnitudes cannot be substituted directly for the
    specimen-plane normalized carrier magnitude used by ``SIMConfig``.  The
    registry therefore controls row direction and nominal phase, while the
    recovered APD-SIM ``k_ratio_xy`` and its domain randomization remain
    unchanged.  Carrier vectors stay in protocol metadata as controller-space
    evidence and diagnostics.
    """
    spec = require_protocol(protocol)
    fg = spec.forward_geometry
    if fg.angle_unit != "degree_mod_180" or fg.phase_unit != "radian":
        raise ProtocolRuntimeError(
            f"Unsupported forward units for {spec.protocol_id}: "
            f"{fg.angle_unit}/{fg.phase_unit}"
        )
    if tuple(fg.orientation_angles) != tuple(spec.orientation_angles):
        raise ProtocolRuntimeError("Top-level and forward-geometry orientation angles disagree")
    if tuple(fg.nominal_phase_values) != tuple(spec.nominal_phase_values):
        raise ProtocolRuntimeError("Top-level and forward-geometry phase values disagree")
    if tuple(tuple(v) for v in fg.carrier_vectors) != tuple(tuple(v) for v in spec.carrier_vectors):
        raise ProtocolRuntimeError("Top-level and forward-geometry carrier vectors disagree")
    # Carrier direction is consumed as an invariant.  Magnitude remains in the
    # APD specimen-plane k_ratio because no DMD-pixel-to-specimen calibration
    # is evidenced.  Modulo 180 handles the conjugate Fourier peak.
    for angle_deg, carrier in zip(spec.orientation_angles, spec.carrier_vectors):
        cx, cy = (float(carrier[0]), float(carrier[1]))
        if abs(cx) + abs(cy) <= 0.0:
            raise ProtocolRuntimeError("Zero controller carrier vector is invalid")
        carrier_angle = math.degrees(math.atan2(cy, cx)) % 180.0
        delta = abs(((carrier_angle - float(angle_deg) + 90.0) % 180.0) - 90.0)
        if delta > 0.2:
            raise ProtocolRuntimeError(
                f"Controller carrier {carrier} angle {carrier_angle:.6f} does not match "
                f"registered orientation {angle_deg:.6f}"
            )
    out = copy.deepcopy(cfg)
    out.angle_list = tuple(float(v) for v in spec.orientation_angles)
    out.phase_list_2d = tuple(float(v) for v in spec.nominal_phase_values)
    out.protocol_id = spec.protocol_id
    out.protocol_hash = spec.protocol_hash
    out.protocol_claim_level = spec.claim_level
    out.protocol_carrier_vectors = tuple(tuple(float(x) for x in v) for v in spec.carrier_vectors)
    out.protocol_forward_geometry = {
        "orientation_angles": tuple(fg.orientation_angles),
        "nominal_phase_values": tuple(fg.nominal_phase_values),
        "carrier_vectors": tuple(tuple(v) for v in fg.carrier_vectors),
        "raw_frame_order": tuple(fg.raw_frame_order),
        "raw_to_slot_mapping": tuple(fg.raw_to_slot_mapping),
        "validity_mask": tuple(fg.validity_mask),
        "phase_source": fg.phase_source,
        "fft_phase_role": fg.fft_phase_role,
    }
    if len(out.angle_list) * len(out.phase_list_2d) != spec.frame_count:
        raise ProtocolRuntimeError(
            f"{spec.protocol_id} is not a complete orientation-major O x P grid"
        )
    return out


def _generator_to_raw_indices(spec: ProtocolSpec) -> Tuple[int, ...]:
    """Map controller raw order onto the forward generator's O-major/P-minor order."""
    orientation_ids = tuple(spec.orientation_ids)
    phase_ids = tuple(spec.phase_ids)
    indices = []
    bindings = sorted(spec.raw_frame_bindings, key=lambda item: item.raw_frame_index)
    for binding in bindings:
        try:
            oi = orientation_ids.index(binding.physical_orientation_id)
            pi = phase_ids.index(binding.physical_phase_id)
        except ValueError as exc:
            raise ProtocolRuntimeError(
                f"Binding {binding.raw_frame_id} references an unknown orientation or phase"
            ) from exc
        indices.append(oi * len(phase_ids) + pi)
    if sorted(indices) != list(range(spec.frame_count)):
        raise ProtocolRuntimeError(
            f"{spec.protocol_id} bindings are not a bijection over the forward grid: {indices}"
        )
    return tuple(indices)


def reorder_generated_to_raw(frames: torch.Tensor, protocol: ProtocolLike) -> torch.Tensor:
    """Reorder forward-generated frames into controller raw-frame order."""
    spec = require_protocol(protocol)
    if frames.ndim != 5 or frames.shape[1] != spec.frame_count:
        raise ProtocolRuntimeError(
            f"Expected (B,{spec.frame_count},Z,H,W) for {spec.protocol_id}, got {tuple(frames.shape)}"
        )
    order = torch.tensor(_generator_to_raw_indices(spec), device=frames.device, dtype=torch.long)
    return frames.index_select(1, order)


def embed_raw_to_slots(
    raw_frames: torch.Tensor,
    protocol: ProtocolLike,
    *,
    kmax: int = KMAX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed raw controller-order frames and return ``(slotted, validity_mask)``.

    Invalid channels are zero-filled only for fixed-shape conditioning; the
    returned mask excludes them from all likelihood and residual operations.
    """
    spec = require_protocol(protocol)
    if int(kmax) != int(spec.kmax):
        raise ProtocolRuntimeError(f"kmax={kmax} does not match registry kmax={spec.kmax}")
    if raw_frames.ndim != 5:
        raise ValueError(f"Expected raw frames (B,K,Z,H,W), got {tuple(raw_frames.shape)}")
    if raw_frames.shape[1] != spec.frame_count:
        raise ProtocolRuntimeError(
            f"{spec.protocol_id} requires exactly {spec.frame_count} observed frames; "
            f"got {raw_frames.shape[1]}"
        )
    b, _k, z, h, w = raw_frames.shape
    slotted = torch.zeros((b, kmax, z, h, w), dtype=raw_frames.dtype, device=raw_frames.device)
    mask = torch.zeros_like(slotted)
    for raw_index, slot in enumerate(spec.raw_to_slot_mapping):
        slotted[:, int(slot)] = raw_frames[:, raw_index]
        mask[:, int(slot)] = 1
    expected = torch.tensor(spec.validity_mask, dtype=mask.dtype, device=mask.device)
    observed = mask[0, :, 0, 0, 0]
    if not torch.equal(observed, expected):
        raise ProtocolRuntimeError("Constructed validity mask disagrees with the protocol registry")
    return slotted, mask


def extract_slots_to_raw(slotted: torch.Tensor, protocol: ProtocolLike) -> torch.Tensor:
    """Select only valid slots and restore controller raw-frame order."""
    spec = require_protocol(protocol)
    if slotted.ndim != 5 or slotted.shape[1] != spec.kmax:
        raise ValueError(f"Expected slotted tensor (B,{spec.kmax},Z,H,W), got {tuple(slotted.shape)}")
    index = torch.tensor(spec.raw_to_slot_mapping, device=slotted.device, dtype=torch.long)
    return slotted.index_select(1, index)


def validate_raw_frame_ids(raw_frame_ids: Sequence[str], protocol: ProtocolLike) -> None:
    """Reject a permutation unless the caller also selects a matching protocol."""
    spec = require_protocol(protocol)
    got = tuple(str(v) for v in raw_frame_ids)
    expected = tuple(str(v) for v in spec.raw_frame_order)
    if got != expected:
        raise RawFrameOrderError(
            f"Raw frame order mismatch for {spec.protocol_id}; expected {expected}, got {got}"
        )


def load_ordered_raw_frames(
    frame_paths: Sequence[Union[str, Path]],
    raw_frame_ids: Sequence[str],
    protocol: ProtocolLike,
    *,
    expected_sha256: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Load a real stack only from an explicit, receipt-order frame list.

    Directory scanning and lexical filename sorting are intentionally absent.
    This supports future K6 acquisitions without pretending that the missing
    historical frame-level K6 receipt exists.
    """
    spec = require_protocol(protocol)
    validate_raw_frame_ids(raw_frame_ids, spec)
    paths = tuple(Path(p).resolve() for p in frame_paths)
    if len(paths) != spec.frame_count:
        raise RawFrameOrderError(
            f"{spec.protocol_id} requires {spec.frame_count} paths, got {len(paths)}"
        )
    if expected_sha256 is not None and len(expected_sha256) != len(paths):
        raise RawFrameOrderError("expected_sha256 length must equal frame path count")
    frames = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_sha256 is not None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != str(expected_sha256[index]).lower():
                raise RawFrameOrderError(f"SHA-256 mismatch for raw frame {index}: {path}")
        frame = np.asarray(read_tiff(path))
        if frame.ndim == 2:
            frame = frame[None, ...]
        if frame.ndim != 3:
            raise RawFrameOrderError(f"Expected frame {path} to be (H,W) or (Z,H,W), got {frame.shape}")
        frames.append(frame)
    shape0 = frames[0].shape
    if any(frame.shape != shape0 for frame in frames):
        raise RawFrameOrderError("Raw frames do not share one Z/H/W shape")
    return np.stack(frames, axis=0)


def forward_protocol_clean(
    x0: torch.Tensor,
    cfg: SIMConfig,
    protocol: ProtocolLike,
    *,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Protocol-bound differentiable noiseless forward operator."""
    spec = require_protocol(protocol)
    bound_cfg = sim_config_for_protocol(cfg, spec)
    generated, theta_used = forward_clean(
        x0, bound_cfg, mode="2d", theta=theta, randomize=randomize, rng=rng
    )
    return reorder_generated_to_raw(generated, spec), theta_used


def forward_protocol_sim(
    x0: torch.Tensor,
    cfg: SIMConfig,
    protocol: ProtocolLike,
    *,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Protocol-bound Poisson-Gaussian measurement synthesis."""
    spec = require_protocol(protocol)
    bound_cfg = sim_config_for_protocol(cfg, spec)
    generated, theta_used = forward_sim(
        x0, bound_cfg, mode="2d", theta=theta, randomize=randomize, rng=rng
    )
    return reorder_generated_to_raw(generated, spec), theta_used


def stage2_forward_protocol(*args, **kwargs):
    """Stage-2 forward hook; intentionally identical to training/validation forward."""
    return forward_protocol_clean(*args, **kwargs)


def diffws_forward_protocol(*args, **kwargs):
    """Protocol-bound forward hook reserved for revised DiffWS evaluation."""
    return forward_protocol_clean(*args, **kwargs)


def physmap_forward_protocol(*args, **kwargs):
    """Protocol-bound forward hook reserved for revised PhysMap evaluation."""
    return forward_protocol_clean(*args, **kwargs)


def evaluation_forward_protocol(*args, **kwargs):
    """Generic future-evaluation hook with the same immutable geometry contract."""
    return forward_protocol_clean(*args, **kwargs)


def masked_poisson_gaussian_likelihood(
    observed_slotted: torch.Tensor,
    predicted_slotted: torch.Tensor,
    protocol: ProtocolLike,
    *,
    photon_scale: Union[float, torch.Tensor],
    read_noise_e: Union[float, torch.Tensor],
    reduce: str = "mean",
) -> torch.Tensor:
    """Poisson-Gaussian NLL over valid protocol slots only."""
    observed = extract_slots_to_raw(observed_slotted, protocol)
    predicted = extract_slots_to_raw(predicted_slotted, protocol)
    return nll_poisson_gaussian_cam(
        observed,
        predicted,
        photon_scale=photon_scale,
        read_noise_e=read_noise_e,
        reduce=reduce,
    )


def checkpoint_protocol_metadata(protocol: ProtocolLike) -> Dict[str, object]:
    """Return the mandatory geometry portion of a revised checkpoint receipt."""
    spec = require_protocol(protocol)
    return {
        "training_protocol_id": spec.protocol_id,
        "training_protocol_hash": spec.protocol_hash,
        "protocol_evidence_level": spec.evidence_level,
        "protocol_claim_level": spec.claim_level,
        "frame_count": spec.frame_count,
        "orientation_count": spec.orientation_count,
        "phases_per_orientation": spec.phases_per_orientation,
        "orientation_ids": list(spec.orientation_ids),
        "orientation_angles": list(spec.orientation_angles),
        "phase_values": list(spec.nominal_phase_values),
        "raw_frame_order": list(spec.raw_frame_order),
        "raw_to_slot_mapping": list(spec.raw_to_slot_mapping),
        "valid_slots": list(spec.valid_slots),
        "validity_mask": list(spec.validity_mask),
        "controller_source_hash": spec.controller_version_hash,
    }


def validate_checkpoint_protocol(
    checkpoint: Mapping[str, object],
    protocol: ProtocolLike,
) -> Mapping[str, object]:
    """Fail closed on legacy, unbound, or cross-geometry checkpoints."""
    spec = require_protocol(protocol)
    metadata_obj = checkpoint.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, Mapping) else checkpoint
    required = tuple(checkpoint_protocol_metadata(spec).keys())
    missing = [key for key in required if key not in metadata]
    if missing:
        raise CheckpointProtocolError(
            "Checkpoint lacks revised protocol metadata (legacy checkpoints are ineligible): "
            + ", ".join(missing)
        )
    if metadata.get("training_protocol_id") != spec.protocol_id:
        raise CheckpointProtocolError(
            f"Checkpoint protocol ID {metadata.get('training_protocol_id')!r} != {spec.protocol_id!r}"
        )
    if metadata.get("training_protocol_hash") != spec.protocol_hash:
        raise CheckpointProtocolError("Checkpoint protocol hash mismatch")
    if tuple(metadata.get("raw_to_slot_mapping", ())) != tuple(spec.raw_to_slot_mapping):
        raise CheckpointProtocolError("Checkpoint raw-to-slot mapping mismatch")
    if tuple(metadata.get("validity_mask", ())) != tuple(spec.validity_mask):
        raise CheckpointProtocolError("Checkpoint validity mask mismatch")
    return metadata


def initialization_compatibility(
    source_protocol: ProtocolLike,
    target_protocol: ProtocolLike,
) -> Tuple[str, str]:
    """Classify full-model initialization from physical row semantics, never K."""
    source = require_protocol(source_protocol)
    target = require_protocol(target_protocol)
    if source.protocol_id == target.protocol_id:
        return "same_protocol_full_model", "protocol hashes and row semantics are identical"
    if source.protocol_id == "DMD_9F_3O3P":
        compatible = (
            bool(target.orientation_subset_of_dmd9)
            and bool(target.orientation_order_compatible_with_dmd9)
            and bool(target.phase_order_compatible_with_dmd9)
        )
        if compatible:
            return (
                "full_model_initialization_from_dmd9_allowed",
                "target orientations, row order, and phase order are a verified DMD9 prefix",
            )
        return (
            "from_scratch_required",
            "target physical row semantics are not a verified ordered subset of DMD9",
        )
    return "from_scratch_required", "no evidence-backed cross-protocol compatibility rule"


__all__ = [
    "CheckpointProtocolError",
    "ProtocolRuntimeError",
    "RawFrameOrderError",
    "checkpoint_protocol_metadata",
    "diffws_forward_protocol",
    "embed_raw_to_slots",
    "evaluation_forward_protocol",
    "extract_slots_to_raw",
    "forward_protocol_clean",
    "forward_protocol_sim",
    "initialization_compatibility",
    "load_ordered_raw_frames",
    "masked_poisson_gaussian_likelihood",
    "physmap_forward_protocol",
    "reorder_generated_to_raw",
    "require_protocol",
    "sim_config_for_protocol",
    "stage2_forward_protocol",
    "validate_checkpoint_protocol",
    "validate_raw_frame_ids",
]

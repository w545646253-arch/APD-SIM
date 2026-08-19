"""Fail-closed checkpoint contract for revised APD-DMD training.

Legacy checkpoints deliberately do not satisfy this schema.  The formal R2
training engine may only load a checkpoint after protocol, source, config and
data-manifest identities have all matched.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Union

import numpy as np
import torch

from .protocol_runtime import (
    CheckpointProtocolError,
    ProtocolLike,
    checkpoint_protocol_metadata,
    require_protocol,
    validate_checkpoint_protocol,
)


MANDATORY_CHECKPOINT_FIELDS = (
    "model_name",
    "architecture_name",
    "architecture_hash",
    "source_snapshot_id",
    "training_protocol_id",
    "training_protocol_hash",
    "protocol_evidence_level",
    "frame_count",
    "orientation_count",
    "phases_per_orientation",
    "orientation_ids",
    "orientation_angles",
    "phase_values",
    "raw_frame_order",
    "raw_to_slot_mapping",
    "valid_slots",
    "validity_mask",
    "controller_source_hash",
    "train_manifest_hash",
    "validation_manifest_hash",
    "sealed_test_no_access_hash",
    "training_config_hash",
    "training_seed",
    "rng_states",
    "global_step",
    "validation_metric",
    "checkpoint_selection_rule",
    "initialization_source",
    "initialization_compatibility_classification",
    "completion_status",
)


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_hash(model: torch.nn.Module) -> str:
    """Hash architecture/class and state schema, never learned values."""
    payload = {
        "module": model.__class__.__module__,
        "class": model.__class__.__qualname__,
        "state_schema": [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_states(
    dataloader_generator: Optional[torch.Generator] = None,
) -> Dict[str, object]:
    """Capture every RNG family required by the R2 receipt."""
    states: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator": (
            dataloader_generator.get_state() if dataloader_generator is not None else None
        ),
    }
    return states


def restore_rng_states(
    states: Mapping[str, object],
    dataloader_generator: Optional[torch.Generator] = None,
) -> None:
    """Restore a complete R2 RNG receipt or fail closed."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "dataloader_generator"}
    missing = sorted(required.difference(states))
    if missing:
        raise CheckpointProtocolError("RNG receipt missing fields: " + ", ".join(missing))
    random.setstate(states["python"])  # type: ignore[arg-type]
    np.random.set_state(states["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(states["torch_cpu"])  # type: ignore[arg-type]
    cuda_states = states["torch_cuda"]
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]
    loader_state = states["dataloader_generator"]
    if loader_state is not None:
        if dataloader_generator is None:
            raise CheckpointProtocolError("Checkpoint has DataLoader RNG state but no generator was supplied")
        dataloader_generator.set_state(loader_state)  # type: ignore[arg-type]


def build_checkpoint_metadata(
    *,
    model: torch.nn.Module,
    protocol: ProtocolLike,
    source_snapshot_id: str,
    train_manifest_hash: str,
    validation_manifest_hash: str,
    sealed_test_no_access_hash: str,
    training_config_hash: str,
    training_seed: int,
    global_step: int,
    validation_metric: Mapping[str, object],
    checkpoint_selection_rule: str,
    initialization_source: str,
    initialization_compatibility_classification: str,
    completion_status: str,
    rng_states: Mapping[str, object],
) -> Dict[str, object]:
    """Construct mandatory metadata without guessing any formal run identity."""
    spec = require_protocol(protocol)
    metadata: Dict[str, object] = checkpoint_protocol_metadata(spec)
    metadata.update(
        {
            "model_name": f"APD-SIM-{spec.protocol_id}",
            "architecture_name": model.__class__.__qualname__,
            "architecture_hash": architecture_hash(model),
            "source_snapshot_id": source_snapshot_id,
            "train_manifest_hash": train_manifest_hash,
            "validation_manifest_hash": validation_manifest_hash,
            "sealed_test_no_access_hash": sealed_test_no_access_hash,
            "training_config_hash": training_config_hash,
            "training_seed": int(training_seed),
            "rng_states": dict(rng_states),
            "global_step": int(global_step),
            "validation_metric": dict(validation_metric),
            "checkpoint_selection_rule": checkpoint_selection_rule,
            "initialization_source": initialization_source,
            "initialization_compatibility_classification": (
                initialization_compatibility_classification
            ),
            "completion_status": completion_status,
        }
    )
    validate_checkpoint_metadata(metadata, spec)
    return metadata


def validate_checkpoint_metadata(
    metadata: Mapping[str, object],
    protocol: ProtocolLike,
    *,
    expected_identities: Optional[Mapping[str, object]] = None,
) -> None:
    spec = require_protocol(protocol)
    missing = [field for field in MANDATORY_CHECKPOINT_FIELDS if field not in metadata]
    if missing:
        raise CheckpointProtocolError("Checkpoint metadata missing: " + ", ".join(missing))
    validate_checkpoint_protocol({"metadata": metadata}, spec)
    if expected_identities:
        mismatches = []
        for field, expected in expected_identities.items():
            if metadata.get(field) != expected:
                mismatches.append(f"{field}: {metadata.get(field)!r} != {expected!r}")
        if mismatches:
            raise CheckpointProtocolError("Checkpoint identity mismatch: " + "; ".join(mismatches))


def save_checkpoint_atomic(
    path: Union[str, Path],
    *,
    model_state: Mapping[str, torch.Tensor],
    ema_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, object],
    scaler_state: Optional[Mapping[str, object]],
    metadata: Mapping[str, object],
    protocol: ProtocolLike,
) -> str:
    """Atomically overwrite best/latest/final and return its SHA-256."""
    target = Path(path).resolve()
    if target.name not in {"best.pt", "latest.pt", "final.pt"}:
        raise CheckpointProtocolError(
            "R2 long-term checkpoint name must be best.pt, latest.pt, or final.pt"
        )
    validate_checkpoint_metadata(metadata, protocol)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    payload = {
        "model": dict(model_state),
        "ema": dict(ema_state),
        "optimizer": dict(optimizer_state),
        "scaler": dict(scaler_state) if scaler_state is not None else None,
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temp)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return sha256_file(target)


def load_checkpoint_bound(
    path: Union[str, Path],
    *,
    protocol: ProtocolLike,
    expected_sha256: Optional[str] = None,
    expected_identities: Optional[Mapping[str, object]] = None,
) -> MutableMapping[str, object]:
    """Hash, deserialize, and validate an R2 checkpoint before state loading."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha = sha256_file(source)
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.lower():
        raise CheckpointProtocolError("Checkpoint file SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, MutableMapping):
        raise CheckpointProtocolError("Checkpoint payload is not a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointProtocolError("Checkpoint lacks an R2 metadata mapping")
    validate_checkpoint_metadata(metadata, protocol, expected_identities=expected_identities)
    for state_key in ("model", "ema", "optimizer"):
        if state_key not in payload:
            raise CheckpointProtocolError(f"Checkpoint lacks {state_key} state")
    return payload


__all__ = [
    "MANDATORY_CHECKPOINT_FIELDS",
    "architecture_hash",
    "build_checkpoint_metadata",
    "capture_rng_states",
    "load_checkpoint_bound",
    "restore_rng_states",
    "save_checkpoint_atomic",
    "sha256_file",
    "validate_checkpoint_metadata",
]

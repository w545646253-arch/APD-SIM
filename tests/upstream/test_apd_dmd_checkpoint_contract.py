from __future__ import annotations

import copy

import pytest
import torch

from unisim.checkpoint_contract import (
    MANDATORY_CHECKPOINT_FIELDS,
    build_checkpoint_metadata,
    capture_rng_states,
    load_checkpoint_bound,
    save_checkpoint_atomic,
)
from unisim.models import UNet3DConditioned
from unisim.protocol_runtime import CheckpointProtocolError
from unisim.protocols import protocol_registry


def _model():
    return UNet3DConditioned(
        in_channels=31,
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        groups=8,
    )


def _metadata(model, protocol_id="DMD_6F_2O3P"):
    return build_checkpoint_metadata(
        model=model,
        protocol=protocol_id,
        source_snapshot_id="a" * 64,
        train_manifest_hash="b" * 64,
        validation_manifest_hash="c" * 64,
        sealed_test_no_access_hash="d" * 64,
        training_config_hash="e" * 64,
        training_seed=20260812,
        global_step=1,
        validation_metric={"name": "validation_loss", "value": 1.0},
        checkpoint_selection_rule="minimum pre-registered validation loss",
        initialization_source="from_scratch",
        initialization_compatibility_classification="from_scratch_required",
        completion_status="INCOMPLETE",
        rng_states=capture_rng_states(),
    )


def test_checkpoint_roundtrip_has_complete_metadata(tmp_path):
    model = _model()
    spec = protocol_registry.require("DMD_6F_2O3P")
    metadata = _metadata(model)
    assert set(MANDATORY_CHECKPOINT_FIELDS).issubset(metadata)
    path = tmp_path / "latest.pt"
    digest = save_checkpoint_atomic(
        path,
        model_state=model.state_dict(),
        ema_state=model.state_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        scaler_state=None,
        metadata=metadata,
        protocol=spec,
    )
    payload = load_checkpoint_bound(path, protocol=spec, expected_sha256=digest)
    assert payload["metadata"]["training_protocol_hash"] == spec.protocol_hash


def test_cross_geometry_and_legacy_checkpoint_rejected(tmp_path):
    model = _model()
    metadata = _metadata(model, "DMD_6F_2O3P")
    path = tmp_path / "latest.pt"
    digest = save_checkpoint_atomic(
        path,
        model_state=model.state_dict(),
        ema_state=model.state_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        scaler_state=None,
        metadata=metadata,
        protocol="DMD_6F_2O3P",
    )
    with pytest.raises(CheckpointProtocolError):
        load_checkpoint_bound(path, protocol="DMD_3F_1O3P", expected_sha256=digest)

    legacy = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict()}, legacy)
    with pytest.raises(CheckpointProtocolError):
        load_checkpoint_bound(legacy, protocol="DMD_6F_2O3P")


def test_identity_mismatch_is_rejected(tmp_path):
    model = _model()
    metadata = _metadata(model)
    path = tmp_path / "latest.pt"
    save_checkpoint_atomic(
        path,
        model_state=model.state_dict(),
        ema_state=model.state_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        scaler_state=None,
        metadata=metadata,
        protocol="DMD_6F_2O3P",
    )
    with pytest.raises(CheckpointProtocolError):
        load_checkpoint_bound(
            path,
            protocol="DMD_6F_2O3P",
            expected_identities={"training_config_hash": "f" * 64},
        )


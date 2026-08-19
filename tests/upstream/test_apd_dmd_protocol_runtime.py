from __future__ import annotations

import copy

import pytest
import torch

from unisim.protocol_runtime import (
    CheckpointProtocolError,
    RawFrameOrderError,
    checkpoint_protocol_metadata,
    diffws_forward_protocol,
    embed_raw_to_slots,
    extract_slots_to_raw,
    forward_protocol_clean,
    initialization_compatibility,
    masked_poisson_gaussian_likelihood,
    physmap_forward_protocol,
    stage2_forward_protocol,
    validate_checkpoint_protocol,
    validate_raw_frame_ids,
)
from unisim.protocols import protocol_registry
from unisim.sim_forward import SIMConfig
from unisim.protocol_runtime import sim_config_for_protocol


@pytest.mark.parametrize(
    "protocol_id,frames,orientations,slots",
    [
        ("DMD_3F_1O3P", 3, 1, (0, 1, 2)),
        ("DMD_6F_2O3P", 6, 2, (0, 1, 2, 3, 4, 5)),
        ("DMD_9F_3O3P", 9, 3, (0, 1, 2, 3, 4, 5, 6, 7, 8)),
    ],
)
def test_raw_slot_embedding_is_bijective(protocol_id, frames, orientations, slots):
    spec = protocol_registry.require(protocol_id)
    assert spec.frame_count == frames
    assert spec.orientation_count == orientations
    assert spec.phases_per_orientation == 3
    assert tuple(spec.valid_slots) == slots
    raw = torch.arange(frames, dtype=torch.float32).view(1, frames, 1, 1, 1)
    slotted, mask = embed_raw_to_slots(raw, spec)
    assert torch.equal(extract_slots_to_raw(slotted, spec), raw)
    assert int(mask.sum()) == frames
    assert torch.equal(mask[0, :, 0, 0, 0], torch.tensor(spec.validity_mask))
    assert torch.count_nonzero(slotted[:, frames:]) == 0


def test_invalid_slots_do_not_affect_masked_likelihood():
    spec = protocol_registry.require("DMD_6F_2O3P")
    raw = torch.full((1, 6, 1, 2, 2), 0.4)
    observed, _ = embed_raw_to_slots(raw, spec)
    predicted = observed.clone()
    loss0 = masked_poisson_gaussian_likelihood(
        observed, predicted, spec, photon_scale=1000.0, read_noise_e=1.0
    )
    predicted[:, 6:] = 1e6
    loss1 = masked_poisson_gaussian_likelihood(
        observed, predicted, spec, photon_scale=1000.0, read_noise_e=1.0
    )
    assert torch.equal(loss0, loss1)


def test_raw_permutation_without_protocol_update_is_rejected():
    spec = protocol_registry.require("DMD_9F_3O3P")
    permuted = list(spec.raw_frame_order)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(RawFrameOrderError):
        validate_raw_frame_ids(permuted, spec)


@pytest.mark.parametrize("protocol_id", ["DMD_3F_1O3P", "DMD_6F_2O3P", "DMD_9F_3O3P"])
def test_training_and_stage2_share_one_forward(protocol_id):
    spec = protocol_registry.require(protocol_id)
    cfg = SIMConfig(
        device="cpu",
        upsample=1,
        psf_size_xy=5,
        psf_size_z=1,
        rand_phase_jitter=0.0,
        rand_angle_jitter=0.0,
    )
    x0 = torch.full((1, 1, 1, 8, 8), 0.5)
    training_mu, theta = forward_protocol_clean(x0, cfg, spec, randomize=False)
    stage2_mu, _ = stage2_forward_protocol(x0, cfg, spec, theta=theta)
    diffws_mu, _ = diffws_forward_protocol(x0, cfg, spec, theta=theta)
    physmap_mu, _ = physmap_forward_protocol(x0, cfg, spec, theta=theta)
    assert training_mu.shape == (1, spec.frame_count, 1, 8, 8)
    assert torch.equal(training_mu, stage2_mu)
    assert torch.equal(training_mu, diffws_mu)
    assert torch.equal(training_mu, physmap_mu)
    assert torch.isfinite(training_mu).all()


def test_checkpoint_contract_rejects_legacy_and_hash_mismatch():
    spec = protocol_registry.require("DMD_6F_2O3P")
    with pytest.raises(CheckpointProtocolError):
        validate_checkpoint_protocol({"model": {}}, spec)

    good = {"metadata": checkpoint_protocol_metadata(spec)}
    assert validate_checkpoint_protocol(good, spec)["training_protocol_id"] == spec.protocol_id

    bad = copy.deepcopy(good)
    bad["metadata"]["training_protocol_hash"] = "0" * 64
    with pytest.raises(CheckpointProtocolError):
        validate_checkpoint_protocol(bad, spec)


def test_initialization_compatibility_uses_physical_sets_not_frame_count():
    k9 = protocol_registry.require("DMD_9F_3O3P")
    k6 = protocol_registry.require("DMD_6F_2O3P")
    k3 = protocol_registry.require("DMD_3F_1O3P")
    classification6, reason6 = initialization_compatibility(k9, k6)
    classification3, reason3 = initialization_compatibility(k9, k3)
    assert classification6 == "from_scratch_required"
    assert "row semantics" in reason6
    assert classification3 == "full_model_initialization_from_dmd9_allowed"
    assert "row order" in reason3


def test_forward_config_binds_carrier_direction_and_nominal_phase():
    cfg = SIMConfig(device="cpu")
    for protocol_id in ("DMD_3F_1O3P", "DMD_6F_2O3P", "DMD_9F_3O3P"):
        spec = protocol_registry.require(protocol_id)
        bound = sim_config_for_protocol(cfg, spec)
        assert tuple(bound.angle_list) == tuple(spec.orientation_angles)
        assert tuple(bound.phase_list_2d) == tuple(spec.nominal_phase_values)
        assert tuple(bound.protocol_carrier_vectors) == tuple(spec.carrier_vectors)
        assert tuple(bound.protocol_forward_geometry["raw_to_slot_mapping"]) == tuple(spec.raw_to_slot_mapping)

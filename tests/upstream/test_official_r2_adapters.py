import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import official_r2_adapters as a


def test_six_frame_roundtrip_is_lossless():
    raw = np.arange(6 * 7 * 9, dtype=np.float32).reshape(6, 7, 9)
    shaped = a.to_orientation_phase(raw)
    assert shaped.shape == (2, 3, 7, 9)
    assert np.array_equal(shaped.reshape(raw.shape), raw)


@pytest.mark.parametrize("shape", [(5, 7, 9), (7, 7, 9), (6, 9), (1, 6, 7, 9)])
def test_adapter_rejects_non_six_frame_shapes(shape):
    with pytest.raises(ValueError):
        a.to_orientation_phase(np.zeros(shape, np.float32))


def test_bundle_hash_and_protocol_are_enforced(tmp_path):
    raw = np.arange(6 * 5 * 5, dtype=np.float32).reshape(6, 5, 5)
    item = tmp_path / "one.npz"
    np.savez(item, raw_stack=raw, frame_order=np.asarray(a.RAW_ORDER), protocol_id=np.asarray([a.PROTOCOL_ID]), protocol_hash=np.asarray([a.PROTOCOL_HASH]))
    meta = {
        "raw_frame_order": list(a.RAW_ORDER), "protocol_id": a.PROTOCOL_ID,
        "protocol_hash": a.PROTOCOL_HASH, "raw_stack_sha256": a.canonical_array_sha256(raw),
    }
    item.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    got, got_meta = a.load_frozen_stack(item)
    assert np.array_equal(got, raw)
    assert got_meta["raw_stack_sha256"] == a.canonical_array_sha256(raw)


def test_bundle_rejects_order_mismatch(tmp_path):
    raw = np.zeros((6, 4, 4), np.float32)
    item = tmp_path / "one.npz"
    np.savez(item, raw_stack=raw, frame_order=np.asarray(a.RAW_ORDER), protocol_id=np.asarray([a.PROTOCOL_ID]), protocol_hash=np.asarray([a.PROTOCOL_HASH]))
    item.with_suffix(".json").write_text(json.dumps({
        "raw_frame_order": list(reversed(a.RAW_ORDER)), "protocol_id": a.PROTOCOL_ID,
        "protocol_hash": a.PROTOCOL_HASH, "raw_stack_sha256": a.canonical_array_sha256(raw),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        a.load_frozen_stack(item)


def test_mlsim_options_freeze_official_architecture_and_dmd6_input():
    opt = a.mlsim_model_options()
    assert (opt.model, opt.nch_in, opt.nch_out) == ("rcan", 6, 1)
    assert (opt.n_resgroups, opt.n_resblocks, opt.n_feats, opt.scale) == (2, 5, 48, 1)


def test_static_smoke_proves_no_frame_synthesis():
    receipt = a.static_smoke()
    assert receipt["status"] == "PASS"
    assert receipt["roundtrip_identical"] is True
    assert receipt["observed_frames_created_or_removed"] is False

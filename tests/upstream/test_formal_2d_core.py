from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from unisim.formal_training_2d import (
    BEST_RULE_ID,
    EMA2D,
    Formal2DTrainingError,
    FormalComponents,
    INPUT_TENSOR_DIMENSIONALITY,
    K3_LOADED_STATUS,
    NORMALIZATION_CONTRACT_HASH,
    SCHEDULED_COUNTER_SEMANTICS,
    DiffusionScheduler2D,
    _checkpoint_payload,
    _checkpoint_event_counters,
    _precheck_resume_transaction_contract,
    _load_history_for_resume,
    _load_verified_k9_initialization,
    _restore_latest_checkpoint,
    _save_committed_checkpoint_pair,
    _snapshot_optimizer_transaction,
    _standard_amp_optimizer_update,
    _transactional_optimizer_update,
    _ssim_local,
    compute_training_losses,
    print_preflight_2d,
    _accepted_json_hashes,
    _reconcile_history_to_latest,
    _sealed_disjoint,
    _validate_validation_bundle,
)
from unisim.model2d import APDConditionedUNet2D, assert_strictly_2d_model
from unisim.protocol_runtime import ProtocolRuntimeError
from unisim.protocols import protocol_registry
from unisim.sim_forward_2d import (
    SIM2DConfig,
    SIM2DContractError,
    embed_raw_to_slots_2d,
    forward_protocol_clean_2d,
    protocol_carrier_unit_vectors_2d,
    sample_theta_2d,
)


@pytest.mark.parametrize("protocol_id,frame_count", [("DMD_3F_1O3P", 3), ("DMD_6F_2O3P", 6), ("DMD_9F_3O3P", 9)])
def test_pure_2d_forward_slot_mask_and_finite_optimizer(protocol_id: str, frame_count: int) -> None:
    device = torch.device("cpu")
    cfg = SIM2DConfig(upsample=1, psf_size_xy=9)
    x0 = torch.rand((1, 1, 32, 32), device=device)
    theta = sample_theta_2d(
        cfg,
        device=device,
        mismatch_scale=0.4,
        snr_scale=0.4,
        rng=np.random.default_rng(7),
    )
    raw, _ = forward_protocol_clean_2d(x0, cfg, protocol_id, theta=theta)
    assert tuple(raw.shape) == (1, frame_count, 32, 32)
    slots, mask = embed_raw_to_slots_2d(raw, protocol_id)
    assert tuple(slots.shape) == (1, 15, 32, 32)
    assert tuple(mask.shape) == (1, 15, 32, 32)
    expected = protocol_registry.require(protocol_id).validity_mask
    assert tuple(int(v) for v in mask[0, :, 0, 0].tolist()) == expected

    model = APDConditionedUNet2D(
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_dim=16,
        groups=8,
    )
    assert_strictly_2d_model(model)
    assert not any(isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)) for module in model.modules())
    scheduler = DiffusionScheduler2D(32, device)
    timestep = torch.tensor([11], dtype=torch.long)
    noise = torch.randn_like(x0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    losses = compute_training_losses(
        model=model,
        scheduler=scheduler,
        sim_config=cfg,
        protocol_id=protocol_id,
        x0=x0,
        timestep=timestep,
        noise=noise,
        theta=theta,
        lambda_phys=0.05,
        physics_active=True,
    )
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()


def test_invalid_slots_do_not_affect_masked_condition_shape() -> None:
    raw = torch.rand((1, 3, 16, 16))
    slots, mask = embed_raw_to_slots_2d(raw, "DMD_3F_1O3P")
    invalid = mask == 0
    assert torch.count_nonzero(slots[invalid]) == 0
    slots[invalid] = 12345.0
    # Applying the mask restores a condition identical to the canonical one.
    canonical, _ = embed_raw_to_slots_2d(raw, "DMD_3F_1O3P")
    assert torch.equal(slots * mask, canonical)


def test_checkpoint_compatibility_contract_is_explicit() -> None:
    assert INPUT_TENSOR_DIMENSIONALITY == "4D_BCHW"
    assert len(NORMALIZATION_CONTRACT_HASH) == 64
    source = protocol_registry.require("DMD_9F_3O3P")
    target = protocol_registry.require("DMD_3F_1O3P")
    assert source.orientation_ids[0] == target.orientation_ids[0] == "X"
    assert source.orientation_angles[0] == target.orientation_angles[0]
    assert source.carrier_vectors[0] == target.carrier_vectors[0]
    assert source.phase_ids == target.phase_ids
    assert source.nominal_phase_values == target.nominal_phase_values


def test_verified_k9_initialization_status_prints_absolute_source_and_hash(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    source = (tmp_path / "best.pt").resolve()
    result = {
        "initialization_status": K3_LOADED_STATUS,
        "initialization_source_path": str(source),
        "initialization_source_sha256": "a" * 64,
    }
    print_preflight_2d(result)
    output = capsys.readouterr().out
    assert "Initialization status: LOADED_VERIFIED_DMD9_EMA_FULL_MODEL" in output
    assert f"Initialization source: {source}" in output
    assert f"Initialization SHA-256: {'a' * 64}" in output


def test_embedded_json_hash_cannot_bypass_computed_payload(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"payload_hash": "0" * 64, "value": 1}), encoding="utf-8")
    with pytest.raises(Formal2DTrainingError, match="Embedded payload_hash"):
        _accepted_json_hashes(path, json.loads(path.read_text()))


def test_sealed_parent_overlap_is_fail_closed() -> None:
    sealed = {"identities": [{"sample_id": f"T{i}", "parent_id": f"TP{i}"} for i in range(30)]}
    sealed["identities"][0]["parent_id"] = "P1"
    with pytest.raises(Formal2DTrainingError, match="sealed-test parent_id overlap"):
        _sealed_disjoint(
            [{"sample_id": "S1", "parent_id": "P1"}],
            [{"sample_id": "S2", "parent_id": "P2"}],
            sealed,
        )


def test_history_crash_window_reconciles_to_durable_latest(tmp_path) -> None:
    path = tmp_path / "validation_history.csv"
    fields = [
        "global_step", "mean_val_diff_loss", "mean_val_phys_loss",
        "mean_val_total_loss", "mean_val_x0_psnr", "mean_val_x0_ssim",
        "validation_entry_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in (2000, 4000):
            writer.writerow({key: (step if key == "global_step" else 1) for key in fields})
    _reconcile_history_to_latest(path, 2000)
    with path.open("r", newline="", encoding="utf-8") as handle:
        assert [int(row["global_step"]) for row in csv.DictReader(handle)] == [2000]


def test_training_lock_acquire_is_idempotent_for_same_owner(tmp_path) -> None:
    from unisim.formal_training_gate import TrainingLock

    lock = TrainingLock(tmp_path, "train9.py", "DMD_9F_3O3P", "cuda:0", "a" * 64)
    lock.acquire()
    lock.acquire()
    assert lock.path.is_file()
    lock.release()
    assert not lock.path.exists()


def test_photon_randomization_uses_full_declared_log_uniform_range() -> None:
    cfg = SIM2DConfig()
    device = torch.device("cpu")
    first_rng = np.random.default_rng(20260813)
    second_rng = np.random.default_rng(20260813)
    first = np.asarray(
        [
            float(
                sample_theta_2d(
                    cfg,
                    device=device,
                    mismatch_scale=0.0,
                    snr_scale=1.0,
                    rng=first_rng,
                )["photon_scale"].item()
            )
            for _ in range(2048)
        ]
    )
    second = np.asarray(
        [
            float(
                sample_theta_2d(
                    cfg,
                    device=device,
                    mismatch_scale=0.0,
                    snr_scale=1.0,
                    rng=second_rng,
                )["photon_scale"].item()
            )
            for _ in range(2048)
        ]
    )
    assert np.array_equal(first, second)
    assert first.min() >= cfg.rand_photon_scale[0]
    assert first.max() <= cfg.rand_photon_scale[1] * (1.0 + 1e-6)
    # Both tails are exercised; in particular, the old nominal-8000 cap fails.
    assert first.min() < 1700.0
    assert first.max() > 19000.0
    expected_log_mean = np.mean(np.log(cfg.rand_photon_scale))
    assert float(np.log(first).mean()) == pytest.approx(expected_log_mean, abs=0.06)

    full = sample_theta_2d(
        cfg,
        device=device,
        mismatch_scale=0.0,
        snr_scale=1.0,
        rng=np.random.default_rng(73),
    )["photon_scale"]
    halfway = sample_theta_2d(
        cfg,
        device=device,
        mismatch_scale=0.0,
        snr_scale=0.5,
        rng=np.random.default_rng(73),
    )["photon_scale"]
    assert halfway.square().item() == pytest.approx(cfg.photon_scale * full.item(), rel=2e-6)


def test_protocol_carrier_directions_follow_registry_and_k6_is_h_then_v() -> None:
    for spec in protocol_registry.all():
        registered = torch.tensor(spec.carrier_vectors, dtype=torch.float32)
        expected = registered / torch.linalg.vector_norm(registered, dim=1, keepdim=True)
        actual = protocol_carrier_unit_vectors_2d(spec.protocol_id)
        assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)

    k6 = protocol_registry.require("DMD_6F_2O3P")
    assert k6.orientation_ids == ("H", "V")
    assert torch.equal(
        protocol_carrier_unit_vectors_2d(k6),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )


def test_mutated_carrier_evidence_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = protocol_registry.require("DMD_6F_2O3P")
    forged_hash = replace(
        registered,
        carrier_vectors=((1.0, 0.0), registered.carrier_vectors[1]),
        protocol_hash="0" * 64,
    )
    with pytest.raises(ProtocolRuntimeError, match="Unregistered protocol payload"):
        protocol_carrier_unit_vectors_2d(forged_hash)

    # Isolate the forward invariant too: even if registry resolution were
    # bypassed, a carrier/registered-angle disagreement fails closed.
    forged_direction = replace(
        registered,
        carrier_vectors=((1.0, 0.0), registered.carrier_vectors[1]),
    )
    monkeypatch.setattr("unisim.sim_forward_2d.require_protocol", lambda _value: forged_direction)
    with pytest.raises(SIM2DContractError, match="direction disagrees"):
        protocol_carrier_unit_vectors_2d("DMD_6F_2O3P")


def test_local_window_ssim_is_finite_reproducible_and_identity_is_one() -> None:
    generator = torch.Generator().manual_seed(812)
    target = torch.rand((2, 1, 32, 32), generator=generator)
    identity_first = _ssim_local(target, target)
    identity_second = _ssim_local(target, target)
    changed = target.clone()
    changed[:, :, 8:24, 8:24] = 1.0 - changed[:, :, 8:24, 8:24]
    changed_score = _ssim_local(changed, target)
    assert torch.isfinite(identity_first)
    assert torch.isfinite(changed_score)
    assert identity_first.item() == pytest.approx(1.0, abs=2e-6)
    assert torch.equal(identity_first, identity_second)
    assert changed_score.item() < identity_first.item()


class _PrivateRNGDataset:
    def __init__(self, seed: int = 19):
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self):
        return copy.deepcopy(self.rng.bit_generator.state)

    def set_rng_state(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state)

    def draw(self) -> float:
        return float(self.rng.random())


class _StatefulScaler:
    def __init__(self, scale: float):
        self._scale = float(scale)

    def state_dict(self):
        return {"scale": self._scale}

    def load_state_dict(self, state):
        self._scale = float(state["scale"])

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def get_scale(self):
        return self._scale

    def unscale_(self, optimizer):
        return None


class _SkippingScaler(_StatefulScaler):
    def step(self, optimizer):
        return None


class _PoisoningScaler(_StatefulScaler):
    def update(self):
        self._scale = float("nan")


class _DynamicOverflowScaler(_StatefulScaler):
    def __init__(self, scale: float):
        super().__init__(scale)
        self.lower_on_next_update = True

    def update(self):
        if self.lower_on_next_update:
            self._scale *= 0.5
            self.lower_on_next_update = False


def _tiny_components() -> FormalComponents:
    model = APDConditionedUNet2D(
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        time_dim=16,
        groups=8,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Materialize optimizer moments so their restoration is observable.
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()
    loader_generator = torch.Generator().manual_seed(313)
    return FormalComponents(
        config={
            "source_snapshot_id": "snapshot-test",
            "train_manifest_hash": "1" * 64,
            "validation_manifest_hash": "2" * 64,
            "sealed_test_manifest_hash": "3" * 64,
            "validation_bundle_hash": "4" * 64,
            "config_payload_hash": "5" * 64,
            "training": {"seed": 101, "total_steps": 10},
            "initialization": {"policy": "from_scratch"},
        },
        preflight={"protocol_id": "DMD_6F_2O3P"},
        device=torch.device("cpu"),
        model=model,
        scheduler=DiffusionScheduler2D(8, torch.device("cpu")),
        sim_config=SIM2DConfig(upsample=1, psf_size_xy=5),
        optimizer=optimizer,
        ema=EMA2D(model, 0.999),
        train_dataset=_PrivateRNGDataset(),  # type: ignore[arg-type]
        train_loader=None,  # type: ignore[arg-type]
        loader_generator=loader_generator,
    )


def test_resume_checkpoint_restores_all_rng_and_training_state(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    components = _tiny_components()
    scaler = _StatefulScaler(4096.0)
    training_generator = torch.Generator().manual_seed(719)
    training_numpy_rng = np.random.default_rng(727)
    payload = _checkpoint_payload(
        components,
        1,
        {"mean_val_total_loss": 0.4},
        scaler=scaler,
        training_generator=training_generator,
        training_numpy_rng=training_numpy_rng,
        best_metrics={"global_step": 1.0, "mean_val_total_loss": 0.4},
    )
    latest = tmp_path / "latest.pt"
    torch.save(payload, latest)
    expected_model = {key: value.clone() for key, value in components.model.state_dict().items()}
    expected_ema = {key: value.clone() for key, value in components.ema.shadow.items()}
    expected = {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": torch.rand(3),
        "loader": torch.rand(3, generator=components.loader_generator),
        "dataset": components.train_dataset.draw(),
        "training_torch": torch.rand(3, generator=training_generator),
        "training_numpy": training_numpy_rng.random(3),
    }

    for parameter in components.model.parameters():
        parameter.data.zero_()
    components.ema.shadow = {key: torch.zeros_like(value) for key, value in components.ema.shadow.items()}
    components.optimizer.state.clear()
    scaler._scale = 1.0
    _ = [random.random(), np.random.random(), torch.rand(3)]
    _ = torch.rand(3, generator=components.loader_generator)
    _ = components.train_dataset.draw()
    _ = torch.rand(3, generator=training_generator)
    _ = training_numpy_rng.random(3)

    start_step, _best, commits, skips, consecutive_skips = _restore_latest_checkpoint(
        latest,
        components,
        scaler=scaler,
        training_generator=training_generator,
        training_numpy_rng=training_numpy_rng,
    )
    assert start_step == 2
    assert commits == 1
    assert skips == 0
    assert consecutive_skips == 0
    assert scaler.get_scale() == 4096.0
    assert components.optimizer.state
    for key, value in expected_model.items():
        assert torch.equal(components.model.state_dict()[key], value)
    for key, value in expected_ema.items():
        assert torch.equal(components.ema.shadow[key], value)
    assert random.random() == expected["python"]
    assert float(np.random.random()) == expected["numpy"]
    assert torch.equal(torch.rand(3), expected["torch"])
    assert torch.equal(torch.rand(3, generator=components.loader_generator), expected["loader"])
    assert components.train_dataset.draw() == expected["dataset"]
    assert torch.equal(torch.rand(3, generator=training_generator), expected["training_torch"])
    assert np.array_equal(training_numpy_rng.random(3), expected["training_numpy"])


def test_resume_checkpoint_fails_closed_on_identity_mismatch(tmp_path: Path) -> None:
    components = _tiny_components()
    scaler = _StatefulScaler(2.0)
    training_generator = torch.Generator().manual_seed(3)
    training_numpy_rng = np.random.default_rng(4)
    payload = _checkpoint_payload(
        components,
        2,
        {},
        scaler=scaler,
        training_generator=training_generator,
        training_numpy_rng=training_numpy_rng,
    )
    payload["metadata"]["training_config_hash"] = "f" * 64
    latest = tmp_path / "latest.pt"
    torch.save(payload, latest)
    with pytest.raises(Formal2DTrainingError, match="identity mismatch") as caught:
        _restore_latest_checkpoint(
            latest,
            components,
            scaler=scaler,
            training_generator=training_generator,
            training_numpy_rng=training_numpy_rng,
        )
    assert caught.value.status == "FORMAL_2D_RESUME_BLOCKED"


def test_checkpoint_metadata_separates_event_and_committed_counters() -> None:
    components = _tiny_components()
    payload = _checkpoint_payload(
        components,
        1,
        {},
        scaler=_StatefulScaler(1.0),
        committed_optimizer_updates=1,
    )
    metadata = payload["metadata"]
    assert metadata["global_step"] == 1
    assert metadata["loop_event_step"] == 1
    assert metadata["data_event_step"] == 1
    assert metadata["committed_optimizer_updates"] == 1
    assert _checkpoint_event_counters(metadata, payload["optimizer"], "DMD_6F_2O3P") == (1, 1)


def test_legacy_dmd3_global_step_commit_mismatch_is_recovery_not_ready() -> None:
    components = _tiny_components()
    legacy = _checkpoint_payload(components, 1, {}, scaler=_StatefulScaler(1.0))
    metadata = dict(legacy["metadata"])
    for key in (
        "loop_event_step",
        "data_event_step",
        "scheduled_iterations",
        "committed_optimizer_updates",
        "global_step_semantics",
    ):
        metadata.pop(key)
    metadata["global_step"] = 6000
    optimizer_state = copy.deepcopy(legacy["optimizer"])
    for state in optimizer_state["state"].values():
        state["step"] = torch.tensor(5990.0)
    with pytest.raises(Formal2DTrainingError, match="global_step=6000.*commits=5990") as caught:
        _checkpoint_event_counters(metadata, optimizer_state, "DMD_3F_1O3P")
    assert caught.value.status == "DMD3_RECOVERY_NOT_READY"


def test_run_blocks_validated_legacy_dmd3_before_smoke(tmp_path: Path, monkeypatch) -> None:
    from unisim import formal_training_2d as module

    components = _tiny_components()
    legacy = _checkpoint_payload(components, 1, {}, scaler=_StatefulScaler(1.0))
    for key in (
        "loop_event_step",
        "data_event_step",
        "committed_optimizer_updates",
        "global_step_semantics",
    ):
        legacy["metadata"].pop(key)
    legacy["metadata"]["global_step"] = 6000
    for state in legacy["optimizer"]["state"].values():
        state["step"] = torch.tensor(5990.0)
    latest = tmp_path / "latest.pt"
    torch.save(legacy, latest)
    config = tmp_path / "train3_formal.json"
    config.write_text(
        json.dumps(
            {
                "protocol_id": "DMD_3F_1O3P",
                "project_root": str(tmp_path),
                "outputs": {"latest_checkpoint_path": str(latest)},
            }
        ),
        encoding="utf-8",
    )
    called = {"preflight": False, "smoke": False}

    def validated_config(*args, **kwargs):
        called["preflight"] = True
        return (
            {
                "project_root": str(tmp_path),
                "outputs": {
                    "checkpoint_dir": str(tmp_path),
                    "validation_history_path": str(tmp_path / "history.csv"),
                    "best_checkpoint_receipt_path": str(tmp_path / "receipt.json"),
                },
                "config_payload_hash": "a" * 64,
                "training": {},
                "validation": {},
            },
            {
                "protocol_id": "DMD_3F_1O3P",
                "initialization_status": K3_LOADED_STATUS,
            },
        )

    def forbidden_smoke(*args, **kwargs):
        called["smoke"] = True
        raise AssertionError("smoke must not run")

    monkeypatch.setattr(module, "_validate_config", validated_config)
    monkeypatch.setattr(module, "run_one_batch_smoke", forbidden_smoke)
    with pytest.raises(Formal2DTrainingError) as caught:
        module.run_formal_training("DMD_3F_1O3P", config)
    assert caught.value.status == "DMD3_RECOVERY_NOT_READY"
    assert called == {"preflight": True, "smoke": False}


def test_completed_final_skips_legacy_latest_prechecks_and_returns_existing(
    tmp_path: Path, monkeypatch
) -> None:
    from unisim import formal_training_2d as module

    (tmp_path / "final.pt").write_bytes(b"completed-final-sentinel")
    # This deliberately cannot be deserialized.  A completed run must route to
    # final validation without consulting its legacy in-progress checkpoint.
    (tmp_path / "latest.pt").write_bytes(b"legacy-latest-must-not-be-read")
    config_path = tmp_path / "train6_formal.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {
        "project_root": str(tmp_path),
        "outputs": {
            "checkpoint_dir": str(tmp_path),
            "validation_history_path": str(tmp_path / "history.csv"),
            "best_checkpoint_receipt_path": str(tmp_path / "receipt.json"),
        },
        "config_payload_hash": "a" * 64,
        "training": {"seed": 101, "amp_cuda": False},
        "validation": {},
    }
    preflight = {
        "protocol_id": "DMD_6F_2O3P",
        "initialization_status": "READY_FROM_SCRATCH",
    }
    events: list[str] = []

    class _FakeLock:
        def __init__(self, *args, **kwargs):
            pass

        def acquire(self):
            events.append("lock")

        def release(self):
            events.append("release")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("unlock")

    def forbidden_precheck(*args, **kwargs):
        events.append("legacy_precheck")
        raise AssertionError("completed final must bypass legacy latest prechecks")

    def smoke(*args, **kwargs):
        events.append("smoke")
        return {"status": "PASS", "total_loss": 0.25}

    def build(*args, **kwargs):
        events.append("build")
        return _tiny_components()

    def finalize(*args, **kwargs):
        events.append("finalize")
        assert kwargs["final_path"] == tmp_path / "final.pt"
        return {"status": "EXISTING_FORMAL_TRAINING_COMPLETE"}

    monkeypatch.setattr(module, "_validate_config", lambda *_args, **_kwargs: (config, preflight))
    monkeypatch.setattr(module, "_precheck_resume_transaction_contract", forbidden_precheck)
    monkeypatch.setattr(module, "TrainingLock", _FakeLock)
    monkeypatch.setattr(module, "run_one_batch_smoke", smoke)
    monkeypatch.setattr(module, "build_formal_components", build)
    monkeypatch.setattr(module, "_finalize_or_validate_completed_run", finalize)

    result = module.run_formal_training("DMD_6F_2O3P", config_path)
    assert result["status"] == "EXISTING_FORMAL_TRAINING_COMPLETE"
    assert "legacy_precheck" not in events
    assert events.index("smoke") < events.index("finalize")


def test_committed_checkpoint_pair_rolls_one_latest_good_with_matching_counters(
    tmp_path: Path,
) -> None:
    components = _tiny_components()
    payload = _checkpoint_payload(
        components,
        1,
        {"mean_val_total_loss": 0.5},
        scaler=_StatefulScaler(1.0),
        committed_optimizer_updates=1,
    )
    hashes = _save_committed_checkpoint_pair(
        payload,
        tmp_path,
        event_step=1,
        committed_optimizer_updates=1,
    )
    assert set(hashes) == {"latest_sha256", "latest_good_sha256"}
    rolled_payload = copy.deepcopy(payload)
    for key in (
        "global_step",
        "loop_event_step",
        "data_event_step",
        "scheduled_iterations",
        "committed_optimizer_updates",
    ):
        rolled_payload["metadata"][key] = 2
    rolled_hashes = _save_committed_checkpoint_pair(
        rolled_payload,
        tmp_path,
        event_step=2,
        committed_optimizer_updates=2,
    )
    assert set(rolled_hashes) == {"latest_sha256", "latest_good_sha256"}
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert sorted(path.name for path in tmp_path.glob("latest_good*.pt")) == ["latest_good.pt"]
    assert sorted(path.name for path in tmp_path.glob("latest*.pt")) == [
        "latest.pt",
        "latest_good.pt",
    ]
    for name in ("latest.pt", "latest_good.pt"):
        saved = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
        metadata = saved["metadata"]
        assert metadata["global_step"] == 2
        assert metadata["loop_event_step"] == 2
        assert metadata["data_event_step"] == 2
        assert metadata["committed_optimizer_updates"] == 2


def test_uncommitted_checkpoint_pair_is_rejected_without_any_write(tmp_path: Path) -> None:
    components = _tiny_components()
    payload = _checkpoint_payload(
        components,
        1,
        {},
        scaler=_StatefulScaler(1.0),
        committed_optimizer_updates=1,
    )
    with pytest.raises(Formal2DTrainingError, match="valid scheduled/committed counters"):
        _save_committed_checkpoint_pair(
            payload,
            tmp_path,
            event_step=2,
            committed_optimizer_updates=1,
        )
    assert list(tmp_path.iterdir()) == []


def test_preflight_only_returns_before_resume_checkpoint_deserialization(tmp_path: Path, monkeypatch) -> None:
    from unisim import formal_training_2d as module

    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"not a torch checkpoint")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_validate_config",
        lambda *_args, **_kwargs: (
            {},
            {
                "protocol_id": "DMD_3F_1O3P",
                "initialization_status": K3_LOADED_STATUS,
                "initialization_source_path": str((tmp_path / "best.pt").resolve()),
                "initialization_source_sha256": "a" * 64,
            },
        ),
    )
    result = module.run_formal_training("DMD_3F_1O3P", config, preflight_only=True)
    assert result["initialization_status"] == K3_LOADED_STATUS


def test_invalid_config_is_rejected_before_any_resume_checkpoint_load(tmp_path: Path, monkeypatch) -> None:
    from unisim import formal_training_2d as module

    config = tmp_path / "tampered.json"
    config.write_text("{}", encoding="utf-8")
    torch_load_called = False
    original_load = torch.load

    def guarded_load(*args, **kwargs):
        nonlocal torch_load_called
        torch_load_called = True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    monkeypatch.setattr(
        module,
        "_validate_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            Formal2DTrainingError("FORMAL_TRAINING_CONFIG_BLOCKED", "tampered")
        ),
    )
    with pytest.raises(Formal2DTrainingError) as caught:
        module.run_formal_training("DMD_3F_1O3P", config)
    assert caught.value.status == "FORMAL_TRAINING_CONFIG_BLOCKED"
    assert torch_load_called is False


def test_nonfinite_gradient_fails_before_optimizer_step_and_writes_receipt(tmp_path: Path) -> None:
    components = _tiny_components()
    before = {key: value.clone() for key, value in components.model.state_dict().items()}
    parameter = next(components.model.parameters())
    loss = parameter.sum()
    loss.register_hook(lambda gradient: gradient * torch.tensor(float("nan")))
    with pytest.raises(Formal2DTrainingError, match="gradients_pre_step"):
        _transactional_optimizer_update(
            loss=loss,
            components=components,
            scaler=_StatefulScaler(1.0),
            amp_enabled=False,
            event_step=2,
            committed_optimizer_updates=1,
            diagnostic_dir=tmp_path,
            context={"sample_id": ["S1"]},
            replay_state={"global_rng_states": {"test": 1}},
        )
    for key, value in before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    receipt = json.loads(next(tmp_path.glob("numeric_gate_event_*.json")).read_text())
    assert receipt["failure_phase"] == "gradients_pre_step"
    assert receipt["committed_optimizer_updates_before_event"] == 1
    assert receipt["checkpoint_written"] is False
    assert receipt["batch_skipped"] is False
    assert not (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "latest_good.pt").exists()
    replay_path = Path(receipt["replay_state_path"])
    assert replay_path.is_file()
    assert hashlib.sha256(replay_path.read_bytes()).hexdigest() == receipt["replay_state_sha256"]
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    assert replay["contains_gt_pixels"] is False
    assert "global_rng_states" in replay["pre_event_rng_states"]


def test_amp_silent_skip_fails_and_does_not_advance_state(tmp_path: Path) -> None:
    components = _tiny_components()
    model_before = {key: value.clone() for key, value in components.model.state_dict().items()}
    optimizer_before = copy.deepcopy(components.optimizer.state_dict())
    loss = sum(parameter.square().mean() for parameter in components.model.parameters())
    with pytest.raises(Formal2DTrainingError, match="amp_silent_skip"):
        _transactional_optimizer_update(
            loss=loss,
            components=components,
            scaler=_SkippingScaler(256.0),
            amp_enabled=True,
            event_step=2,
            committed_optimizer_updates=1,
            diagnostic_dir=tmp_path,
        )
    for key, value in model_before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    assert _checkpoint_event_counters(
        {
            "global_step": 1,
            "loop_event_step": 1,
            "data_event_step": 1,
            "committed_optimizer_updates": 1,
            "global_step_semantics": "LOOP_AND_DATA_EVENT_STEP_NOT_INFERRED_OPTIMIZER_COMMITS",
        },
        optimizer_before,
        "DMD_6F_2O3P",
    ) == (1, 1)
    receipt = json.loads(next(tmp_path.glob("numeric_gate_event_*.json")).read_text())
    assert receipt["failure_phase"] == "amp_silent_skip"


def test_standard_amp_overflow_skip_lowers_scale_then_finite_iteration_commits(
    tmp_path: Path,
) -> None:
    components = _tiny_components()
    components.preflight["protocol_id"] = "DMD_3F_1O3P"
    components.preflight["initialization_status"] = K3_LOADED_STATUS
    scaler = _DynamicOverflowScaler(256.0)
    model_before = {key: value.clone() for key, value in components.model.state_dict().items()}
    ema_before = {key: value.clone() for key, value in components.ema.shadow.items()}

    parameter = next(components.model.parameters())
    overflow_loss = parameter.sum()
    overflow_loss.register_hook(lambda gradient: gradient * torch.tensor(float("inf")))
    skipped = _standard_amp_optimizer_update(
        loss=overflow_loss,
        components=components,
        scaler=scaler,
        amp_enabled=True,
        scheduled_iteration=2,
        committed_optimizer_updates=1,
        diagnostic_dir=tmp_path,
        context={"sample_id": ["TRAIN_ONLY"]},
    )
    assert skipped.amp_overflow_skipped is True
    assert skipped.committed_optimizer_updates == 1
    assert skipped.previous_scale == 256.0
    assert skipped.new_scale == 128.0
    assert all(parameter.grad is None for parameter in components.model.parameters())
    for key, value in model_before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    for key, value in ema_before.items():
        assert torch.equal(components.ema.shadow[key], value)
    receipt = json.loads((tmp_path / "amp_overflow_skip_iteration_000002.json").read_text())
    assert receipt["status"] == "AMP_OVERFLOW_SKIP"
    assert receipt["optimizer_updated"] is False
    assert receipt["ema_updated"] is False
    assert receipt["checkpoint_written"] is False
    assert receipt["committed_optimizer_updates_before"] == 1
    assert receipt["committed_optimizer_updates_after"] == 1

    finite_loss = sum(value.square().mean() for value in components.model.parameters())
    committed = _standard_amp_optimizer_update(
        loss=finite_loss,
        components=components,
        scaler=scaler,
        amp_enabled=True,
        scheduled_iteration=3,
        committed_optimizer_updates=1,
        diagnostic_dir=tmp_path,
    )
    assert committed.amp_overflow_skipped is False
    assert committed.committed_optimizer_updates == 2
    assert committed.gradient_norm is not None
    components.ema.update(components.model)
    assert not (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "best.pt").exists()

    progress = _checkpoint_payload(
        components,
        3,
        {},
        scaler=scaler,
        committed_optimizer_updates=2,
        amp_overflow_skips=1,
        consecutive_amp_overflow_skips=0,
    )
    assert _checkpoint_event_counters(progress["metadata"], progress["optimizer"], "DMD_3F_1O3P") == (3, 2)
    assert progress["metadata"]["global_step_semantics"] == SCHEDULED_COUNTER_SEMANTICS


def test_post_step_nonfinite_state_is_rolled_back(tmp_path: Path, monkeypatch) -> None:
    components = _tiny_components()
    model_before = {key: value.clone() for key, value in components.model.state_dict().items()}
    optimizer_before = copy.deepcopy(components.optimizer.state_dict())
    original_step = components.optimizer.step

    def corrupting_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        next(components.model.parameters()).data.fill_(float("inf"))
        return result

    monkeypatch.setattr(components.optimizer, "step", corrupting_step)
    loss = sum(parameter.square().mean() for parameter in components.model.parameters())
    with pytest.raises(Formal2DTrainingError, match="state_post_step"):
        _transactional_optimizer_update(
            loss=loss,
            components=components,
            scaler=_StatefulScaler(1.0),
            amp_enabled=False,
            event_step=2,
            committed_optimizer_updates=1,
            diagnostic_dir=tmp_path,
        )
    for key, value in model_before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    assert _checkpoint_event_counters(
        {
            "global_step": 1,
            "loop_event_step": 1,
            "data_event_step": 1,
            "committed_optimizer_updates": 1,
            "global_step_semantics": "LOOP_AND_DATA_EVENT_STEP_NOT_INFERRED_OPTIMIZER_COMMITS",
        },
        optimizer_before,
        "DMD_6F_2O3P",
    ) == (1, 1)


def test_transaction_snapshot_preserves_tensor_devices() -> None:
    components = _tiny_components()
    snapshot = _snapshot_optimizer_transaction(components, _StatefulScaler(1.0))
    model_devices = {value.device for value in components.model.state_dict().values()}
    snapshot_devices = {value.device for value in snapshot["model"].values()}
    assert snapshot_devices == model_devices
    optimizer_devices = {
        value.device
        for state in components.optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    }
    snapshot_optimizer_devices = {
        value.device
        for state in snapshot["optimizer"]["state"].values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    }
    assert snapshot_optimizer_devices == optimizer_devices


def test_nonfinite_scaler_after_step_is_detected_and_rolled_back(tmp_path: Path) -> None:
    components = _tiny_components()
    scaler = _PoisoningScaler(256.0)
    model_before = {key: value.clone() for key, value in components.model.state_dict().items()}
    optimizer_before = copy.deepcopy(components.optimizer.state_dict())
    loss = sum(parameter.square().mean() for parameter in components.model.parameters())
    with pytest.raises(Formal2DTrainingError, match="state_post_step"):
        _transactional_optimizer_update(
            loss=loss,
            components=components,
            scaler=scaler,
            amp_enabled=True,
            event_step=2,
            committed_optimizer_updates=1,
            diagnostic_dir=tmp_path,
        )
    assert scaler.get_scale() == 256.0
    for key, value in model_before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    assert _checkpoint_event_counters(
        {
            "global_step": 1,
            "loop_event_step": 1,
            "data_event_step": 1,
            "committed_optimizer_updates": 1,
            "global_step_semantics": "LOOP_AND_DATA_EVENT_STEP_NOT_INFERRED_OPTIMIZER_COMMITS",
        },
        optimizer_before,
        "DMD_6F_2O3P",
    ) == (1, 1)
    receipt = json.loads(next(tmp_path.glob("numeric_gate_event_*.json")).read_text())
    assert receipt["failure_phase"] == "state_post_step"
    assert receipt["context"]["scaler_state"] == ["scaler.scale"]


def test_nonfinite_optimizer_param_group_is_detected_and_rolled_back(
    tmp_path: Path, monkeypatch
) -> None:
    components = _tiny_components()
    scaler = _StatefulScaler(1.0)
    model_before = {key: value.clone() for key, value in components.model.state_dict().items()}
    original_step = components.optimizer.step

    def poisoning_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        components.optimizer.param_groups[0]["lr"] = float("nan")
        return result

    monkeypatch.setattr(components.optimizer, "step", poisoning_step)
    loss = sum(parameter.square().mean() for parameter in components.model.parameters())
    with pytest.raises(Formal2DTrainingError, match="state_post_step"):
        _transactional_optimizer_update(
            loss=loss,
            components=components,
            scaler=scaler,
            amp_enabled=False,
            event_step=2,
            committed_optimizer_updates=1,
            diagnostic_dir=tmp_path,
        )
    assert components.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    for key, value in model_before.items():
        assert torch.equal(components.model.state_dict()[key], value)
    receipt = json.loads(next(tmp_path.glob("numeric_gate_event_*.json")).read_text())
    assert receipt["context"]["optimizer_state"]
    assert any("param_groups" in path for path in receipt["context"]["optimizer_state"])


def test_resume_history_preserves_best_and_rejects_checkpoint_conflict(tmp_path: Path) -> None:
    fields = [
        "global_step",
        "mean_val_diff_loss",
        "mean_val_phys_loss",
        "mean_val_total_loss",
        "mean_val_x0_psnr",
        "mean_val_x0_ssim",
        "validation_entry_count",
    ]
    rows = [
        dict(zip(fields, (2, 0.8, 0.2, 1.0, 20.0, 0.7, 3))),
        dict(zip(fields, (4, 0.5, 0.1, 0.6, 21.0, 0.8, 3))),
    ]
    history = tmp_path / "validation_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = _load_history_for_resume(history, start_step=5)
    assert best is not None
    assert best["global_step"] == 4.0
    assert best["mean_val_total_loss"] == pytest.approx(0.6)

    with history.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(
            dict(zip(fields, (5, 0.4, 0.1, 0.5, 22.0, 0.85, 3)))
        )
    with pytest.raises(Formal2DTrainingError, match="history step sequence conflicts"):
        _load_history_for_resume(history, start_step=5)


def test_k3_null_configured_hash_resolves_from_completed_k9_receipt(tmp_path: Path) -> None:
    source = APDConditionedUNet2D(
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        time_dim=16,
        groups=8,
    )
    target = APDConditionedUNet2D(
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        time_dim=16,
        groups=8,
    )
    for parameter in source.parameters():
        parameter.data.fill_(0.125)
    k9 = protocol_registry.require("DMD_9F_3O3P")
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "ema": source.state_dict(),
            "metadata": {
                "completion_status": "FORMAL_TRAINING_COMPLETE",
                "training_protocol_id": k9.protocol_id,
                "training_protocol_hash": k9.protocol_hash,
                "architecture_hash": hashlib.sha256(b"unused").hexdigest(),
                "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
                "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
            },
        },
        checkpoint,
    )
    # Architecture hashes depend on schema, not weights.
    from unisim.checkpoint_contract import architecture_hash

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["metadata"]["architecture_hash"] = architecture_hash(target)
    torch.save(saved, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = tmp_path / "completion_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "completion_status": "FORMAL_TRAINING_COMPLETE",
                "checkpoint_sha256": checkpoint_hash,
                "protocol_id": k9.protocol_id,
                "protocol_hash": k9.protocol_hash,
                "architecture_hash": architecture_hash(target),
                "normalization_contract_hash": NORMALIZATION_CONTRACT_HASH,
                "input_tensor_dimensionality": INPUT_TENSOR_DIMENSIONALITY,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "initialization": {
            "checkpoint_path": str(checkpoint),
            "completion_receipt_path": str(receipt),
            "checkpoint_sha256": None,
        }
    }
    _load_verified_k9_initialization(target, config, tmp_path)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)

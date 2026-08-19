#!/usr/bin/env python3
"""Fail-closed APD-SIM-6 evaluation on the frozen Official R2 test bundle.

This entrypoint is intentionally independent from the training process.  While
the formal APD CUDA lock is owned by a live process it exits before importing
PyTorch, opening a checkpoint, or opening a test NPZ.  Once training has
naturally completed, it accepts only the formal DMD_6F_2O3P best checkpoint
bound to its completion receipt and immutable training identities.

The executable path consumes, but never regenerates, the thirty shared raw
stacks.  It writes native float32 reconstructions and an auditable manifest; it
does not open GT images or compute test metrics.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "apd_dmd_r2" / "train6_formal.json").is_file():
            return parent
    raise RuntimeError(f"cannot locate DIFF-SIM project root from {here}")


PROJECT_ROOT = _project_root()
PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"
REGISTRY_HASH = "5186ebd2a17c5e39ccf486f3e7b61fb3cf7f86c907c9460740fbc23385fa2968"
RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
EXPECTED_CLASSES = {"CCP": 10, "ER": 10, "MT": 10}
COMPLETION_STATUS = "FORMAL_TRAINING_COMPLETE"
BEST_RULE_ID = "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"
NORMALIZATION_HASH = "a148bcb41ab149285435bc0e0bd57526c6346fd905a6abece6721f204e1cd2d3"
LOCK_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "apd_dmd_geometry_r2"
    / "_device_locks"
    / "cuda"
    / "training.lock"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json"
CURRENT_POINTER = PROJECT_ROOT / "outputs" / "OFFICIAL_BASELINES_DMD6_R2_CURRENT.json"

# Frozen before formal test evaluation.  One deterministic reconstruction per
# FOV is permitted; there is no best-of-N selection.
EVALUATION_POLICY = {
    "policy_id": "APD6_OFFICIAL_R2_WARMSTART_MAP_V1",
    "weights": "ema",
    "widefield_initialization": "arithmetic_mean_of_exactly_six_observed_frames_then_clip_0_1",
    "diffusion_seed": "uint64_be(SHA256(policy_id|checkpoint_sha256|raw_stack_sha256)[:8]) mod 2^63",
    "diffusion_init_t": 600,
    "diffusion_steps_including_endpoints": 80,
    "ddim_eta": 0.0,
    "padding": "reflect_bottom_right_to_multiple_16_then_exact_unpad",
    "physics_refinement_iterations": 40,
    "physics_refinement_optimizer": "Adam",
    "physics_refinement_lr": 0.005,
    "physics_data_term": "Poisson-Gaussian camera NLL on all six observed frames",
    "physics_prior_weight": 0.0,
    "output_clip": [0.0, 1.0],
    "test_gt_access": False,
    "best_of_n": False,
}


class EvaluationBlocked(RuntimeError):
    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


EVALUATION_POLICY_HASH = hashlib.sha256(canonical_json_bytes(EVALUATION_POLICY)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    header = canonical_json_bytes({"dtype": value.dtype.str, "shape": list(value.shape)})
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationBlocked("APD6_FINALIZATION_PENDING", f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise EvaluationBlocked("APD6_FINALIZATION_BLOCKED", f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationBlocked("APD6_FINALIZATION_BLOCKED", f"{label} is not a JSON object")
    return value


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # type: ignore[attr-defined]
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_training_guard(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    """Inspect the training lock without modifying or renaming it."""
    if not lock_path.exists():
        return {"active": False, "lock_path": str(lock_path), "lock_exists": False}
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except Exception as exc:
        raise EvaluationBlocked(
            "APD_GPU_STATE_UNRESOLVED",
            f"training lock is unreadable and was left untouched: {lock_path}: {exc}",
        ) from exc
    running = _pid_is_running(pid)
    result = {
        "active": running,
        "lock_path": str(lock_path),
        "lock_exists": True,
        "pid": pid,
        "protocol_id": payload.get("protocol_id"),
        "script_name": payload.get("script_name"),
        "config_hash": payload.get("config_hash"),
    }
    if running:
        raise EvaluationBlocked(
            "APD_GPU_ACTIVE",
            f"PID {pid} owns {lock_path}; no checkpoint, test bundle, or CUDA runtime was accessed",
        )
    # A stale lock is not ours to alter.  Treat it as unresolved rather than
    # silently assuming that formal GPU work is safe.
    raise EvaluationBlocked(
        "APD_GPU_STATE_UNRESOLVED",
        f"stale-looking lock remains at {lock_path}; evaluator will not modify it",
    )


def resolve_output_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
    else:
        pointer = _load_json(CURRENT_POINTER, "Official R2 current pointer")
        root = Path(str(pointer.get("output_root", ""))).resolve()
    expected_parent = (PROJECT_ROOT / "outputs").resolve()
    if root.parent != expected_parent or not root.name.startswith("OFFICIAL_BASELINES_DMD6_R2_"):
        raise EvaluationBlocked("APD6_FINALIZATION_BLOCKED", f"invalid Official R2 output root: {root}")
    return root


def validate_static_contracts(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = _load_json(config_path, "formal APD6 config")
    config_payload = dict(config)
    embedded = str(config_payload.pop("config_payload_hash", ""))
    actual_payload_hash = hashlib.sha256(canonical_json_bytes(config_payload)).hexdigest()
    if embedded != actual_payload_hash:
        raise EvaluationBlocked("APD6_CONFIG_HASH_MISMATCH", "formal config payload hash changed")
    if (
        config.get("config_type") != "APD_DMD_R2_FORMAL_2D"
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("protocol_hash") != PROTOCOL_HASH
        or config.get("protocol_registry_hash") != REGISTRY_HASH
    ):
        raise EvaluationBlocked("APD6_PROTOCOL_MISMATCH", "formal config is not frozen DMD_6F_2O3P")
    model = config.get("model", {})
    normalization = config.get("forward", {}).get("normalization", {})
    if (
        model.get("architecture") != "APDConditionedUNet2D"
        or int(model.get("in_channels", -1)) != 31
        or int(model.get("kmax", -1)) != 15
        or normalization != {"clip": [0.0, 1.0], "lower_percentile": 0.5, "upper_percentile": 99.5}
    ):
        raise EvaluationBlocked("APD6_CONFIG_HASH_MISMATCH", "model/normalization contract drift")
    protocol_path = PROJECT_ROOT / "protocols" / "dmd_6f_2o3p.json"
    protocol = _load_json(protocol_path, "DMD6 protocol")
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("protocol_hash") != PROTOCOL_HASH
        or protocol.get("frame_count") != 6
        or protocol.get("orientation_count") != 2
        or protocol.get("phases_per_orientation") != 3
        or tuple(protocol.get("raw_frame_order", ())) != RAW_ORDER
    ):
        raise EvaluationBlocked("APD6_PROTOCOL_MISMATCH", "protocol file identity/geometry drift")
    return {
        "config": config,
        "config_file_sha256": sha256_file(config_path),
        "config_payload_hash": actual_payload_hash,
        "protocol_file_sha256": sha256_file(protocol_path),
        "evaluation_policy_hash": EVALUATION_POLICY_HASH,
    }


def validate_completion_receipt(
    config: Mapping[str, Any], checkpoint_override: str | None = None
) -> dict[str, Any]:
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EvaluationBlocked("APD6_FINALIZATION_BLOCKED", "formal config lacks outputs mapping")
    checkpoint = Path(
        checkpoint_override if checkpoint_override else str(outputs.get("best_checkpoint_path", ""))
    ).resolve()
    receipt_path = Path(str(outputs.get("best_checkpoint_receipt_path", ""))).resolve()
    final_path = Path(str(outputs.get("checkpoint_dir", ""))).resolve() / "final.pt"
    receipt = _load_json(receipt_path, "APD6 best checkpoint receipt")
    if receipt.get("completion_status") != COMPLETION_STATUS:
        raise EvaluationBlocked(
            "APD6_FINALIZATION_PENDING",
            f"best receipt is {receipt.get('completion_status')!r}, not {COMPLETION_STATUS}",
        )
    if receipt.get("test_data_used_for_selection") is not False:
        raise EvaluationBlocked("APD6_TEST_LEAKAGE_BLOCKED", "receipt does not deny test-based selection")
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("protocol_hash") != PROTOCOL_HASH
        or receipt.get("selection_rule") != BEST_RULE_ID
        or receipt.get("normalization_contract_hash") != NORMALIZATION_HASH
        or receipt.get("architecture_contract") != "APD_DMD_R2_STRICT_2D_CONV_V1"
        or receipt.get("input_tensor_dimensionality") != "4D_BCHW"
    ):
        raise EvaluationBlocked("APD6_CHECKPOINT_PROVENANCE_BLOCKED", "completion receipt identity mismatch")
    if Path(str(receipt.get("checkpoint_path", ""))).resolve() != checkpoint:
        raise EvaluationBlocked("APD6_CHECKPOINT_PROVENANCE_BLOCKED", "receipt/checkpoint path mismatch")
    if not checkpoint.is_file() or not final_path.is_file():
        raise EvaluationBlocked("APD6_FINALIZATION_PENDING", "best.pt or formal final.pt is absent")
    checkpoint_hash = sha256_file(checkpoint)
    final_hash = sha256_file(final_path)
    if checkpoint_hash != receipt.get("checkpoint_sha256"):
        raise EvaluationBlocked("APD6_CHECKPOINT_HASH_MISMATCH", "best checkpoint hash mismatch")
    if final_hash != receipt.get("formal_final_checkpoint_sha256"):
        raise EvaluationBlocked("APD6_CHECKPOINT_HASH_MISMATCH", "final checkpoint hash mismatch")
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "final_checkpoint": final_path,
        "final_checkpoint_sha256": final_hash,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": sha256_file(receipt_path),
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvaluationBlocked("APD_SHARED_BUNDLE_MISSING", f"missing manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _scalar(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", "expected scalar NPZ metadata")
    return array.reshape(-1)[0].item()


def _aggregate_hash(values: Iterable[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _expected_aggregate_from_audit(output_root: Path) -> str:
    path = output_root / "01_shared_contract" / "roundtrip_audit.md"
    if not path.is_file():
        raise EvaluationBlocked("APD_SHARED_BUNDLE_MISSING", f"missing roundtrip audit: {path}")
    match = re.search(r"Test aggregate raw-stack hash:\s*`([0-9a-f]{64})`", path.read_text(encoding="utf-8"))
    if not match:
        raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", "test aggregate absent from audit")
    return match.group(1)


def validate_test_bundle(output_root: Path, *, open_payloads: bool) -> dict[str, Any]:
    import numpy as np

    contract_root = output_root / "01_shared_contract"
    bundle_root = (contract_root / "test30_dmd6_bundle").resolve()
    manifest_path = contract_root / "test30_dmd6_manifest.tsv"
    rows = _read_tsv(manifest_path)
    if len(rows) != 30:
        raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", f"expected 30 rows, got {len(rows)}")
    classes: dict[str, int] = {}
    raw_hashes: list[str] = []
    expected_keys = {
        "acquisition_noise_seed", "frame_order", "protocol_hash", "protocol_id", "raw_stack",
        "sample_id", "source_file_sha256", "source_identity_digest", "theta_fields", "theta_values",
    }
    seen: set[str] = set()
    for index, row in enumerate(rows):
        structure = row.get("structure_class", "")
        classes[structure] = classes.get(structure, 0) + 1
        if (
            int(row.get("order", -1)) != index
            or row.get("sample_id") in seen
            or row.get("protocol_id") != PROTOCOL_ID
            or row.get("protocol_hash") != PROTOCOL_HASH
            or int(row.get("frame_count", -1)) != 6
            or tuple(row.get("frame_order", "").split("/")) != RAW_ORDER
            or row.get("raw_shape") != "6 1004 1004"
            or row.get("raw_dtype") != "float32"
            or row.get("split") != "sealed_test"
            or row.get("test_gt_embedded_in_npz", "").lower() != "false"
            or row.get("roundtrip_status") != "PASS"
        ):
            raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", f"manifest row {index} drift")
        seen.add(row["sample_id"])
        raw_hashes.append(row["raw_stack_sha256"])
        path = (contract_root / Path(row["npz_path"])).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", "NPZ escapes frozen bundle") from exc
        if not path.is_file() or sha256_file(path) != row.get("npz_sha256"):
            raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", f"NPZ hash mismatch: {path}")
        if open_payloads:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != expected_keys or any("gt" in key.lower() for key in archive.files):
                    raise EvaluationBlocked("APD_TEST_GT_ACCESS_BLOCKED", f"unexpected NPZ members: {path}")
                raw = np.ascontiguousarray(archive["raw_stack"])
                if (
                    raw.shape != (6, 1004, 1004)
                    or raw.dtype != np.float32
                    or not bool(np.isfinite(raw).all())
                    or array_sha256(raw) != row["raw_stack_sha256"]
                    or _scalar(archive["protocol_id"]) != PROTOCOL_ID
                    or _scalar(archive["protocol_hash"]) != PROTOCOL_HASH
                    or tuple(str(value) for value in archive["frame_order"].tolist()) != RAW_ORDER
                    or _scalar(archive["sample_id"]) != row["sample_id"]
                ):
                    raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", f"NPZ payload drift: {path}")
    if classes != EXPECTED_CLASSES:
        raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", f"test class counts drift: {classes}")
    aggregate = _aggregate_hash(raw_hashes)
    expected_aggregate = _expected_aggregate_from_audit(output_root)
    if aggregate != expected_aggregate:
        raise EvaluationBlocked("APD_SHARED_BUNDLE_HASH_MISMATCH", "aggregate raw-stack hash mismatch")
    return {
        "manifest": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "bundle_root": bundle_root,
        "rows": rows,
        "entry_count": 30,
        "class_counts": classes,
        "aggregate_raw_stack_sha256": aggregate,
        "payloads_opened": bool(open_payloads),
    }


def load_bound_model(
    config: Mapping[str, Any], completion: Mapping[str, Any], device: Any
) -> tuple[Any, Any, Mapping[str, Any]]:
    import torch

    from unisim.checkpoint_contract import architecture_hash, load_checkpoint_bound
    from unisim.formal_training_2d import DiffusionScheduler2D
    from unisim.model2d import APDConditionedUNet2D, assert_strictly_2d_model

    model_cfg = config["model"]
    model = APDConditionedUNet2D(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(int(v) for v in model_cfg["channel_mults"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        dropout=float(model_cfg["dropout"]),
        time_dim=int(model_cfg["time_dim"]),
        groups=int(model_cfg["groups"]),
    )
    assert_strictly_2d_model(model)
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
        "completion_status": COMPLETION_STATUS,
    }
    payload = load_checkpoint_bound(
        completion["checkpoint"], protocol=PROTOCOL_ID,
        expected_sha256=str(completion["checkpoint_sha256"]), expected_identities=expected,
    )
    state = payload.get("ema")
    if not isinstance(state, Mapping):
        raise EvaluationBlocked("APD6_CHECKPOINT_PROVENANCE_BLOCKED", "EMA state is absent")
    if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in state.values()):
        raise EvaluationBlocked("APD6_CHECKPOINT_PROVENANCE_BLOCKED", "EMA contains non-finite values")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    scheduler = DiffusionScheduler2D(
        int(config["training"]["diffusion_steps"]), device, str(config["training"]["beta_schedule"])
    )
    return model, scheduler, payload["metadata"]


def _seed(checkpoint_hash: str, raw_hash: str) -> int:
    label = f"{EVALUATION_POLICY['policy_id']}|{checkpoint_hash}|{raw_hash}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") & ((1 << 63) - 1)


def _ddim_reconstruct(raw: Any, model: Any, scheduler: Any, seed: int, device: Any) -> Any:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from unisim.sim_forward_2d import embed_raw_to_slots_2d

    observed = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
    wide = observed.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    slotted, mask = embed_raw_to_slots_2d(observed, PROTOCOL_ID)
    height, width = observed.shape[-2:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        wide = F.pad(wide, (0, pad_w, 0, pad_h), mode="reflect")
        slotted = F.pad(slotted, (0, pad_w, 0, pad_h), mode="reflect")
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(wide.shape, generator=generator, device=device, dtype=wide.dtype)
    init_t = int(EVALUATION_POLICY["diffusion_init_t"])
    steps = int(EVALUATION_POLICY["diffusion_steps_including_endpoints"])
    timesteps = np.rint(np.linspace(init_t, 0, steps)).astype(np.int64).tolist()
    if len(set(timesteps)) != steps or timesteps[0] != init_t or timesteps[-1] != 0:
        raise EvaluationBlocked("APD6_INFERENCE_POLICY_BLOCKED", "invalid frozen DDIM timestep sequence")
    timestep = torch.full((1,), init_t, device=device, dtype=torch.long)
    x = scheduler.q_sample(wide, timestep, noise)
    x0 = wide
    with torch.inference_mode():
        for index, current in enumerate(timesteps):
            t = torch.full((1,), int(current), device=device, dtype=torch.long)
            epsilon = model(torch.cat((x, slotted, mask), dim=1), t)
            x0 = scheduler.predict_x0(x, t, epsilon).clamp(0.0, 1.0)
            previous = int(timesteps[index + 1]) if index + 1 < len(timesteps) else -1
            if previous < 0:
                x = x0
            else:
                alpha = scheduler.alpha_bar[previous]
                x = alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * epsilon
    return x0[..., :height, :width]


def _physics_refine(x_init: Any, raw: Any, theta_fields: Sequence[str], theta_values: Any, config: Mapping[str, Any], device: Any) -> Any:
    import numpy as np
    import torch

    from unisim.formal_training_2d import _make_sim_config
    from unisim.sim_forward_2d import forward_protocol_clean_2d

    theta_source = dict(zip((str(v) for v in theta_fields), (float(v) for v in theta_values)))
    theta = {
        "k_ratio_xy": torch.tensor([theta_source["k_ratio_xy"]], device=device),
        "mod_depth": torch.tensor([theta_source["mod_depth"]], device=device),
        "phase_offsets": torch.tensor([theta_source["phase_offset_rad"]], device=device),
        "angle_offsets": torch.tensor([theta_source["angle_offset_deg"]], device=device),
        "background": torch.tensor([theta_source["background"]], device=device),
        "psf_sigma_scale": torch.tensor([theta_source["psf_sigma_scale"]], device=device),
        "photon_scale": torch.tensor([theta_source["photon_scale"]], device=device),
        "read_noise_e": torch.tensor([theta_source["read_noise_e"]], device=device),
    }
    observed = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
    estimate = x_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=float(EVALUATION_POLICY["physics_refinement_lr"]))
    sim_config = _make_sim_config(config)
    for _ in range(int(EVALUATION_POLICY["physics_refinement_iterations"])):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = forward_protocol_clean_2d(estimate, sim_config, PROTOCOL_ID, theta=theta)
        photon = theta["photon_scale"].clamp_min(1e-12)
        variance = predicted.clamp_min(0.0) / photon + (theta["read_noise_e"] / photon).square() + 1e-8
        loss = 0.5 * (torch.log(variance) + (observed - predicted).square() / variance).mean()
        if not bool(torch.isfinite(loss)):
            raise EvaluationBlocked("APD6_NUMERICAL_FAILURE", "non-finite physics-refinement loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            estimate.clamp_(0.0, 1.0)
    return estimate.detach()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def execute(
    output_root: Path,
    static: Mapping[str, Any],
    completion: Mapping[str, Any],
    bundle: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise EvaluationBlocked("APD6_GPU_RUNTIME_BLOCKED", "CUDA is unavailable after the APD lock gate")
    device = torch.device("cuda")
    model, scheduler, checkpoint_metadata = load_bound_model(static["config"], completion, device)
    output_dir = output_dir.resolve()
    expected_parent = (output_root / "10_apd6_finalization").resolve()
    try:
        output_dir.relative_to(expected_parent)
    except ValueError as exc:
        raise EvaluationBlocked("APD6_OUTPUT_PATH_BLOCKED", "output must stay under 10_apd6_finalization") from exc
    rows_out: list[dict[str, Any]] = []
    fields = (
        "order", "sample_id", "structure_class", "protocol_id", "protocol_hash", "frame_order",
        "input_npz_path", "input_npz_sha256", "input_raw_stack_sha256", "checkpoint_path",
        "checkpoint_sha256", "checkpoint_global_step", "evaluation_policy_hash", "diffusion_seed",
        "recon_path", "recon_file_sha256", "recon_array_sha256", "recon_shape", "recon_dtype",
        "runtime_seconds", "test_gt_accessed", "status",
    )
    in_progress = output_dir / "apd6_recon_manifest.in_progress.tsv"
    for row in bundle["rows"]:
        input_path = (Path(bundle["manifest"]).parent / row["npz_path"]).resolve()
        with np.load(input_path, allow_pickle=False) as archive:
            raw = np.ascontiguousarray(archive["raw_stack"], dtype=np.float32)
            theta_fields = tuple(str(v) for v in archive["theta_fields"].tolist())
            theta_values = np.asarray(archive["theta_values"], dtype=np.float64)
        seed = _seed(str(completion["checkpoint_sha256"]), row["raw_stack_sha256"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        warm = _ddim_reconstruct(raw, model, scheduler, seed, device)
        recon = _physics_refine(warm, raw, theta_fields, theta_values, static["config"], device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        array = np.ascontiguousarray(recon[0, 0].detach().cpu().numpy(), dtype=np.float32)
        if array.shape != (1004, 1004) or not bool(np.isfinite(array).all()):
            raise EvaluationBlocked("APD6_NUMERICAL_FAILURE", f"invalid output for {row['sample_id']}")
        recon_path = output_dir / "recon" / f"{int(row['order']):03d}_{row['sample_id']}.npy"
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        _atomic_write(recon_path, stream.getvalue())
        rows_out.append({
            "order": row["order"], "sample_id": row["sample_id"], "structure_class": row["structure_class"],
            "protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH, "frame_order": "/".join(RAW_ORDER),
            "input_npz_path": str(input_path), "input_npz_sha256": row["npz_sha256"],
            "input_raw_stack_sha256": row["raw_stack_sha256"], "checkpoint_path": str(completion["checkpoint"]),
            "checkpoint_sha256": completion["checkpoint_sha256"],
            "checkpoint_global_step": checkpoint_metadata["global_step"],
            "evaluation_policy_hash": EVALUATION_POLICY_HASH, "diffusion_seed": seed,
            "recon_path": str(recon_path), "recon_file_sha256": sha256_file(recon_path),
            "recon_array_sha256": array_sha256(array), "recon_shape": "1004 1004", "recon_dtype": "float32",
            "runtime_seconds": f"{elapsed:.9f}", "test_gt_accessed": False, "status": "PASS",
        })
        _atomic_write(in_progress, _tsv_bytes(rows_out, fields))
    final_manifest = output_dir / "apd6_recon_manifest.tsv"
    _atomic_write(final_manifest, _tsv_bytes(rows_out, fields))
    output_aggregate = _aggregate_hash(row["recon_array_sha256"] for row in rows_out)
    handoff = {
        "schema_version": 2,
        "status": "APD6_SHARED_BUNDLE_RECONSTRUCTION_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "frame_order": list(RAW_ORDER),
        "test_bundle_hash": bundle["aggregate_raw_stack_sha256"],
        "test_manifest_sha256": bundle["manifest_sha256"],
        "apd6_checkpoint": str(completion["checkpoint"]),
        "apd6_checkpoint_sha256": completion["checkpoint_sha256"],
        "completion_receipt_sha256": completion["receipt_sha256"],
        "evaluation_policy": EVALUATION_POLICY,
        "evaluation_policy_hash": EVALUATION_POLICY_HASH,
        "reconstruction_manifest": str(final_manifest),
        "reconstruction_manifest_sha256": sha256_file(final_manifest),
        "output_aggregate_array_sha256": output_aggregate,
        "entry_count": 30,
        "test_gt_access_count": 0,
        "test_metrics_computed": False,
    }
    handoff_path = output_root / "10_apd6_finalization" / "APD6_SHARED_BUNDLE_HANDOFF.json"
    _atomic_write(handoff_path, canonical_json_bytes(handoff) + b"\n")
    return handoff


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", help="Official R2 output root; defaults to CURRENT pointer")
    parser.add_argument("--checkpoint", help="Explicit formal best.pt; must match completion receipt")
    parser.add_argument("--verify-bundle-payloads", action="store_true", help="Open/hash all 30 NPZ payloads")
    parser.add_argument("--execute", action="store_true", help="Run one frozen reconstruction per FOV on CUDA")
    parser.add_argument("--output-dir", help="Defaults below 10_apd6_finalization/apd6_reconstruction")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_accessed = False
    bundle_accessed = False
    try:
        active_training_guard()
        static = validate_static_contracts()
        completion = validate_completion_receipt(static["config"], args.checkpoint)
        checkpoint_accessed = True
        root = resolve_output_root(args.output_root)
        bundle = validate_test_bundle(root, open_payloads=bool(args.execute or args.verify_bundle_payloads))
        bundle_accessed = True
        if not args.execute:
            result = {
                "status": "APD6_EVALUATION_PREFLIGHT_PASS",
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "completion_receipt_sha256": completion["receipt_sha256"],
                "test30_bundle_hash": bundle["aggregate_raw_stack_sha256"],
                "bundle_payloads_opened": bundle["payloads_opened"],
                "evaluation_policy_hash": EVALUATION_POLICY_HASH,
                "cuda_initialized": False,
                "formal_reconstruction_executed": False,
            }
        else:
            destination = (
                Path(args.output_dir).resolve()
                if args.output_dir
                else root / "10_apd6_finalization" / "apd6_reconstruction"
            )
            result = execute(root, static, completion, bundle, destination)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except EvaluationBlocked as exc:
        result = {
            "status": exc.status,
            "detail": exc.detail,
            "checkpoint_accessed": checkpoint_accessed,
            "test_bundle_accessed": bundle_accessed,
            "cuda_initialized": False,
            "formal_reconstruction_executed": False,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 3 if exc.status in {"APD_GPU_ACTIVE", "APD_GPU_STATE_UNRESOLVED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

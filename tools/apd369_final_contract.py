"""Fail-closed contracts for final APD-SIM 3/6/9 re-evaluation.

This module is inference-only.  It validates immutable checkpoints, freezes the
existing 30-FOV identity grid, and exposes the three protocol plans.  Model,
forward, metric, and refinement mathematics remain in the existing production
modules and are not reimplemented here.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "APD369_FINAL_REEVALUATION_20260817_090027"
FORMAL_DMD6_ROOT = ROOT / "outputs" / "OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
FORMAL_TEST_LEDGER = FORMAL_DMD6_ROOT / "01_shared_contract" / "test30_dmd6_manifest.tsv"
TRAIN_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "train_manifest.json"
VALIDATION_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "validation_manifest.json"
SEALED_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "sealed_test_manifest.json"
PRINCIPAL_DIFFUSION_SEED_BASE = 20260812
R2_SELECTION_RULE = "R2_MIN_TOTAL_THEN_PSNR_SSIM_EARLIEST_V1"
R3_SELECTION_RULE = "R3_MAX_FULL_PSNR_THEN_FULL_SSIM_THEN_STAGE1_PSNR_THEN_MIN_TOTAL_THEN_EARLIER_V1"


class APD369ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalPlan:
    method: str
    label: str
    protocol_id: str
    protocol_hash: str
    raw_order: tuple[str, ...]
    validity_mask: tuple[int, ...]
    config_path: Path
    checkpoint_path: Path
    receipt_path: Path
    history_path: Path
    selection_rule: str


PLANS = (
    FinalPlan(
        "APD-SIM-3", "DMD-3F", "DMD_3F_1O3P",
        "e1e70fcfab3b97359fb0b9a44dfcace166922eaf8585927de5b9c9091fdc79e9",
        ("X0", "X120", "X240"),
        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ROOT / "configs" / "apd_dmd_r2" / "train3_formal_restart_simple_r1.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "best.pt",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "best_checkpoint_receipt.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd3_restart_simple_r1" / "validation_history.csv",
        R2_SELECTION_RULE,
    ),
    FinalPlan(
        "APD-SIM-6", "DMD-6F", "DMD_6F_2O3P",
        "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83",
        ("H0", "H120", "H240", "V0", "V120", "V240"),
        (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ROOT / "configs" / "apd_dmd_r2" / "train6_formal.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best.pt",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "best_checkpoint_receipt.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r2" / "dmd6" / "validation_history.csv",
        R2_SELECTION_RULE,
    ),
    FinalPlan(
        "APD-SIM-9", "DMD-9F", "DMD_9F_3O3P",
        "449670667c6ecb043fc55a303872a9e47cddeceb9ef97204b087ca3d45b095e3",
        ("X0", "X120", "X240", "Y0", "Y120", "Y240", "Z0", "Z120", "Z240"),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
        ROOT / "configs" / "apd_dmd_r3" / "train9_retrain_r1.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r3" / "dmd9_retrain_r1" / "best.pt",
        ROOT / "checkpoints" / "apd_dmd_geometry_r3" / "dmd9_retrain_r1" / "formal_receipt.json",
        ROOT / "checkpoints" / "apd_dmd_geometry_r3" / "dmd9_retrain_r1" / "validation_history.csv",
        R3_SELECTION_RULE,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        raise APD369ContractError(f"refusing empty TSV: {path}")
    fieldnames = list(fields or materialized[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise APD369ContractError(f"expected JSON object: {path}")
    return value


def strict_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise APD369ContractError(f"{field}: bool is not an integer field")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise APD369ContractError(f"{field}: invalid integer {value!r}") from None
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise APD369ContractError(f"{field}: expected nonnegative integral value, got {value!r}")
    return int(number)


def _finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _tensor_state_digest(state: Mapping[str, Any]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
        digest.update(b"\n")
        digest.update(tensor.numpy().tobytes(order="C"))
        count += 1
    return count, digest.hexdigest()


def _history_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise APD369ContractError(f"empty validation history: {path}")
    return rows


def _select_history(plan: FinalPlan) -> tuple[dict[str, str], int]:
    rows = _history_rows(plan.history_path)
    if plan.selection_rule == R2_SELECTION_RULE:
        selected = min(
            rows,
            key=lambda row: (
                float(row["mean_val_total_loss"]),
                -float(row["mean_val_x0_psnr"]),
                -float(row["mean_val_x0_ssim"]),
                strict_nonnegative_int(row["global_step"], "global_step"),
            ),
        )
    else:
        full = [row for row in rows if row.get("validation_kind") == "FULL_PIPELINE"]
        if not full or any(not math.isfinite(float(row["mean_full_pipeline_psnr"])) for row in full):
            raise APD369ContractError("DMD9 full-pipeline validation history incomplete")
        selected = min(
            full,
            key=lambda row: (
                -float(row["mean_full_pipeline_psnr"]),
                -float(row["mean_full_pipeline_ssim"]),
                -float(row["mean_stage1_psnr"]),
                float(row["mean_val_total_loss"]),
                strict_nonnegative_int(row["scheduled_iteration"], "scheduled_iteration"),
            ),
        )
    return selected, len(rows)


def _selection_step(plan: FinalPlan, row: Mapping[str, str]) -> int:
    return strict_nonnegative_int(
        row["scheduled_iteration"] if plan.selection_rule == R3_SELECTION_RULE else row["global_step"],
        "selected_iteration",
    )


def _selection_metric(plan: FinalPlan, row: Mapping[str, str]) -> tuple[str, float]:
    if plan.selection_rule == R3_SELECTION_RULE:
        return "mean_full_pipeline_psnr", float(row["mean_full_pipeline_psnr"])
    return "mean_val_total_loss", float(row["mean_val_total_loss"])


def _receipt_checks(plan: FinalPlan, receipt: Mapping[str, Any], checkpoint_sha: str, selected: Mapping[str, str]) -> None:
    selected_step = _selection_step(plan, selected)
    if plan.selection_rule == R3_SELECTION_RULE:
        metrics = receipt.get("selection_metrics", {})
        checks = {
            "status": receipt.get("status") == "FORMAL_TRAINING_COMPLETE",
            "generation": receipt.get("run_generation_id") == "DMD9_RETRAIN_R1",
            "scheduled_target": strict_nonnegative_int(receipt.get("scheduled_iterations"), "scheduled_iterations") == 100000,
            "protocol": receipt.get("protocol_id") == plan.protocol_id and receipt.get("protocol_hash") == plan.protocol_hash,
            "selection_rule": receipt.get("selection_rule") == plan.selection_rule,
            "selected_step": strict_nonnegative_int(metrics.get("scheduled_iteration"), "receipt selected step") == selected_step,
            "selected_psnr": math.isclose(float(metrics.get("mean_full_pipeline_psnr")), float(selected["mean_full_pipeline_psnr"]), rel_tol=0.0, abs_tol=1e-12),
            "checkpoint_hash": receipt.get("checkpoint_sha256") == checkpoint_sha,
            "test_access": strict_nonnegative_int(receipt.get("test_access_count"), "test_access_count") == 0,
        }
    else:
        metrics = receipt.get("metrics", {})
        checks = {
            "status": receipt.get("completion_status") == "FORMAL_TRAINING_COMPLETE",
            "protocol": receipt.get("protocol_id") == plan.protocol_id and receipt.get("protocol_hash") == plan.protocol_hash,
            "selection_rule": receipt.get("selection_rule") == plan.selection_rule,
            "selected_step": strict_nonnegative_int(metrics.get("global_step"), "receipt selected step") == selected_step,
            "selected_total": math.isclose(float(metrics.get("mean_val_total_loss")), float(selected["mean_val_total_loss"]), rel_tol=0.0, abs_tol=1e-12),
            "checkpoint_hash": receipt.get("checkpoint_sha256") == checkpoint_sha,
            "test_access": receipt.get("test_data_used_for_selection") is False,
        }
    if not all(checks.values()):
        raise APD369ContractError(f"receipt/history conflict for {plan.method}: {checks}")


def audit_checkpoint(plan: FinalPlan) -> dict[str, Any]:
    from unisim.protocol_runtime import require_protocol

    for path in (plan.config_path, plan.checkpoint_path, plan.receipt_path, plan.history_path):
        if not path.is_file():
            raise APD369ContractError(f"required checkpoint artifact absent: {path}")
    config = read_json(plan.config_path)
    receipt = read_json(plan.receipt_path)
    spec = require_protocol(plan.protocol_id)
    checks = {
        "config_protocol": config.get("protocol_id") == plan.protocol_id,
        "config_protocol_hash": config.get("protocol_hash") == plan.protocol_hash,
        "registry_protocol_hash": spec.protocol_hash == plan.protocol_hash,
        "raw_order": tuple(spec.raw_frame_order) == plan.raw_order,
        "validity_mask": tuple(spec.validity_mask) == plan.validity_mask,
        "sealed_marker": config.get("sealed_test_no_access_marker") == "NO_ACCESS_DURING_TRAINING",
    }
    if not all(checks.values()):
        raise APD369ContractError(f"config/protocol conflict for {plan.method}: {checks}")
    checkpoint_sha = sha256_file(plan.checkpoint_path)
    selected, history_count = _select_history(plan)
    _receipt_checks(plan, receipt, checkpoint_sha, selected)
    payload = torch.load(plan.checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    selected_step = _selection_step(plan, selected)
    payload_checks = {
        "protocol": metadata.get("training_protocol_id") == plan.protocol_id,
        "protocol_hash": metadata.get("training_protocol_hash") == plan.protocol_hash,
        "selected_step": strict_nonnegative_int(metadata.get("global_step"), "checkpoint global_step") == selected_step,
        "selection_rule": metadata.get("checkpoint_selection_rule") == plan.selection_rule,
        "complete": metadata.get("completion_status") == "FORMAL_TRAINING_COMPLETE",
        "model_finite": _finite_tree(payload.get("model", {})),
        "ema_finite": _finite_tree(payload.get("ema", {})),
        "optimizer_finite": _finite_tree(payload.get("optimizer", {})),
        "scaler_finite": _finite_tree(payload.get("scaler", {})),
        "test_access": strict_nonnegative_int(metadata.get("test_access_count", 0), "metadata test_access_count") == 0,
    }
    model_count, model_hash = _tensor_state_digest(payload.get("model", {}))
    ema_count, ema_hash = _tensor_state_digest(payload.get("ema", {}))
    payload_checks["state_counts"] = model_count == 270 and ema_count == 270
    if plan.protocol_id != "DMD_9F_3O3P":
        payload_checks["raw_order"] = tuple(metadata.get("raw_frame_order", ())) == plan.raw_order
        payload_checks["validity_mask"] = tuple(metadata.get("validity_mask", ())) == plan.validity_mask
    if not all(payload_checks.values()):
        raise APD369ContractError(f"checkpoint payload conflict for {plan.method}: {payload_checks}")
    metric_name, metric_value = _selection_metric(plan, selected)
    row = {
        "method": plan.method,
        "protocol_id": plan.protocol_id,
        "protocol_hash": plan.protocol_hash,
        "raw_order": list(plan.raw_order),
        "validity_mask": list(plan.validity_mask),
        "config_path": str(plan.config_path.resolve()),
        "config_sha256": sha256_file(plan.config_path),
        "config_payload_hash": config["config_payload_hash"],
        "checkpoint_path": str(plan.checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "receipt_path": str(plan.receipt_path.resolve()),
        "receipt_sha256": sha256_file(plan.receipt_path),
        "history_path": str(plan.history_path.resolve()),
        "history_sha256": sha256_file(plan.history_path),
        "history_rows": history_count,
        "selected_validation_iteration": selected_step,
        "selection_rule": plan.selection_rule,
        "selection_metric": metric_name,
        "selection_metric_value": metric_value,
        "model_state_sha256": model_hash,
        "ema_state_sha256": ema_hash,
        "model_tensor_count": model_count,
        "ema_tensor_count": ema_count,
        "model_finite": True,
        "ema_finite": True,
        "optimizer_finite": True,
        "scaler_finite": True,
        "test_access_count": 0,
        "status": "PASS",
    }
    del payload
    return row


def _audit_dmd9_completion() -> dict[str, Any]:
    root = ROOT / "checkpoints" / "apd_dmd_geometry_r3" / "dmd9_retrain_r1"
    receipt = read_json(root / "formal_receipt.json")
    final_path = root / "final.pt"
    if receipt.get("status") != "FORMAL_TRAINING_COMPLETE" or not final_path.is_file():
        raise APD369ContractError("DMD9 R1 formal completion receipt/final checkpoint absent")
    final_sha = sha256_file(final_path)
    if receipt.get("formal_final_checkpoint_sha256") != final_sha:
        raise APD369ContractError("DMD9 final checkpoint hash conflicts with formal receipt")
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    checks = {
        "generation": metadata.get("run_generation_id") == "DMD9_RETRAIN_R1",
        "protocol": metadata.get("training_protocol_id") == "DMD_9F_3O3P",
        "protocol_hash": metadata.get("training_protocol_hash") == PLANS[2].protocol_hash,
        "final_step": strict_nonnegative_int(metadata.get("global_step"), "DMD9 final global_step") == 100000,
        "complete": metadata.get("completion_status") == "FORMAL_TRAINING_COMPLETE",
        "model_finite": _finite_tree(payload.get("model", {})),
        "ema_finite": _finite_tree(payload.get("ema", {})),
        "optimizer_finite": _finite_tree(payload.get("optimizer", {})),
        "scaler_finite": _finite_tree(payload.get("scaler", {})),
        "test_access": strict_nonnegative_int(metadata.get("test_access_count", 0), "DMD9 final test access") == 0,
    }
    if not all(checks.values()):
        raise APD369ContractError(f"DMD9 final completion payload invalid: {checks}")
    del payload
    return {"status": "PASS", "final_path": str(final_path.resolve()), "final_sha256": final_sha, "checks": checks}


def freeze_checkpoint_contract(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    target = output_root / "00_checkpoint_contract"
    rows = [audit_checkpoint(plan) for plan in PLANS]
    completion = _audit_dmd9_completion()
    forward_hashes = []
    model_hashes = []
    for plan in PLANS:
        config = read_json(plan.config_path)
        forward_hashes.append(hashlib.sha256(canonical_bytes(config["forward"])).hexdigest())
        model_hashes.append(hashlib.sha256(canonical_bytes({
            "model": config["model"],
            "diffusion_steps": config["training"]["diffusion_steps"],
            "beta_schedule": config["training"]["beta_schedule"],
        })).hexdigest())
    if len(set(forward_hashes)) != 1 or len(set(model_hashes)) != 1:
        raise APD369ContractError("cross-protocol non-geometry forward/model configuration drift")
    payload = {
        "schema_version": 1,
        "status": "FINAL_APD369_CHECKPOINTS_FROZEN",
        "inference_weight_branch": "ema",
        "training_executed": False,
        "test_used_for_checkpoint_selection": False,
        "shared_forward_config_sha256": forward_hashes[0],
        "shared_model_diffusion_config_sha256": model_hashes[0],
        "dmd9_training_completion": completion,
        "checkpoints": {row["method"]: row for row in rows},
    }
    atomic_json(target / "FINAL_APD369_CHECKPOINTS.json", payload)
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "method": row["method"], "protocol_id": row["protocol_id"], "protocol_hash": row["protocol_hash"],
            "selected_validation_iteration": row["selected_validation_iteration"], "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"], "model_finite": row["model_finite"],
            "ema_finite": row["ema_finite"], "optimizer_finite": row["optimizer_finite"],
            "scaler_finite": row["scaler_finite"], "test_access_count": row["test_access_count"], "status": row["status"],
        })
    write_tsv(target / "checkpoint_hashes.tsv", flat_rows)
    report = [
        "# Final APD-SIM 3/6/9 checkpoint selection", "",
        "DMD9 R1 completed 100000 scheduled iterations with exit artifacts and zero test access.",
        "The frozen DMD9 rule selects validation iteration 5000, not final iteration 100000.", "",
    ]
    for row in rows:
        report.extend([
            f"## {row['method']}", "",
            f"- Protocol: `{row['protocol_id']}` / `{row['protocol_hash']}`",
            f"- Selected iteration: {row['selected_validation_iteration']}",
            f"- Checkpoint: `{row['checkpoint_path']}`",
            f"- SHA-256: `{row['checkpoint_sha256']}`",
            f"- Selection: `{row['selection_rule']}` on `{row['selection_metric']}` = {row['selection_metric_value']}",
            "- Model, EMA, optimizer, and scaler finite: PASS", "- Test access: 0", "",
        ])
    (target / "checkpoint_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def _manifest_samples(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    samples = payload.get("samples", payload.get("records"))
    if not isinstance(samples, list):
        raise APD369ContractError(f"manifest samples absent: {path}")
    return [dict(row) for row in samples]


def freeze_test_manifest(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    from unisim.revision_r1 import frame_budget_r1c2 as fb

    target = output_root / "01_test_manifest"
    rows = fb.audit_dataset(target)
    if len(rows) != 30:
        raise APD369ContractError("sealed test manifest must contain exactly 30 identities")
    counts = {name: sum(row["class"] == name for row in rows) for name in ("CCP", "ER", "MT")}
    if counts != {"CCP": 10, "ER": 10, "MT": 10}:
        raise APD369ContractError(f"test structure count conflict: {counts}")
    with FORMAL_TEST_LEDGER.open("r", encoding="utf-8-sig", newline="") as handle:
        formal = list(csv.DictReader(handle, delimiter="\t"))
    formal_by_id = {row["sample_id"]: row for row in formal}
    if len(formal_by_id) != 30:
        raise APD369ContractError("formal DMD6 test ledger is not a unique 30-FOV grid")
    frozen = []
    for row in rows:
        source = formal_by_id.get(row["sample_id"])
        if source is None:
            raise APD369ContractError(f"formal identity absent: {row['sample_id']}")
        checks = {
            "file": row["file_sha256"] == source["source_file_sha256"],
            "normalized": row["normalized_array_sha256"] == source["gt_normalized_array_sha256"],
            "class": row["class"] == source["structure_class"],
        }
        if not all(checks.values()):
            raise APD369ContractError(f"formal identity mismatch {row['sample_id']}: {checks}")
        frozen.append({
            **row,
            "measurement_seed": strict_nonnegative_int(source["acquisition_noise_seed"], "measurement_seed"),
            "diffusion_seed": PRINCIPAL_DIFFUSION_SEED_BASE + int(row["order"]),
            "formal_dmd6_raw_sha256": source["raw_stack_sha256"],
            "formal_dmd6_npz_path": str((FORMAL_DMD6_ROOT / "01_shared_contract" / source["npz_path"]).resolve()),
            "formal_dmd6_npz_sha256": source["npz_sha256"],
        })
    train = _manifest_samples(TRAIN_MANIFEST)
    validation = _manifest_samples(VALIDATION_MANIFEST)
    test_sets = {
        "sample_id": {row["sample_id"] for row in frozen},
        "parent_id": {row["parent_id"] for row in frozen},
        "file_sha256": {row["file_sha256"] for row in frozen},
        "normalized_array_sha256": {row["normalized_array_sha256"] for row in frozen},
    }
    overlap: dict[str, dict[str, int]] = {}
    for split_name, split_rows in (("train", train), ("validation", validation)):
        overlap[split_name] = {}
        for key, values in test_sets.items():
            aliases = {
                "normalized_array_sha256": ("normalized_array_sha256", "normalized_pixel_sha256"),
            }.get(key, (key,))
            candidates = {
                str(next((row[name] for name in aliases if row.get(name)), ""))
                for row in split_rows
            }
            candidates.discard("")
            overlap[split_name][key] = len(values & candidates)
    if any(value for split in overlap.values() for value in split.values()):
        raise APD369ContractError(f"train/validation/test identity overlap: {overlap}")
    write_tsv(target / "test30_manifest.tsv", frozen)
    payload = {
        "schema_version": 1,
        "status": "TEST30_MANIFEST_FROZEN",
        "count": 30,
        "class_counts": counts,
        "source_formal_ledger": str(FORMAL_TEST_LEDGER.resolve()),
        "source_formal_ledger_sha256": sha256_file(FORMAL_TEST_LEDGER),
        "train_validation_overlap": overlap,
        "samples": frozen,
    }
    atomic_json(target / "test30_manifest.json", payload)
    digest = sha256_file(target / "test30_manifest.json")
    (target / "test30_manifest_sha256.txt").write_text(digest + "\n", encoding="ascii")
    return payload


def load_frozen_contract(output_root: Path = DEFAULT_OUTPUT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = read_json(output_root / "00_checkpoint_contract" / "FINAL_APD369_CHECKPOINTS.json")
    manifest = read_json(output_root / "01_test_manifest" / "test30_manifest.json")
    if checkpoints.get("status") != "FINAL_APD369_CHECKPOINTS_FROZEN" or manifest.get("status") != "TEST30_MANIFEST_FROZEN":
        raise APD369ContractError("final checkpoint/test manifest contracts are not frozen")
    return checkpoints, manifest


__all__ = [
    "APD369ContractError", "DEFAULT_OUTPUT_ROOT", "FORMAL_TEST_LEDGER", "FORMAL_DMD6_ROOT",
    "PLANS", "PRINCIPAL_DIFFUSION_SEED_BASE", "FinalPlan", "array_sha256", "atomic_json",
    "freeze_checkpoint_contract", "freeze_test_manifest", "load_frozen_contract", "read_json",
    "sha256_file", "strict_nonnegative_int", "write_tsv",
]

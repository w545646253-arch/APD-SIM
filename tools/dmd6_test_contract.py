"""Fail-closed shared contract for matched DMD-6F test entry points.

This module validates and delegates.  It does not reimplement APD-SIM,
ML-SIM, mcSIM, fairSIM, the production forward model, or metric formulas.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = PROJECT_ROOT / "outputs/OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
SHARED_ROOT = FORMAL_ROOT / "01_shared_contract"
FORMAL_MANIFEST = SHARED_ROOT / "test30_dmd6_manifest.tsv"
FORMAL_BASE_RESULTS = FORMAL_ROOT / "17_revision_experiments/01_matched_baselines"
FORMAL_BASE_METRICS = FORMAL_BASE_RESULTS / "per_fov_metrics.csv"
FAIRSIM_POINTER = PROJECT_ROOT / "outputs/FAIRSIM6_VALIDATION_CONFIG_R2_CURRENT.json"
FORMAL_FORWARD_CONTRACT = SHARED_ROOT / "forward_contract.json"
FROZEN_FORWARD_SOURCE = (
    PROJECT_ROOT
    / "audit/dmd3_nonfinite_recovery_20260814_010345/source_backups/unisim/sim_forward_2d.py"
)
FROZEN_FORWARD_SHA256 = "f067a832c2dbac2da32fb4c3a73ac39047a754985f27ecafa31071786321fdd8"

PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"
RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
VALIDITY_MASK = ((1, 1, 1), (1, 1, 1), (0, 0, 0))
SCIENTIFIC_METHODS = (
    "WF-6",
    "ML-SIM-6R",
    "mcSIM-Wiener-6",
    "fairSIM-6-native",
    "APD-SIM-6",
)
BASE_METHODS = tuple(method for method in SCIENTIFIC_METHODS if method != "fairSIM-6-native")
FIGURE_LABELS = {**{method: method for method in SCIENTIFIC_METHODS}, "fairSIM-6-native": "fairSIM-6"}
LEGACY_FAIRSIM_TOKENS = (
    "FAIRSIM6_EXTENSION_R1_",
    "07_fairsim_formal",
    "fairsim6_per_fov.csv",
    "FROZEN_CONFIG",
    "validation_selection_receipt.json",
)


class DMD6ContractError(RuntimeError):
    """Raised when an immutable DMD-6F test contract is violated."""


@dataclass(frozen=True)
class FairSimSelection:
    root: Path
    pointer: Mapping[str, Any]
    selection: Mapping[str, Any]
    formal_receipt: Mapping[str, Any]
    output_hashes: Mapping[str, str]

    @property
    def method(self) -> str:
        return "fairSIM-6-native"

    @property
    def config(self) -> Mapping[str, Any]:
        return self.selection["selected"]["config"]

    @property
    def harmonization(self) -> Mapping[str, Any]:
        return self.selection["selected"]["harmonization"]


@dataclass(frozen=True)
class FormalIndex:
    manifest_rows: tuple[Mapping[str, str], ...]
    manifest_by_id: Mapping[str, Mapping[str, str]]
    output_rows: Mapping[tuple[str, str], Mapping[str, str]]
    fair: FairSimSelection
    bundle_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


def fairsim_array_sha256(value: np.ndarray) -> str:
    """Hash definition frozen by the current fairSIM R2 output ledger."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(map(str, array.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_rows(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise DMD6ContractError(f"Required ledger is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise DMD6ContractError(f"Required ledger is empty: {path}")
    return rows


def write_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str = ",",
    fields: Sequence[str] | None = None,
) -> None:
    if not rows and fields is None:
        raise DMD6ContractError(f"Cannot infer fields for empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assert_protocol_contract() -> dict[str, Any]:
    from unisim.protocols import protocol_registry

    spec = protocol_registry.require(PROTOCOL_ID)
    if spec.protocol_hash != PROTOCOL_HASH:
        raise DMD6ContractError(f"Protocol hash mismatch: {spec.protocol_hash}")
    if tuple(spec.raw_frame_order) != RAW_ORDER:
        raise DMD6ContractError(f"Raw order mismatch: {tuple(spec.raw_frame_order)}")
    if len(spec.raw_frame_bindings) != 6:
        raise DMD6ContractError("DMD-6F must have exactly six raw-frame bindings")
    if tuple(int(x) for x in spec.validity_mask) != (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0):
        raise DMD6ContractError("DMD-6F canonical validity mask is not six leading valid slots")
    orientations = {binding.physical_orientation_id for binding in spec.raw_frame_bindings}
    phases = Counter(binding.physical_orientation_id for binding in spec.raw_frame_bindings)
    if len(orientations) != 2 or set(phases.values()) != {3}:
        raise DMD6ContractError("DMD-6F must be exactly two orientations by three phases")
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "raw_order": list(RAW_ORDER),
        "frame_count": 6,
        "orientation_count": 2,
        "phases_per_orientation": 3,
        "validity_mask_3x3": [list(row) for row in VALIDITY_MASK],
    }


def _require_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DMD6ContractError(f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DMD6ContractError(f"Expected a JSON object: {path}")
    return value


def resolve_current_fairsim() -> FairSimSelection:
    pointer = _require_json(FAIRSIM_POINTER)
    if pointer.get("schema_version") != 1 or pointer.get("status") != "FAIRSIM6_NATIVE_MAIN_TEXT_READY":
        raise DMD6ContractError("Current fairSIM pointer status/schema is not eligible")
    if pointer.get("selected_method") != "fairSIM-6-native":
        raise DMD6ContractError("Current fairSIM pointer does not select fairSIM-6-native")
    root = Path(str(pointer.get("output_root", ""))).expanduser().resolve()
    if not root.is_dir() or "FAIRSIM6_VALIDATION_CONFIG_R2_" not in root.name:
        raise DMD6ContractError(f"Current fairSIM R2 root is invalid: {root}")
    if any(token.lower() in str(root).lower() for token in LEGACY_FAIRSIM_TOKENS):
        raise DMD6ContractError(f"Legacy fairSIM path rejected: {root}")

    final_audit_path = root / "10_final_audit/final_audit.json"
    if sha256_file(final_audit_path) != pointer.get("final_audit_sha256"):
        raise DMD6ContractError("fairSIM current pointer final-audit hash mismatch")
    final_audit = _require_json(final_audit_path)
    if final_audit.get("selected_method") != "fairSIM-6-native" or final_audit.get("P0") != 0:
        raise DMD6ContractError("fairSIM final audit does not authorize native main-text output")

    selection_path = root / "05_selection/frozen_selection_receipt.json"
    selection = _require_json(selection_path)
    if selection.get("main_text_method_label") != "fairSIM-6-native":
        raise DMD6ContractError("Frozen fairSIM scientific method ID mismatch")
    if selection.get("main_text_eligible_before_test") is not True:
        raise DMD6ContractError("fairSIM selection was not eligible before formal test")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("config"), dict):
        raise DMD6ContractError("fairSIM frozen selected config is missing")
    config = selected["config"]
    if config.get("method") != "fairSIM-6C" and selection.get("main_text_method_label") != "fairSIM-6-native":
        raise DMD6ContractError("fairSIM configuration identity is inconsistent")
    if config.get("geometry_mode") != "NATIVE_ESTIMATION":
        raise DMD6ContractError("fairSIM-6C cannot replace selected fairSIM-6-native")
    if config.get("protocol_id") != PROTOCOL_ID or config.get("protocol_hash") != PROTOCOL_HASH:
        raise DMD6ContractError("fairSIM selected config protocol mismatch")
    if tuple(config.get("raw_order", ())) != RAW_ORDER:
        raise DMD6ContractError("fairSIM selected config raw order mismatch")
    if config.get("test_specific_tuning") is not False or config.get("per_fov_tuning") is not False:
        raise DMD6ContractError("fairSIM selected config contains forbidden test/per-FOV tuning")

    formal_path = root / "06_formal_test/formal_test_receipt.json"
    formal = _require_json(formal_path)
    if formal.get("status") != "FAIRSIM6_FORMAL_TEST_COMPLETE" or formal.get("method") != "fairSIM-6-native":
        raise DMD6ContractError("fairSIM formal-test receipt is not complete/native")
    if formal.get("n") != 30 or formal.get("test_run_count") != 1:
        raise DMD6ContractError("fairSIM formal test must contain one run over 30 FOVs")
    if formal.get("test_specific_parameter_changes") is not False:
        raise DMD6ContractError("fairSIM formal receipt reports post-freeze test changes")
    if formal.get("selection_receipt_sha256_at_gate") != sha256_file(selection_path):
        raise DMD6ContractError("fairSIM formal test is not bound to current frozen selection")
    if formal.get("test_bundle_manifest_sha256") != sha256_file(FORMAL_MANIFEST):
        raise DMD6ContractError("fairSIM formal test bundle-manifest hash mismatch")

    ledger_path = root / "06_formal_test/output_hashes.tsv"
    output_hashes = {
        row["relative_path"].replace("\\", "/"): row["sha256"]
        for row in read_rows(ledger_path, delimiter="\t")
    }
    return FairSimSelection(root, pointer, selection, formal, output_hashes)


def _validate_output_path(path: Path, allowed_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError as error:
        raise DMD6ContractError(f"Formal output escapes its frozen root: {resolved}") from error
    if not resolved.is_file():
        raise DMD6ContractError(f"Formal output is missing: {resolved}")
    return resolved


def resolve_formal_index() -> FormalIndex:
    assert_protocol_contract()
    fair = resolve_current_fairsim()
    manifest_rows = read_rows(FORMAL_MANIFEST, delimiter="\t")
    if len(manifest_rows) != 30:
        raise DMD6ContractError(f"Formal manifest must contain 30 rows, found {len(manifest_rows)}")
    if Counter(row["structure_class"] for row in manifest_rows) != Counter({"CCP": 10, "ER": 10, "MT": 10}):
        raise DMD6ContractError("Formal manifest is not exactly 10 CCP / 10 ER / 10 MT")
    if len({row["sample_id"] for row in manifest_rows}) != 30:
        raise DMD6ContractError("Formal manifest sample IDs are not unique")
    for expected_order, row in enumerate(manifest_rows):
        if int(row["order"]) != expected_order:
            raise DMD6ContractError("Formal manifest order is not contiguous")
        if row["protocol_id"] != PROTOCOL_ID or row["protocol_hash"] != PROTOCOL_HASH:
            raise DMD6ContractError(f"Formal manifest protocol mismatch: {row['sample_id']}")
        if row["frame_count"] != "6" or tuple(row["frame_order"].split("/")) != RAW_ORDER:
            raise DMD6ContractError(f"Formal manifest raw contract mismatch: {row['sample_id']}")
        bundle = _validate_output_path(SHARED_ROOT / row["npz_path"], SHARED_ROOT)
        if sha256_file(bundle) != row["npz_sha256"]:
            raise DMD6ContractError(f"Formal NPZ file hash mismatch: {row['sample_id']}")

    output_rows: dict[tuple[str, str], Mapping[str, str]] = {}
    base_rows = read_rows(FORMAL_BASE_METRICS, delimiter=",")
    if len(base_rows) != 30 * len(BASE_METHODS):
        raise DMD6ContractError("Formal base-method ledger does not contain 4 x 30 rows")
    for row in base_rows:
        method = row["method"]
        if method not in BASE_METHODS:
            raise DMD6ContractError(f"Unexpected base method in formal ledger: {method}")
        if row["protocol_id"] != PROTOCOL_ID or row["protocol_hash"] != PROTOCOL_HASH:
            raise DMD6ContractError("Formal base output protocol mismatch")
        path = _validate_output_path(Path(row["harmonized_path"]), FORMAL_BASE_RESULTS)
        native = _validate_output_path(Path(row["native_path"]), FORMAL_BASE_RESULTS)
        output_rows[(row["sample_id"], method)] = {**row, "harmonized_path": str(path), "native_path": str(native)}

    fair_metrics_path = fair.root / "06_formal_test/per_fov_metrics.csv"
    fair_rows = read_rows(fair_metrics_path, delimiter=",")
    if len(fair_rows) != 30 or {row["method"] for row in fair_rows} != {"fairSIM-6-native"}:
        raise DMD6ContractError("Current fairSIM formal ledger must contain native 30-FOV output only")
    fair_test_root = fair.root / "06_formal_test"
    for row in fair_rows:
        if row["protocol_id"] != PROTOCOL_ID or row["protocol_hash"] != PROTOCOL_HASH:
            raise DMD6ContractError("Current fairSIM formal output protocol mismatch")
        if tuple(row["raw_order"].split("/")) != RAW_ORDER:
            raise DMD6ContractError("Current fairSIM formal output raw order mismatch")
        path = _validate_output_path(Path(row["harmonized_path"]), fair_test_root)
        native = _validate_output_path(Path(row["native_path"]), fair_test_root)
        for item in (path, native):
            relative = item.relative_to(fair_test_root).as_posix()
            expected = fair.output_hashes.get(relative)
            if expected is None or sha256_file(item) != expected:
                raise DMD6ContractError(f"Current fairSIM file ledger mismatch: {item}")
        output_rows[(row["sample_id"], "fairSIM-6-native")] = {
            **row,
            "harmonized_path": str(path),
            "native_path": str(native),
        }

    expected_keys = {
        (row["sample_id"], method)
        for row in manifest_rows
        for method in SCIENTIFIC_METHODS
    }
    if set(output_rows) != expected_keys:
        raise DMD6ContractError("Formal output ledger is not a complete 30 x 5 matrix")
    identity = {
        "formal_manifest_sha256": sha256_file(FORMAL_MANIFEST),
        "formal_base_metrics_sha256": sha256_file(FORMAL_BASE_METRICS),
        "fair_pointer_sha256": sha256_file(FAIRSIM_POINTER),
        "fair_selection_sha256": sha256_file(fair.root / "05_selection/frozen_selection_receipt.json"),
        "fair_formal_receipt_sha256": sha256_file(fair.root / "06_formal_test/formal_test_receipt.json"),
        "fair_metrics_sha256": sha256_file(fair_metrics_path),
    }
    return FormalIndex(
        tuple(manifest_rows),
        {row["sample_id"]: row for row in manifest_rows},
        output_rows,
        fair,
        canonical_json_sha256(identity),
    )


def normalize_gt(native: np.ndarray) -> np.ndarray:
    from revision_dmd6_common import normalize_gt as production_normalize_gt

    return production_normalize_gt(native)


def read_gt(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DMD6ContractError(f"GT TIFF not found: {resolved}")
    native = np.asarray(tifffile.imread(resolved))
    if native.ndim != 2 or native.shape != (1004, 1004):
        raise DMD6ContractError(f"GT must be one 1004 x 1004 channel: {resolved} -> {native.shape}")
    if not np.isfinite(native).all() or float(np.max(native)) <= float(np.min(native)):
        raise DMD6ContractError(f"GT must be finite and nonconstant: {resolved}")
    normalized = validate_harmonized(normalize_gt(native), name="normalized GT")
    identity = {
        "sample_id": resolved.stem,
        "absolute_path": str(resolved),
        "file_sha256": sha256_file(resolved),
        "pixel_sha256": array_sha256(native),
        "normalized_array_sha256": array_sha256(normalized),
    }
    return native, normalized, identity


def match_formal_gt(identity: Mapping[str, str], index: FormalIndex) -> Mapping[str, str] | None:
    row = index.manifest_by_id.get(identity["sample_id"])
    if row is None:
        return None
    required = (
        ("file_sha256", "source_file_sha256"),
        ("pixel_sha256", "source_pixel_sha256"),
        ("normalized_array_sha256", "gt_normalized_array_sha256"),
    )
    if all(identity[left] == row[right] for left, right in required):
        return row
    return None


def _load_npy_verified(
    path: Path,
    expected_array_hash: str,
    *,
    name: str,
    expected_shape: tuple[int, int] | None = (1004, 1004),
    hash_kind: str = "official_r2",
) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    array = np.ascontiguousarray(value)
    actual_hash = fairsim_array_sha256(array) if hash_kind == "fairsim_r2" else array_sha256(array)
    if actual_hash != expected_array_hash:
        raise DMD6ContractError(f"Formal output array hash mismatch: {name}: {path}")
    if array.ndim != 2 or not np.isfinite(array).all():
        raise DMD6ContractError(f"Formal output is not a finite 2-D array: {name}")
    if expected_shape is not None and array.shape != expected_shape:
        raise DMD6ContractError(f"Formal output shape mismatch for {name}: {array.shape}")
    return array


def load_formal_sample(
    row: Mapping[str, str],
    index: FormalIndex,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    sample_id = row["sample_id"]
    bundle_path = _validate_output_path(SHARED_ROOT / row["npz_path"], SHARED_ROOT)
    if sha256_file(bundle_path) != row["npz_sha256"]:
        raise DMD6ContractError(f"Formal bundle file hash mismatch: {sample_id}")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        if str(bundle["protocol_id"].reshape(-1)[0]) != PROTOCOL_ID:
            raise DMD6ContractError(f"Formal raw protocol ID mismatch: {sample_id}")
        if str(bundle["protocol_hash"].reshape(-1)[0]) != PROTOCOL_HASH:
            raise DMD6ContractError(f"Formal raw protocol hash mismatch: {sample_id}")
        if tuple(str(x) for x in bundle["frame_order"].tolist()) != RAW_ORDER:
            raise DMD6ContractError(f"Formal raw frame order mismatch: {sample_id}")
        if str(bundle["source_file_sha256"].reshape(-1)[0]) != row["source_file_sha256"]:
            raise DMD6ContractError(f"Formal raw source-file identity mismatch: {sample_id}")
        raw = np.ascontiguousarray(bundle["raw_stack"], dtype=np.float32)
    validate_raw_stack(raw, expected_hash=row["raw_stack_sha256"])

    harmonized: dict[str, np.ndarray] = {}
    native: dict[str, np.ndarray] = {}
    output_paths: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for method in SCIENTIFIC_METHODS:
        ledger = index.output_rows[(sample_id, method)]
        if ledger["raw_stack_sha256"] != row["raw_stack_sha256"]:
            raise DMD6ContractError(f"Method/raw hash mismatch: {sample_id}/{method}")
        harm_path = Path(ledger["harmonized_path"])
        native_path = Path(ledger["native_path"])
        hash_kind = "fairsim_r2" if method == "fairSIM-6-native" else "official_r2"
        harmonized[method] = validate_harmonized(
            _load_npy_verified(
                harm_path,
                ledger["harmonized_array_sha256"],
                name=f"{sample_id}/{method}",
                hash_kind=hash_kind,
            ),
            name=f"{sample_id}/{method}",
        )
        native[method] = _load_npy_verified(
            native_path,
            ledger["native_array_sha256"],
            name=f"{sample_id}/{method} native",
            expected_shape=None,
            hash_kind=hash_kind,
        )
        output_paths[method] = str(harm_path)
        output_hashes[method] = ledger["harmonized_array_sha256"]

    expected_wf = wf6_arithmetic_mean(raw)
    if not np.array_equal(native["WF-6"], expected_wf):
        raise DMD6ContractError(f"Formal WF-6 is not bitwise raw.mean(axis=0): {sample_id}")
    return raw, native, harmonized, {
        "sample_id": sample_id,
        "raw_stack_sha256": row["raw_stack_sha256"],
        "raw_bundle_path": str(bundle_path),
        "raw_bundle_file_sha256": row["npz_sha256"],
        "method_output_paths": output_paths,
        "method_output_array_sha256": output_hashes,
        "formal_bundle_hash": index.bundle_hash,
        "model_execution_count": 0,
    }


def validate_raw_stack(raw: np.ndarray, *, expected_hash: str | None = None) -> str:
    value = np.asarray(raw)
    if value.shape != (6, 1004, 1004) and not (value.ndim == 3 and value.shape[0] == 6):
        raise DMD6ContractError(f"DMD-6F raw stack must have six 2-D frames: {value.shape}")
    if value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all():
        raise DMD6ContractError("DMD-6F raw stack must be contiguous finite float32")
    digest = array_sha256(value)
    if expected_hash is not None and digest != expected_hash:
        raise DMD6ContractError(f"DMD-6F raw-stack hash mismatch: {digest} != {expected_hash}")
    return digest


def wf6_arithmetic_mean(raw: np.ndarray) -> np.ndarray:
    validate_raw_stack(np.ascontiguousarray(raw, dtype=np.float32))
    result = np.ascontiguousarray(raw.mean(axis=0), dtype=np.float32)
    reference = np.ascontiguousarray(raw.mean(axis=0), dtype=np.float32)
    if not np.array_equal(result, reference):
        raise DMD6ContractError("WF-6 arithmetic-mean bitwise identity failed")
    return result


def validate_harmonized(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise DMD6ContractError(f"{name} must be a finite 2-D harmonized float array")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1e-7 or maximum > 1.0 + 1e-7:
        raise DMD6ContractError(f"{name} violates frozen [0,1] harmonization: {minimum}, {maximum}")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _load_frozen_forward_module() -> Any:
    if not FROZEN_FORWARD_SOURCE.is_file() or sha256_file(FROZEN_FORWARD_SOURCE) != FROZEN_FORWARD_SHA256:
        raise DMD6ContractError("Frozen production forward source identity is unavailable")
    module_name = "unisim._dmd6_formal_forward_f067a832"
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_FORWARD_SOURCE)
    if spec is None or spec.loader is None:
        raise DMD6ContractError("Cannot load frozen production forward source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def generator_contract() -> dict[str, Any]:
    forward = _require_json(FORMAL_FORWARD_CONTRACT)
    expected = forward.get("implementation_files", {}).get("forward_2d", {}).get("sha256")
    if expected != FROZEN_FORWARD_SHA256:
        raise DMD6ContractError("Formal forward contract does not bind the frozen source")
    source = FROZEN_FORWARD_SOURCE.read_text(encoding="utf-8")
    required_tokens = (
        "object_up[:, 0].unsqueeze(1) * patterns[:, 0].unsqueeze(0)",
        "gaussian_psf_2d",
        "F.conv2d",
        'F.interpolate(raw, size=x0.shape[-2:], mode="area")',
        "background",
        "torch.poisson",
        "torch.randn",
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise DMD6ContractError(f"Frozen production forward source-order evidence missing: {missing}")
    return {
        "status": "PASS",
        "source_path": str(FROZEN_FORWARD_SOURCE.resolve()),
        "source_sha256": FROZEN_FORWARD_SHA256,
        "formal_contract_path": str(FORMAL_FORWARD_CONTRACT.resolve()),
        "formal_contract_sha256": sha256_file(FORMAL_FORWARD_CONTRACT),
        "operation_order": [
            "object_x_illumination",
            "psf_convolution",
            "camera_grid_sampling",
            "background",
            "poisson_gaussian_camera_noise",
        ],
        "implementation_note": (
            "The frozen source adds a spatially constant background immediately before "
            "linear area resampling; this is mathematically identical to adding that "
            "same constant after camera-grid sampling."
        ),
        "implementation_role": "frozen production forward used for official R2 bundle",
    }


def generate_dmd6_raw(gt: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate custom raw data with the exact frozen official-R2 forward source."""
    import torch
    from unisim.protocols import protocol_registry

    module = _load_frozen_forward_module()
    config_path = PROJECT_ROOT / "configs/apd_dmd_r2/train6_formal.json"
    config = _require_json(config_path)
    values = dict(config["forward"])
    allowed = set(module.SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in values.items() if key in allowed}
    for key in tuple(values):
        if key.startswith("rand_") and isinstance(values[key], list):
            values[key] = tuple(float(item) for item in values[key])
    sim_config = module.SIM2DConfig(**values)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))[None, None].to(device)
    theta = module.nominal_theta_2d(sim_config, device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        raw, _ = module.forward_protocol_sim_2d(
            tensor,
            sim_config,
            PROTOCOL_ID,
            theta=dict(theta),
            randomize=False,
            noise_generator=generator,
        )
    result = np.ascontiguousarray(raw[0].cpu().numpy(), dtype=np.float32)
    spec = protocol_registry.require(PROTOCOL_ID)
    if tuple(spec.raw_frame_order) != RAW_ORDER:
        raise DMD6ContractError("Frozen generator registry raw order mismatch")
    raw_hash = validate_raw_stack(result)
    return result, {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "raw_order": list(RAW_ORDER),
        "raw_frame_order": list(RAW_ORDER),
        "seed": int(seed),
        "raw_stack_sha256": raw_hash,
        "generator_source_path": str(FROZEN_FORWARD_SOURCE.resolve()),
        "generator_source_sha256": FROZEN_FORWARD_SHA256,
        "forward_config_path": str(config_path.resolve()),
        "forward_config_sha256": sha256_file(config_path),
        "production_forward": True,
    }


def run_serial_methods(
    raw: np.ndarray,
    *,
    seed: int,
    work_dir: Path,
    fair: FairSimSelection,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Delegate all five reconstructions, once each, against one in-memory raw stack."""
    from compare_single_dmd6 import _mlsim_checkpoint, _run_mcsim
    from fairsim6_extension_common import harmonize_fairsim, reconstruct_fairsim6
    from revision_dmd6_common import (
        MCSIM_CALIBRATION,
        apd6_reconstruct,
        harmonize,
        mlsim_reconstruct,
    )

    raw = np.ascontiguousarray(raw, dtype=np.float32)
    raw_hash = validate_raw_stack(raw)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "raw_dmd6_stack.npy"
    np.save(raw_path, raw, allow_pickle=False)
    native: dict[str, np.ndarray] = {"WF-6": wf6_arithmetic_mean(raw)}
    checkpoint, _ = _mlsim_checkpoint()
    native["ML-SIM-6R"] = np.ascontiguousarray(mlsim_reconstruct(raw, checkpoint), dtype=np.float32)
    native["mcSIM-Wiener-6"] = np.ascontiguousarray(_run_mcsim(raw_path, work_dir / "mcsim_native.npy"), dtype=np.float32)
    fair_native, fair_receipt = reconstruct_fairsim6(raw, work_dir / "fairsim_native", fair.config)
    native["fairSIM-6-native"] = np.ascontiguousarray(fair_native, dtype=np.float32)
    native["APD-SIM-6"] = np.ascontiguousarray(apd6_reconstruct(raw, int(seed), stage2=True), dtype=np.float32)

    calibration = _require_json(MCSIM_CALIBRATION)
    harmonized: dict[str, np.ndarray] = {}
    for method in SCIENTIFIC_METHODS:
        value = native[method]
        if value.ndim != 2 or not np.isfinite(value).all():
            raise DMD6ContractError(f"Serial method returned invalid output: {method}")
        if method == "fairSIM-6-native":
            mapped = harmonize_fairsim(value, fair.harmonization)
        else:
            mapped = harmonize(method, value, calibration.get("methods", {}).get(method))
        harmonized[method] = validate_harmonized(mapped, name=method)
    return native, harmonized, {
        "raw_stack_sha256": raw_hash,
        "method_execution_count": 5,
        "fairSIM_run_receipt": fair_receipt,
    }


def compute_metrics(gt: np.ndarray, method: str, harmonized: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from revision_dmd6_common import gt_frc, metrics_module

    reference = validate_harmonized(gt, name="metric GT")
    reconstruction = validate_harmonized(harmonized, name=f"metric {method}")
    if reference.shape != reconstruction.shape:
        raise DMD6ContractError(f"Metric shape mismatch for {method}")
    metric_api = metrics_module()
    frc, curve = gt_frc(reference, reconstruction)
    period_px = frc["cutoff_derived_spatial_period_px"]
    period_um = frc["cutoff_derived_spatial_period_um"]
    if period_um is None and bool(frc["right_censored_at_nyquist"]):
        # Frozen R2 table convention: a right-censored curve is reported at the
        # Nyquist upper bound, without changing the curve or crossing formula.
        period_px = 2.0
        period_um = 2.0 * (6.5 / 60.0)
    row = {
        "method": method,
        "psnr": float(metric_api.psnr_native(reference, reconstruction)),
        "ssim": float(metric_api.ssim_native(reference, reconstruction)),
        "gt_frc_cutoff_cycles_per_pixel": frc["cutoff_cycles_per_pixel"],
        "gt_frc_period_px": period_px,
        "gt_frc_period_um": period_um,
        "frc_auc": float(frc["frc_auc_to_cutoff_or_nyquist"]),
        "frc_right_censored": bool(frc["right_censored_at_nyquist"]),
        "metric_input_dtype": str(reconstruction.dtype),
        "metric_input_role": "harmonized_float32",
        "frc_reporting_convention": "right-censored_at_nyquist_reports_2px_upper_bound",
    }
    for key in ("psnr", "ssim", "frc_auc"):
        if not np.isfinite(float(row[key])):
            raise DMD6ContractError(f"Non-finite {key} for {method}")
    return row, curve


def classify_structure(sample_id: str) -> str:
    if sample_id.startswith("CCP"):
        return "CCP"
    if sample_id.startswith("ER_"):
        return "ER"
    if sample_id.startswith("microtubules"):
        return "Microtubule"
    raise DMD6ContractError(f"Cannot classify formal structure: {sample_id}")


def scan_gt_folder(root: Path) -> list[dict[str, str]]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise DMD6ContractError(f"GT folder not found: {resolved}")
    paths = sorted(
        (path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}),
        key=lambda path: str(path).lower(),
    )
    rows: list[dict[str, str]] = []
    seen_pixels: set[str] = set()
    for path in paths:
        try:
            native, _, identity = read_gt(path)
        except DMD6ContractError:
            continue
        pixel_hash = array_sha256(native)
        if pixel_hash in seen_pixels:
            continue
        seen_pixels.add(pixel_hash)
        rows.append(
            {
                "order": str(len(rows)),
                "sample_id": identity["sample_id"],
                "structure_class": classify_structure(identity["sample_id"]),
                "folder": path.parent.relative_to(resolved).as_posix() or ".",
                "absolute_path": identity["absolute_path"],
                "file_sha256": identity["file_sha256"],
                "pixel_sha256": identity["pixel_sha256"],
                "normalized_array_sha256": identity["normalized_array_sha256"],
                "shape": "1004x1004",
                "dtype": str(native.dtype),
            }
        )
    if len(rows) != 30:
        raise DMD6ContractError(f"Expected exactly 30 unique eligible GT TIFFs, found {len(rows)}")
    counts = Counter(row["structure_class"] for row in rows)
    if counts != Counter({"CCP": 10, "ER": 10, "Microtubule": 10}):
        raise DMD6ContractError(f"GT composition must be 10/10/10, found {dict(counts)}")
    return rows


def formal_manifest_matches(scanned: Sequence[Mapping[str, str]], index: FormalIndex) -> bool:
    if {row["sample_id"] for row in scanned} != set(index.manifest_by_id):
        return False
    for item in scanned:
        formal = index.manifest_by_id[item["sample_id"]]
        if item["file_sha256"] != formal["source_file_sha256"]:
            return False
        if item["pixel_sha256"] != formal["source_pixel_sha256"]:
            return False
        if item["normalized_array_sha256"] != formal["gt_normalized_array_sha256"]:
            return False
    return True


def write_output_hashes(root: Path, destination: Path) -> None:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item != destination), key=lambda item: item.as_posix()):
        rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(destination, rows, delimiter="\t")


def formal_evidence_rows(index: FormalIndex) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest = [dict(row) for row in index.manifest_rows]
    paths: list[dict[str, str]] = []
    hashes: list[dict[str, str]] = []
    for row in index.manifest_rows:
        sample = row["sample_id"]
        bundle = (SHARED_ROOT / row["npz_path"]).resolve()
        paths.append({"sample_id": sample, "method": "RAW_DMD6", "path": str(bundle)})
        hashes.append({"sample_id": sample, "method": "RAW_DMD6", "file_sha256": row["npz_sha256"], "array_sha256": row["raw_stack_sha256"]})
        for method in SCIENTIFIC_METHODS:
            output = index.output_rows[(sample, method)]
            path = Path(output["harmonized_path"])
            paths.append({"sample_id": sample, "method": method, "path": str(path)})
            hashes.append({"sample_id": sample, "method": method, "file_sha256": sha256_file(path), "array_sha256": output["harmonized_array_sha256"]})
    return manifest, paths, hashes


__all__ = [
    "DMD6ContractError",
    "FAIRSIM_POINTER",
    "FIGURE_LABELS",
    "FORMAL_ROOT",
    "FROZEN_FORWARD_SHA256",
    "FROZEN_FORWARD_SOURCE",
    "PROTOCOL_HASH",
    "PROTOCOL_ID",
    "RAW_ORDER",
    "SCIENTIFIC_METHODS",
    "array_sha256",
    "assert_protocol_contract",
    "atomic_json",
    "classify_structure",
    "compute_metrics",
    "formal_evidence_rows",
    "fairsim_array_sha256",
    "formal_manifest_matches",
    "generate_dmd6_raw",
    "generator_contract",
    "load_formal_sample",
    "match_formal_gt",
    "read_gt",
    "resolve_current_fairsim",
    "resolve_formal_index",
    "run_serial_methods",
    "scan_gt_folder",
    "sha256_file",
    "validate_harmonized",
    "validate_raw_stack",
    "wf6_arithmetic_mean",
    "write_output_hashes",
    "write_rows",
]

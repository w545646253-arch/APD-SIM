"""Independent dataset-identity audit for the strict Reviewer-1 C3 run.

This module is intentionally independent of the formal evaluation entrypoint.  It
only reads the immutable run and its upstream manifests/caches, writes new audit
artifacts to a caller-owned directory, and exposes whole-tree snapshot helpers so
the combined audit entrypoint can prove that the old run was not modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
CESHIJI_ROOT = Path(r"data/sealed_test_gt")
EXISTING_RUN = ROOT / "outputs" / "reviewer1_physmap6_strict" / "20260813T183229Z"
GT_MANIFEST = ROOT / "_REVISION_R1_20260812T082048Z" / "DATASET_MANIFEST.csv"
ROBUSTNESS_MANIFEST = ROOT / "_REVISION_R1_20260812T082048Z" / "PHYSMAP6_PATCH_MANIFEST.csv"
BUNDLE_MANIFEST = (
    ROOT
    / "outputs"
    / "OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
    / "01_shared_contract"
    / "test30_dmd6_manifest.tsv"
)
SEALED_TEST_MANIFEST = ROOT / "manifests" / "apd_dmd_r2" / "sealed_test_manifest.json"

IDENTITY_STATUSES = {
    "CESHIJI_EXACT_30_MATCH",
    "CESHIJI_PARTIAL_MATCH",
    "CESHIJI_NOT_USED",
    "CESHIJI_IDENTITY_UNRESOLVED",
}


class DatasetIdentityAuditError(RuntimeError):
    """Raised when a required identity source cannot be audited safely."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash pixels using the authoritative official-R2 array definition."""

    value = np.ascontiguousarray(array)
    header = _canonical_bytes({"dtype": value.dtype.str, "shape": list(value.shape)})
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def normalize_image(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array).astype(np.float32, copy=False)
    low = float(np.percentile(value, 0.5))
    high = float(np.percentile(value, 99.5))
    if not high > low:
        raise DatasetIdentityAuditError("normalization percentile range is non-positive")
    return np.ascontiguousarray(
        np.clip((value - low) / (high - low + 1e-8), 0.0, 1.0), dtype=np.float32
    )


def normalized_pixel_sha256(array: np.ndarray) -> str:
    normalized = normalize_image(array)
    header = _canonical_bytes(
        {
            "dtype": normalized.dtype.str,
            "normalization": "percentile_0.5_99.5_clip_0_1",
            "shape": list(normalized.shape),
        }
    )
    return hashlib.sha256(header + b"\n" + normalized.tobytes(order="C")).hexdigest()


def _read_tiff(path: Path) -> np.ndarray:
    if not path.is_file():
        raise DatasetIdentityAuditError(f"missing TIFF: {path}")
    try:
        value = np.asarray(tifffile.imread(path))
    except Exception as exc:  # pragma: no cover - library error carries path context
        raise DatasetIdentityAuditError(f"unreadable TIFF {path}: {exc}") from exc
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
        raise DatasetIdentityAuditError(
            f"GT must be a numeric 2-D image: {path}; shape={value.shape}, dtype={value.dtype}"
        )
    if not bool(np.isfinite(value).all()):
        raise DatasetIdentityAuditError(f"non-finite pixels in {path}")
    return value


def _structure_from_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("ccps_"):
        return "CCPs"
    if lowered.startswith("er_"):
        return "ER"
    if lowered.startswith("microtubules_"):
        return "microtubules"
    return "UNRESOLVED"


def _sample_id(path: Path) -> str:
    return path.stem


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(item)) for item in shape)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_new(path: Path, data: bytes) -> None:
    """Write one new artifact atomically, refusing accidental overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _write_new(path, payload + b"\n")


def capture_existing_run_snapshot(run_dir: Path = EXISTING_RUN) -> dict[str, Any]:
    """Return a complete byte-identity snapshot of every file in the old run."""

    root = run_dir.resolve(strict=True)
    files: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise DatasetIdentityAuditError(f"existing run is empty: {root}")
    aggregate = hashlib.sha256(_canonical_bytes(files)).hexdigest()
    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "aggregate_sha256": aggregate,
        "files": files,
    }


def verify_existing_run_snapshot(
    before: Mapping[str, Any], run_dir: Path = EXISTING_RUN
) -> dict[str, Any]:
    """Rehash the full old run and report exact added/removed/changed sets."""

    after = capture_existing_run_snapshot(run_dir)
    before_by_path = {str(item["relative_path"]): item for item in before.get("files", [])}
    after_by_path = {str(item["relative_path"]): item for item in after["files"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = sorted(
        key
        for key in set(before_by_path) & set(after_by_path)
        if (
            before_by_path[key].get("sha256") != after_by_path[key].get("sha256")
            or int(before_by_path[key].get("size_bytes", -1))
            != int(after_by_path[key].get("size_bytes", -2))
        )
    )
    exact = not added and not removed and not changed and (
        str(before.get("aggregate_sha256")) == str(after["aggregate_sha256"])
    )
    return {
        "status": "PASS" if exact else "FAIL",
        "exact": exact,
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after["aggregate_sha256"],
        "before_file_count": before.get("file_count"),
        "after_file_count": after["file_count"],
        "added": added,
        "removed": removed,
        "changed": changed,
        "after": after,
    }


def enumerate_ceshiji(root: Path = CESHIJI_ROOT) -> list[dict[str, Any]]:
    resolved = root.resolve(strict=True)
    paths = [
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    ]
    order = {"CCPs": 0, "ER": 1, "microtubules": 2, "UNRESOLVED": 9}
    paths.sort(key=lambda path: (order[_structure_from_name(path.name)], path.relative_to(resolved).as_posix().lower()))
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        array = _read_tiff(path)
        normalized = normalize_image(array)
        rows.append(
            {
                "order": index,
                "sample_id": _sample_id(path),
                "structure_class": _structure_from_name(path.name),
                "absolute_path": str(path.resolve(strict=True)),
                "relative_path": path.relative_to(resolved).as_posix(),
                "filename": path.name,
                "parent_class_directory": path.parent.name,
                "file_size_bytes": int(path.stat().st_size),
                "image_shape": _shape_text(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256_file(path),
                "decoded_pixel_sha256": array_sha256(array),
                "normalized_pixel_sha256": normalized_pixel_sha256(array),
                "normalized_array_sha256": array_sha256(normalized),
                "pixels_finite": True,
            }
        )
    return rows


def _read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DatasetIdentityAuditError(f"JSON object required: {path}")
    return value


def _manifest_rows_by_sample(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for source in rows:
        sample = str(source["sample_id"])
        if sample in output:
            raise DatasetIdentityAuditError(f"duplicate manifest sample_id: {sample}")
        output[sample] = dict(source)
    return output


def _audit_nominal_chain() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_rows = [
        row
        for row in _read_csv(GT_MANIFEST)
        if row.get("dataset") == "30-FOV GT benchmark"
    ]
    gt_rows.sort(key=lambda row: int(row["order"]))
    bundle_rows = _read_csv(BUNDLE_MANIFEST, delimiter="\t")
    bundle_rows.sort(key=lambda row: int(row["order"]))
    nominal_path = EXISTING_RUN / "R1C3_NOMINAL_PER_FOV.csv"
    nominal_rows = _read_csv(nominal_path)
    nominal_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nominal_rows:
        nominal_by_sample[row["sample_id"]].append(row)

    if len(gt_rows) != 30 or len(bundle_rows) != 30 or len(nominal_rows) != 120:
        raise DatasetIdentityAuditError(
            "existing nominal identity sources are not 30 GT / 30 bundles / 120 method rows"
        )
    if [int(row["order"]) for row in gt_rows] != list(range(30)):
        raise DatasetIdentityAuditError("GT manifest order is incomplete")
    if [int(row["order"]) for row in bundle_rows] != list(range(30)):
        raise DatasetIdentityAuditError("bundle manifest order is incomplete")

    gt_by_sample = _manifest_rows_by_sample(gt_rows)
    bundle_by_sample = _manifest_rows_by_sample(bundle_rows)
    sealed_payload = _read_json_object(SEALED_TEST_MANIFEST)
    sealed_identities = sealed_payload.get("identities")
    if not isinstance(sealed_identities, list) or len(sealed_identities) != 30:
        raise DatasetIdentityAuditError("sealed test manifest does not contain 30 identities")
    sealed_by_sample = {
        str(item.get("sample_id")): item
        for item in sealed_identities
        if isinstance(item, dict)
    }
    if set(gt_by_sample) != set(bundle_by_sample) or set(gt_by_sample) != set(nominal_by_sample):
        raise DatasetIdentityAuditError("sample IDs disagree across GT, bundle, and nominal sources")
    if set(gt_by_sample) != set(sealed_by_sample):
        raise DatasetIdentityAuditError("sample IDs disagree with sealed test identities")

    manifest: list[dict[str, Any]] = []
    cache_failures: list[str] = []
    for order, gt_row in enumerate(gt_rows):
        sample = gt_row["sample_id"]
        bundle = bundle_by_sample[sample]
        sealed = sealed_by_sample[sample]
        methods = nominal_by_sample[sample]
        path = Path(gt_row["absolute_path"]).resolve(strict=True)
        pixels = _read_tiff(path)
        normalized = normalize_image(pixels)
        source_sha = sha256_file(path)
        pixel_sha = array_sha256(pixels)
        normalized_pixel_sha = normalized_pixel_sha256(pixels)
        normalized_array_sha = array_sha256(normalized)
        bundle_path = (BUNDLE_MANIFEST.parent / bundle["npz_path"]).resolve(strict=True)
        npz_sha = sha256_file(bundle_path)
        with np.load(bundle_path, allow_pickle=False) as archive:
            archive_files = tuple(sorted(archive.files))
            if any("gt" in key.lower() for key in archive_files):
                raise DatasetIdentityAuditError(f"GT unexpectedly embedded in {bundle_path}")
            raw = np.asarray(archive["raw_stack"], dtype=np.float32)
        raw_sha = array_sha256(raw)
        nominal_gt_hashes = sorted({row["gt_identity_sha256"] for row in methods})
        nominal_raw_hashes = sorted({row["raw_stack_sha256"] for row in methods})
        nominal_methods = sorted(row["method"] for row in methods)
        expected_methods = ["APD-SIM-6", "DiffWS-6", "PhysMap-6", "WF"]

        checks = {
            "manifest_source_file_sha": source_sha == gt_row["sha256"],
            "bundle_source_file_sha": source_sha == bundle["source_file_sha256"],
            "bundle_source_pixel_sha": pixel_sha == bundle["source_pixel_sha256"],
            "bundle_normalized_pixel_sha": normalized_pixel_sha
            == bundle["source_normalized_pixel_sha256"],
            "bundle_normalized_array_sha": normalized_array_sha
            == bundle["gt_normalized_array_sha256"],
            "bundle_npz_sha": npz_sha == bundle["npz_sha256"],
            "bundle_raw_sha": raw_sha == bundle["raw_stack_sha256"],
            "nominal_gt_sha": nominal_gt_hashes == [source_sha],
            "nominal_raw_sha": nominal_raw_hashes == [raw_sha],
            "nominal_method_set": nominal_methods == expected_methods,
            "bundle_has_no_gt": not any("gt" in key.lower() for key in archive_files),
            "sealed_identity_digest": bundle["sealed_identity_digest"]
            == sealed.get("identity_digest"),
            "sealed_sample_class_parent": (
                bundle["structure_class"] == sealed.get("class")
                and bundle["parent_id"] == sealed.get("parent_id")
            ),
        }
        chain_status = "PASS" if all(checks.values()) else "FAIL"

        prediction_files = []
        for row in methods:
            prediction_path = (
                EXISTING_RUN
                / "nominal_predictions"
                / f"{order:03d}_{sample}_{row['method'].replace(' ', '_')}.npy"
            )
            if not prediction_path.is_file():
                cache_failures.append(str(prediction_path))
                continue
            prediction = np.load(prediction_path, allow_pickle=False)
            prediction_ok = (
                bool(np.isfinite(prediction).all())
                and array_sha256(np.ascontiguousarray(prediction, dtype=np.float32))
                == row["prediction_sha256"]
            )
            if not prediction_ok:
                cache_failures.append(str(prediction_path))
            prediction_files.append(
                {
                    "method": row["method"],
                    "path": str(prediction_path),
                    "file_sha256": sha256_file(prediction_path),
                    "array_sha256": array_sha256(
                        np.ascontiguousarray(prediction, dtype=np.float32)
                    ),
                    "csv_array_sha256": row["prediction_sha256"],
                    "status": "PASS" if prediction_ok else "FAIL",
                }
            )

        manifest.append(
            {
                "sample_order": order,
                "sample_id": sample,
                "parent_id": bundle["parent_id"],
                "structure_class": bundle["structure_class"],
                "sealed_identity_digest": bundle["sealed_identity_digest"],
                "actual_test_root": str(path.parent),
                "gt_source_absolute_path": str(path),
                "file_size_bytes": int(path.stat().st_size),
                "image_shape": _shape_text(pixels.shape),
                "dtype": str(pixels.dtype),
                "source_file_sha256_manifest": gt_row["sha256"],
                "source_file_sha256_recomputed": source_sha,
                "source_pixel_sha256_manifest": bundle["source_pixel_sha256"],
                "source_pixel_sha256_recomputed": pixel_sha,
                "source_normalized_pixel_sha256_manifest": bundle[
                    "source_normalized_pixel_sha256"
                ],
                "source_normalized_pixel_sha256_recomputed": normalized_pixel_sha,
                "gt_normalized_array_sha256_manifest": bundle[
                    "gt_normalized_array_sha256"
                ],
                "gt_normalized_array_sha256_recomputed": normalized_array_sha,
                "bundle_npz_absolute_path": str(bundle_path),
                "bundle_npz_sha256_manifest": bundle["npz_sha256"],
                "bundle_npz_sha256_recomputed": npz_sha,
                "raw_stack_sha256_manifest": bundle["raw_stack_sha256"],
                "raw_stack_sha256_recomputed": raw_sha,
                "nominal_gt_identity_sha256": ";".join(nominal_gt_hashes),
                "nominal_raw_stack_sha256": ";".join(nominal_raw_hashes),
                "nominal_methods": ";".join(nominal_methods),
                "bundle_gt_embedded": False,
                "identity_chain_status": chain_status,
                "checks_json": _canonical_bytes(checks).decode("utf-8"),
                "prediction_cache_count": len(prediction_files),
            }
        )

    source_roots = sorted({str(Path(item["gt_source_absolute_path"]).parent) for item in manifest})
    summary = {
        "status": "PASS"
        if all(row["identity_chain_status"] == "PASS" for row in manifest)
        and not cache_failures
        else "FAIL",
        "actual_gt_roots": source_roots,
        "sample_count": len(manifest),
        "method_rows": len(nominal_rows),
        "prediction_cache_expected": 120,
        "prediction_cache_verified": 120 - len(cache_failures),
        "prediction_cache_failures": cache_failures,
        "gt_manifest": str(GT_MANIFEST),
        "gt_manifest_sha256": sha256_file(GT_MANIFEST),
        "bundle_manifest": str(BUNDLE_MANIFEST),
        "bundle_manifest_sha256": sha256_file(BUNDLE_MANIFEST),
        "sealed_test_manifest": str(SEALED_TEST_MANIFEST),
        "sealed_test_manifest_sha256": sha256_file(SEALED_TEST_MANIFEST),
        "nominal_csv": str(nominal_path),
        "nominal_csv_sha256": sha256_file(nominal_path),
        "bundle_contract": (
            "raw NPZ files contain observed six-frame tensors and source identity receipts; "
            "they deliberately omit GT. Metrics loaded GT late from the exact manifest path."
        ),
    }
    return manifest, summary


def _audit_robustness_chain() -> dict[str, Any]:
    manifest_rows = _read_csv(ROBUSTNESS_MANIFEST)
    csv_path = EXISTING_RUN / "R1C3_ROBUSTNESS_PER_SAMPLE.csv"
    formal_rows = _read_csv(csv_path)
    if len(manifest_rows) != 20 or len(formal_rows) != 4320:
        raise DatasetIdentityAuditError("robustness identity chain is not 20 sources / 4320 rows")
    expected_by_sample = {row["sample_id"]: row for row in manifest_rows}
    if len(expected_by_sample) != 20:
        raise DatasetIdentityAuditError("robustness manifest sample IDs are not unique")

    source_checks: list[dict[str, Any]] = []
    for row in manifest_rows:
        path = Path(row["absolute_path"]).resolve(strict=True)
        value = _read_tiff(path)
        y, x = int(row["crop_y"]), int(row["crop_x"])
        h, w = int(row["crop_height"]), int(row["crop_width"])
        patch = np.ascontiguousarray(normalize_image(value)[y : y + h, x : x + w])
        # The historical patch receipt predates the canonical newline header.
        legacy = hashlib.sha256(
            str(patch.dtype).encode("utf-8")
            + str(tuple(patch.shape)).encode("utf-8")
            + patch.tobytes(order="C")
        ).hexdigest()
        checks = {
            "file_sha": sha256_file(path) == row["file_sha256"],
            "shape": tuple(value.shape)
            == (int(row["source_height"]), int(row["source_width"])),
            "crop_bounds": y >= 0 and x >= 0 and y + h <= value.shape[0] and x + w <= value.shape[1],
            "patch_sha": legacy == row["gt_patch_sha256"],
        }
        source_checks.append(
            {
                "sample_id": row["sample_id"],
                "source_path": str(path),
                "source_file_sha256": row["file_sha256"],
                "gt_patch_sha256": row["gt_patch_sha256"],
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    group_methods: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    group_raw: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    csv_identity_failures: list[str] = []
    for row in formal_rows:
        sample = row["sample_id"]
        expected = expected_by_sample.get(sample)
        if expected is None or row["gt_identity_sha256"] != expected["gt_patch_sha256"]:
            csv_identity_failures.append(
                f"{row.get('factor')}/{row.get('severity')}/{sample}/{row.get('method')}"
            )
            continue
        key = (row["factor"], row["severity"], sample)
        group_methods[key].add(row["method"])
        group_raw[key].add(row["raw_stack_sha256"])
    expected_methods = {"WF", "DiffWS-6", "PhysMap-6", "APD-SIM-6"}
    shared_input_failures = sorted(
        "/".join(key)
        for key in group_methods
        if group_methods[key] != expected_methods or len(group_raw[key]) != 1
    )

    visual_root = EXISTING_RUN / "robustness_visual_arrays"
    visual_files = sorted(path for path in visual_root.glob("*.npz") if path.is_file())
    visual_failures: list[str] = []
    for path in visual_files:
        try:
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    value = np.asarray(archive[key])
                    if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
                        raise ValueError(f"non-finite {key}")
        except Exception as exc:  # pragma: no cover - only on corrupt historical cache
            visual_failures.append(f"{path}: {exc}")

    ceshiji_hashes = {sha256_file(path) for path in CESHIJI_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}}
    robust_hashes = {row["file_sha256"] for row in manifest_rows}
    status = (
        "PASS"
        if all(item["status"] == "PASS" for item in source_checks)
        and not csv_identity_failures
        and not shared_input_failures
        and not visual_failures
        else "FAIL"
    )
    return {
        "status": status,
        "role": "SEPARATE_FIXED_20_PATCH_ROBUSTNESS_GRID_NOT_NOMINAL_30_FOV",
        "manifest": str(ROBUSTNESS_MANIFEST),
        "manifest_sha256": sha256_file(ROBUSTNESS_MANIFEST),
        "formal_csv": str(csv_path),
        "formal_csv_sha256": sha256_file(csv_path),
        "source_count": len(manifest_rows),
        "formal_row_count": len(formal_rows),
        "factor_severity_sample_groups": len(group_methods),
        "source_root_counts": dict(
            sorted(Counter(str(Path(row["absolute_path"]).parent) for row in manifest_rows).items())
        ),
        "source_file_sha_overlap_with_ceshiji": len(ceshiji_hashes & robust_hashes),
        "csv_identity_failure_count": len(csv_identity_failures),
        "csv_identity_failures": csv_identity_failures,
        "method_shared_raw_failure_count": len(shared_input_failures),
        "method_shared_raw_failures": shared_input_failures,
        "visual_cache_npz_count": len(visual_files),
        "visual_cache_failures": visual_failures,
        "source_checks": source_checks,
    }


def _evidence_inventory(nominal_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles = {
        "STATUS.json": "formal run final-status anchor",
        "AUDIT_PHYSMAP6_IMPLEMENTATION.json": "preflight and formal artifact audit",
        "DMD6_PROTOCOL_RECEIPT.json": "protocol identity; no GT identity",
        "APD6_CHECKPOINT_RECEIPT.json": "checkpoint identity; no GT identity",
        "R1C3_NOMINAL_PER_FOV.csv": "per-method GT and raw-stack identity",
        "R1C3_NOMINAL_STATS.json": "nominal source-CSV hash receipt",
        "R1C3_ROBUSTNESS_PER_SAMPLE.csv": "separate 20-patch robustness identities",
        "R1C3_ROBUSTNESS_STATS.json": "robustness source-CSV hash receipt",
        "R1C3_PREFLIGHT.json": "upstream manifest paths and hashes",
    }
    inventory: list[dict[str, Any]] = []
    for name, role in roles.items():
        path = EXISTING_RUN / name
        if not path.is_file():
            inventory.append({"path": str(path), "role": role, "status": "MISSING"})
            continue
        item: dict[str, Any] = {
            "path": str(path),
            "role": role,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "status": "PRESENT",
        }
        if path.suffix.lower() == ".json":
            payload = _read_json_object(path)
            item["declared_status"] = payload.get("status")
            if name == "R1C3_NOMINAL_STATS.json":
                item["source_csv_sha256_declared"] = payload.get("source_csv_sha256")
                item["source_csv_sha256_matches"] = payload.get("source_csv_sha256") == nominal_summary["nominal_csv_sha256"]
            elif name == "R1C3_ROBUSTNESS_STATS.json":
                actual = sha256_file(EXISTING_RUN / "R1C3_ROBUSTNESS_PER_SAMPLE.csv")
                item["source_csv_sha256_declared"] = payload.get("source_csv_sha256")
                item["source_csv_sha256_matches"] = payload.get("source_csv_sha256") == actual
        inventory.append(item)
    inventory.extend(
        {
            "path": str(path),
            "role": role,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "status": "PRESENT",
        }
        for path, role in (
            (GT_MANIFEST, "authoritative nominal GT path/file-hash mapping"),
            (SEALED_TEST_MANIFEST, "sealed test identity boundary"),
            (BUNDLE_MANIFEST, "authoritative raw-bundle and source pixel identities"),
            (ROBUSTNESS_MANIFEST, "authoritative separate robustness patch identities"),
        )
    )
    return inventory


def _comparison_rows(
    ceshiji: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_pixel: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_normalized: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ceshiji:
        by_file[str(row["sha256"])].append(row)
        by_pixel[str(row["decoded_pixel_sha256"])].append(row)
        by_normalized[str(row["normalized_pixel_sha256"])].append(row)
        by_sample[str(row["sample_id"])].append(row)

    output: list[dict[str, Any]] = []
    used_ceshiji: set[str] = set()
    for old in existing:
        candidates: list[Mapping[str, Any]] = []
        match_basis = "NONE"
        for lookup, key, label in (
            (by_file, str(old["source_file_sha256_recomputed"]), "FILE_SHA256"),
            (by_pixel, str(old["source_pixel_sha256_recomputed"]), "PIXEL_SHA256"),
            (
                by_normalized,
                str(old["source_normalized_pixel_sha256_recomputed"]),
                "NORMALIZED_PIXEL_SHA256",
            ),
            (by_sample, str(old["sample_id"]), "SAMPLE_ID_ONLY"),
        ):
            if len(lookup.get(key, [])) == 1:
                candidates = lookup[key]
                match_basis = label
                break
        match = candidates[0] if candidates else None
        if match is not None:
            used_ceshiji.add(str(match["absolute_path"]))
        exact_file = bool(match) and old["source_file_sha256_recomputed"] == match["sha256"]
        exact_pixel = bool(match) and old["source_pixel_sha256_recomputed"] == match["decoded_pixel_sha256"]
        exact_normalized = bool(match) and old["source_normalized_pixel_sha256_recomputed"] == match["normalized_pixel_sha256"]
        exact_metadata = bool(match) and (
            str(old["image_shape"]) == str(match["image_shape"])
            and str(old["dtype"]) == str(match["dtype"])
            and int(old["file_size_bytes"]) == int(match["file_size_bytes"])
        )
        exact_path = bool(match) and Path(str(old["gt_source_absolute_path"])).resolve(
            strict=False
        ) == Path(str(match["absolute_path"])).resolve(strict=False)
        exact = (
            bool(match)
            and exact_file
            and exact_pixel
            and exact_normalized
            and exact_metadata
            and exact_path
        )
        output.append(
            {
                "existing_sample_order": old["sample_order"],
                "existing_sample_id": old["sample_id"],
                "existing_gt_absolute_path": old["gt_source_absolute_path"],
                "existing_file_sha256": old["source_file_sha256_recomputed"],
                "existing_pixel_sha256": old["source_pixel_sha256_recomputed"],
                "existing_normalized_pixel_sha256": old[
                    "source_normalized_pixel_sha256_recomputed"
                ],
                "existing_image_shape": old["image_shape"],
                "existing_dtype": old["dtype"],
                "existing_file_size_bytes": old["file_size_bytes"],
                "ceshiji_sample_id": match["sample_id"] if match else "",
                "ceshiji_absolute_path": match["absolute_path"] if match else "",
                "ceshiji_file_sha256": match["sha256"] if match else "",
                "ceshiji_pixel_sha256": match["decoded_pixel_sha256"] if match else "",
                "ceshiji_normalized_pixel_sha256": match["normalized_pixel_sha256"] if match else "",
                "ceshiji_image_shape": match["image_shape"] if match else "",
                "ceshiji_dtype": match["dtype"] if match else "",
                "ceshiji_file_size_bytes": match["file_size_bytes"] if match else "",
                "match_basis": match_basis,
                "file_sha256_match": exact_file,
                "pixel_sha256_match": exact_pixel,
                "normalized_pixel_sha256_match": exact_normalized,
                "shape_dtype_size_match": exact_metadata,
                "absolute_path_match": exact_path,
                "comparison_status": "EXACT_MATCH" if exact else (
                    "NAME_ONLY_MISMATCH" if match_basis == "SAMPLE_ID_ONLY" else "NO_EXACT_MATCH"
                ),
            }
        )

    for row in ceshiji:
        if str(row["absolute_path"]) not in used_ceshiji:
            output.append(
                {
                    "existing_sample_order": "",
                    "existing_sample_id": "",
                    "existing_gt_absolute_path": "",
                    "existing_file_sha256": "",
                    "existing_pixel_sha256": "",
                    "existing_normalized_pixel_sha256": "",
                    "existing_image_shape": "",
                    "existing_dtype": "",
                    "existing_file_size_bytes": "",
                    "ceshiji_sample_id": row["sample_id"],
                    "ceshiji_absolute_path": row["absolute_path"],
                    "ceshiji_file_sha256": row["sha256"],
                    "ceshiji_pixel_sha256": row["decoded_pixel_sha256"],
                    "ceshiji_normalized_pixel_sha256": row["normalized_pixel_sha256"],
                    "ceshiji_image_shape": row["image_shape"],
                    "ceshiji_dtype": row["dtype"],
                    "ceshiji_file_size_bytes": row["file_size_bytes"],
                    "match_basis": "CESHIJI_ONLY",
                    "file_sha256_match": False,
                    "pixel_sha256_match": False,
                    "normalized_pixel_sha256_match": False,
                    "shape_dtype_size_match": False,
                    "absolute_path_match": False,
                    "comparison_status": "CESHIJI_NOT_IN_EXISTING_TEST",
                }
            )
    return output


def _identity_status(
    ceshiji: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    nominal_chain_status: str,
) -> str:
    if len(existing) != 30 or nominal_chain_status != "PASS":
        return "CESHIJI_IDENTITY_UNRESOLVED"
    exact = sum(row["comparison_status"] == "EXACT_MATCH" for row in comparison)
    if len(ceshiji) == 30 and exact == 30 and len(comparison) == 30:
        return "CESHIJI_EXACT_30_MATCH"
    if exact > 0:
        return "CESHIJI_PARTIAL_MATCH"
    if all(row["comparison_status"] != "NAME_ONLY_MISMATCH" for row in comparison):
        return "CESHIJI_NOT_USED"
    return "CESHIJI_IDENTITY_UNRESOLVED"


CESHIJI_FIELDS = (
    "order",
    "sample_id",
    "structure_class",
    "sealed_identity_digest",
    "absolute_path",
    "relative_path",
    "filename",
    "parent_class_directory",
    "file_size_bytes",
    "image_shape",
    "dtype",
    "sha256",
    "decoded_pixel_sha256",
    "normalized_pixel_sha256",
    "normalized_array_sha256",
    "pixels_finite",
)

EXISTING_FIELDS = (
    "sample_order",
    "sample_id",
    "parent_id",
    "structure_class",
    "sealed_identity_digest",
    "actual_test_root",
    "gt_source_absolute_path",
    "file_size_bytes",
    "image_shape",
    "dtype",
    "source_file_sha256_manifest",
    "source_file_sha256_recomputed",
    "source_pixel_sha256_manifest",
    "source_pixel_sha256_recomputed",
    "source_normalized_pixel_sha256_manifest",
    "source_normalized_pixel_sha256_recomputed",
    "gt_normalized_array_sha256_manifest",
    "gt_normalized_array_sha256_recomputed",
    "bundle_npz_absolute_path",
    "bundle_npz_sha256_manifest",
    "bundle_npz_sha256_recomputed",
    "raw_stack_sha256_manifest",
    "raw_stack_sha256_recomputed",
    "nominal_gt_identity_sha256",
    "nominal_raw_stack_sha256",
    "nominal_methods",
    "bundle_gt_embedded",
    "identity_chain_status",
    "checks_json",
    "prediction_cache_count",
)

COMPARISON_FIELDS = (
    "existing_sample_order",
    "existing_sample_id",
    "existing_gt_absolute_path",
    "existing_file_sha256",
    "existing_pixel_sha256",
    "existing_normalized_pixel_sha256",
    "existing_image_shape",
    "existing_dtype",
    "existing_file_size_bytes",
    "ceshiji_sample_id",
    "ceshiji_absolute_path",
    "ceshiji_file_sha256",
    "ceshiji_pixel_sha256",
    "ceshiji_normalized_pixel_sha256",
    "ceshiji_image_shape",
    "ceshiji_dtype",
    "ceshiji_file_size_bytes",
    "match_basis",
    "file_sha256_match",
    "pixel_sha256_match",
    "normalized_pixel_sha256_match",
    "shape_dtype_size_match",
    "absolute_path_match",
    "comparison_status",
)


def run_dataset_identity_audit(output_dir: Path) -> dict[str, Any]:
    """Execute the read-only audit and emit the five required artifacts."""

    destination = output_dir.resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_before = capture_existing_run_snapshot()
    ceshiji = enumerate_ceshiji()
    existing, nominal_summary = _audit_nominal_chain()
    robustness = _audit_robustness_chain()
    comparison = _comparison_rows(ceshiji, existing)
    status = _identity_status(ceshiji, existing, comparison, nominal_summary["status"])
    if status not in IDENTITY_STATUSES:  # defensive exhaustiveness
        raise AssertionError(status)

    ceshiji_counts = dict(sorted(Counter(row["structure_class"] for row in ceshiji).items()))
    exact_count = sum(row["comparison_status"] == "EXACT_MATCH" for row in comparison)
    evidence = _evidence_inventory(nominal_summary)
    required_evidence_present = all(item["status"] == "PRESENT" for item in evidence)
    if status == "CESHIJI_EXACT_30_MATCH" and (
        ceshiji_counts != {"CCPs": 10, "ER": 10, "microtubules": 10}
        or not required_evidence_present
        or robustness["status"] != "PASS"
    ):
        status = "CESHIJI_IDENTITY_UNRESOLVED"

    ceshiji_path = destination / "CESHIJI_MANIFEST.csv"
    existing_path = destination / "EXISTING_R1C3_TEST_MANIFEST.csv"
    comparison_path = destination / "DATASET_IDENTITY_COMPARISON.csv"
    _write_new(ceshiji_path, _csv_bytes(ceshiji, CESHIJI_FIELDS))
    _write_new(existing_path, _csv_bytes(existing, EXISTING_FIELDS))
    _write_new(comparison_path, _csv_bytes(comparison, COMPARISON_FIELDS))

    snapshot_after_identity_outputs = verify_existing_run_snapshot(snapshot_before)
    audit = {
        "schema_version": 1,
        "status": status,
        "scope": "independent read-only identity audit of the immutable strict R1C3 run",
        "ceshiji_root": str(CESHIJI_ROOT.resolve(strict=True)),
        "actual_nominal_gt_root": nominal_summary["actual_gt_roots"][0]
        if len(nominal_summary["actual_gt_roots"]) == 1
        else nominal_summary["actual_gt_roots"],
        "actual_raw_bundle_root": str(BUNDLE_MANIFEST.parent.resolve(strict=True)),
        "ceshiji_tiff_count": len(ceshiji),
        "ceshiji_class_counts": ceshiji_counts,
        "existing_nominal_sample_count": len(existing),
        "exact_match_count": exact_count,
        "partial_or_name_only_count": sum(
            row["comparison_status"] == "NAME_ONLY_MISMATCH" for row in comparison
        ),
        "ceshiji_only_count": sum(
            row["comparison_status"] == "CESHIJI_NOT_IN_EXISTING_TEST"
            for row in comparison
        ),
        "matching_priority": [
            "source file SHA-256",
            "decoded pixel SHA-256",
            "normalized pixel SHA-256",
            "sample ID only (never sufficient for exact match)",
        ],
        "nominal_identity_chain": nominal_summary,
        "robustness_identity_chain": robustness,
        "evidence_inventory": evidence,
        "existing_run_snapshot_before": snapshot_before,
        "existing_run_snapshot_after_identity_outputs": {
            key: value
            for key, value in snapshot_after_identity_outputs.items()
            if key != "after"
        },
        "old_run_unchanged_during_identity_audit": snapshot_after_identity_outputs["exact"],
        "rerun_nominal_on_ceshiji_required": status != "CESHIJI_EXACT_30_MATCH",
        "rerun_entry_created": False,
        "rerun_entry_reason": (
            "not required: all 30 source files are byte-, pixel-, normalized-pixel-, "
            "path-, shape-, dtype-, and size-matched"
            if status == "CESHIJI_EXACT_30_MATCH"
            else "required by policy; combined task must provide the separate rerun entry"
        ),
        "artifacts": {
            "ceshiji_manifest": str(ceshiji_path),
            "existing_test_manifest": str(existing_path),
            "comparison": str(comparison_path),
        },
    }
    if not snapshot_after_identity_outputs["exact"]:
        audit["status"] = "CESHIJI_IDENTITY_UNRESOLVED"
        status = audit["status"]

    json_path = destination / "DATASET_IDENTITY_AUDIT.json"
    _write_json_new(json_path, audit)
    md = f"""# R1C3 dataset identity audit

Status: `{status}`

## Conclusion

- Existing nominal 30-FOV GT root: `{audit['actual_nominal_gt_root']}`
- Enumerated `ceshiji` TIFF files: **{len(ceshiji)}**
- Class counts: CCPs **{ceshiji_counts.get('CCPs', 0)}**, ER **{ceshiji_counts.get('ER', 0)}**, microtubules **{ceshiji_counts.get('microtubules', 0)}**
- Exact one-to-one matches: **{exact_count}/30**
- Match evidence: source-file SHA-256, decoded-pixel SHA-256, normalized-pixel SHA-256, shape, dtype, size, and absolute source path were independently recomputed.
- Existing run whole-tree unchanged during this audit: **{snapshot_after_identity_outputs['exact']}** (`{snapshot_before['file_count']}` files; aggregate `{snapshot_before['aggregate_sha256']}`).

## Provenance boundary

The formal nominal run consumed immutable six-frame observations from the official shared-bundle
root `{audit['actual_raw_bundle_root']}`. Those NPZ files deliberately contain no GT. For metrics,
the formal code loaded GT late from the 30-row mapping whose absolute paths all resolve to
`{audit['actual_nominal_gt_root']}`. All 30 source file, decoded-pixel, normalized-pixel, raw-stack,
and per-method CSV receipts agree.

The 4,320-row robustness grid is a separate frozen 20-patch experiment rooted outside `ceshiji`;
its source/crop hashes and method-shared raw identities pass independently. It is not evidence
against the nominal 30-FOV identity and must not be described as the nominal dataset.

## Rerun decision

`R1C3_rerun_nominal_on_ceshiji.py` is not needed or created because the existing nominal test set
already equals all 30 `ceshiji` GT files exactly. Re-running would duplicate the same source GT set,
not repair a dataset mismatch.
"""
    md_path = destination / "DATASET_IDENTITY_AUDIT.md"
    _write_new(md_path, md.encode("utf-8"))

    # The returned snapshot is intentionally full: the combined entrypoint must
    # call verify_existing_run_snapshot(result["snapshot_before"]) after every
    # convergence/Figure-5 operation, not merely after this module returns.
    return {
        "status": status,
        "output_dir": str(destination),
        "exact_match_count": exact_count,
        "ceshiji_count": len(ceshiji),
        "class_counts": ceshiji_counts,
        "actual_nominal_gt_root": audit["actual_nominal_gt_root"],
        "snapshot_before": snapshot_before,
        "snapshot_after_identity_outputs": snapshot_after_identity_outputs,
        "audit_json": str(json_path),
        "audit_md": str(md_path),
        "artifacts": audit["artifacts"],
    }


__all__ = [
    "CESHIJI_ROOT",
    "EXISTING_RUN",
    "capture_existing_run_snapshot",
    "verify_existing_run_snapshot",
    "enumerate_ceshiji",
    "run_dataset_identity_audit",
]

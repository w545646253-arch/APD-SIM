from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from unisim.revision_r1 import dataset_fig5_audit_identity as identity


def test_authoritative_array_hash_definition_is_byte_and_metadata_sensitive() -> None:
    value = np.arange(12, dtype=np.uint16).reshape(3, 4)
    contiguous = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = hashlib.sha256(header + b"\n" + contiguous.tobytes(order="C")).hexdigest()
    assert identity.array_sha256(value) == expected
    assert identity.array_sha256(value.astype(np.float32)) != expected
    assert identity.array_sha256(value.reshape(4, 3)) != expected


def test_snapshot_detects_no_change_and_one_changed_file(tmp_path: Path) -> None:
    root = tmp_path / "old-run"
    root.mkdir()
    first = root / "a.txt"
    second = root / "nested" / "b.bin"
    second.parent.mkdir()
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    before = identity.capture_existing_run_snapshot(root)
    assert identity.verify_existing_run_snapshot(before, root)["status"] == "PASS"
    second.write_bytes(b"changed")
    check = identity.verify_existing_run_snapshot(before, root)
    assert check["status"] == "FAIL"
    assert check["changed"] == ["nested/b.bin"]


def test_real_ceshiji_is_exact_30_and_class_balanced() -> None:
    rows = identity.enumerate_ceshiji()
    assert len(rows) == 30
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["structure_class"]] = counts.get(row["structure_class"], 0) + 1
        assert row["image_shape"] == "1004x1004"
        assert row["pixels_finite"] is True
        assert len(row["sha256"]) == 64
        assert len(row["decoded_pixel_sha256"]) == 64
    assert counts == {"CCPs": 10, "ER": 10, "microtubules": 10}


def test_full_identity_audit_writes_exact_manifests_and_preserves_old_run(
    tmp_path: Path,
) -> None:
    result = identity.run_dataset_identity_audit(tmp_path / "identity")
    assert result["status"] == "CESHIJI_EXACT_30_MATCH"
    assert result["exact_match_count"] == 30
    assert result["class_counts"] == {"CCPs": 10, "ER": 10, "microtubules": 10}
    assert result["snapshot_after_identity_outputs"]["exact"] is True

    destination = Path(result["output_dir"])
    required = {
        "CESHIJI_MANIFEST.csv",
        "EXISTING_R1C3_TEST_MANIFEST.csv",
        "DATASET_IDENTITY_COMPARISON.csv",
        "DATASET_IDENTITY_AUDIT.json",
        "DATASET_IDENTITY_AUDIT.md",
    }
    assert {path.name for path in destination.iterdir()} == required
    with (destination / "DATASET_IDENTITY_COMPARISON.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        comparison = list(csv.DictReader(handle))
    assert len(comparison) == 30
    assert all(row["comparison_status"] == "EXACT_MATCH" for row in comparison)
    assert all(row["file_sha256_match"] == "True" for row in comparison)
    assert all(row["pixel_sha256_match"] == "True" for row in comparison)
    assert all(row["absolute_path_match"] == "True" for row in comparison)

    audit = json.loads((destination / "DATASET_IDENTITY_AUDIT.json").read_text("utf-8"))
    assert audit["status"] == "CESHIJI_EXACT_30_MATCH"
    assert audit["robustness_identity_chain"]["status"] == "PASS"
    assert audit["robustness_identity_chain"]["role"].startswith("SEPARATE_FIXED_20")
    assert audit["rerun_nominal_on_ceshiji_required"] is False
    assert audit["old_run_unchanged_during_identity_audit"] is True

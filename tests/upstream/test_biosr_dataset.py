from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

import unisim.datasets as datasets_module
from unisim.datasets import BioSRGT2DDataset, BioSRManifestError


def _manifest(tmp_path: Path, array: np.ndarray, *, split: str = "train") -> Path:
    image_path = tmp_path / "sample.tif"
    tifffile.imwrite(image_path, array)
    payload = {
        "records": [
            {
                "sample_id": "S1",
                "parent_id": "P1",
                "class": "CCP",
                "absolute_path": str(image_path),
                "relative_path": image_path.name,
                "file_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "split": split,
                "split_seed": 20260813,
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_biosr_manifest_only_strict_2d_crop_and_metadata(tmp_path: Path) -> None:
    array = np.arange(1004 * 1004, dtype=np.uint32).reshape(1004, 1004) % 65535
    manifest = _manifest(tmp_path, array.astype(np.uint16))
    accessed = []
    dataset = BioSRGT2DDataset(
        manifest,
        patch_size=320,
        augment=True,
        expected_split="train",
        access_logger=lambda path: accessed.append(str(path)),
        rng_seed=19,
    )
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (1, 320, 320)
    assert sample["image"].dtype.is_floating_point
    assert sample["sample_id"] == "S1"
    assert sample["parent_id"] == "P1"
    assert sample["class"] == "CCP"
    assert accessed == [str((tmp_path / "sample.tif").resolve())]
    assert dataset.accessed_tiff_paths == tuple(accessed)
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 1004, 1004), dtype=np.uint8),
        np.zeros((1004, 1004, 1), dtype=np.uint8),
        np.zeros((1003, 1004), dtype=np.uint8),
        np.zeros((1004, 1004), dtype=np.uint8),
    ],
)
def test_biosr_rejects_non_2d_wrong_shape_and_constant(tmp_path: Path, array: np.ndarray) -> None:
    manifest = _manifest(tmp_path, array)
    if list(array.shape) != [1004, 1004]:
        with pytest.raises(BioSRManifestError):
            BioSRGT2DDataset(manifest, expected_split="train")
    else:
        dataset = BioSRGT2DDataset(manifest, expected_split="train")
        with pytest.raises(BioSRManifestError):
            _ = dataset[0]


def test_biosr_rejects_nonfinite_without_sanitizing(tmp_path: Path) -> None:
    array = np.linspace(0, 1, 1004 * 1004, dtype=np.float32).reshape(1004, 1004)
    array[0, 0] = np.nan
    dataset = BioSRGT2DDataset(_manifest(tmp_path, array), expected_split="train")
    with pytest.raises(BioSRManifestError, match="NaN or Inf"):
        _ = dataset[0]


def test_biosr_file_hash_is_verified_once_and_private_rng_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array = (np.arange(1004 * 1004, dtype=np.uint32).reshape(1004, 1004) % 65521).astype(
        np.uint16
    )
    calls = []
    original = datasets_module._sha256_file

    def counted(path: Path) -> str:
        calls.append(str(Path(path).resolve()))
        return original(path)

    monkeypatch.setattr(datasets_module, "_sha256_file", counted)
    dataset = BioSRGT2DDataset(
        _manifest(tmp_path, array),
        patch_size=320,
        augment=True,
        expected_split="train",
        verify_file_sha256=True,
        rng_seed=93,
    )
    initial_state = dataset.get_rng_state()
    first = dataset[0]
    dataset.set_rng_state(initial_state)
    second = dataset[0]

    assert calls == [str((tmp_path / "sample.tif").resolve())]
    assert first["crop_top"] == second["crop_top"]
    assert first["crop_left"] == second["crop_left"]
    assert first["image"].equal(second["image"])
    # Returned state is defensive: mutating it must not mutate the dataset.
    external = dataset.get_rng_state()
    assert external is not None
    external["state"]["state"] = 0
    assert dataset.get_rng_state() != external


def test_biosr_file_hash_mismatch_fails_before_tiff_decode(tmp_path: Path) -> None:
    array = np.arange(1004 * 1004, dtype=np.uint32).reshape(1004, 1004).astype(np.float32)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["file_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    dataset = BioSRGT2DDataset(
        manifest,
        expected_split="train",
        verify_file_sha256=True,
        rng_seed=1,
    )
    with pytest.raises(BioSRManifestError, match="File SHA-256 mismatch"):
        _ = dataset[0]

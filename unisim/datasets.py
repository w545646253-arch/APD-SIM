# -*- coding: utf-8 -*-
"""
unisim/datasets.py

Fixes + improvements (reasonable, not minimal):
1) ✅ Fix torch.from_numpy negative-stride crash:
   - Any flip/rot90 can create numpy views with negative strides.
   - We enforce contiguous arrays via np.ascontiguousarray(...) before converting to torch.

2) More robust image/volume reading:
   - Handles tif/tiff plus common image formats for 2D GT.
   - Handles (Z,H,W) or (T,Z,H,W) volumes for 3D GT.

3) Canonical output shapes:
   - 2D: (C=1, Z=1, H, W)
   - 3D: (C=1, Z,   H, W)

4) Augmentations are safe and SIM-friendly:
   - Random flips
   - Optional 90-degree rotations in XY (does not change shape for square patches)
   - Mild gamma augmentation
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

import imageio.v2 as imageio
import tifffile

from .utils import (
    list_image_files,
    read_tiff,
    normalize_percentile,
    random_crop_2d,
    random_crop_3d,
)


@dataclass
class DatasetConfig:
    patch2d: int = 256
    patch3d_xy: int = 96
    patch3d_z: int = 16
    p_low: float = 0.5
    p_high: float = 99.5
    augment: bool = True


def _to_float32_2d_gray(arr: np.ndarray) -> np.ndarray:
    """Convert arbitrary image array to 2D float32 grayscale."""
    arr = np.asarray(arr)

    # If tif is (T,H,W) or (Z,H,W) accidentally fed into 2D dataset, take the first slice.
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    if arr.ndim == 3:
        # Common color layout: (H,W,C)
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        # Less common: (C,H,W)
        elif arr.shape[0] in (3, 4):
            arr = arr[0, ...]
        else:
            # Fallback: average channels
            arr = arr.mean(axis=-1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after conversion, got shape {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    # sanitize
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr


def _augment_2d(img: np.ndarray) -> np.ndarray:
    """Safe 2D augmentations. Returns a numpy array (may be non-contiguous; caller will fix)."""
    # Random flips
    if np.random.rand() < 0.5:
        img = img[:, ::-1]  # can create negative stride
    if np.random.rand() < 0.5:
        img = img[::-1, :]

    # Random 90-degree rotations (safe because patches are square)
    if np.random.rand() < 0.3:
        k = np.random.randint(0, 4)
        if k:
            img = np.rot90(img, k=k)

    # Mild gamma
    if np.random.rand() < 0.3:
        g = np.random.uniform(0.8, 1.2)
        img = np.clip(img ** g, 0.0, 1.0)

    return img


def _augment_3d(vol: np.ndarray) -> np.ndarray:
    """Safe 3D augmentations. Returns a numpy array (may be non-contiguous; caller will fix)."""
    # Flips
    if np.random.rand() < 0.5:
        vol = vol[:, :, ::-1]  # W flip
    if np.random.rand() < 0.5:
        vol = vol[:, ::-1, :]  # H flip
    if np.random.rand() < 0.3:
        vol = vol[::-1, :, :]  # Z flip

    # Rot90 in XY plane (axes H,W)
    if np.random.rand() < 0.3:
        k = np.random.randint(0, 4)
        if k:
            vol = np.rot90(vol, k=k, axes=(1, 2))

    # Mild gamma
    if np.random.rand() < 0.3:
        g = np.random.uniform(0.8, 1.2)
        vol = np.clip(vol ** g, 0.0, 1.0)

    return vol


class GT2DDataset(Dataset):
    """Loads 2D GT images (BIOSR GT etc.) and returns tensor (C=1, Z=1, H, W)."""

    def __init__(self, root: Union[str, Path], cfg: DatasetConfig):
        self.root = Path(root)
        self.cfg = cfg
        self.files: List[Path] = list_image_files(
            self.root, exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg")
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        f = self.files[idx]
        if f.suffix.lower() in (".tif", ".tiff"):
            img = read_tiff(f)
        else:
            img = imageio.imread(str(f))

        img = _to_float32_2d_gray(img)
        img = normalize_percentile(img, self.cfg.p_low, self.cfg.p_high)
        img = random_crop_2d(img, (self.cfg.patch2d, self.cfg.patch2d))

        if self.cfg.augment:
            img = _augment_2d(img)

        # ✅ critical fix: make contiguous (no negative strides) before torch.from_numpy
        x = np.ascontiguousarray(img[None, None, :, :], dtype=np.float32)  # (1,1,H,W)
        return torch.from_numpy(x)


class GT3DDataset(Dataset):
    """Loads 3D GT volumes (tif stacks). Returns tensor (C=1, Z, H, W)."""

    def __init__(self, root: Union[str, Path], cfg: DatasetConfig):
        self.root = Path(root)
        self.cfg = cfg
        self.files: List[Path] = list_image_files(self.root, exts=(".tif", ".tiff"))
        if len(self.files) == 0:
            raise RuntimeError(f"No tif volumes found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        f = self.files[idx]
        vol = read_tiff(f)
        vol = np.asarray(vol)

        # common shapes: (Z,H,W) or (T,Z,H,W). If T exists, take first.
        if vol.ndim == 4:
            vol = vol[0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume (Z,H,W), got shape {vol.shape} from {f}")

        vol = vol.astype(np.float32, copy=False)
        if not np.isfinite(vol).all():
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        vol = normalize_percentile(vol, self.cfg.p_low, self.cfg.p_high)
        vol = random_crop_3d(vol, (self.cfg.patch3d_z, self.cfg.patch3d_xy, self.cfg.patch3d_xy))

        if self.cfg.augment:
            vol = _augment_3d(vol)

        # ✅ critical fix: contiguous before torch.from_numpy
        x = np.ascontiguousarray(vol[None, :, :, :], dtype=np.float32)  # (1,Z,H,W)
        return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Formal APD-DMD R2 two-dimensional data path
# ---------------------------------------------------------------------------

BIOSR_REQUIRED_SHAPE: Tuple[int, int] = (1004, 1004)


class BioSRManifestError(RuntimeError):
    """Raised when a frozen BioSR manifest violates the formal 2-D contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_payload_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    for key in ("manifest_hash", "manifest_sha256", "payload_hash"):
        unhashed.pop(key, None)
    return hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()


def _manifest_records(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = None
        for key in ("samples", "records", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            raise BioSRManifestError("Manifest must contain a top-level samples list")
    else:
        raise BioSRManifestError("Manifest root must be an object or list")
    if not all(isinstance(record, Mapping) for record in records):
        raise BioSRManifestError("Every manifest sample must be a JSON object")
    return records


def _strict_biosr_array(path: Path) -> np.ndarray:
    """Read one TIFF without squeezing, sanitizing, resizing, or channel selection."""
    try:
        array = np.asarray(tifffile.imread(str(path)))
    except Exception as exc:
        raise BioSRManifestError(f"tifffile could not read {path}: {exc}") from exc
    if array.ndim != 2:
        raise BioSRManifestError(
            f"Formal BioSR GT must be exactly 2-D; {path} has ndim={array.ndim}, shape={array.shape}"
        )
    if tuple(int(v) for v in array.shape) != BIOSR_REQUIRED_SHAPE:
        raise BioSRManifestError(
            f"Formal BioSR GT must be {BIOSR_REQUIRED_SHAPE}; {path} has {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise BioSRManifestError(f"Formal BioSR GT must be numeric; {path} has dtype={array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise BioSRManifestError(f"Formal BioSR GT contains NaN or Inf: {path}")
    if float(np.min(array)) == float(np.max(array)):
        raise BioSRManifestError(f"Formal BioSR GT is constant: {path}")
    return array


class BioSRGT2DDataset(Dataset):
    """Manifest-only loader for the formal 2-D BioSR APD-SIM training path.

    The returned image has shape ``(1, patch, patch)``.  There is deliberately
    no singleton Z axis.  The class never scans ``dataset_root`` and never
    accepts image paths outside the frozen manifest.  ``access_logger`` is a
    small audit hook called immediately before each TIFF read.
    """

    REQUIRED_RECORD_FIELDS: Tuple[str, ...] = (
        "sample_id",
        "parent_id",
        "class",
        "absolute_path",
        "relative_path",
        "file_sha256",
        "pixel_sha256",
        "shape",
        "dtype",
        "split",
        "split_seed",
    )

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        patch_size: int = 320,
        augment: bool = True,
        p_low: float = 0.5,
        p_high: float = 99.5,
        expected_manifest_hash: Optional[str] = None,
        expected_split: Optional[str] = None,
        verify_file_sha256: bool = False,
        access_logger: Optional[Callable[[Path], None]] = None,
        rng_seed: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if int(patch_size) <= 0 or int(patch_size) > min(BIOSR_REQUIRED_SHAPE):
            raise BioSRManifestError(f"Invalid patch_size={patch_size}")
        if not (0.0 <= float(p_low) < float(p_high) <= 100.0):
            raise BioSRManifestError(f"Invalid percentiles: {p_low}, {p_high}")

        raw_bytes = self.manifest_path.read_bytes()
        try:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))
        except Exception as exc:
            raise BioSRManifestError(f"Invalid JSON manifest {self.manifest_path}: {exc}") from exc
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        payload_hash = _manifest_payload_hash(payload) if isinstance(payload, Mapping) else None
        embedded_hash = None
        if isinstance(payload, Mapping):
            embedded_hash = payload.get("manifest_hash", payload.get("manifest_sha256"))
            if embedded_hash is not None and str(embedded_hash).lower() != payload_hash:
                raise BioSRManifestError("Embedded manifest hash does not match canonical payload")
        if expected_manifest_hash is not None:
            expected = str(expected_manifest_hash).lower()
            accepted = {file_hash}
            if payload_hash:
                accepted.add(payload_hash)
            if expected not in accepted:
                raise BioSRManifestError(
                    f"Manifest SHA-256 mismatch: expected {expected}, file={file_hash}, payload={payload_hash}"
                )

        manifest_dataset_root = Path(str(payload.get("dataset_root", ""))) if isinstance(payload, Mapping) else Path()
        if not manifest_dataset_root.is_absolute():
            manifest_dataset_root = (self.manifest_path.parents[2] / manifest_dataset_root).resolve()
        records = [dict(record) for record in _manifest_records(payload)]
        if not records:
            raise BioSRManifestError("Formal training/validation manifest is empty")
        seen_ids = set()
        for index, record in enumerate(records):
            missing = [key for key in self.REQUIRED_RECORD_FIELDS if key not in record]
            if missing:
                raise BioSRManifestError(f"Manifest record {index} is missing: {', '.join(missing)}")
            sample_id = str(record["sample_id"])
            if sample_id in seen_ids:
                raise BioSRManifestError(f"Duplicate sample_id in manifest: {sample_id}")
            seen_ids.add(sample_id)
            if expected_split is not None and str(record["split"]) != str(expected_split):
                raise BioSRManifestError(
                    f"Record {sample_id} split={record['split']!r}, expected {expected_split!r}"
                )
            shape = tuple(int(v) for v in record["shape"])
            if shape != BIOSR_REQUIRED_SHAPE:
                raise BioSRManifestError(
                    f"Record {sample_id} declares shape={shape}, expected {BIOSR_REQUIRED_SHAPE}"
                )
            raw_path = Path(str(record["absolute_path"])); path = (manifest_dataset_root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
            if path.suffix.lower() not in (".tif", ".tiff"):
                raise BioSRManifestError(f"Record {sample_id} is not TIFF: {path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            record["absolute_path"] = str(path)

        self.records: List[Dict[str, Any]] = records
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.verify_file_sha256 = bool(verify_file_sha256)
        self.access_logger = access_logger
        self.manifest_file_sha256 = file_hash
        self.manifest_payload_sha256 = payload_hash
        self._accessed_paths: List[str] = []
        self._verified_file_sha256: Dict[str, str] = {}
        self._rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def accessed_tiff_paths(self) -> Tuple[str, ...]:
        return tuple(self._accessed_paths)

    def get_rng_state(self) -> Optional[Dict[str, Any]]:
        """Return a deep-copyable augmentation/crop RNG receipt."""
        return copy.deepcopy(self._rng.bit_generator.state) if self._rng is not None else None

    def set_rng_state(self, state: Optional[Mapping[str, Any]]) -> None:
        """Restore the private generator exactly; reject state for global-RNG mode."""
        if state is None:
            if self._rng is not None:
                raise BioSRManifestError("Cannot restore an absent RNG state into seeded dataset mode")
            return
        if self._rng is None:
            raise BioSRManifestError("Dataset was not constructed with a private RNG")
        try:
            self._rng.bit_generator.state = copy.deepcopy(dict(state))
        except Exception as exc:
            raise BioSRManifestError(f"Invalid dataset RNG state: {exc}") from exc

    def _read_record(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        record = self.records[int(idx)]
        path = Path(record["absolute_path"])
        self._accessed_paths.append(str(path))
        if self.access_logger is not None:
            self.access_logger(path)
        if self.verify_file_sha256:
            path_key = str(path.resolve())
            expected = str(record["file_sha256"]).lower()
            actual = self._verified_file_sha256.get(path_key)
            if actual is None:
                actual = _sha256_file(path).lower()
                if actual == expected:
                    self._verified_file_sha256[path_key] = actual
            if actual != expected:
                raise BioSRManifestError(f"File SHA-256 mismatch for {path}")
        return _strict_biosr_array(path), record

    def load_full_normalized(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        array, record = self._read_record(idx)
        image = normalize_percentile(array.astype(np.float32, copy=False), self.p_low, self.p_high)
        image = np.ascontiguousarray(image, dtype=np.float32)
        if image.ndim != 2 or image.shape != BIOSR_REQUIRED_SHAPE or not np.isfinite(image).all():
            raise BioSRManifestError(f"Normalization violated the 2-D contract for {record['sample_id']}")
        return image, dict(record)

    def _rand(self) -> float:
        return float(self._rng.random()) if self._rng is not None else float(np.random.rand())

    def _randint(self, low: int, high: int) -> int:
        if self._rng is not None:
            return int(self._rng.integers(low, high))
        return int(np.random.randint(low, high))

    def _uniform(self, low: float, high: float) -> float:
        if self._rng is not None:
            return float(self._rng.uniform(low, high))
        return float(np.random.uniform(low, high))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, record = self.load_full_normalized(idx)
        max_top = image.shape[0] - self.patch_size
        max_left = image.shape[1] - self.patch_size
        top = self._randint(0, max_top + 1)
        left = self._randint(0, max_left + 1)
        patch = image[top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if self._rand() < 0.5:
                patch = patch[:, ::-1]
            if self._rand() < 0.5:
                patch = patch[::-1, :]
            if self._rand() < 0.3:
                k = self._randint(0, 4)
                if k:
                    patch = np.rot90(patch, k=k)
            if self._rand() < 0.3:
                gamma = self._uniform(0.8, 1.2)
                patch = np.clip(patch**gamma, 0.0, 1.0)

        tensor = torch.from_numpy(np.ascontiguousarray(patch[None, :, :], dtype=np.float32))
        return {
            "image": tensor,
            "sample_id": str(record["sample_id"]),
            "parent_id": str(record["parent_id"]),
            "class": str(record["class"]),
            "source_path": str(record["absolute_path"]),
            "crop_top": int(top),
            "crop_left": int(left),
        }


__all__ = [
    "BIOSR_REQUIRED_SHAPE",
    "BioSRGT2DDataset",
    "BioSRManifestError",
    "DatasetConfig",
    "GT2DDataset",
    "GT3DDataset",
]

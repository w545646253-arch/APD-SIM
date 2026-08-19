
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Union

import numpy as np
import tifffile as tiff
import torch


def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_image_files(root: Union[str, Path], exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def read_tiff(path: Union[str, Path]) -> np.ndarray:
    """Read a tif/tiff. Returns np.ndarray.
    For multi-page tif, shape can be (T,H,W) or (Z,H,W) or (T,Z,H,W).
    """
    path = str(path)
    arr = tiff.imread(path)
    return np.asarray(arr)


def save_tiff(path: Union[str, Path], arr: np.ndarray, dtype: Optional[np.dtype] = None) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if dtype is not None:
        arr = arr.astype(dtype)
    tiff.imwrite(path, arr)


def normalize_percentile(x: np.ndarray, p_low: float = 0.5, p_high: float = 99.5, eps: float = 1e-8) -> np.ndarray:
    """Robust normalization to [0,1] using percentiles."""
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = (x - lo) / (hi - lo + eps)
    x = np.clip(x, 0.0, 1.0)
    return x


def center_crop_2d(img: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = crop_hw
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return img[..., y0:y0+ch, x0:x0+cw]


def random_crop_2d(img: np.ndarray, crop_hw: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[-2], img.shape[-1]
    ch, cw = crop_hw
    if h < ch or w < cw:
        # pad
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        img = np.pad(img, ((0,0),(0,pad_h),(0,pad_w)) if img.ndim==3 else ((0,pad_h),(0,pad_w)), mode="reflect")
        h, w = img.shape[-2], img.shape[-1]
    y0 = np.random.randint(0, h - ch + 1)
    x0 = np.random.randint(0, w - cw + 1)
    return img[..., y0:y0+ch, x0:x0+cw]


def random_crop_3d(vol: np.ndarray, crop_zyx: Tuple[int, int, int]) -> np.ndarray:
    """vol shape (Z,H,W)."""
    z, h, w = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    cz, ch, cw = crop_zyx
    if z < cz or h < ch or w < cw:
        pad_z = max(0, cz - z)
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        vol = np.pad(vol, ((0,pad_z),(0,pad_h),(0,pad_w)), mode="reflect")
        z, h, w = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    z0 = np.random.randint(0, z - cz + 1)
    y0 = np.random.randint(0, h - ch + 1)
    x0 = np.random.randint(0, w - cw + 1)
    return vol[z0:z0+cz, y0:y0+ch, x0:x0+cw]


def to_torch(x: np.ndarray, device: torch.device, add_batch: bool = True) -> torch.Tensor:
    """Convert numpy to torch float32 tensor."""
    t = torch.from_numpy(x).float().to(device)
    if add_batch:
        t = t.unsqueeze(0)
    return t

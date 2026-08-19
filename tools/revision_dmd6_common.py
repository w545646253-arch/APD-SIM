"""Shared, audit-friendly utilities for the DMD-6F revision experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "outputs/OFFICIAL_BASELINES_DMD6_R2_20260813_162020"
REVISION_ROOT = BASELINE_ROOT / "17_revision_experiments"
PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"
RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
PIXEL_SIZE_UM = 6.5 / 60.0
DEFAULT_SEED = 20260812
MCSIM_PYTHON = Path(r"data/external_input")
MLSIM_SOURCE = ROOT / "external/official_r2_worktrees/mlsim"
MCSIM_SOURCE = ROOT / "external/official_r2_worktrees/mcsim"
MCSIM_CALIBRATION = REVISION_ROOT / "01_matched_baselines/validation_calibration.json"


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None, *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("fields are required for an empty CSV")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_gt(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"GT must be a single-channel 2-D image, got {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("GT contains NaN or Inf")
    low, high = np.percentile(array, (0.5, 99.5))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise ValueError("GT is constant or has an invalid intensity range")
    return np.ascontiguousarray(np.clip((array - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def metrics_module() -> Any:
    path = BASELINE_ROOT / "11_metrics/official_r2_common_metrics.py"
    spec = importlib.util.spec_from_file_location("revision_official_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gt_frc(reference: np.ndarray, reconstruction: np.ndarray, *, pixel_size_um: float = PIXEL_SIZE_UM) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Frozen GT-referenced FRC: 5% crop, Tukey alpha=.20, 100 nonzero annuli."""
    ref = np.asarray(reference, dtype=np.float64)
    rec = np.asarray(reconstruction, dtype=np.float64)
    if ref.shape != rec.shape or ref.ndim != 2 or not np.isfinite(ref).all() or not np.isfinite(rec).all():
        raise ValueError("GT-FRC requires finite, shape-matched 2-D arrays")
    crop_y = int(math.floor(ref.shape[0] * 0.05))
    crop_x = int(math.floor(ref.shape[1] * 0.05))
    ref = ref[crop_y: ref.shape[0] - crop_y, crop_x: ref.shape[1] - crop_x]
    rec = rec[crop_y: rec.shape[0] - crop_y, crop_x: rec.shape[1] - crop_x]
    from scipy.signal.windows import tukey
    window = np.outer(tukey(ref.shape[0], alpha=0.20), tukey(ref.shape[1], alpha=0.20))
    fa = np.fft.fftshift(np.fft.fft2((ref - ref.mean()) * window))
    fb = np.fft.fftshift(np.fft.fft2((rec - rec.mean()) * window))
    fy = np.fft.fftshift(np.fft.fftfreq(ref.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(ref.shape[1]))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    edges = np.linspace(0.0, 0.5, 101, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(100, np.nan, dtype=np.float64)
    counts = np.zeros(100, dtype=np.int64)
    cross = fa * np.conj(fb)
    pa, pb = np.abs(fa) ** 2, np.abs(fb) ** 2
    for index in range(100):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        counts[index] = int(mask.sum())
        denominator = math.sqrt(float(pa[mask].sum()) * float(pb[mask].sum()))
        if denominator > 0.0:
            values[index] = float(np.clip(np.real(cross[mask].sum()) / denominator, -1.0, 1.0))
    threshold = 1.0 / 7.0
    cutoff = None
    crossing = None
    for left in range(99):
        a, b = values[left], values[left + 1]
        if np.isfinite(a) and np.isfinite(b) and a >= threshold and b < threshold:
            cutoff = float(centers[left] + (threshold - a) * (centers[left + 1] - centers[left]) / (b - a))
            crossing = [left, left + 1]
            break
    right_censored = cutoff is None and bool(np.isfinite(values).all()) and bool(np.all(values >= threshold))
    unresolved = cutoff is None and not right_censored
    cutoff_for_auc = 0.5 if cutoff is None else cutoff
    valid_auc = np.isfinite(values) & (centers <= cutoff_for_auc)
    auc = float(np.trapz(np.clip(values[valid_auc], -1.0, 1.0), centers[valid_auc])) if valid_auc.sum() > 1 else float("nan")
    period_px = None if cutoff is None else float(1.0 / cutoff)
    period_um = None if period_px is None else float(period_px * pixel_size_um)
    meta = {
        "frc_type": "GT-referenced FRC",
        "crop_each_edge_fraction": 0.05,
        "mean_centering": True,
        "window": "separable 2D Tukey",
        "tukey_alpha": 0.20,
        "radial_annuli": 100,
        "smoothing": "none",
        "threshold": threshold,
        "crossing_rule": "first downward crossing with adjacent-bin linear interpolation",
        "crossing_bins": crossing,
        "cutoff_cycles_per_pixel": cutoff,
        "cutoff_derived_spatial_period_px": period_px,
        "cutoff_derived_spatial_period_um": period_um,
        "right_censored_at_nyquist": right_censored,
        "unresolved_no_crossing": unresolved,
        "frc_auc_to_cutoff_or_nyquist": auc,
        "pixel_size_um": pixel_size_um,
    }
    return meta, {"frequency_cycles_per_pixel": centers, "frc": values, "count": counts}


def generate_dmd6_raw(gt: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from unisim.protocols import protocol_registry
    from unisim.sim_forward_2d import SIM2DConfig, forward_protocol_sim_2d, nominal_theta_2d

    config = json.loads((ROOT / "configs/apd_dmd_r2/train6_formal.json").read_text(encoding="utf-8"))
    values = dict(config["forward"])
    allowed = set(SIM2DConfig.__dataclass_fields__)
    values = {key: value for key, value in values.items() if key in allowed}
    for key in tuple(values):
        if key.startswith("rand_") and isinstance(values[key], list):
            values[key] = tuple(float(x) for x in values[key])
    sim_config = SIM2DConfig(**values)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(np.ascontiguousarray(gt))[None, None].to(device=device, dtype=torch.float32)
    theta = nominal_theta_2d(sim_config, device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        raw, _ = forward_protocol_sim_2d(tensor, sim_config, PROTOCOL_ID, theta=dict(theta), randomize=False, noise_generator=generator)
    result = np.ascontiguousarray(raw[0].cpu().numpy(), dtype=np.float32)
    spec = protocol_registry.require(PROTOCOL_ID)
    if result.shape[0] != 6 or tuple(spec.raw_frame_order) != RAW_ORDER or not np.isfinite(result).all():
        raise RuntimeError("formal DMD-6F generation contract failed")
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "raw_frame_order": list(RAW_ORDER),
        "seed": int(seed),
        "raw_shape": list(result.shape),
        "raw_dtype": str(result.dtype),
        "raw_stack_sha256": array_sha256(result),
        "forward_config_path": str((ROOT / "configs/apd_dmd_r2/train6_formal.json").resolve()),
        "forward_config_sha256": sha256_file(ROOT / "configs/apd_dmd_r2/train6_formal.json"),
    }
    return result, receipt


def apd6_reconstruct(raw: np.ndarray, seed: int, *, stage2: bool = True) -> np.ndarray:
    import torch
    from unisim.revision_r1.physmap6_core import RefinementConfig, masked_refine
    from unisim.revision_r1.physmap6_pipeline import load_stage1_registered, make_sim_config, stage1_reconstruct_registered
    from unisim.sim_forward_2d import embed_raw_to_slots_2d, nominal_theta_2d

    config_path = ROOT / "configs/apd_dmd_r2/train6_formal.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "checkpoints/apd_dmd_geometry_r2/dmd6/best_checkpoint_receipt.json").read_text(encoding="utf-8"))
    checkpoint = Path(receipt["checkpoint_path"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, scheduler, _ = load_stage1_registered(config, checkpoint, receipt["checkpoint_sha256"], device, protocol_id=PROTOCOL_ID)
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32)
    x_ws, _, _ = stage1_reconstruct_registered(raw_tensor, model, scheduler, protocol_id=PROTOCOL_ID, seed=int(seed))
    if not stage2:
        return np.ascontiguousarray(x_ws[0, 0].detach().cpu().numpy(), dtype=np.float32)
    sim_config = make_sim_config(config)
    theta = nominal_theta_2d(sim_config, device)
    _, mask = embed_raw_to_slots_2d(raw_tensor, PROTOCOL_ID)
    geometry = {"protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH, "raw_frame_order": list(RAW_ORDER)}
    result = masked_refine(x_ws, raw_tensor, mask[0, :, 0, 0], geometry, {"sim_config": sim_config, "theta": theta}, RefinementConfig())
    output = np.ascontiguousarray(result.final_reconstruction[0, 0].detach().cpu().numpy(), dtype=np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("APD-SIM-6 returned non-finite values")
    return output


def mlsim_reconstruct(raw: np.ndarray, checkpoint: Path) -> np.ndarray:
    import torch
    from official_r2_adapters import build_mlsim_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_mlsim_model(MLSIM_SOURCE)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        output = model(torch.from_numpy(np.ascontiguousarray(raw))[None].to(device=device, dtype=torch.float32))
    array = np.ascontiguousarray(output[0, 0].detach().cpu().numpy(), dtype=np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError("ML-SIM-6R returned non-finite values")
    return array


def harmonize(method: str, native: np.ndarray, calibration: Mapping[str, Any] | None = None) -> np.ndarray:
    array = np.asarray(native, dtype=np.float32)
    if method == "mcSIM-Wiener-6":
        if array.ndim != 2 or array.shape[0] % 2 or array.shape[1] % 2:
            raise ValueError(f"mcSIM native output must be an even 2-D grid, got {array.shape}")
        # Official mcSIM reconstructs on a fixed two-times super-resolved grid.
        # Map it to the frozen GT evaluation support with a source-independent
        # 2x2 area average; native output remains preserved separately.
        array = array.reshape(array.shape[0] // 2, 2, array.shape[1] // 2, 2).mean(axis=(1, 3))
        if calibration is None:
            calibration = json.loads(MCSIM_CALIBRATION.read_text(encoding="utf-8"))["methods"][method]
        array = float(calibration["slope"]) * array + float(calibration["intercept"])
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)


def fit_global_affine(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> dict[str, float]:
    sum_x = sum_y = sum_xx = sum_xy = 0.0
    count = 0
    for prediction, target in zip(predictions, targets):
        x = np.asarray(prediction, dtype=np.float64).ravel()
        y = np.asarray(target, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError("calibration arrays do not match")
        sum_x += float(x.sum()); sum_y += float(y.sum())
        sum_xx += float(np.dot(x, x)); sum_xy += float(np.dot(x, y)); count += x.size
    denominator = count * sum_xx - sum_x * sum_x
    if count == 0 or denominator == 0.0:
        raise ValueError("global affine calibration is singular")
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return {"slope": float(slope), "intercept": float(intercept), "pixel_count": int(count)}

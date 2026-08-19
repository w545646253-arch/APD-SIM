#!/usr/bin/env python3
"""Thin, auditable adapters for official DMD-6F baseline sources.

The module deliberately contains no reconstruction algorithm.  It validates a
frozen six-frame bundle, performs the one-to-one H/V reshape required by an
official implementation, and delegates to the pristine/worktree source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


RAW_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"


def canonical_array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": a.dtype.str, "shape": list(a.shape)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + a.tobytes(order="C")).hexdigest()


def load_frozen_stack(path: pathlib.Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load exactly six observed frames and validate their frozen metadata."""
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("shared bundle item must be an .npz file")
    with np.load(path, allow_pickle=False) as data:
        key = "raw_stack" if "raw_stack" in data else "raw" if "raw" in data else None
        if key is None:
            raise ValueError("bundle item lacks raw_stack array")
        raw = np.asarray(data[key], dtype=np.float32)
        meta = {
            "raw_frame_order": [str(x) for x in data["frame_order"].tolist()],
            "protocol_id": str(data["protocol_id"].reshape(-1)[0]),
            "protocol_hash": str(data["protocol_hash"].reshape(-1)[0]),
            "raw_stack_sha256": canonical_array_sha256(raw),
        }
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        declared = json.loads(sidecar.read_text(encoding="utf-8"))
        for field in ("raw_frame_order", "protocol_id", "protocol_hash", "raw_stack_sha256"):
            if field in declared:
                meta[field] = declared[field]
    if raw.ndim != 3 or raw.shape[0] != 6:
        raise ValueError(f"expected [6,H,W], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw stack contains non-finite values")
    if tuple(meta.get("raw_frame_order", ())) != RAW_ORDER:
        raise ValueError("raw-frame order mismatch")
    if meta.get("protocol_id") != PROTOCOL_ID or meta.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("DMD-6F protocol mismatch")
    actual = canonical_array_sha256(raw)
    if meta.get("raw_stack_sha256") != actual:
        raise ValueError("raw-stack hash mismatch")
    return raw, meta


def to_orientation_phase(raw: np.ndarray) -> np.ndarray:
    """Lossless view/copy: [H0,H120,H240,V0,V120,V240] -> [2,3,H,W]."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 6:
        raise ValueError("DMD-6F input must have exactly six frames")
    shaped = np.ascontiguousarray(raw.reshape(2, 3, *raw.shape[-2:]))
    if not np.array_equal(shaped.reshape(raw.shape), raw):
        raise RuntimeError("roundtrip failed; adapter may not synthesize observed frames")
    return shaped


def mlsim_model_options() -> SimpleNamespace:
    """Frozen RCAN definition from the official ML-SIM publication command."""
    return SimpleNamespace(
        model="rcan", nch_in=6, nch_out=1, n_resgroups=2, n_resblocks=5,
        n_feats=48, reduction=16, narch=0, scale=1, task="simin_gtout", cpu=True,
    )


def build_mlsim_model(source_root: pathlib.Path):
    """Instantiate the official RCAN with the officially parameterized 6 inputs."""
    source_root = source_root.resolve()
    module_path = source_root / "models.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("official_mlsim_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GetModel(mlsim_model_options())


def reconstruct_mcsim_cpu(
    raw: np.ndarray,
    source_root: pathlib.Path,
    *,
    pixel_size_um: float,
    wavelength_um: float = 0.488,
    na: float = 1.4,
    wiener_parameter: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Delegate a 2O3P stack to official mcSIM's CPU Wiener path."""
    shaped = to_orientation_phase(raw)
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    try:
        from mcsim.analysis.sim_reconstruction import SimImageSet  # type: ignore
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
    phases = np.tile(np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]), (2, 1))
    # Physical carrier guesses follow H (fy) then V (fx), in cycles/um.
    # The frozen APD forward contract defines k_ratio_xy relative to
    # k_max=NA/lambda (not the incoherent 2*NA/lambda cutoff).
    frq = 0.85 * (na / wavelength_um)
    frq_guess = np.array([[0.0, frq], [frq, 0.0]], dtype=float)
    sim = SimImageSet.initialize(
        physical_params={"pixel_size": pixel_size_um, "na": na, "wavelength": wavelength_um},
        imgs=shaped,
        wiener_parameter=wiener_parameter,
        frq_estimation_mode="band-correlation",
        frq_guess=frq_guess,
        phase_estimation_mode="wicker-iterative",
        phases_guess=phases,
        combine_bands_mode="fairSIM",
        background=0.02,
        use_gpu=False,
        print_to_terminal=False,
    )
    sim.reconstruct()
    output = np.asarray(sim.sim_sr.compute() if hasattr(sim.sim_sr, "compute") else sim.sim_sr, dtype=np.float32)
    output = np.squeeze(output)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise RuntimeError(f"official mcSIM returned invalid output {output.shape}")
    receipt = {
        "implementation": "QI2lab/mcSIM SimImageSet",
        "input_shape": list(shaped.shape), "input_hash": canonical_array_sha256(raw),
        "output_shape": list(output.shape), "output_hash": canonical_array_sha256(output),
        "use_gpu": False, "wiener_parameter": wiener_parameter,
        "phase_estimation_mode": "wicker-iterative", "carrier_estimation_mode": "band-correlation",
    }
    return output, receipt


def static_smoke() -> dict[str, Any]:
    sample = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    shaped = to_orientation_phase(sample)
    return {
        "status": "PASS", "protocol_id": PROTOCOL_ID, "raw_order": list(RAW_ORDER),
        "input_shape": list(sample.shape), "orientation_phase_shape": list(shaped.shape),
        "roundtrip_identical": bool(np.array_equal(shaped.reshape(sample.shape), sample)),
        "observed_frames_created_or_removed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-smoke", action="store_true")
    parser.add_argument("--bundle-item", type=pathlib.Path)
    args = parser.parse_args()
    if args.static_smoke:
        print(json.dumps(static_smoke(), indent=2))
        return 0
    if args.bundle_item:
        raw, meta = load_frozen_stack(args.bundle_item)
        print(json.dumps({"status": "PASS", "shape": list(raw.shape), "raw_stack_sha256": meta["raw_stack_sha256"]}, indent=2))
        return 0
    parser.error("choose --static-smoke or --bundle-item")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

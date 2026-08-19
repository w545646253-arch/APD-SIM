from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from revision_dmd6_common import gt_frc, harmonize, normalize_gt


def test_normalize_gt_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="single-channel 2-D"):
        normalize_gt(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="constant"):
        normalize_gt(np.ones((32, 32), dtype=np.float32))
    invalid = np.zeros((32, 32), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_gt(invalid)


def test_mcsim_harmonization_is_fixed_area_map_then_global_affine() -> None:
    native = np.arange(64, dtype=np.float32).reshape(8, 8) / 63.0
    calibration = {"slope": 0.5, "intercept": 0.125}
    actual = harmonize("mcSIM-Wiener-6", native, calibration)
    expected = native.reshape(4, 2, 4, 2).mean(axis=(1, 3)) * 0.5 + 0.125
    assert actual.shape == (4, 4)
    np.testing.assert_array_equal(actual, np.clip(expected, 0.0, 1.0).astype(np.float32))


def test_gt_frc_frozen_protocol_and_nyquist_censor() -> None:
    rng = np.random.default_rng(20260812)
    image = rng.normal(size=(128, 128)).astype(np.float32)
    result, curve = gt_frc(image, image, pixel_size_um=0.1)
    assert result["frc_type"] == "GT-referenced FRC"
    assert result["crop_each_edge_fraction"] == 0.05
    assert result["tukey_alpha"] == 0.20
    assert result["radial_annuli"] == 100
    assert result["threshold"] == pytest.approx(1.0 / 7.0)
    assert result["right_censored_at_nyquist"] is True
    assert result["cutoff_cycles_per_pixel"] is None
    assert curve["frc"].shape == (100,)
    assert np.nanmin(curve["frc"]) > 0.999999


def test_single_input_script_has_required_defaults_and_no_plotting_imports() -> None:
    source_path = ROOT / "compare_single_dmd6.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "matplotlib" not in imported
    assert "DEFAULT_SEED = 20260812" in source
    assert "microtubules_Cell_046_SIM_gt.tif" in source
    assert '"APD-SIM-6"' in source
    assert '"ML-SIM-6R"' in source
    assert '"mcSIM-Wiener-6"' in source
    assert "raw_dmd6_stack.npy" in source
    assert "selection_receipt.json" in source


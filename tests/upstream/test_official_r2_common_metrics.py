from __future__ import annotations

import math

import numpy as np
import pytest

from tools.official_r2_common_metrics import (
    DEFAULT_BOOTSTRAP_SEED,
    MetricValidationError,
    class_summaries,
    holm_correction,
    paired_wilcoxon,
    parent_image_bootstrap_ci,
    psnr_native,
    reference_frc_1over7,
    ssim_native,
)


def test_psnr_ssim_known_values_and_alias_contract() -> None:
    reference = np.zeros((8, 8), dtype=np.float32)
    reconstruction = np.full((8, 8), 0.5, dtype=np.float32)
    assert psnr_native(reference, reconstruction) == pytest.approx(
        10.0 * math.log10(4.0)
    )
    assert math.isinf(psnr_native(reference, reference.copy()))
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    assert ssim_native(ramp, ramp.copy()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reference", "reconstruction", "match"),
    [
        (np.zeros((8, 8)), np.zeros((7, 8)), "shapes differ"),
        (np.full((8, 8), np.nan), np.zeros((8, 8)), "NaN or Inf"),
        (np.zeros((8, 8)), np.full((8, 8), 1.01), "outside"),
        (np.zeros((8, 8), dtype=complex), np.zeros((8, 8)), "real numeric"),
    ],
)
def test_image_metrics_fail_closed_on_invalid_arrays(
    reference: np.ndarray, reconstruction: np.ndarray, match: str
) -> None:
    with pytest.raises(MetricValidationError, match=match):
        psnr_native(reference, reconstruction)


def test_image_metrics_require_explicit_valid_range_and_ssim_window() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(MetricValidationError, match="strictly positive"):
        psnr_native(image, image, data_range=0.0)
    with pytest.raises(MetricValidationError, match="odd integer"):
        ssim_native(image, image, win_size=4)
    with pytest.raises(MetricValidationError, match="at least 3"):
        ssim_native(np.zeros((2, 8)), np.zeros((2, 8)))


def _low_pass_pair(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0.0, 1.0, size=(128, 128))
    fy = np.fft.fftfreq(128)[:, None]
    fx = np.fft.fftfreq(128)[None, :]
    low_pass = np.exp(-((fx * fx + fy * fy) / (2.0 * 0.08**2)))
    reconstruction = np.fft.ifft2(np.fft.fft2(reference) * low_pass).real
    reconstruction = (reconstruction - reconstruction.min()) / (
        reconstruction.max() - reconstruction.min()
    )
    return reference, reconstruction


def test_frc_first_downward_crossing_excludes_dc() -> None:
    reference, reconstruction = _low_pass_pair()
    metadata, curves = reference_frc_1over7(
        reference,
        reconstruction,
        apod_px=10,
        min_samples_per_bin=8,
    )
    assert metadata["dc_excluded"] is True
    assert metadata["cutoff_rule"].startswith("first downward")
    assert metadata["right_censored_at_nyquist"] is False
    assert metadata["unresolved_no_crossing"] is False
    assert 0.0 < metadata["cutoff_cycles_per_pixel"] < 0.5
    left, right = metadata["crossing_bins"]
    assert left > 0
    assert curves["frc"][left] >= 1.0 / 7.0
    assert curves["frc"][right] < 1.0 / 7.0


def test_frc_identity_is_right_censored_without_fabricated_cutoff() -> None:
    rng = np.random.default_rng(11)
    reference = rng.uniform(0.0, 1.0, size=(128, 128))
    metadata, _curves = reference_frc_1over7(
        reference,
        reference.copy(),
        apod_px=10,
        min_samples_per_bin=8,
    )
    assert metadata["right_censored_at_nyquist"] is True
    assert metadata["unresolved_no_crossing"] is False
    assert metadata["cutoff_cycles_per_pixel"] is None
    assert metadata["cutoff_derived_spatial_period_px"] is None


def test_frc_rejects_invalid_apodization_and_nonfinite_input() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(MetricValidationError, match="half the shortest"):
        reference_frc_1over7(image, image, apod_px=9, min_samples_per_bin=1)
    bad = image.copy()
    bad[0, 0] = np.inf
    with pytest.raises(MetricValidationError, match="NaN or Inf"):
        reference_frc_1over7(bad, image, apod_px=2, min_samples_per_bin=1)


def test_parent_bootstrap_uses_parent_not_row_and_is_deterministic() -> None:
    # P1 has ten rows and must still have the same inferential weight as P2/P3.
    a = np.asarray([1.0] * 10 + [3.0, 5.0])
    b = np.zeros_like(a)
    parents = ["P1"] * 10 + ["P2", "P3"]
    classes = ["A"] * 11 + ["B"]
    first = parent_image_bootstrap_ci(
        a,
        b,
        parents,
        class_labels=classes,
        n_resamples=2_000,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    second = parent_image_bootstrap_ci(
        a,
        b,
        parents,
        class_labels=classes,
        n_resamples=2_000,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(3.0)
    assert first["n_rows"] == 12
    assert first["n_parent_images"] == 3
    assert first["parent_differences"] == [1.0, 3.0, 5.0]
    assert first["parent_class_counts"] == {"A": 2, "B": 1}
    assert first["stratified_by_class"] is True


def test_parent_bootstrap_rejects_nonfinite_and_cross_class_parent() -> None:
    with pytest.raises(MetricValidationError, match="NaN or Inf"):
        parent_image_bootstrap_ci([1.0, np.nan], [0.0, 0.0], ["P1", "P2"])
    with pytest.raises(MetricValidationError, match="multiple classes"):
        parent_image_bootstrap_ci(
            [1.0, 2.0],
            [0.0, 0.0],
            ["P1", "P1"],
            class_labels=["A", "B"],
        )


def test_two_sided_wilcoxon_and_all_zero_policy() -> None:
    result = paired_wilcoxon([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert result["alternative"] == "two-sided"
    assert result["n_nonzero_differences"] == 3
    assert result["p_value"] == pytest.approx(0.25)
    zeros = paired_wilcoxon([1.0, 2.0], [1.0, 2.0])
    assert zeros["statistic"] == 0.0
    assert zeros["p_value"] == 1.0
    assert zeros["n_zero_differences"] == 2


def test_wilcoxon_can_aggregate_repeated_rows_by_parent() -> None:
    result = paired_wilcoxon(
        [1.0, 3.0, -2.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        parent_ids=["P1", "P1", "P2", "P3"],
    )
    assert result["unit"] == "parent_image"
    assert result["n_rows"] == 4
    assert result["n_parent_images"] == 3
    assert result["n_positive_differences"] == 2
    assert result["n_negative_differences"] == 1


def test_holm_correction_is_monotone_and_preserves_input_order() -> None:
    result = holm_correction({"m1": 0.01, "m2": 0.04, "m3": 0.03}, alpha=0.05)
    assert result["ordered_labels_by_raw_p"] == ["m1", "m3", "m2"]
    by_label = {item["label"]: item for item in result["results"]}
    assert by_label["m1"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert by_label["m2"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert by_label["m3"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert by_label["m1"]["reject_at_alpha"] is True
    assert by_label["m2"]["reject_at_alpha"] is False


@pytest.mark.parametrize("p_values", [[], [0.1, np.nan], [-0.1], [1.1]])
def test_holm_rejects_empty_nonfinite_or_out_of_range_p_values(p_values: list[float]) -> None:
    with pytest.raises(MetricValidationError):
        holm_correction(p_values)


def test_class_summaries_parent_unit_and_order() -> None:
    result = class_summaries(
        [1.0, 3.0, 2.0, 4.0],
        ["CCP", "CCP", "ER", "ER"],
        parent_ids=["P1", "P1", "P2", "P3"],
        class_order=["CCP", "ER"],
    )
    assert result["unit"] == "parent_image"
    assert result["n_rows"] == 4
    assert result["n_parent_images"] == 3
    assert list(result["classes"]) == ["CCP", "ER"]
    assert result["classes"]["CCP"]["mean"] == pytest.approx(2.0)
    assert result["classes"]["CCP"]["n"] == 1
    assert result["classes"]["CCP"]["sd"] is None
    assert result["classes"]["CCP"]["n_rows"] == 2
    assert result["classes"]["ER"]["mean"] == pytest.approx(3.0)
    assert result["classes"]["ER"]["sd"] == pytest.approx(math.sqrt(2.0))


def test_class_summaries_fail_closed_on_bad_order_and_parent_class() -> None:
    with pytest.raises(MetricValidationError, match="each and only"):
        class_summaries([1.0, 2.0], ["A", "B"], class_order=["A", "C"])
    with pytest.raises(MetricValidationError, match="multiple classes"):
        class_summaries(
            [1.0, 2.0],
            ["A", "B"],
            parent_ids=["P1", "P1"],
        )

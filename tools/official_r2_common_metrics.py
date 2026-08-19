"""Common, fail-closed metrics for the official DMD-6F R2 comparison.

The functions in this module operate only on arrays and metadata supplied by
the caller.  They do not discover data, read manifests, select methods, or
write results.  This keeps the sealed-test boundary and the metric definition
separate.

All image metrics require matching, finite, real-valued 2-D arrays inside the
explicit interval ``[data_min, data_min + data_range]``.  Statistical helpers
likewise reject missing or non-finite values instead of silently dropping
pairs.  Returned mappings contain only JSON-serializable scalar/list values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any, Literal

import numpy as np


DEFAULT_BOOTSTRAP_SEED = 20260813
FRC_THRESHOLD = 1.0 / 7.0


class MetricValidationError(ValueError):
    """Raised when a metric/statistical contract is not satisfied."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MetricValidationError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetricValidationError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise MetricValidationError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MetricValidationError(f"{name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise MetricValidationError(f"{name} must be {relation}")
    return result


def _data_interval(data_range: Any, data_min: Any) -> tuple[float, float, float]:
    width = _finite_float(data_range, "data_range")
    lower = _finite_float(data_min, "data_min")
    if width <= 0.0:
        raise MetricValidationError("data_range must be strictly positive")
    upper = lower + width
    if not math.isfinite(upper):
        raise MetricValidationError("data_min + data_range must be finite")
    return width, lower, upper


def _numeric_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise MetricValidationError(
            f"{name} must be {ndim}-D, got shape {tuple(array.shape)}"
        )
    if array.size == 0:
        raise MetricValidationError(f"{name} must not be empty")
    if array.dtype.kind not in "iuf":
        raise MetricValidationError(
            f"{name} must have a real numeric dtype, got {array.dtype}"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise MetricValidationError(f"{name} contains NaN or Inf")
    return result


def _image_pair(
    reference: Any,
    reconstruction: Any,
    *,
    data_range: Any,
    data_min: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    width, lower, upper = _data_interval(data_range, data_min)
    ref = _numeric_array(reference, "reference", ndim=2)
    rec = _numeric_array(reconstruction, "reconstruction", ndim=2)
    if ref.shape != rec.shape:
        raise MetricValidationError(
            f"image shapes differ: reference={ref.shape}, reconstruction={rec.shape}"
        )
    for name, array in (("reference", ref), ("reconstruction", rec)):
        observed_min = float(np.min(array))
        observed_max = float(np.max(array))
        if observed_min < lower or observed_max > upper:
            raise MetricValidationError(
                f"{name} values [{observed_min}, {observed_max}] fall outside "
                f"the declared interval [{lower}, {upper}]"
            )
    return ref, rec, width


def psnr_native(
    reference: Any,
    reconstruction: Any,
    *,
    data_range: float = 1.0,
    data_min: float = 0.0,
) -> float:
    """Return full-field PSNR on the declared native intensity interval."""

    ref, rec, width = _image_pair(
        reference, reconstruction, data_range=data_range, data_min=data_min
    )
    mse = float(np.mean(np.square(ref - rec), dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    result = float(10.0 * math.log10((width * width) / mse))
    if not math.isfinite(result):
        raise MetricValidationError("PSNR computation produced a non-finite value")
    return result


def ssim_native(
    reference: Any,
    reconstruction: Any,
    *,
    data_range: float = 1.0,
    data_min: float = 0.0,
    win_size: int | None = None,
) -> float:
    """Return full-field single-channel SSIM with an explicit data range."""

    ref, rec, width = _image_pair(
        reference, reconstruction, data_range=data_range, data_min=data_min
    )
    shortest_side = int(min(ref.shape))
    if win_size is None:
        if shortest_side < 3:
            raise MetricValidationError("SSIM requires both image dimensions to be at least 3")
        selected_win = min(7, shortest_side if shortest_side % 2 else shortest_side - 1)
    else:
        selected_win = _positive_int(win_size, "win_size")
        if selected_win < 3 or selected_win % 2 == 0:
            raise MetricValidationError("win_size must be an odd integer of at least 3")
        if selected_win > shortest_side:
            raise MetricValidationError("win_size must not exceed the shortest image dimension")

    from skimage.metrics import structural_similarity

    result = float(
        structural_similarity(
            ref,
            rec,
            data_range=width,
            win_size=selected_win,
            channel_axis=None,
        )
    )
    if not math.isfinite(result):
        raise MetricValidationError("SSIM computation produced a non-finite value")
    return result


# Concise aliases for callers that do not use the historical ``_native`` names.
psnr = psnr_native
ssim = ssim_native


def _cosine_edge_window(length: int, edge_pixels: int) -> np.ndarray:
    window = np.ones(length, dtype=np.float64)
    if edge_pixels == 0:
        return window
    ramp = 0.5 - 0.5 * np.cos(
        np.linspace(0.0, np.pi, edge_pixels + 1, dtype=np.float64)
    )
    window[:edge_pixels] = ramp[:-1]
    window[-edge_pixels:] = ramp[:-1][::-1]
    return window


def reference_frc_1over7(
    reference: Any,
    reconstruction: Any,
    *,
    data_range: float = 1.0,
    data_min: float = 0.0,
    apod_px: int = 20,
    min_samples_per_bin: int = 64,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute GT-referenced radial FRC using a prespecified 1/7 cutoff.

    Radial bins are one Fourier pixel wide.  DC is excluded from cutoff
    finding, the curve is not smoothed, and the cutoff is the first adjacent
    above-to-below 1/7 crossing with linear interpolation.  If uninterrupted
    above-threshold support reaches the terminal Nyquist annulus, the result is
    right-censored and no artificial numerical cutoff or period is assigned.
    Other no-crossing shapes are marked unresolved.
    """

    ref, rec, _ = _image_pair(
        reference, reconstruction, data_range=data_range, data_min=data_min
    )
    edge = _positive_int(apod_px, "apod_px", allow_zero=True)
    minimum_samples = _positive_int(min_samples_per_bin, "min_samples_per_bin")
    if edge > min(ref.shape) // 2:
        raise MetricValidationError("apod_px must not exceed half the shortest dimension")

    height, width = (int(value) for value in ref.shape)
    window = np.outer(
        _cosine_edge_window(height, edge), _cosine_edge_window(width, edge)
    )
    a = (ref - float(np.mean(ref))) * window
    b = (rec - float(np.mean(rec))) * window
    fa = np.fft.fftshift(np.fft.fft2(a))
    fb = np.fft.fftshift(np.fft.fft2(b))
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(np.square(xx) + np.square(yy))
    bin_width = 1.0 / float(max(height, width))
    bin_index = np.floor(radius / bin_width + 1e-12).astype(np.int64)
    # The bin containing frequencies immediately below 0.5 is the terminal
    # Nyquist annulus for both even and odd dimensions.
    terminal_bin = int(math.floor(np.nextafter(0.5, 0.0) / bin_width))

    frequency = np.full(terminal_bin + 1, np.nan, dtype=np.float64)
    frc = np.full(terminal_bin + 1, np.nan, dtype=np.float64)
    counts = np.zeros(terminal_bin + 1, dtype=np.int64)
    cross = fa * np.conj(fb)
    power_a = np.square(np.abs(fa))
    power_b = np.square(np.abs(fb))
    nyquist_support = radius <= 0.5 + 1e-12
    for index in range(terminal_bin + 1):
        mask = (bin_index == index) & nyquist_support
        count = int(np.count_nonzero(mask))
        counts[index] = count
        if count < minimum_samples:
            continue
        denominator = math.sqrt(
            float(np.sum(power_a[mask], dtype=np.float64))
            * float(np.sum(power_b[mask], dtype=np.float64))
        )
        frequency[index] = float(np.mean(radius[mask], dtype=np.float64))
        if denominator > 0.0:
            value = float(np.real(np.sum(cross[mask])) / denominator)
            frc[index] = float(np.clip(value, -1.0, 1.0))

    valid = np.flatnonzero(
        np.isfinite(frequency)
        & np.isfinite(frc)
        & (np.arange(len(frequency), dtype=np.int64) > 0)
    )
    cutoff: float | None = None
    crossing_bins: list[int] | None = None
    for left, right in zip(valid[:-1], valid[1:]):
        if int(right) != int(left) + 1:
            continue
        value_left = float(frc[left])
        value_right = float(frc[right])
        if value_left >= FRC_THRESHOLD and value_right < FRC_THRESHOLD:
            frequency_left = float(frequency[left])
            frequency_right = float(frequency[right])
            cutoff = frequency_left + (FRC_THRESHOLD - value_left) * (
                frequency_right - frequency_left
            ) / (value_right - value_left)
            crossing_bins = [int(left), int(right)]
            break

    reaches_terminal = bool(valid.size and int(valid[-1]) == terminal_bin)
    starts_above = bool(valid.size and float(frc[valid[0]]) >= FRC_THRESHOLD)
    contiguous = bool(valid.size and np.all(np.diff(valid) == 1))
    all_above = bool(valid.size and np.all(frc[valid] >= FRC_THRESHOLD))
    right_censored = bool(
        cutoff is None and reaches_terminal and starts_above and contiguous and all_above
    )
    unresolved = bool(cutoff is None and not right_censored)
    period = None if cutoff is None else float(1.0 / cutoff)
    metadata: dict[str, Any] = {
        "frc_type": "GT-referenced radial FRC",
        "threshold": FRC_THRESHOLD,
        "cutoff_rule": "first downward crossing after excluded DC, linear interpolation",
        "dc_excluded": True,
        "apodization": "separable cosine edge window",
        "apod_px": edge,
        "bin_width_cycles_per_pixel": bin_width,
        "min_samples_per_bin": minimum_samples,
        "smoothing": "none",
        "nyquist_cycles_per_pixel": 0.5,
        "terminal_annulus_bin": terminal_bin,
        "terminal_ring_reaches_nyquist": reaches_terminal,
        "right_censored_at_nyquist": right_censored,
        "unresolved_no_crossing": unresolved,
        "cutoff_cycles_per_pixel": cutoff,
        "cutoff_derived_spatial_period_px": period,
        "crossing_bins": crossing_bins,
        "n_valid_non_dc_bins": int(valid.size),
    }
    curves = {
        "frequency_cycles_per_pixel": frequency,
        "frc": frc,
        "count": counts,
    }
    return metadata, curves


def _numeric_vector(value: Any, name: str) -> np.ndarray:
    return _numeric_array(value, name, ndim=1)


def _labels(value: Any, name: str, expected_length: int) -> list[str]:
    array = np.asarray(value, dtype=object)
    if array.ndim != 1 or len(array) != expected_length:
        raise MetricValidationError(
            f"{name} must be a 1-D sequence of length {expected_length}"
        )
    result: list[str] = []
    for item in array.tolist():
        if not isinstance(item, (str, np.str_)) or not str(item).strip():
            raise MetricValidationError(f"{name} entries must be non-empty strings")
        result.append(str(item))
    return result


def _paired_parent_differences(
    a: Any,
    b: Any,
    parent_ids: Any,
    class_labels: Any | None = None,
) -> tuple[np.ndarray, list[str], list[str] | None, int]:
    left = _numeric_vector(a, "a")
    right = _numeric_vector(b, "b")
    if left.shape != right.shape:
        raise MetricValidationError(f"paired vector shapes differ: a={left.shape}, b={right.shape}")
    parents = _labels(parent_ids, "parent_ids", len(left))
    classes = None if class_labels is None else _labels(class_labels, "class_labels", len(left))

    rows_by_parent: dict[str, list[int]] = defaultdict(list)
    for index, parent in enumerate(parents):
        rows_by_parent[parent].append(index)
    ordered_parents = sorted(rows_by_parent)
    differences = np.asarray(
        [
            float(np.mean(left[rows_by_parent[parent]] - right[rows_by_parent[parent]]))
            for parent in ordered_parents
        ],
        dtype=np.float64,
    )
    parent_classes: list[str] | None = None
    if classes is not None:
        parent_classes = []
        for parent in ordered_parents:
            observed = {classes[index] for index in rows_by_parent[parent]}
            if len(observed) != 1:
                raise MetricValidationError(
                    f"parent {parent!r} maps to multiple classes: {sorted(observed)}"
                )
            parent_classes.append(next(iter(observed)))
    return differences, ordered_parents, parent_classes, len(left)


def parent_image_bootstrap_ci(
    a: Any,
    b: Any,
    parent_ids: Any,
    *,
    class_labels: Any | None = None,
    statistic: Literal["mean", "median"] = "mean",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a paired ``a - b`` CI using parent image as the sampling unit.

    Multiple rows/crops belonging to one parent are first averaged, so parents
    with more rows cannot receive more inferential weight.  When
    ``class_labels`` are supplied, parents are resampled within class while the
    observed number of parents per class is preserved.
    """

    if statistic not in {"mean", "median"}:
        raise MetricValidationError("statistic must be 'mean' or 'median'")
    resamples = _positive_int(n_resamples, "n_resamples")
    bootstrap_seed = _positive_int(seed, "seed", allow_zero=True)
    confidence = _finite_float(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise MetricValidationError("confidence_level must lie strictly between 0 and 1")

    differences, parents, parent_classes, n_rows = _paired_parent_differences(
        a, b, parent_ids, class_labels
    )
    reducer = np.mean if statistic == "mean" else np.median
    estimate = float(reducer(differences))
    strata: dict[str, np.ndarray] = {}
    if parent_classes is None:
        strata["__all__"] = np.arange(len(parents), dtype=np.int64)
    else:
        for class_name in sorted(set(parent_classes)):
            strata[class_name] = np.asarray(
                [index for index, item in enumerate(parent_classes) if item == class_name],
                dtype=np.int64,
            )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_statistics = np.empty(resamples, dtype=np.float64)
    # Chunking bounds memory for large requested resample counts while keeping
    # a deterministic call order and seed.
    chunk_size = 4096
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        count = stop - start
        blocks: list[np.ndarray] = []
        for class_name in sorted(strata):
            indices = strata[class_name]
            sampled_positions = rng.integers(
                0, len(indices), size=(count, len(indices)), endpoint=False
            )
            blocks.append(differences[indices[sampled_positions]])
        sampled = np.concatenate(blocks, axis=1)
        bootstrap_statistics[start:stop] = reducer(sampled, axis=1)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_statistics, [alpha, 1.0 - alpha], method="linear")
    class_counts = (
        None
        if parent_classes is None
        else {
            class_name: int(sum(item == class_name for item in parent_classes))
            for class_name in sorted(set(parent_classes))
        }
    )
    return {
        "contrast": "a_minus_b",
        "statistic": statistic,
        "estimate": estimate,
        "confidence_level": confidence,
        "confidence_interval": [float(low), float(high)],
        "n_rows": n_rows,
        "n_parent_images": len(parents),
        "parent_ids": parents,
        "parent_differences": [float(value) for value in differences],
        "resampling_unit": "parent_image",
        "stratified_by_class": parent_classes is not None,
        "parent_class_counts": class_counts,
        "n_resamples": resamples,
        "seed": bootstrap_seed,
        "quantile_method": "linear",
    }


def paired_wilcoxon(
    a: Any,
    b: Any,
    *,
    parent_ids: Any | None = None,
) -> dict[str, Any]:
    """Return a paired two-sided Wilcoxon signed-rank test for ``a - b``.

    If ``parent_ids`` are provided, repeated rows are averaged within parent
    before testing.  An all-zero contrast is defined as statistic 0 and
    p-value 1 rather than delegated to SciPy's version-dependent error path.
    """

    if parent_ids is None:
        left = _numeric_vector(a, "a")
        right = _numeric_vector(b, "b")
        if left.shape != right.shape:
            raise MetricValidationError(
                f"paired vector shapes differ: a={left.shape}, b={right.shape}"
            )
        differences = left - right
        n_rows = len(left)
        n_parents: int | None = None
        unit = "row"
    else:
        differences, parents, _classes, n_rows = _paired_parent_differences(
            a, b, parent_ids
        )
        n_parents = len(parents)
        unit = "parent_image"

    nonzero = differences != 0.0
    n_nonzero = int(np.count_nonzero(nonzero))
    if n_nonzero == 0:
        statistic_value = 0.0
        p_value = 1.0
    else:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        statistic_value = float(result.statistic)
        p_value = float(result.pvalue)
        if not math.isfinite(statistic_value) or not math.isfinite(p_value):
            raise MetricValidationError("Wilcoxon computation produced a non-finite result")
        if not 0.0 <= p_value <= 1.0:
            raise MetricValidationError("Wilcoxon computation produced a p-value outside [0, 1]")
    return {
        "test": "Wilcoxon signed-rank",
        "alternative": "two-sided",
        "contrast": "a_minus_b",
        "zero_method": "wilcox",
        "continuity_correction": False,
        "method_requested": "auto",
        "unit": unit,
        "n_rows": int(n_rows),
        "n_parent_images": n_parents,
        "n_nonzero_differences": n_nonzero,
        "n_zero_differences": int(len(differences) - n_nonzero),
        "n_positive_differences": int(np.count_nonzero(differences > 0.0)),
        "n_negative_differences": int(np.count_nonzero(differences < 0.0)),
        "statistic": statistic_value,
        "p_value": p_value,
    }


def holm_correction(
    p_values: Mapping[str, Any] | Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply Holm's family-wise correction and preserve input order."""

    significance = _finite_float(alpha, "alpha")
    if not 0.0 < significance < 1.0:
        raise MetricValidationError("alpha must lie strictly between 0 and 1")
    if isinstance(p_values, Mapping):
        if labels is not None:
            raise MetricValidationError("labels must be omitted when p_values is a mapping")
        input_labels = [str(item) for item in p_values.keys()]
        raw_values = list(p_values.values())
    else:
        raw_values = list(p_values)
        if labels is None:
            input_labels = [f"H{index + 1}" for index in range(len(raw_values))]
        else:
            input_labels = _labels(labels, "labels", len(raw_values))
    if not raw_values:
        raise MetricValidationError("p_values must not be empty")
    if len(set(input_labels)) != len(input_labels) or any(not item.strip() for item in input_labels):
        raise MetricValidationError("hypothesis labels must be unique non-empty strings")

    values = _numeric_vector(raw_values, "p_values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise MetricValidationError("p_values must all lie in [0, 1]")
    count = len(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    scale = count - np.arange(count, dtype=np.int64)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_values * scale))
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    results = [
        {
            "label": input_labels[index],
            "p_value": float(values[index]),
            "holm_adjusted_p_value": float(adjusted[index]),
            "reject_at_alpha": bool(adjusted[index] <= significance),
        }
        for index in range(count)
    ]
    return {
        "method": "Holm",
        "alpha": significance,
        "n_hypotheses": count,
        "ordered_labels_by_raw_p": [input_labels[int(index)] for index in order],
        "results": results,
    }


def _descriptive(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise MetricValidationError("descriptive statistics require a non-empty finite vector")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": None if values.size == 1 else float(np.std(values, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def class_summaries(
    values: Any,
    class_labels: Any,
    *,
    parent_ids: Any | None = None,
    class_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return overall and class-stratified descriptive summaries.

    With ``parent_ids``, repeated rows are averaged within parent before the
    summaries are computed.  A parent associated with more than one class is
    rejected.
    """

    vector = _numeric_vector(values, "values")
    classes = _labels(class_labels, "class_labels", len(vector))
    if parent_ids is None:
        units = vector
        unit_classes = classes
        n_parent_images: int | None = None
        unit = "row"
        row_counts = {
            class_name: int(sum(item == class_name for item in classes))
            for class_name in sorted(set(classes))
        }
    else:
        parents = _labels(parent_ids, "parent_ids", len(vector))
        rows_by_parent: dict[str, list[int]] = defaultdict(list)
        for index, parent in enumerate(parents):
            rows_by_parent[parent].append(index)
        ordered_parents = sorted(rows_by_parent)
        parent_values: list[float] = []
        parent_classes: list[str] = []
        for parent in ordered_parents:
            indices = rows_by_parent[parent]
            observed_classes = {classes[index] for index in indices}
            if len(observed_classes) != 1:
                raise MetricValidationError(
                    f"parent {parent!r} maps to multiple classes: {sorted(observed_classes)}"
                )
            parent_values.append(float(np.mean(vector[indices])))
            parent_classes.append(next(iter(observed_classes)))
        units = np.asarray(parent_values, dtype=np.float64)
        unit_classes = parent_classes
        n_parent_images = len(ordered_parents)
        unit = "parent_image"
        row_counts = {
            class_name: int(sum(item == class_name for item in classes))
            for class_name in sorted(set(classes))
        }

    observed = set(unit_classes)
    if class_order is None:
        ordered_classes = sorted(observed)
    else:
        ordered_classes = _labels(class_order, "class_order", len(class_order))
        if len(set(ordered_classes)) != len(ordered_classes):
            raise MetricValidationError("class_order entries must be unique")
        if set(ordered_classes) != observed:
            raise MetricValidationError(
                "class_order must contain each and only the observed classes"
            )

    by_class: dict[str, Any] = {}
    unit_class_array = np.asarray(unit_classes, dtype=object)
    for class_name in ordered_classes:
        selected = units[unit_class_array == class_name]
        summary = _descriptive(selected)
        summary["n_rows"] = row_counts[class_name]
        summary["n_parent_images"] = (
            None if parent_ids is None else int(np.count_nonzero(unit_class_array == class_name))
        )
        by_class[class_name] = summary
    return {
        "unit": unit,
        "n_rows": int(len(vector)),
        "n_parent_images": n_parent_images,
        "overall": _descriptive(units),
        "classes": by_class,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "FRC_THRESHOLD",
    "MetricValidationError",
    "class_summaries",
    "holm_correction",
    "paired_wilcoxon",
    "parent_image_bootstrap_ci",
    "psnr",
    "psnr_native",
    "reference_frc_1over7",
    "ssim",
    "ssim_native",
]

"""Strict, CPU-only reporting for the Reviewer #1 DMD six-frame ablation.

This module consumes only the *new* R1C3 sample-level CSV/NPZ/JSON artifacts
inside one timestamped run directory.  It never discovers or reads historical
PhysMap results.  All validation is fail-closed: incomplete Cartesian products,
identity drift, non-finite values, stale labels, or an existing non-identical
output abort publication.

The preferred integration point is :func:`generate_all_reports`.  The pure
builders are public so the experiment runner and tests can independently
recompute every table value.  No function in this module imports torch, loads a
checkpoint, accesses a GPU, or reads ground truth outside the explicitly saved
Figure-5 NPZ receipts.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from tools.official_r2_common_metrics import parent_image_bootstrap_ci


PROTOCOL_ID = "DMD_6F_2O3P"
PROTOCOL_HASH = "580e8ac305e665a7bbe127f1b89c61c0d571c949880673d168d21a04f31d3e83"
RAW_FRAME_ORDER = ("H0", "H120", "H240", "V0", "V120", "V240")
VALIDITY_MASK = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
METHODS = ("WF", "DiffWS-6", "PhysMap-6", "APD-SIM-6")
BASELINES = ("WF", "DiffWS-6", "PhysMap-6")
FIGURE_METHODS = ("APD-SIM-6", "PhysMap-6", "DiffWS-6")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260814

DEFAULT_FACTOR_LEVELS: "OrderedDict[str, tuple[float, ...]]" = OrderedDict(
    (
        ("kxy_mismatch", (0.0, 0.05, 0.1, 0.15)),
        ("photon_scale_mul", (1.0, 0.5, 0.25, 0.125)),
        ("read_noise_mul", (1.0, 2.0, 4.0, 8.0)),
        ("background_add", (0.0, 0.01, 0.02, 0.05)),
        ("psf_blur", (0.0, 0.1, 0.2, 0.3)),
        ("mod_depth_drop", (0.0, 0.1, 0.2, 0.3)),
        ("phase_jitter_rad", (0.0, 0.1, 0.2, 0.4, 0.6)),
        ("angle_jitter_deg", (0.0, 0.5, 1.0, 2.0, 3.0)),
        ("aberr_defocus", (0.0, 0.025, 0.05, 0.075, 0.1)),
        ("aberr_astig_x", (0.0, 0.025, 0.05, 0.075, 0.1)),
        ("aberr_coma_x", (0.0, 0.025, 0.05, 0.075, 0.1)),
        ("aberr_spherical", (0.0, 0.025, 0.05, 0.075, 0.1)),
    )
)

FACTOR_LABELS = {
    "kxy_mismatch": "Spatial-frequency mismatch",
    "photon_scale_mul": "Photon-scale reduction",
    "read_noise_mul": "Read-noise multiplier",
    "background_add": "Background offset",
    "psf_blur": "PSF blur",
    "mod_depth_drop": "Modulation-depth reduction",
    "phase_jitter_rad": "Phase jitter",
    "angle_jitter_deg": "Pattern-angle jitter",
    "aberr_defocus": "Defocus",
    "aberr_astig_x": "Astigmatism",
    "aberr_coma_x": "Coma",
    "aberr_spherical": "Spherical aberration",
}

_REQUIRED_NOMINAL_FIELDS = {
    "sample_order", "sample_id", "parent_id", "structure", "method",
    "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
    "forward_parameters_sha256", "normalization_sha256", "gt_identity_sha256",
    "noise_seed", "diffusion_seed", "refinement_config_sha256", "psnr", "ssim",
    "frc_status", "frc_cutoff_cycles_per_pixel", "frc_spatial_period_px",
    "observed_nrmse", "poisson_gaussian_objective", "runtime_seconds",
    "peak_gpu_memory_bytes", "gradient_finite", "output_finite", "prediction_sha256",
}
_REQUIRED_ROBUST_FIELDS = {
    "factor", "severity", "sample_order", "sample_id", "parent_id", "structure",
    "method", "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
    "forward_parameters_sha256", "normalization_sha256", "gt_identity_sha256",
    "noise_seed", "diffusion_seed", "refinement_config_sha256", "theta_true_json",
    "theta_inverse_json", "psnr", "ssim", "observed_nrmse",
    "poisson_gaussian_objective", "runtime_seconds", "peak_gpu_memory_bytes",
    "gradient_finite", "output_finite", "prediction_sha256", "status",
}
_REQUIRED_RUNTIME_FIELDS = {
    "sample_order", "sample_id", "parent_id", "structure", "repeat_index", "method",
    "component", "measurement_kind", "warmup_runs_before_measurement",
    "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
    "forward_parameters_sha256", "normalization_sha256", "noise_seed",
    "diffusion_seed", "refinement_config_sha256", "runtime_seconds",
    "peak_gpu_memory_bytes",
}

_IDENTITY_FIELDS = (
    "raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
    "forward_parameters_sha256", "normalization_sha256", "gt_identity_sha256",
    "noise_seed", "diffusion_seed",
)
_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_OUTPUT_TEXT = (
    "PhysMap-9", "MAP-9", "PhysMap upper bound", "full observation set",
)


class ReportingValidationError(RuntimeError):
    """Fail-closed reporting error carrying an R1C3-compatible status."""

    def __init__(self, detail: str, status: str = "R1C3_FORMAL_EVALUATION_INCOMPLETE"):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Figure5Spec:
    """Prespecified Figure-5 selection; it is independent of GT and methods."""

    factors: tuple[str, ...] = ("phase_jitter_rad", "psf_blur", "photon_scale_mul")
    severities: tuple[tuple[float, ...], ...] = (
        (0.1, 0.4, 0.6),
        (0.1, 0.2, 0.3),
        (0.5, 0.25, 0.125),
    )
    methods: tuple[str, ...] = FIGURE_METHODS
    sample_order: int = 0
    display_min: float = 0.0
    display_max: float = 1.0
    profile_row_fraction: float = 0.5
    profile_severity_index: int = 2

    def __post_init__(self) -> None:
        _require(
            self.factors == ("phase_jitter_rad", "psf_blur", "photon_scale_mul"),
            "Figure 5 factors/order must be phase jitter, PSF blur, photon-scale reduction",
        )
        _require(
            self.severities == ((0.1, 0.4, 0.6), (0.1, 0.2, 0.3), (0.5, 0.25, 0.125)),
            "Figure 5 fixed severities differ from the preregistered display specification",
        )
        _require(self.methods == FIGURE_METHODS,
                 "Figure 5 columns must be APD-SIM-6, PhysMap-6, DiffWS-6")
        _require(self.display_min < self.display_max, "invalid fixed display range")
        _require(0.0 <= self.profile_row_fraction <= 1.0,
                 "profile row fraction must be in [0, 1]")
        _require(self.profile_severity_index in (0, 1, 2),
                 "profile severity index must identify a prespecified Figure-5 level")


def _require(condition: bool, detail: str, status: str = "R1C3_FORMAL_EVALUATION_INCOMPLETE") -> None:
    if not condition:
        raise ReportingValidationError(detail, status)


DEFAULT_FIGURE5_SPEC = Figure5Spec()


def _finite(value: Any, name: str, *, nonnegative: bool = False, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportingValidationError(f"{name} is not numeric: {value!r}") from exc
    _require(math.isfinite(number), f"{name} contains NaN/Inf", "R1C3_NONFINITE_RESULT")
    if nonnegative:
        _require(number >= 0.0, f"{name} must be non-negative")
    if positive:
        _require(number > 0.0, f"{name} must be positive")
    return number


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportingValidationError(f"{name} is not an integer: {value!r}") from exc
    _require(str(value).strip() in {str(number), f"{number}.0"} or isinstance(value, int),
             f"{name} is not an exact integer: {value!r}")
    if minimum is not None:
        _require(number >= minimum, f"{name} must be >= {minimum}")
    return number


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    _require(text in {"true", "false"}, f"{name} must be true/false")
    return text == "true"


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in _HEX for character in text)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    """Match the experiment runner's stable dtype/shape-aware array hash."""
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _load_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"CSV header missing: {path}")
        _require(required_fields.issubset(set(reader.fieldnames)),
                 f"CSV schema missing fields {sorted(required_fields - set(reader.fieldnames))}: {path}")
        rows = list(reader)
    _require(rows, f"CSV contains no rows: {path}")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingValidationError(f"invalid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    _assert_json_finite(value, str(path))
    return value


def _assert_json_finite(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{name}[{index}]")
    elif isinstance(value, (float, np.floating)):
        _require(math.isfinite(float(value)), f"{name} contains NaN/Inf", "R1C3_NONFINITE_RESULT")


def _descriptive(values: Sequence[Any], name: str) -> dict[str, Any]:
    array = np.asarray([_finite(value, name) for value in values], dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, f"empty vector: {name}")
    q1, median, q3 = np.quantile(array, (0.25, 0.5, 0.75), method="linear")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sample_sd": None if array.size == 1 else float(np.std(array, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _factor_levels(value: Mapping[str, Sequence[float]] | None) -> "OrderedDict[str, tuple[float, ...]]":
    source = DEFAULT_FACTOR_LEVELS if value is None else value
    result: "OrderedDict[str, tuple[float, ...]]" = OrderedDict()
    for factor, levels in source.items():
        _require(factor in FACTOR_LABELS, f"unknown robustness factor: {factor}")
        converted = tuple(_finite(level, f"severity {factor}") for level in levels)
        _require(converted and len(set(converted)) == len(converted),
                 f"empty/duplicate levels for {factor}")
        result[str(factor)] = converted
    _require(tuple(result) == tuple(DEFAULT_FACTOR_LEVELS),
             "formal robustness factors/order differ from the preregistered 12-factor design")
    _require(result == DEFAULT_FACTOR_LEVELS,
             "formal severity definitions differ from the preregistered design")
    return result


def _same_identity(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], case: str) -> None:
    for field in fields:
        values = {str(row.get(field, "")) for row in rows}
        _require(len(values) == 1 and next(iter(values)) != "",
                 f"input identity mismatch for {case}: {field}",
                 "R1C3_INPUT_IDENTITY_MISMATCH")


def _check_method(method: Any) -> str:
    text = str(method)
    _require(text in METHODS, f"unexpected method in primary results: {text!r}")
    return text


def validate_nominal_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and type the exact 30-FOV x four-method nominal grid."""
    _require(len(rows) == 120, f"nominal grid must contain 120 rows, got {len(rows)}")
    typed: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        order = _integer(row["sample_order"], "sample_order", minimum=0)
        method = _check_method(row["method"])
        key = (order, method)
        _require(key not in seen, f"duplicate nominal key: {key}")
        seen.add(key)
        for field in _IDENTITY_FIELDS[:-1]:
            if field.endswith("sha256"):
                _require(_is_sha256(row[field]), f"invalid SHA-256 in nominal {field}: {key}")
        _require(_is_sha256(row["prediction_sha256"]), f"invalid prediction SHA-256: {key}")
        row.update(
            sample_order=order,
            method=method,
            psnr=_finite(row["psnr"], f"nominal {key} psnr"),
            ssim=_finite(row["ssim"], f"nominal {key} ssim"),
            runtime_seconds=_finite(row["runtime_seconds"], f"nominal {key} runtime", nonnegative=True),
            peak_gpu_memory_bytes=_integer(row["peak_gpu_memory_bytes"], "peak_gpu_memory_bytes", minimum=0),
            gradient_finite=_boolean(row["gradient_finite"], "gradient_finite"),
            output_finite=_boolean(row["output_finite"], "output_finite"),
        )
        _require(row["gradient_finite"] and row["output_finite"],
                 f"non-finite nominal result: {key}", "R1C3_NONFINITE_RESULT")
        for field in ("observed_nrmse", "poisson_gaussian_objective"):
            text = str(row[field]).strip()
            row[field] = None if text == "" else _finite(text, f"nominal {key} {field}")
        status = str(row["frc_status"])
        _require(status in {"CUTOFF", "RIGHT_CENSORED", "UNRESOLVED"},
                 f"invalid FRC status for {key}: {status}")
        cutoff_text = str(row["frc_cutoff_cycles_per_pixel"]).strip()
        period_text = str(row["frc_spatial_period_px"]).strip()
        if status == "CUTOFF":
            cutoff = _finite(cutoff_text, f"nominal {key} FRC cutoff", positive=True)
            period = _finite(period_text, f"nominal {key} FRC period", positive=True)
            _require(math.isclose(cutoff * period, 1.0, rel_tol=5e-6, abs_tol=5e-6),
                     f"FRC cutoff/period inconsistency: {key}")
            row["frc_cutoff_cycles_per_pixel"] = cutoff
            row["frc_spatial_period_px"] = period
        else:
            _require(cutoff_text == "" and period_text == "",
                     f"non-cutoff FRC row has invented numeric value: {key}")
            row["frc_cutoff_cycles_per_pixel"] = None
            row["frc_spatial_period_px"] = None
        typed.append(row)
        by_order[order].append(row)
    _require(sorted(by_order) == list(range(30)), "nominal sample orders must be exactly 0..29")
    _require(len({str(row["sample_id"]) for row in typed}) == 30,
             "nominal grid must contain 30 unique sample IDs")
    for order, case_rows in by_order.items():
        _require({row["method"] for row in case_rows} == set(METHODS),
                 f"nominal method grid incomplete at sample {order}")
        _same_identity(case_rows, _IDENTITY_FIELDS, f"nominal sample {order}")
        refined = [row for row in case_rows if row["method"] in {"PhysMap-6", "APD-SIM-6"}]
        _same_identity(refined, ("refinement_config_sha256",), f"nominal refine sample {order}")
        _require(_is_sha256(refined[0]["refinement_config_sha256"]),
                 f"invalid refinement configuration hash at sample {order}")
        for row in case_rows:
            if row["method"] in {"PhysMap-6", "APD-SIM-6"}:
                _require(row["observed_nrmse"] is not None and row["poisson_gaussian_objective"] is not None,
                         f"refinement diagnostics missing: sample {order}/{row['method']}")
    return sorted(typed, key=lambda row: (row["sample_order"], METHODS.index(row["method"])))


def build_nominal_statistics(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return nominal JSON and paired-difference CSV rows (10,000 official resamples)."""
    typed = validate_nominal_rows(rows)
    summaries: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in typed if row["method"] == method]
        payload: dict[str, Any] = {
            metric: _descriptive([row[metric] for row in selected], f"{method}/{metric}")
            for metric in ("psnr", "ssim", "runtime_seconds", "peak_gpu_memory_bytes")
        }
        for metric in ("observed_nrmse", "poisson_gaussian_objective"):
            values = [row[metric] for row in selected if row[metric] is not None]
            payload[metric] = None if not values else _descriptive(values, f"{method}/{metric}")
            payload[f"{metric}_missing_count"] = len(selected) - len(values)
        numeric_frc = [row for row in selected if row["frc_status"] == "CUTOFF"]
        payload["frc_status_counts"] = dict(Counter(row["frc_status"] for row in selected))
        payload["frc_cutoff_cycles_per_pixel"] = (
            None if not numeric_frc else _descriptive(
                [row["frc_cutoff_cycles_per_pixel"] for row in numeric_frc], f"{method}/frc cutoff"
            )
        )
        payload["frc_spatial_period_px"] = (
            None if not numeric_frc else _descriptive(
                [row["frc_spatial_period_px"] for row in numeric_frc], f"{method}/frc period"
            )
        )
        payload["frc_numeric_count"] = len(numeric_frc)
        payload["frc_non_numeric_count"] = len(selected) - len(numeric_frc)
        summaries[method] = payload

    paired_rows: list[dict[str, Any]] = []
    paired_summaries: dict[str, Any] = {}
    apd = {row["sample_order"]: row for row in typed if row["method"] == "APD-SIM-6"}
    for comparator_index, comparator in enumerate(("PhysMap-6", "DiffWS-6")):
        other = {row["sample_order"]: row for row in typed if row["method"] == comparator}
        contrast = f"APD-SIM-6_minus_{comparator}"
        paired_summaries[contrast] = {}
        for metric_index, metric in enumerate(("psnr", "ssim")):
            left = [apd[index][metric] for index in range(30)]
            right = [other[index][metric] for index in range(30)]
            parents = [str(apd[index]["parent_id"]) for index in range(30)]
            classes = [str(apd[index]["structure"]) for index in range(30)]
            for index in range(30):
                _require(apd[index]["sample_id"] == other[index]["sample_id"] and
                         apd[index]["parent_id"] == other[index]["parent_id"] and
                         apd[index]["structure"] == other[index]["structure"],
                         f"paired alignment mismatch: {contrast}/{metric}/sample {index}")
            seed = int(bootstrap_seed + comparator_index * 101 + metric_index)
            ci = parent_image_bootstrap_ci(
                left, right, parents, class_labels=classes, statistic="mean",
                n_resamples=BOOTSTRAP_RESAMPLES, confidence_level=0.95, seed=seed,
            )
            paired_summaries[contrast][metric] = ci
            for index in range(30):
                paired_rows.append({
                    "sample_order": index,
                    "sample_id": apd[index]["sample_id"],
                    "parent_id": apd[index]["parent_id"],
                    "structure": apd[index]["structure"],
                    "contrast": contrast,
                    "metric": metric,
                    "apd_value": left[index],
                    "comparator_value": right[index],
                    "paired_difference": left[index] - right[index],
                    "bootstrap_mean_difference": ci["estimate"],
                    "bootstrap_ci_low": ci["confidence_interval"][0],
                    "bootstrap_ci_high": ci["confidence_interval"][1],
                    "bootstrap_resamples": ci["n_resamples"],
                    "bootstrap_seed": ci["seed"],
                    "resampling_unit": ci["resampling_unit"],
                })
    stats = {
        "schema_version": 1,
        "status": "COMPLETE_VALIDATED",
        "protocol_id": PROTOCOL_ID,
        "n_fov": 30,
        "method_order": list(METHODS),
        "native_normalized_reconstruction_metrics": True,
        "method_summaries": summaries,
        "paired_contrasts": paired_summaries,
        "bootstrap_contract": {
            "implementation": "tools.official_r2_common_metrics.parent_image_bootstrap_ci",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "statistic": "mean",
            "stratified_by_structure": True,
            "base_seed": int(bootstrap_seed),
        },
    }
    _assert_json_finite(stats, "nominal statistics")
    return stats, paired_rows


def validate_robustness_rows(
    rows: Sequence[Mapping[str, Any]],
    factor_levels: Mapping[str, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Validate the exact 12-factor x 54-level x 20-sample x four-method grid."""
    levels = _factor_levels(factor_levels)
    expected = sum(len(item) for item in levels.values()) * 20 * len(METHODS)
    _require(len(rows) == expected, f"robustness grid must contain {expected} rows, got {len(rows)}")
    typed: list[dict[str, Any]] = []
    seen: set[tuple[str, float, int, str]] = set()
    cases: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        factor = str(row["factor"])
        _require(factor in levels, f"unknown robustness factor: {factor}")
        severity = _finite(row["severity"], f"{factor} severity")
        matched = [candidate for candidate in levels[factor] if math.isclose(severity, candidate, abs_tol=1e-12)]
        _require(len(matched) == 1, f"unregistered severity {severity} for {factor}")
        severity = matched[0]
        order = _integer(row["sample_order"], "sample_order", minimum=0)
        method = _check_method(row["method"])
        key = (factor, severity, order, method)
        _require(key not in seen, f"duplicate robustness key: {key}")
        seen.add(key)
        _require(str(row["status"]) == "PASS", f"formal failure recorded at {key}")
        for field in ("raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
                      "forward_parameters_sha256", "normalization_sha256",
                      "gt_identity_sha256", "prediction_sha256"):
            _require(_is_sha256(row[field]), f"invalid robustness SHA-256 {field}: {key}")
        row.update(
            factor=factor, severity=severity, sample_order=order, method=method,
            psnr=_finite(row["psnr"], f"robustness {key} psnr"),
            ssim=_finite(row["ssim"], f"robustness {key} ssim"),
            runtime_seconds=_finite(row["runtime_seconds"], f"robustness {key} runtime", nonnegative=True),
            peak_gpu_memory_bytes=_integer(row["peak_gpu_memory_bytes"], "peak_gpu_memory_bytes", minimum=0),
            gradient_finite=_boolean(row["gradient_finite"], "gradient_finite"),
            output_finite=_boolean(row["output_finite"], "output_finite"),
        )
        _require(row["gradient_finite"] and row["output_finite"],
                 f"non-finite robustness result: {key}", "R1C3_NONFINITE_RESULT")
        for field in ("observed_nrmse", "poisson_gaussian_objective"):
            text = str(row[field]).strip()
            row[field] = None if text == "" else _finite(text, f"robustness {key} {field}")
        typed.append(row)
        cases[(factor, severity, order)].append(row)
    expected_cases = sum(len(item) for item in levels.values()) * 20
    _require(len(cases) == expected_cases, f"robustness case count must be {expected_cases}")
    for factor, factor_severities in levels.items():
        for severity in factor_severities:
            orders = sorted(order for f, s, order in cases if f == factor and s == severity)
            _require(orders == list(range(20)), f"sample orders incomplete for {factor}/{severity}")
            for order in orders:
                case_rows = cases[(factor, severity, order)]
                _require({row["method"] for row in case_rows} == set(METHODS),
                         f"method grid incomplete: {factor}/{severity}/{order}")
                _same_identity(case_rows, _IDENTITY_FIELDS, f"{factor}/{severity}/{order}")
                refined = [row for row in case_rows if row["method"] in {"PhysMap-6", "APD-SIM-6"}]
                _same_identity(refined, ("refinement_config_sha256",), f"refine {factor}/{severity}/{order}")
                _require(_is_sha256(refined[0]["refinement_config_sha256"]),
                         f"invalid refinement hash: {factor}/{severity}/{order}")
    return sorted(
        typed,
        key=lambda row: (
            list(levels).index(row["factor"]), levels[row["factor"]].index(row["severity"]),
            row["sample_order"], METHODS.index(row["method"]),
        ),
    )


def build_robustness_statistics(
    rows: Sequence[Mapping[str, Any]],
    factor_levels: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return complete robustness statistics and explicit failed-case rows."""
    levels = _factor_levels(factor_levels)
    typed = validate_robustness_rows(rows, levels)
    groups: dict[str, Any] = {}
    for factor, factor_severities in levels.items():
        groups[factor] = {}
        for severity in factor_severities:
            groups[factor][format(severity, ".12g")] = {}
            for method in METHODS:
                selected = [
                    row for row in typed
                    if row["factor"] == factor and row["severity"] == severity and row["method"] == method
                ]
                _require(len(selected) == 20, f"robustness summary group is not n=20: {factor}/{severity}/{method}")
                payload = {
                    metric: _descriptive([row[metric] for row in selected], f"{factor}/{severity}/{method}/{metric}")
                    for metric in ("psnr", "ssim", "runtime_seconds", "peak_gpu_memory_bytes")
                }
                for metric in ("observed_nrmse", "poisson_gaussian_objective"):
                    values = [row[metric] for row in selected if row[metric] is not None]
                    payload[metric] = None if not values else _descriptive(values, f"{factor}/{severity}/{method}/{metric}")
                    payload[f"{metric}_missing_count"] = 20 - len(values)
                groups[factor][format(severity, ".12g")][method] = payload
    stats = {
        "schema_version": 1,
        "status": "COMPLETE_VALIDATED",
        "protocol_id": PROTOCOL_ID,
        "factor_order": list(levels),
        "factor_levels": {factor: list(values) for factor, values in levels.items()},
        "factor_count": len(levels),
        "factor_level_count": sum(len(item) for item in levels.values()),
        "sample_count": 20,
        "method_order": list(METHODS),
        "row_count": len(typed),
        "all_cases_present": True,
        "silent_skips": 0,
        "failed_case_count": 0,
        "groups": groups,
    }
    _assert_json_finite(stats, "robustness statistics")
    return stats, []


def _format_severity(factor: str, severity: float, *, latex: bool) -> str:
    if factor in {"read_noise_mul", "photon_scale_mul"}:
        return f"{severity:g}$\\times$" if latex else f"{severity:g}x"
    if factor == "background_add":
        return f"+{severity:g}"
    if factor in {"mod_depth_drop", "kxy_mismatch", "psf_blur"}:
        return f"{100.0 * severity:g}\\%" if latex else f"{100.0 * severity:g}%"
    if factor == "angle_jitter_deg":
        return f"{severity:g}$^\\circ$" if latex else f"{severity:g} deg"
    if factor == "phase_jitter_rad":
        return f"{severity:g} rad"
    if factor.startswith("aberr_"):
        return f"{severity:.3f} waves RMS"
    return f"{severity:g}"


def build_table2(
    robustness_rows: Sequence[Mapping[str, Any]],
    factor_levels: Mapping[str, Sequence[float]] | None = None,
    *, bootstrap_seed: int = BOOTSTRAP_SEED + 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute Table 2 from sample rows, including every best-baseline label."""
    levels = _factor_levels(factor_levels)
    typed = validate_robustness_rows(robustness_rows, levels)
    output: list[dict[str, Any]] = []
    model: dict[str, Any] = {}
    for factor_index, (factor, factor_severities) in enumerate(levels.items()):
        severity = factor_severities[-1]
        model[factor] = {"severity": severity, "metrics": {}}
        for metric_index, metric in enumerate(("psnr", "ssim")):
            arrays: dict[str, list[float]] = {}
            selected_rows: dict[str, list[dict[str, Any]]] = {}
            for method in METHODS:
                chosen = sorted(
                    (row for row in typed if row["factor"] == factor and
                     row["severity"] == severity and row["method"] == method),
                    key=lambda row: row["sample_order"],
                )
                _require(len(chosen) == 20, f"Table 2 group incomplete: {factor}/{metric}/{method}")
                selected_rows[method] = chosen
                arrays[method] = [row[metric] for row in chosen]
            baseline = max(BASELINES, key=lambda method: float(np.mean(arrays[method])))
            apd_rows = selected_rows["APD-SIM-6"]
            base_rows = selected_rows[baseline]
            for left, right in zip(apd_rows, base_rows):
                _require(left["sample_id"] == right["sample_id"] and left["parent_id"] == right["parent_id"],
                         f"Table 2 paired alignment mismatch: {factor}/{metric}")
            seed = int(bootstrap_seed + factor_index * 17 + metric_index)
            ci = parent_image_bootstrap_ci(
                arrays["APD-SIM-6"], arrays[baseline],
                [row["parent_id"] for row in apd_rows],
                class_labels=[row["structure"] for row in apd_rows],
                statistic="mean", n_resamples=BOOTSTRAP_RESAMPLES,
                confidence_level=0.95, seed=seed,
            )
            summaries = {method: _descriptive(arrays[method], f"Table2/{factor}/{metric}/{method}") for method in METHODS}
            item = {
                "factor": factor,
                "factor_label": FACTOR_LABELS[factor],
                "severity": severity,
                "severity_label": _format_severity(factor, severity, latex=False),
                "metric": metric,
                "n_paired_samples": 20,
                "wf_mean": summaries["WF"]["mean"],
                "wf_sample_sd": summaries["WF"]["sample_sd"],
                "diffws6_mean": summaries["DiffWS-6"]["mean"],
                "diffws6_sample_sd": summaries["DiffWS-6"]["sample_sd"],
                "physmap6_mean": summaries["PhysMap-6"]["mean"],
                "physmap6_sample_sd": summaries["PhysMap-6"]["sample_sd"],
                "apd6_mean": summaries["APD-SIM-6"]["mean"],
                "apd6_sample_sd": summaries["APD-SIM-6"]["sample_sd"],
                "best_matched_six_frame_baseline": baseline,
                "apd_minus_best_baseline_mean": ci["estimate"],
                "apd_minus_best_baseline_ci_low": ci["confidence_interval"][0],
                "apd_minus_best_baseline_ci_high": ci["confidence_interval"][1],
                "bootstrap_resamples": ci["n_resamples"],
                "bootstrap_seed": ci["seed"],
            }
            output.append(item)
            model[factor]["metrics"][metric] = {"row": item, "method_summaries": summaries, "bootstrap": ci}
    _require(len(output) == 24, "Table 2 must contain 12 factors x PSNR/SSIM")
    return output, model


def load_figure5_data(path: Path | str) -> list[dict[str, Any]]:
    """Public CSV-only Figure-5 data loader for an independent audit.

    This intentionally performs no historical result discovery.  It exposes
    exactly the three registered factors and three matched six-frame columns
    present in the supplied sample-level CSV snapshot.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"factor", "severity", "sample_order", "sample_id", "method", "psnr", "ssim"}
    _require(rows and required.issubset(rows[0]), f"Figure-5 CSV schema incomplete: {source}")
    selected: list[dict[str, Any]] = []
    for row in rows:
        factor = str(row["factor"])
        method = str(row["method"])
        if factor in DEFAULT_FIGURE5_SPEC.factors and method in FIGURE_METHODS:
            selected.append({
                "factor": factor,
                "severity": _finite(row["severity"], "Figure-5 severity"),
                "sample_order": _integer(row["sample_order"], "Figure-5 sample_order", minimum=0),
                "sample_id": str(row["sample_id"]),
                "method": method,
                "psnr": _finite(row["psnr"], "Figure-5 PSNR"),
                "ssim": _finite(row["ssim"], "Figure-5 SSIM"),
            })
    _require(selected, "Figure-5 CSV has no registered rows")
    return selected


def compute_table2_rows(
    nominal_csv: Path | str, robustness_csv: Path | str
) -> list[dict[str, Any]]:
    """Small independent CSV recomputation helper used by regression tests.

    Formal publication uses :func:`build_table2` on the full 4,320-row grid.
    This helper also supports a compact audit fixture and reports nominal plus
    available robustness endpoints without inventing missing factors.
    """
    def read(path: Path | str) -> list[dict[str, str]]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    nominal = read(nominal_csv)
    robust = read(robustness_csv)
    _require(nominal and robust, "independent Table-2 CSV input is empty")
    output: list[dict[str, Any]] = []
    for label, rows in (("nominal", nominal), ("robustness", robust)):
        groups: dict[tuple[str, float | None], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            factor = str(row.get("factor", label))
            severity = None if label == "nominal" else _finite(row["severity"], "Table-2 severity")
            groups[(factor, severity)].append(row)
        for (factor, severity), group in groups.items():
            for metric in ("psnr", "ssim"):
                means: dict[str, float] = {}
                for method in ("DiffWS-6", "PhysMap-6", "APD-SIM-6"):
                    values = [_finite(row[metric], f"Table-2 {metric}") for row in group if row["method"] == method]
                    _require(values, f"Table-2 compact group lacks {method}: {factor}/{metric}")
                    means[method] = float(np.mean(values))
                baseline = max(("DiffWS-6", "PhysMap-6"), key=lambda item: means[item])
                output.append({
                    "condition": factor,
                    "severity": severity,
                    "metric": metric,
                    "best_matched_six_frame_baseline": baseline,
                    "best_baseline_mean": means[baseline],
                    "apd6_mean": means["APD-SIM-6"],
                    "apd_minus_best_baseline_mean": means["APD-SIM-6"] - means[baseline],
                })
    return output


def _pm(mean: float, sd: float, metric: str) -> str:
    digits = 2 if metric == "psnr" else 4
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def table2_caption_text() -> str:
    return (
        "Strict matched six-frame robustness at the strongest preregistered severity of each "
        "perturbation. Values are mean $\\pm$ sample SD across the same 20 fixed held-out patches. "
        "For each metric and factor, the best matched six-frame baseline is the highest group mean "
        "among WF, DiffWS-6, and PhysMap-6. The final columns report APD-SIM-6 minus that labeled "
        "baseline with a 95\\% structure-stratified paired-parent percentile-bootstrap confidence "
        "interval from 10,000 resamples."
    )


def render_table2_tex(table_rows: Sequence[Mapping[str, Any]]) -> str:
    _require(len(table_rows) == 24, "cannot render incomplete Table 2")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        f"\\caption{{{table2_caption_text()}}}", r"\label{tab:physmap6-strict}",
        r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        "Factor (severity) & Metric & WF & DiffWS-6 & PhysMap-6 & APD-SIM-6 & Best baseline & "
        "$\\Delta$ [95\\% CI] \\\\",
        r"\midrule",
    ]
    for index in range(0, len(table_rows), 2):
        pair = table_rows[index:index + 2]
        _require([row["metric"] for row in pair] == ["psnr", "ssim"], "Table 2 metric ordering drift")
        for metric_index, row in enumerate(pair):
            metric = str(row["metric"])
            digits = 2 if metric == "psnr" else 4
            factor = str(row["factor"])
            factor_cell = (
                f"{row['factor_label']} ({_format_severity(factor, float(row['severity']), latex=True)})"
                if metric_index == 0 else ""
            )
            cells = [
                _pm(float(row["wf_mean"]), float(row["wf_sample_sd"]), metric),
                _pm(float(row["diffws6_mean"]), float(row["diffws6_sample_sd"]), metric),
                _pm(float(row["physmap6_mean"]), float(row["physmap6_sample_sd"]), metric),
                _pm(float(row["apd6_mean"]), float(row["apd6_sample_sd"]), metric),
                str(row["best_matched_six_frame_baseline"]),
                f"{float(row['apd_minus_best_baseline_mean']):+.{digits}f} "
                f"[{float(row['apd_minus_best_baseline_ci_low']):+.{digits}f}, "
                f"{float(row['apd_minus_best_baseline_ci_high']):+.{digits}f}]",
            ]
            label = "PSNR (dB)" if metric == "psnr" else "SSIM"
            lines.append(f"{factor_cell} & {label} & " + " & ".join(cells) + r" \\")
        if index + 2 < len(table_rows):
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def validate_runtime_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate 30 FOV x three repeats x six requested runtime components."""
    groups = (
        ("WF", "six-frame mean"), ("DiffWS-6", "Stage 1"), ("PhysMap-6", "total"),
        ("APD-SIM-6", "Stage 1"), ("APD-SIM-6", "Stage 2"), ("APD-SIM-6", "total"),
    )
    expected_kinds = {
        ("WF", "six-frame mean"): "direct_cuda_timing",
        ("DiffWS-6", "Stage 1"): "direct_cuda_timing",
        ("PhysMap-6", "total"): "direct_cuda_timing",
        ("APD-SIM-6", "Stage 1"): "alias_of_same_repeat_diffws_stage1",
        ("APD-SIM-6", "Stage 2"): "direct_cuda_timing",
        ("APD-SIM-6", "total"): "derived_same_repeat_component_sum",
    }
    _require(len(rows) == 30 * 3 * len(groups), f"runtime grid must contain 540 rows, got {len(rows)}")
    typed: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, str]] = set()
    by_repeat: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        order = _integer(row["sample_order"], "sample_order", minimum=0)
        repeat = _integer(row["repeat_index"], "repeat_index", minimum=0)
        method = _check_method(row["method"])
        component = str(row["component"])
        _require((method, component) in groups, f"unexpected runtime component: {method}/{component}")
        key = (order, repeat, method, component)
        _require(key not in seen, f"duplicate runtime key: {key}")
        seen.add(key)
        _require(_integer(row["warmup_runs_before_measurement"], "warmup count") == 1,
                 f"runtime warm-up contract changed: {key}")
        _require(str(row["measurement_kind"]) == expected_kinds[(method, component)],
                 f"runtime measurement-kind contract changed: {key}")
        for field in ("raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
                      "forward_parameters_sha256", "normalization_sha256"):
            _require(_is_sha256(row[field]), f"invalid runtime SHA-256 {field}: {key}")
        row.update(
            sample_order=order, repeat_index=repeat, method=method, component=component,
            runtime_seconds=_finite(row["runtime_seconds"], f"runtime {key}", nonnegative=True),
            peak_gpu_memory_bytes=_integer(row["peak_gpu_memory_bytes"], "peak memory", minimum=0),
        )
        typed.append(row)
        by_repeat[(order, repeat)].append(row)
    _require(sorted({row["sample_order"] for row in typed}) == list(range(30)),
             "runtime sample orders must be exactly 0..29")
    _require({row["repeat_index"] for row in typed} == {0, 1, 2},
             "runtime repeat indices must be exactly 0,1,2")
    for case, case_rows in by_repeat.items():
        _require({(row["method"], row["component"]) for row in case_rows} == set(groups),
                 f"runtime component grid incomplete: {case}")
        _same_identity(
            case_rows,
            ("raw_stack_sha256", "validity_mask_sha256", "geometry_sha256",
             "forward_parameters_sha256", "normalization_sha256", "noise_seed", "diffusion_seed"),
            f"runtime sample/repeat {case}",
        )
    return sorted(typed, key=lambda row: (row["sample_order"], row["repeat_index"], groups.index((row["method"], row["component"]))))


def build_runtime_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    typed = validate_runtime_rows(rows)
    groups = (
        ("WF", "six-frame mean"), ("DiffWS-6", "Stage 1"), ("PhysMap-6", "total"),
        ("APD-SIM-6", "Stage 1"), ("APD-SIM-6", "Stage 2"), ("APD-SIM-6", "total"),
    )
    summaries: list[dict[str, Any]] = []
    for method, component in groups:
        selected = [row for row in typed if row["method"] == method and row["component"] == component]
        _require(len(selected) == 90, f"runtime summary group is not n=90: {method}/{component}")
        by_fov: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            by_fov[row["sample_order"]].append(row["runtime_seconds"])
        _require(all(len(values) == 3 for values in by_fov.values()),
                 f"runtime repeats incomplete: {method}/{component}")
        run_stats = _descriptive([row["runtime_seconds"] for row in selected], f"runtime/{method}/{component}")
        fov_stats = _descriptive([float(np.mean(by_fov[index])) for index in range(30)],
                                 f"runtime FOV means/{method}/{component}")
        summaries.append({
            "method": method, "component": component, "n_fov": 30,
            "repeats_per_fov": 3, "warmup_per_fov": 1,
            "mean_seconds_across_90_runs": run_stats["mean"],
            "sample_sd_seconds_across_90_runs": run_stats["sample_sd"],
            "mean_of_fov_means_seconds": fov_stats["mean"],
            "sample_sd_of_fov_means_seconds": fov_stats["sample_sd"],
            "peak_allocated_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in selected),
            "measurement_kinds": sorted({str(row["measurement_kind"]) for row in selected}),
        })
    result = {
        "schema_version": 1, "status": "COMPLETE_VALIDATED", "protocol_id": PROTOCOL_ID,
        "device_batch_size": 1, "cuda_synchronized_by_producer_contract": True, "n_fov": 30,
        "warmup_per_fov": 1, "recorded_repeats_per_fov": 3, "summaries": summaries,
    }
    _assert_json_finite(result, "runtime statistics")
    return result


def render_runtime_tex(stats: Mapping[str, Any]) -> str:
    summaries = list(stats["summaries"])
    _require(len(summaries) == 6, "runtime table requires six rows")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Strict DMD six-frame runtime benchmark at batch size one. Each of 30 FOVs received one warm-up followed by three CUDA-synchronized recorded repeats. Time is the mean $\pm$ sample SD of the 30 per-FOV repeat means; memory is peak allocated GPU memory.}",
        r"\label{tab:physmap6-runtime}", r"\begin{tabular}{llrr}", r"\toprule",
        "Method & Component & Time (s) & Peak memory (MiB) \\\\", r"\midrule",
    ]
    for item in summaries:
        lines.append(
            f"{item['method']} & {item['component']} & "
            f"{float(item['mean_of_fov_means_seconds']):.3f} $\\pm$ "
            f"{float(item['sample_sd_of_fov_means_seconds']):.3f} & "
            f"{int(item['peak_allocated_gpu_memory_bytes']) / 2**20:.1f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _figure_npz_name(factor: str, severity: float, method: str) -> str:
    return f"{factor}_{severity:g}_{method.replace(' ', '_')}.npz"


def _load_figure_arrays(
    visual_dir: Path,
    robust_rows: Sequence[Mapping[str, Any]],
    spec: Figure5Spec,
) -> tuple[dict[tuple[str, float, str], np.ndarray], dict[str, Any]]:
    typed = validate_robustness_rows(robust_rows)
    arrays: dict[tuple[str, float, str], np.ndarray] = {}
    sources: list[dict[str, Any]] = []
    shape: tuple[int, int] | None = None
    for factor, severities in zip(spec.factors, spec.severities):
        _require(factor in DEFAULT_FACTOR_LEVELS, f"Figure 5 factor not registered: {factor}")
        for severity in severities:
            _require(any(math.isclose(severity, level, abs_tol=1e-12) for level in DEFAULT_FACTOR_LEVELS[factor]),
                     f"Figure 5 severity not registered: {factor}/{severity}")
            gt_hashes: set[str] = set()
            for method in spec.methods:
                path = visual_dir / _figure_npz_name(factor, severity, method)
                _require(path.is_file(), f"Figure 5 NPZ missing: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    _require(set(archive.files) == {"prediction", "gt"},
                             f"Figure 5 NPZ keys drift: {path}")
                    prediction = np.asarray(archive["prediction"])
                    gt = np.asarray(archive["gt"])
                _require(prediction.ndim == 2 and gt.shape == prediction.shape and prediction.size > 0,
                         f"Figure 5 array shape invalid: {path}")
                _require(prediction.dtype.kind == "f" and gt.dtype.kind == "f",
                         f"Figure 5 arrays must be floating point: {path}")
                _require(np.isfinite(prediction).all() and np.isfinite(gt).all(),
                         f"Figure 5 array contains NaN/Inf: {path}", "R1C3_NONFINITE_RESULT")
                _require(float(np.min(prediction)) >= spec.display_min - 1e-6 and
                         float(np.max(prediction)) <= spec.display_max + 1e-6,
                         f"Figure 5 prediction outside fixed native display interval: {path}")
                if shape is None:
                    shape = tuple(int(value) for value in prediction.shape)
                _require(tuple(prediction.shape) == shape, f"Figure 5 shape drift: {path}")
                prediction = np.ascontiguousarray(prediction)
                pred_hash = sha256_array(prediction)
                gt_hashes.add(sha256_array(np.ascontiguousarray(gt)))
                matching = [
                    row for row in typed
                    if row["factor"] == factor and math.isclose(row["severity"], severity, abs_tol=1e-12)
                    and row["sample_order"] == spec.sample_order and row["method"] == method
                ]
                _require(len(matching) == 1, f"Figure 5 CSV source row unresolved: {factor}/{severity}/{method}")
                _require(matching[0]["prediction_sha256"] == pred_hash,
                         f"Figure 5 NPZ/CSV prediction hash mismatch: {factor}/{severity}/{method}")
                arrays[(factor, severity, method)] = prediction
                sources.append({
                    "factor": factor, "severity": severity, "method": method,
                    "sample_order": spec.sample_order, "npz_path": str(path.resolve()),
                    "npz_sha256": sha256_file(path), "prediction_sha256": pred_hash,
                    "gt_sha256": sha256_array(np.ascontiguousarray(gt)),
                    "csv_prediction_sha256": matching[0]["prediction_sha256"],
                })
            _require(len(gt_hashes) == 1,
                     f"Figure 5 GT identity differs across method columns: {factor}/{severity}",
                     "R1C3_INPUT_IDENTITY_MISMATCH")
    assert shape is not None
    profile_row = int(round(spec.profile_row_fraction * (shape[0] - 1)))
    receipt = {
        "schema_version": 1, "status": "PASS", "sample_order": spec.sample_order,
        "factor_order": list(spec.factors),
        "severities": {factor: list(levels) for factor, levels in zip(spec.factors, spec.severities)},
        "column_order": list(spec.methods),
        "display_mapping": "fixed linear native reconstruction range; no per-method remapping",
        "display_range": [spec.display_min, spec.display_max],
        "profile_selection_basis": "prespecified shape-only row fraction; GT and method outputs not consulted",
        "profile_row_fraction": spec.profile_row_fraction, "profile_row_index": profile_row,
        "profile_uses_native_reconstruction": True,
        "profile_severity_index": spec.profile_severity_index,
        "image_shape": list(shape), "source_count": len(sources), "sources": sources,
    }
    return arrays, receipt


def render_figure5(
    visual_dir: Path | str,
    robustness_rows: Sequence[Mapping[str, Any]],
    *, spec: Figure5Spec = DEFAULT_FIGURE5_SPEC,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Render 3 factors x 3 fixed severity rows and GT-independent profiles."""
    arrays, receipt = _load_figure_arrays(Path(visual_dir), robustness_rows, spec)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    height_ratios: list[float] = []
    for _ in spec.factors:
        height_ratios.extend((1.0, 1.0, 1.0, 0.62))
    fig = plt.figure(figsize=(9.4, 23.0), constrained_layout=True)
    grid = fig.add_gridspec(12, 3, height_ratios=height_ratios)
    colors = {"APD-SIM-6": "#0072B2", "PhysMap-6": "#D55E00", "DiffWS-6": "#009E73"}
    profile_row = int(receipt["profile_row_index"])
    for factor_index, (factor, severities) in enumerate(zip(spec.factors, spec.severities)):
        row0 = factor_index * 4
        for severity_index, severity in enumerate(severities):
            for method_index, method in enumerate(spec.methods):
                axis = fig.add_subplot(grid[row0 + severity_index, method_index])
                axis.imshow(
                    arrays[(factor, severity, method)], cmap="gray",
                    vmin=spec.display_min, vmax=spec.display_max, interpolation="nearest",
                )
                axis.axhline(profile_row, color="#F0E442", linewidth=0.55, linestyle="--", alpha=0.9)
                axis.set_xticks([]); axis.set_yticks([])
                if row0 + severity_index == 0:
                    axis.set_title(method, fontsize=11, fontweight="bold")
                if method_index == 0:
                    axis.set_ylabel(
                        f"{FACTOR_LABELS[factor]}\n{_format_severity(factor, severity, latex=False)}",
                        fontsize=8.5,
                    )
                for spine in axis.spines.values():
                    spine.set_linewidth(0.6); spine.set_color("#444444")
        profile_axis = fig.add_subplot(grid[row0 + 3, :])
        severity = severities[spec.profile_severity_index]
        x = np.arange(receipt["image_shape"][1])
        for method in spec.methods:
            profile_axis.plot(
                x, arrays[(factor, severity, method)][profile_row, :],
                color=colors[method], linewidth=1.0, label=method,
            )
        profile_axis.set_xlim(0, receipt["image_shape"][1] - 1)
        profile_axis.set_ylim(spec.display_min, spec.display_max)
        profile_axis.set_ylabel("Native intensity", fontsize=8)
        profile_axis.set_xlabel(
            f"Fixed center-row profile at {_format_severity(factor, severity, latex=False)} (pixel)",
            fontsize=8,
        )
        profile_axis.grid(True, color="#d8d8d8", linewidth=0.45)
        profile_axis.spines[["top", "right"]].set_visible(False)
        profile_axis.tick_params(labelsize=7)
        if factor_index == 0:
            profile_axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("Strict matched DMD six-frame robustness", fontsize=14)
    with tempfile.TemporaryDirectory(prefix="r1c3-figure5-") as temporary:
        folder = Path(temporary)
        png_path = folder / "figure.png"
        pdf_path = folder / "figure.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches="tight", metadata={"Software": "R1C3 strict reporting"})
        fig.savefig(
            pdf_path, bbox_inches="tight",
            metadata={"Creator": "R1C3 strict reporting", "CreationDate": None, "ModDate": None},
        )
        png = png_path.read_bytes(); pdf = pdf_path.read_bytes()
    plt.close(fig)
    _require(png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 10_000,
             "Figure 5 PNG render invalid")
    _require(pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-1024:] and len(pdf) > 10_000,
             "Figure 5 PDF render invalid")
    receipt["png_sha256"] = hashlib.sha256(png).hexdigest()
    receipt["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
    return png, pdf, receipt


def figure5_caption_text(spec: Figure5Spec = DEFAULT_FIGURE5_SPEC) -> str:
    return (
        "Strict matched six-frame reconstructions under phase jitter, PSF blur, and photon-scale "
        "reduction. Each factor is shown at three fixed preregistered severities; columns are "
        "APD-SIM-6, PhysMap-6, and DiffWS-6 in that order. Every row uses the identical six-frame "
        f"measurement and the fixed linear native display interval [{spec.display_min:g},"
        f"{spec.display_max:g}] for all methods. The dashed "
        f"line marks the prespecified shape-only profile row ({spec.profile_row_fraction:g} of image "
        "height), selected without GT or any reconstruction. Profiles use native reconstructions at "
        "the third displayed severity and are not computed from display-remapped images."
    )


def _validate_protocol_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    _require(receipt.get("status") == "PASS", "protocol receipt is not PASS", "R1C3_PROTOCOL_UNRESOLVED")
    _require(receipt.get("protocol_id") == PROTOCOL_ID and receipt.get("protocol_hash") == PROTOCOL_HASH,
             "protocol receipt identity mismatch", "R1C3_PROTOCOL_UNRESOLVED")
    _require(tuple(receipt.get("raw_frame_order", ())) == RAW_FRAME_ORDER,
             "protocol receipt raw order mismatch", "R1C3_PROTOCOL_UNRESOLVED")
    mask = receipt.get("validity_mask_15_slots", receipt.get("validity_mask"))
    _require(tuple(int(item) for item in mask) == VALIDITY_MASK,
             "protocol receipt validity mask mismatch", "R1C3_PROTOCOL_UNRESOLVED")
    return receipt


def _validate_checkpoint_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    _require(receipt.get("status") == "PASS", "checkpoint receipt is not PASS", "R1C3_APD6_CHECKPOINT_UNRESOLVED")
    _require(_is_sha256(receipt.get("checkpoint_sha256")),
             "checkpoint receipt SHA-256 invalid", "R1C3_APD6_CHECKPOINT_UNRESOLVED")
    _require(receipt.get("protocol_id") == PROTOCOL_ID and receipt.get("protocol_hash") == PROTOCOL_HASH,
             "checkpoint protocol identity mismatch", "R1C3_APD6_CHECKPOINT_UNRESOLVED")
    _require(receipt.get("test_data_used_for_selection") is False,
             "checkpoint was not proven validation-only", "R1C3_APD6_CHECKPOINT_UNRESOLVED")
    _require(bool(receipt.get("all_model_parameters_finite")) and bool(receipt.get("all_ema_parameters_finite")),
             "checkpoint parameter finiteness unresolved", "R1C3_APD6_CHECKPOINT_UNRESOLVED")
    selected_step = _integer(receipt.get("selected_step"), "selected checkpoint step", minimum=0)
    committed = _integer(
        receipt.get("optimizer_committed_updates_at_selected_checkpoint"),
        "selected checkpoint committed optimizer updates",
        minimum=0,
    )
    delta = _integer(receipt.get("global_minus_optimizer_commits"), "event/commit delta", minimum=0)
    _require(selected_step - committed == delta, "checkpoint event/commit semantics mismatch")
    _require(
        receipt.get("selected_step_semantics")
        == "loop/data event step; not the number of committed optimizer updates",
        "checkpoint selected-step semantics are ambiguous",
    )
    _finite(receipt.get("validation_metric_value"), "checkpoint validation metric")
    return receipt


def render_methods_tex(protocol: Mapping[str, Any]) -> str:
    return (
        "PhysMap-6 required no network training and had no learnable checkpoint. For each sample, "
        "we formed its initialization by averaging the six raw measurements in registered order "
        "$H0,H120,H240,V0,V120,V240$. We then performed per-sample physics-only masked-likelihood "
        "optimization. PhysMap-6 and APD-SIM-6 Stage~2 called the same refinement function with the "
        "same six-frame tensor, raw-to-slot mapping, validity mask, acquisition geometry, forward "
        "operator, Poisson--Gaussian objective, Adam configuration, learning rate, 40-update stopping "
        "rule, $[0,1]$ intensity bounds, numerical precision, and random seed. Their sole difference "
        "was initialization: the six-frame mean for PhysMap-6 and the frozen APD Stage~1 output "
        "$x_{\\mathrm{ws}}$ for APD-SIM-6. The explicit prior weight was $\\lambda_{\\mathrm{prior}}=0$; "
        "therefore this procedure is described as physics-only masked-likelihood optimization, not "
        "as optimization with a nonzero learned or handcrafted prior. DiffWS-6 denotes Stage~1 alone, "
        "and WF denotes the six-frame mean. Metrics were computed on native normalized reconstructions.\n"
    )


def _paired_summary(stats: Mapping[str, Any], comparator: str, metric: str) -> Mapping[str, Any]:
    return stats["paired_contrasts"][f"APD-SIM-6_minus_{comparator}"][metric]


def render_results_tex(
    nominal_stats: Mapping[str, Any], table_rows: Sequence[Mapping[str, Any]],
    runtime_stats: Mapping[str, Any],
) -> str:
    apd = nominal_stats["method_summaries"]["APD-SIM-6"]
    phrases = []
    for comparator in ("PhysMap-6", "DiffWS-6"):
        psnr = _paired_summary(nominal_stats, comparator, "psnr")
        ssim = _paired_summary(nominal_stats, comparator, "ssim")
        phrases.append(
            f"versus {comparator}, PSNR {float(psnr['estimate']):+.2f} dB "
            f"(95\\% CI {float(psnr['confidence_interval'][0]):+.2f} to {float(psnr['confidence_interval'][1]):+.2f}) "
            f"and SSIM {float(ssim['estimate']):+.4f} "
            f"(95\\% CI {float(ssim['confidence_interval'][0]):+.4f} to {float(ssim['confidence_interval'][1]):+.4f})"
        )
    figure_factors = set(DEFAULT_FIGURE5_SPEC.factors)
    endpoints = []
    for row in table_rows:
        if row["factor"] in figure_factors and row["metric"] == "psnr":
            endpoints.append(
                f"{str(row['factor_label']).lower()} {_format_severity(str(row['factor']), float(row['severity']), latex=True)}: "
                f"{float(row['apd6_mean']):.2f} $\\pm$ {float(row['apd6_sample_sd']):.2f} dB"
            )
    apd_total = next(item for item in runtime_stats["summaries"] if item["method"] == "APD-SIM-6" and item["component"] == "total")
    return (
        f"Across the 30 sealed nominal FOVs, APD-SIM-6 achieved "
        f"{float(apd['psnr']['mean']):.2f} $\\pm$ {float(apd['psnr']['sample_sd']):.2f} dB PSNR and "
        f"SSIM {float(apd['ssim']['mean']):.4f} $\\pm$ {float(apd['ssim']['sample_sd']):.4f}. "
        + "; ".join(phrases) + ".\n\n"
        + "At the strongest displayed robustness severities, APD-SIM-6 PSNR was "
        + "; ".join(endpoints) + ". All 12 perturbations, including defocus, astigmatism, coma, "
        "and spherical aberration, are reported in revised Table~2 with the labeled strongest matched "
        "six-frame baseline for each metric.\n\n"
        f"The synchronized 30-FOV runtime benchmark gave an APD-SIM-6 total time of "
        f"{float(apd_total['mean_of_fov_means_seconds']):.3f} $\\pm$ "
        f"{float(apd_total['sample_sd_of_fov_means_seconds']):.3f} s per FOV.\n"
    )


def render_response_tex(
    protocol: Mapping[str, Any], checkpoint: Mapping[str, Any], nominal_stats: Mapping[str, Any],
    robustness_stats: Mapping[str, Any], table_rows: Sequence[Mapping[str, Any]],
) -> str:
    phys = _paired_summary(nominal_stats, "PhysMap-6", "psnr")
    diff = _paired_summary(nominal_stats, "DiffWS-6", "psnr")
    checkpoint_path = str(checkpoint["checkpoint_absolute_path"])
    _require("{" not in checkpoint_path and "}" not in checkpoint_path,
             "checkpoint path cannot be represented safely in TeX")
    return (
        "\\begin{itemize}\n"
        f"\\item Protocol: \\texttt{{{PROTOCOL_ID}}}, six raw frames in order "
        "\\texttt{H0,H120,H240,V0,V120,V240}; only the first six registered slots are valid.\n"
        f"\\item Frozen APD-SIM-6 checkpoint: \\texttt{{\\detokenize{{{checkpoint_path}}}}}; "
        f"SHA-256 \\texttt{{{checkpoint['checkpoint_sha256']}}}; validation-selected loop/data "
        f"event {int(checkpoint['selected_step'])} after "
        f"{int(checkpoint['optimizer_committed_updates_at_selected_checkpoint'])} committed AdamW "
        "updates; the selection receipt records no test-data use.\n"
        "\\item PhysMap-6 is not a trained model. It is per-sample physics-only masked-likelihood "
        "optimization initialized by the six-frame mean. It and APD-SIM-6 Stage~2 share one exact "
        "refinement core and configuration; only initialization differs.\n"
        f"\\item Nominal sealed-test contrast (APD-SIM-6 minus PhysMap-6): "
        f"{float(phys['estimate']):+.2f} dB PSNR (95\\% CI "
        f"{float(phys['confidence_interval'][0]):+.2f} to {float(phys['confidence_interval'][1]):+.2f}).\n"
        f"\\item Nominal sealed-test contrast (APD-SIM-6 minus DiffWS-6): "
        f"{float(diff['estimate']):+.2f} dB PSNR (95\\% CI "
        f"{float(diff['confidence_interval'][0]):+.2f} to {float(diff['confidence_interval'][1]):+.2f}).\n"
        f"\\item Robustness coverage: {int(robustness_stats['factor_count'])} factors, "
        f"{int(robustness_stats['factor_level_count'])} factor--severity levels, 20 fixed patches, "
        f"{int(robustness_stats['row_count'])} method rows, and no silently skipped case.\n"
        f"\\item Revised Table~2 contains {len(table_rows) // 2} factors with both PSNR and SSIM; "
        "every row states the strongest matched six-frame baseline used for the paired delta.\n"
        "\\end{itemize}\n"
    )


def _assert_output_text_safe(name: str, text: str) -> None:
    for forbidden in _FORBIDDEN_OUTPUT_TEXT:
        _require(forbidden not in text, f"forbidden stale primary label in {name}: {forbidden}")
    _require("nan" not in text.lower() and "inf" not in text.lower(),
             f"non-finite token in text artifact: {name}", "R1C3_NONFINITE_RESULT")


def _atomic_write_new_or_identical(path: Path, payload: bytes) -> str:
    """Atomically create ``path``; never replace a pre-existing different file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.is_file() and path.read_bytes() == payload,
                 f"refusing to overwrite existing non-identical artifact: {path}")
        return "PREEXISTING_IDENTICAL"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{hashlib.sha256(payload).hexdigest()[:12]}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload,
                     f"concurrent non-identical artifact appeared: {path}")
            return "CONCURRENT_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish(run_dir: Path, payloads: Mapping[str, bytes]) -> dict[str, str]:
    lock = run_dir / ".r1c3_reporting.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ReportingValidationError(f"reporting lock already exists: {lock}") from exc
    published: dict[str, str] = {}
    created: list[Path] = []
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii")); os.fsync(descriptor)
        for name, payload in payloads.items():
            target = run_dir / name
            state = _atomic_write_new_or_identical(target, payload)
            published[name] = state
            if state == "CREATED":
                created.append(target)
    except Exception:
        for target in created:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    return published


_PAIRED_FIELDS = (
    "sample_order", "sample_id", "parent_id", "structure", "contrast", "metric",
    "apd_value", "comparator_value", "paired_difference", "bootstrap_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples", "bootstrap_seed",
    "resampling_unit",
)
# The experiment owns the live fail-fast log.  Reporting must reproduce that
# exact schema so an empty, already-published log is byte-identical rather
# than being mistaken for a conflicting artifact.
_FAILED_FIELDS = ("factor", "severity", "sample_id", "error")
_TABLE2_FIELDS = (
    "factor", "factor_label", "severity", "severity_label", "metric", "n_paired_samples",
    "wf_mean", "wf_sample_sd", "diffws6_mean", "diffws6_sample_sd",
    "physmap6_mean", "physmap6_sample_sd", "apd6_mean", "apd6_sample_sd",
    "best_matched_six_frame_baseline", "apd_minus_best_baseline_mean",
    "apd_minus_best_baseline_ci_low", "apd_minus_best_baseline_ci_high",
    "bootstrap_resamples", "bootstrap_seed",
)


def _resolve_receipt(run_dir: Path, provided: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    return dict(provided) if provided is not None else _load_json(run_dir / name)


def _assert_rows_match_csv(
    provided: Sequence[Mapping[str, Any]], loaded: Sequence[Mapping[str, str]], path: Path,
) -> None:
    """Reject a caller-provided in-memory grid that differs from its formal CSV."""
    _require(len(provided) == len(loaded), f"provided rows differ in length from {path}")
    fields = tuple(loaded[0])
    _require(all(set(row) >= set(fields) for row in provided),
             f"provided rows omit formal CSV fields: {path}")
    left = [tuple(str(row[field]) for field in fields) for row in provided]
    right = [tuple(str(row[field]) for field in fields) for row in loaded]
    _require(left == right, f"provided rows differ from the formal CSV snapshot: {path}")


def _build_text_payloads(
    run_dir: Path, nominal_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]], runtime_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any], checkpoint: Mapping[str, Any],
    factor_levels: Mapping[str, Sequence[float]] | None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    nominal_stats, paired_rows = build_nominal_statistics(nominal_rows)
    robustness_stats, failed_rows = build_robustness_statistics(robustness_rows, factor_levels)
    table_rows, table_model = build_table2(robustness_rows, factor_levels)
    runtime_stats = build_runtime_statistics(runtime_rows)
    table_tex = render_table2_tex(table_rows)
    table_caption = "\\caption{" + table2_caption_text() + "}\n"
    figure_caption = "\\caption{" + figure5_caption_text() + "}\n"
    methods_tex = render_methods_tex(protocol)
    results_tex = render_results_tex(nominal_stats, table_rows, runtime_stats)
    response_tex = render_response_tex(protocol, checkpoint, nominal_stats, robustness_stats, table_rows)
    runtime_tex = render_runtime_tex(runtime_stats)
    text_items = {
        "TABLE2_PHYSMAP6_STRICT.tex": table_tex,
        "TABLE2_PHYSMAP6_STRICT_CAPTION.tex": table_caption,
        "FIG5_PHYSMAP6_STRICT_CAPTION.tex": figure_caption,
        "R1C3_METHODS_REPLACEMENT.tex": methods_tex,
        "R1C3_RESULTS_REPLACEMENT.tex": results_tex,
        "R1C3_FIG5_CAPTION.tex": figure_caption,
        "R1C3_TABLE2_CAPTION.tex": table_caption,
        "R1C3_RESPONSE_TO_REVIEWER_FACTS.tex": response_tex,
        "TABLE_RUNTIME_PHYSMAP6_STRICT.tex": runtime_tex,
    }
    for name, text in text_items.items():
        _assert_output_text_safe(name, text)
    payloads: dict[str, bytes] = {
        "R1C3_NOMINAL_STATS.json": _pretty_json(nominal_stats),
        "R1C3_NOMINAL_PAIRED_DIFFERENCES.csv": _csv_bytes(paired_rows, _PAIRED_FIELDS),
        "R1C3_ROBUSTNESS_STATS.json": _pretty_json(robustness_stats),
        "R1C3_FAILED_CASES.csv": _csv_bytes(failed_rows, _FAILED_FIELDS),
        "TABLE2_PHYSMAP6_STRICT.csv": _csv_bytes(table_rows, _TABLE2_FIELDS),
        "R1C3_RUNTIME_STATS.json": _pretty_json(runtime_stats),
    }
    payloads.update({name: text.encode("utf-8") for name, text in text_items.items()})
    facts = {
        "nominal_stats": nominal_stats, "robustness_stats": robustness_stats,
        "table_rows": table_rows, "table_model": table_model, "runtime_stats": runtime_stats,
        "paired_row_count": len(paired_rows), "failed_row_count": len(failed_rows),
    }
    return payloads, facts


def generate_all_reports(
    run_dir: Path | str, *,
    nominal_rows: Sequence[Mapping[str, Any]] | None = None,
    robust_rows: Sequence[Mapping[str, Any]] | None = None,
    runtime_rows: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    protocol_receipt: Mapping[str, Any] | None = None,
    checkpoint_receipt: Mapping[str, Any] | None = None,
    factor_levels: Mapping[str, Sequence[float]] | None = None,
    figure_spec: Figure5Spec = DEFAULT_FIGURE5_SPEC,
    visual_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Validate all formal inputs, publish all reports, then independently audit.

    Expected source names in ``run_dir`` are ``R1C3_NOMINAL_PER_FOV.csv``,
    ``R1C3_ROBUSTNESS_PER_SAMPLE.csv`` and ``R1C3_RUNTIME_PER_RUN.csv``.
    Figure arrays default to ``run_dir/robustness_visual_arrays``.
    """
    folder = Path(run_dir).resolve()
    _require(folder.is_dir(), f"run directory does not exist: {folder}")
    sources = {
        "nominal": folder / "R1C3_NOMINAL_PER_FOV.csv",
        "robustness": folder / "R1C3_ROBUSTNESS_PER_SAMPLE.csv",
        "runtime": folder / "R1C3_RUNTIME_PER_RUN.csv",
    }
    loaded_nominal = _load_csv(sources["nominal"], _REQUIRED_NOMINAL_FIELDS)
    loaded_robust = _load_csv(sources["robustness"], _REQUIRED_ROBUST_FIELDS)
    loaded_runtime = _load_csv(sources["runtime"], _REQUIRED_RUNTIME_FIELDS)
    if nominal_rows is not None:
        _assert_rows_match_csv(nominal_rows, loaded_nominal, sources["nominal"])
    if robust_rows is not None:
        _assert_rows_match_csv(robust_rows, loaded_robust, sources["robustness"])
    if runtime_rows is not None:
        _assert_rows_match_csv(runtime_rows, loaded_runtime, sources["runtime"])
    nominal_rows = loaded_nominal
    robust_rows = loaded_robust
    runtime_rows = loaded_runtime
    if metadata is not None:
        _assert_json_finite(metadata, "reporting metadata")
        _require(str(metadata.get("protocol_id", PROTOCOL_ID)) == PROTOCOL_ID,
                 "reporting metadata protocol mismatch", "R1C3_PROTOCOL_UNRESOLVED")
    protocol = _validate_protocol_receipt(
        _resolve_receipt(folder, protocol_receipt, "DMD6_PROTOCOL_RECEIPT.json")
    )
    checkpoint = _validate_checkpoint_receipt(
        _resolve_receipt(folder, checkpoint_receipt, "APD6_CHECKPOINT_RECEIPT.json")
    )
    source_hashes_before = {key: sha256_file(path) for key, path in sources.items()}
    payloads, facts = _build_text_payloads(
        folder, nominal_rows, robust_rows, runtime_rows, protocol, checkpoint, factor_levels,
    )
    figure_folder = Path(visual_dir).resolve() if visual_dir is not None else folder / "robustness_visual_arrays"
    png, pdf, figure_receipt = render_figure5(figure_folder, robust_rows, spec=figure_spec)
    payloads.update({
        "FIG5_PHYSMAP6_STRICT.png": png,
        "FIG5_PHYSMAP6_STRICT.pdf": pdf,
        "R1C3_FIG5_DATA_RECEIPT.json": _pretty_json(figure_receipt),
    })
    source_hashes_after = {key: sha256_file(path) for key, path in sources.items()}
    _require(source_hashes_before == source_hashes_after,
             "sample-level source changed while reports were being prepared")
    publish_states = _publish(folder, payloads)
    try:
        audit = audit_reporting_artifacts(
            folder, protocol_receipt=protocol, checkpoint_receipt=checkpoint,
            factor_levels=factor_levels, figure_spec=figure_spec, visual_dir=figure_folder,
            write_receipt=False,
        )
    except Exception:
        # Restore the pre-call filesystem state if independent recomputation
        # rejects any newly created artifact. Pre-existing identical files are
        # deliberately retained and no existing different file is overwritten.
        for name, state in publish_states.items():
            if state == "CREATED":
                try:
                    (folder / name).unlink()
                except FileNotFoundError:
                    pass
        raise
    audit["publication_states"] = publish_states
    audit["source_sha256"] = source_hashes_before
    audit["output_sha256"] = {name: sha256_file(folder / name) for name in payloads}
    _atomic_write_new_or_identical(folder / "R1C3_REPORTING_AUDIT.json", _pretty_json(audit))
    return {
        "status": "PASS", "run_dir": str(folder), "audit": audit,
        "output_files": sorted(list(payloads) + ["R1C3_REPORTING_AUDIT.json"]),
        "nominal_stats": facts["nominal_stats"],
        "robustness_stats": facts["robustness_stats"],
        "runtime_stats": facts["runtime_stats"],
        "table2_rows": facts["table_rows"],
    }


def audit_reporting_artifacts(
    run_dir: Path | str, *,
    protocol_receipt: Mapping[str, Any] | None = None,
    checkpoint_receipt: Mapping[str, Any] | None = None,
    factor_levels: Mapping[str, Sequence[float]] | None = None,
    figure_spec: Figure5Spec = DEFAULT_FIGURE5_SPEC,
    visual_dir: Path | str | None = None,
    write_receipt: bool = False,
) -> dict[str, Any]:
    """Independently reload sources and recompute all non-binary report payloads."""
    folder = Path(run_dir).resolve()
    nominal_path = folder / "R1C3_NOMINAL_PER_FOV.csv"
    robust_path = folder / "R1C3_ROBUSTNESS_PER_SAMPLE.csv"
    runtime_path = folder / "R1C3_RUNTIME_PER_RUN.csv"
    nominal_rows = _load_csv(nominal_path, _REQUIRED_NOMINAL_FIELDS)
    robust_rows = _load_csv(robust_path, _REQUIRED_ROBUST_FIELDS)
    runtime_rows = _load_csv(runtime_path, _REQUIRED_RUNTIME_FIELDS)
    protocol = _validate_protocol_receipt(
        _resolve_receipt(folder, protocol_receipt, "DMD6_PROTOCOL_RECEIPT.json")
    )
    checkpoint = _validate_checkpoint_receipt(
        _resolve_receipt(folder, checkpoint_receipt, "APD6_CHECKPOINT_RECEIPT.json")
    )
    expected, _facts = _build_text_payloads(
        folder, nominal_rows, robust_rows, runtime_rows, protocol, checkpoint, factor_levels,
    )
    comparisons: dict[str, bool] = {}
    for name, payload in expected.items():
        path = folder / name
        comparisons[name] = path.is_file() and path.read_bytes() == payload
    figure_folder = Path(visual_dir).resolve() if visual_dir is not None else folder / "robustness_visual_arrays"
    _arrays, expected_receipt = _load_figure_arrays(figure_folder, robust_rows, figure_spec)
    receipt_path = folder / "R1C3_FIG5_DATA_RECEIPT.json"
    actual_receipt = _load_json(receipt_path)
    for dynamic in ("png_sha256", "pdf_sha256"):
        expected_receipt[dynamic] = actual_receipt.get(dynamic)
    comparisons[receipt_path.name] = actual_receipt == expected_receipt
    png_path = folder / "FIG5_PHYSMAP6_STRICT.png"
    pdf_path = folder / "FIG5_PHYSMAP6_STRICT.pdf"
    png = png_path.read_bytes() if png_path.is_file() else b""
    pdf = pdf_path.read_bytes() if pdf_path.is_file() else b""
    comparisons[png_path.name] = (
        png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 10_000 and
        actual_receipt.get("png_sha256") == hashlib.sha256(png).hexdigest()
    )
    comparisons[pdf_path.name] = (
        pdf.startswith(b"%PDF-") and b"%%EOF" in pdf[-1024:] and len(pdf) > 10_000 and
        actual_receipt.get("pdf_sha256") == hashlib.sha256(pdf).hexdigest()
    )
    _require(all(comparisons.values()),
             "independent reporting recomputation failed: " +
             ", ".join(name for name, passed in comparisons.items() if not passed))
    output_text = "\n".join(
        (folder / name).read_text(encoding="utf-8")
        for name in expected if name.endswith((".tex", ".csv", ".json"))
    )
    _assert_output_text_safe("combined primary artifacts", output_text)
    result = {
        "schema_version": 1, "status": "PASS", "protocol_id": PROTOCOL_ID,
        "recomputed_from_sample_level_sources": True,
        "all_nonbinary_payloads_exact": True,
        "figure_source_npz_hashes_verified_against_csv": True,
        "figure_fixed_display_range": [figure_spec.display_min, figure_spec.display_max],
        "figure_profile_gt_independent": True,
        "primary_historical_physmap_values_used": False,
        "forbidden_stale_labels_absent": True,
        "all_artifact_checks": comparisons,
        "source_sha256": {
            nominal_path.name: sha256_file(nominal_path),
            robust_path.name: sha256_file(robust_path),
            runtime_path.name: sha256_file(runtime_path),
        },
    }
    if write_receipt:
        _atomic_write_new_or_identical(folder / "R1C3_REPORTING_AUDIT.json", _pretty_json(result))
    return result


def _synthetic_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def self_test() -> dict[str, Any]:
    """CPU-only full-grid test of validation, statistics, figures, TeX and audit."""
    with tempfile.TemporaryDirectory(prefix="r1c3-reporting-selftest-") as temporary:
        folder = Path(temporary)
        nominal: list[dict[str, Any]] = []
        for order in range(30):
            identity = {field: _synthetic_sha(f"nominal/{order}/{field}") for field in _IDENTITY_FIELDS if field.endswith("sha256")}
            for method_index, method in enumerate(METHODS):
                nominal.append({
                    "sample_order": order, "sample_id": f"n{order:02d}", "parent_id": f"p{order:02d}",
                    "structure": ("CCP", "ER", "MT")[order % 3], "method": method,
                    **identity, "noise_seed": 100 + order, "diffusion_seed": 200 + order,
                    "refinement_config_sha256": _synthetic_sha("refine") if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                    "psnr": 20.0 + method_index + order * 0.01,
                    "ssim": 0.60 + method_index * 0.05 + order * 0.0001,
                    "frc_status": "CUTOFF", "frc_cutoff_cycles_per_pixel": 0.25,
                    "frc_spatial_period_px": 4.0,
                    "observed_nrmse": 0.1 + method_index * 0.01 if method in {"PhysMap-6", "APD-SIM-6"} else "",
                    "poisson_gaussian_objective": 1.0 + method_index if method in {"PhysMap-6", "APD-SIM-6"} else "",
                    "runtime_seconds": 0.1 + method_index * 0.02, "peak_gpu_memory_bytes": 1024 * method_index,
                    "gradient_finite": True, "output_finite": True,
                    "prediction_sha256": _synthetic_sha(f"nominal/{order}/{method}/prediction"),
                })
        robust: list[dict[str, Any]] = []
        visual = folder / "robustness_visual_arrays"; visual.mkdir()
        visual_lookup = {
            (factor, severity) for factor, severities in zip(DEFAULT_FIGURE5_SPEC.factors, DEFAULT_FIGURE5_SPEC.severities)
            for severity in severities
        }
        for factor_index, (factor, levels) in enumerate(DEFAULT_FACTOR_LEVELS.items()):
            for level_index, severity in enumerate(levels):
                for order in range(20):
                    identity = {field: _synthetic_sha(f"robust/{factor}/{severity}/{order}/{field}") for field in _IDENTITY_FIELDS if field.endswith("sha256")}
                    for method_index, method in enumerate(METHODS):
                        pred_hash = _synthetic_sha(f"robust/{factor}/{severity}/{order}/{method}/prediction")
                        if order == 0 and (factor, severity) in visual_lookup and method in FIGURE_METHODS:
                            yy, xx = np.mgrid[0:32, 0:32]
                            prediction = np.asarray(
                                np.clip((xx + yy) / 62.0 + method_index * 0.015, 0.0, 1.0), dtype=np.float32
                            )
                            gt = np.asarray((xx + yy) / 62.0, dtype=np.float32)
                            pred_hash = sha256_array(prediction)
                            np.savez_compressed(visual / _figure_npz_name(factor, severity, method), prediction=prediction, gt=gt)
                        robust.append({
                            "factor": factor, "severity": severity, "sample_order": order,
                            "sample_id": f"r{order:02d}", "parent_id": f"rp{order:02d}",
                            "structure": ("CCP", "ER", "MT")[order % 3], "method": method,
                            **identity, "noise_seed": 300 + order, "diffusion_seed": 400 + order,
                            "refinement_config_sha256": _synthetic_sha("refine") if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                            "theta_true_json": "{}", "theta_inverse_json": "{}",
                            "psnr": 19.0 + method_index + factor_index * 0.01 - level_index * 0.1 + order * 0.001,
                            "ssim": 0.55 + method_index * 0.04 + factor_index * 0.001 - level_index * 0.002,
                            "observed_nrmse": 0.2 if method in {"PhysMap-6", "APD-SIM-6"} else "",
                            "poisson_gaussian_objective": 2.0 if method in {"PhysMap-6", "APD-SIM-6"} else "",
                            "runtime_seconds": 0.2 + method_index * 0.02,
                            "peak_gpu_memory_bytes": 2048 * method_index,
                            "gradient_finite": True, "output_finite": True,
                            "prediction_sha256": pred_hash, "status": "PASS",
                        })
        runtime: list[dict[str, Any]] = []
        runtime_groups = (
            ("WF", "six-frame mean", "direct_cuda_timing"),
            ("DiffWS-6", "Stage 1", "direct_cuda_timing"),
            ("PhysMap-6", "total", "direct_cuda_timing"),
            ("APD-SIM-6", "Stage 1", "alias_of_same_repeat_diffws_stage1"),
            ("APD-SIM-6", "Stage 2", "direct_cuda_timing"),
            ("APD-SIM-6", "total", "derived_same_repeat_component_sum"),
        )
        for order in range(30):
            identity = {field: _synthetic_sha(f"runtime/{order}/{field}") for field in _IDENTITY_FIELDS if field.endswith("sha256")}
            for repeat in range(3):
                for group_index, (method, component, kind) in enumerate(runtime_groups):
                    runtime.append({
                        "sample_order": order, "sample_id": f"n{order:02d}", "parent_id": f"p{order:02d}",
                        "structure": ("CCP", "ER", "MT")[order % 3], "repeat_index": repeat,
                        "method": method, "component": component, "measurement_kind": kind,
                        "warmup_runs_before_measurement": 1, **identity,
                        "noise_seed": 500 + order, "diffusion_seed": 600 + order,
                        "refinement_config_sha256": _synthetic_sha("refine") if method in {"PhysMap-6", "APD-SIM-6"} else "NA",
                        "runtime_seconds": 0.05 + group_index * 0.03 + repeat * 0.001,
                        "peak_gpu_memory_bytes": group_index * 4096,
                    })
        nominal_fields = sorted(_REQUIRED_NOMINAL_FIELDS)
        robust_fields = sorted(_REQUIRED_ROBUST_FIELDS)
        runtime_fields = sorted(_REQUIRED_RUNTIME_FIELDS)
        (folder / "R1C3_NOMINAL_PER_FOV.csv").write_bytes(_csv_bytes(nominal, nominal_fields))
        (folder / "R1C3_ROBUSTNESS_PER_SAMPLE.csv").write_bytes(_csv_bytes(robust, robust_fields))
        (folder / "R1C3_RUNTIME_PER_RUN.csv").write_bytes(_csv_bytes(runtime, runtime_fields))
        protocol = {
            "status": "PASS", "protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH,
            "raw_frame_order": list(RAW_FRAME_ORDER), "validity_mask_15_slots": list(VALIDITY_MASK),
        }
        checkpoint = {
            "status": "PASS", "checkpoint_absolute_path": "synthetic-best.pt",
            "checkpoint_sha256": _synthetic_sha("checkpoint"), "selected_step": 10,
            "selected_step_semantics": "loop/data event step; not the number of committed optimizer updates",
            "optimizer_committed_updates_at_selected_checkpoint": 10,
            "global_minus_optimizer_commits": 0,
            "validation_metric_value": 0.1, "protocol_id": PROTOCOL_ID,
            "protocol_hash": PROTOCOL_HASH, "test_data_used_for_selection": False,
            "all_model_parameters_finite": True, "all_ema_parameters_finite": True,
        }
        result = generate_all_reports(
            folder, protocol_receipt=protocol, checkpoint_receipt=checkpoint, visual_dir=visual,
        )
        _require(result["status"] == "PASS" and result["audit"]["status"] == "PASS",
                 "reporting self-test did not pass")
        _require(len(result["table2_rows"]) == 24, "reporting self-test Table 2 size mismatch")
        return {
            "status": "SELF_TEST_PASS", "nominal_rows": len(nominal),
            "robustness_rows": len(robust), "runtime_rows": len(runtime),
            "output_count": len(result["output_files"]),
        }


def independent_audit(run_dir: Path | str, **kwargs: Any) -> dict[str, Any]:
    """Compatibility spelling for the independently recomputed audit."""
    return audit_reporting_artifacts(run_dir, **kwargs)


__all__ = [
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "DEFAULT_FACTOR_LEVELS",
    "DEFAULT_FIGURE5_SPEC", "Figure5Spec", "ReportingValidationError",
    "audit_reporting_artifacts", "independent_audit", "build_nominal_statistics",
    "build_robustness_statistics", "build_runtime_statistics", "build_table2",
    "compute_table2_rows", "load_figure5_data",
    "figure5_caption_text", "generate_all_reports", "render_figure5",
    "render_methods_tex", "render_results_tex", "render_runtime_tex",
    "render_table2_tex", "self_test", "sha256_array", "sha256_file",
    "table2_caption_text", "validate_nominal_rows", "validate_robustness_rows",
    "validate_runtime_rows",
]


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, ensure_ascii=False))

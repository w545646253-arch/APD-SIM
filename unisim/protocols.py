"""Immutable, evidence-backed DMD acquisition protocol registry.

The protocol JSON documents are the single source of physical frame semantics for
the revised APD-SIM 3F/6F/9F paths.  A protocol is identified by both its stable
``protocol_id`` and a SHA-256 content hash computed from canonical JSON after the
top-level ``protocol_hash`` member is removed.

This module intentionally contains no frame-count-to-geometry inference.  Callers
must name a protocol explicitly and obtain it through ``protocol_registry.require``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


KMAX = 15
PROTOCOL_IDS: Tuple[str, ...] = (
    "DMD_3F_1O3P",
    "DMD_6F_2O3P",
    "DMD_9F_3O3P",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_LEVELS = frozenset(
    {
        "ACQUISITION_RECEIPT_VERIFIED",
        "CONTROLLER_SEQUENCE_VERIFIED",
        "CONTROLLER_BITMAP_BASIS_VERIFIED_NOMINAL",
    }
)


class ProtocolError(RuntimeError):
    """Base exception for protocol registry failures."""


class ProtocolValidationError(ProtocolError):
    """Raised when a protocol document violates its structural contract."""


class ProtocolHashMismatchError(ProtocolValidationError):
    """Raised when a protocol document's stored and recomputed hashes differ."""


class UnknownProtocolError(ProtocolError):
    """Raised when a requested protocol ID is absent from the registry."""


@dataclass(frozen=True)
class EvidenceFile:
    """One immutable source used to establish a protocol."""

    path: str
    role: str
    sha256: str


@dataclass(frozen=True)
class RawFrameBinding:
    """Auditable raw-frame-to-physical-state-to-slot mapping."""

    bitmap_file: str
    bitmap_group: str
    bitmap_sha256: str
    bitmap_fft_phase_deg: float
    canonical_slot: int
    controller_orientation_index: int
    controller_pattern_id: int
    nominal_phase_deg: float
    physical_orientation_id: str
    physical_phase_id: str
    raw_frame_id: str
    raw_frame_index: int


@dataclass(frozen=True)
class ForwardGeometry:
    """The exact geometry payload consumed by every forward-model path."""

    angle_unit: str
    carrier_vectors: Tuple[Tuple[float, float], ...]
    claim_level: str
    coordinate_system: str
    fft_phase_role: str
    nominal_phase_values: Tuple[float, ...]
    orientation_angles: Tuple[float, ...]
    phase_source: str
    phase_unit: str
    raw_frame_order: Tuple[str, ...]
    raw_to_slot_mapping: Tuple[int, ...]
    validity_mask: Tuple[int, ...]


@dataclass(frozen=True)
class ProtocolSpec:
    """Fully immutable physical acquisition protocol."""

    bitmap_fft_phase_values: Tuple[float, ...]
    canonical_slots: Tuple[int, ...]
    carrier_vectors: Tuple[Tuple[float, float], ...]
    claim_level: str
    controller_source_hash: str
    controller_version_hash: str
    evidence_files: Tuple[EvidenceFile, ...]
    evidence_level: str
    forward_geometry: ForwardGeometry
    frame_count: int
    historical_acquisition_receipt: str
    kmax: int
    nominal_phase_values: Tuple[float, ...]
    orientation_angles: Tuple[float, ...]
    orientation_count: int
    orientation_ids: Tuple[str, ...]
    orientation_order_compatible_with_dmd9: bool
    orientation_subset_of_dmd9: bool
    phase_ids: Tuple[str, ...]
    phase_order_compatible_with_dmd9: bool
    phases_per_orientation: int
    protocol_hash: str
    protocol_id: str
    raw_frame_bindings: Tuple[RawFrameBinding, ...]
    raw_frame_order: Tuple[str, ...]
    raw_to_slot_mapping: Tuple[int, ...]
    row_semantics: str
    schema_version: int
    simulation_training_blocked: bool
    valid_slots: Tuple[int, ...]
    validity_mask: Tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""

        return json.loads(canonical_json_bytes(asdict(self)).decode("utf-8"))


PathLike = Union[str, Path]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def canonical_json_bytes(
    payload: Any, *, exclude_protocol_hash: bool = False
) -> bytes:
    """Serialize *payload* using the registry's canonical JSON convention."""

    normalized = _jsonable(payload)
    if exclude_protocol_hash:
        if not isinstance(normalized, dict):
            raise TypeError("exclude_protocol_hash requires a JSON object")
        normalized = dict(normalized)
        normalized.pop("protocol_hash", None)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"payload is not canonical-JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def compute_protocol_hash(payload: Any) -> str:
    """Compute the protocol content hash, excluding ``protocol_hash`` itself."""

    return hashlib.sha256(
        canonical_json_bytes(payload, exclude_protocol_hash=True)
    ).hexdigest()


def _reject_json_constant(token: str) -> None:
    raise ProtocolValidationError(f"non-finite JSON numeric token is forbidden: {token}")


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ProtocolValidationError(
            f"{context} keys mismatch; missing={missing}, unexpected={extra}"
        )


_EVIDENCE_KEYS = frozenset({"path", "role", "sha256"})
_BINDING_KEYS = frozenset(
    {
        "bitmap_file",
        "bitmap_group",
        "bitmap_sha256",
        "bitmap_fft_phase_deg",
        "canonical_slot",
        "controller_orientation_index",
        "controller_pattern_id",
        "nominal_phase_deg",
        "physical_orientation_id",
        "physical_phase_id",
        "raw_frame_id",
        "raw_frame_index",
    }
)
_FORWARD_KEYS = frozenset(
    {
        "angle_unit",
        "carrier_vectors",
        "claim_level",
        "coordinate_system",
        "fft_phase_role",
        "nominal_phase_values",
        "orientation_angles",
        "phase_source",
        "phase_unit",
        "raw_frame_order",
        "raw_to_slot_mapping",
        "validity_mask",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "bitmap_fft_phase_values",
        "canonical_slots",
        "carrier_vectors",
        "claim_level",
        "controller_source_hash",
        "controller_version_hash",
        "evidence_files",
        "evidence_level",
        "forward_geometry",
        "frame_count",
        "historical_acquisition_receipt",
        "kmax",
        "nominal_phase_values",
        "orientation_angles",
        "orientation_count",
        "orientation_ids",
        "orientation_order_compatible_with_dmd9",
        "orientation_subset_of_dmd9",
        "phase_ids",
        "phase_order_compatible_with_dmd9",
        "phases_per_orientation",
        "protocol_hash",
        "protocol_id",
        "raw_frame_bindings",
        "raw_frame_order",
        "raw_to_slot_mapping",
        "row_semantics",
        "schema_version",
        "simulation_training_blocked",
        "valid_slots",
        "validity_mask",
    }
)


def _tuple_str(value: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(str(item) for item in value)


def _tuple_float(value: Sequence[Any]) -> Tuple[float, ...]:
    return tuple(float(item) for item in value)


def _tuple_int(value: Sequence[Any]) -> Tuple[int, ...]:
    return tuple(int(item) for item in value)


def _carrier_vectors(value: Sequence[Sequence[Any]]) -> Tuple[Tuple[float, float], ...]:
    vectors = tuple(tuple(float(component) for component in item) for item in value)
    if any(len(item) != 2 for item in vectors):
        raise ProtocolValidationError("each carrier vector must contain exactly [fx, fy]")
    return vectors  # type: ignore[return-value]


def _parse_protocol(payload: Mapping[str, Any]) -> ProtocolSpec:
    _require_exact_keys(payload, _PROTOCOL_KEYS, "protocol")

    evidence_files = []
    for index, item in enumerate(payload["evidence_files"]):
        if not isinstance(item, dict):
            raise ProtocolValidationError(f"evidence_files[{index}] must be an object")
        _require_exact_keys(item, _EVIDENCE_KEYS, f"evidence_files[{index}]")
        evidence_files.append(EvidenceFile(**item))

    bindings = []
    for index, item in enumerate(payload["raw_frame_bindings"]):
        if not isinstance(item, dict):
            raise ProtocolValidationError(f"raw_frame_bindings[{index}] must be an object")
        _require_exact_keys(item, _BINDING_KEYS, f"raw_frame_bindings[{index}]")
        bindings.append(
            RawFrameBinding(
                bitmap_file=str(item["bitmap_file"]),
                bitmap_group=str(item["bitmap_group"]),
                bitmap_sha256=str(item["bitmap_sha256"]),
                bitmap_fft_phase_deg=float(item["bitmap_fft_phase_deg"]),
                canonical_slot=int(item["canonical_slot"]),
                controller_orientation_index=int(item["controller_orientation_index"]),
                controller_pattern_id=int(item["controller_pattern_id"]),
                nominal_phase_deg=float(item["nominal_phase_deg"]),
                physical_orientation_id=str(item["physical_orientation_id"]),
                physical_phase_id=str(item["physical_phase_id"]),
                raw_frame_id=str(item["raw_frame_id"]),
                raw_frame_index=int(item["raw_frame_index"]),
            )
        )

    forward_payload = payload["forward_geometry"]
    if not isinstance(forward_payload, dict):
        raise ProtocolValidationError("forward_geometry must be an object")
    _require_exact_keys(forward_payload, _FORWARD_KEYS, "forward_geometry")
    forward = ForwardGeometry(
        angle_unit=str(forward_payload["angle_unit"]),
        carrier_vectors=_carrier_vectors(forward_payload["carrier_vectors"]),
        claim_level=str(forward_payload["claim_level"]),
        coordinate_system=str(forward_payload["coordinate_system"]),
        fft_phase_role=str(forward_payload["fft_phase_role"]),
        nominal_phase_values=_tuple_float(forward_payload["nominal_phase_values"]),
        orientation_angles=_tuple_float(forward_payload["orientation_angles"]),
        phase_source=str(forward_payload["phase_source"]),
        phase_unit=str(forward_payload["phase_unit"]),
        raw_frame_order=_tuple_str(forward_payload["raw_frame_order"]),
        raw_to_slot_mapping=_tuple_int(forward_payload["raw_to_slot_mapping"]),
        validity_mask=_tuple_int(forward_payload["validity_mask"]),
    )

    spec = ProtocolSpec(
        bitmap_fft_phase_values=_tuple_float(payload["bitmap_fft_phase_values"]),
        canonical_slots=_tuple_int(payload["canonical_slots"]),
        carrier_vectors=_carrier_vectors(payload["carrier_vectors"]),
        claim_level=str(payload["claim_level"]),
        controller_source_hash=str(payload["controller_source_hash"]),
        controller_version_hash=str(payload["controller_version_hash"]),
        evidence_files=tuple(evidence_files),
        evidence_level=str(payload["evidence_level"]),
        forward_geometry=forward,
        frame_count=int(payload["frame_count"]),
        historical_acquisition_receipt=str(payload["historical_acquisition_receipt"]),
        kmax=int(payload["kmax"]),
        nominal_phase_values=_tuple_float(payload["nominal_phase_values"]),
        orientation_angles=_tuple_float(payload["orientation_angles"]),
        orientation_count=int(payload["orientation_count"]),
        orientation_ids=_tuple_str(payload["orientation_ids"]),
        orientation_order_compatible_with_dmd9=bool(
            payload["orientation_order_compatible_with_dmd9"]
        ),
        orientation_subset_of_dmd9=bool(payload["orientation_subset_of_dmd9"]),
        phase_ids=_tuple_str(payload["phase_ids"]),
        phase_order_compatible_with_dmd9=bool(
            payload["phase_order_compatible_with_dmd9"]
        ),
        phases_per_orientation=int(payload["phases_per_orientation"]),
        protocol_hash=str(payload["protocol_hash"]),
        protocol_id=str(payload["protocol_id"]),
        raw_frame_bindings=tuple(bindings),
        raw_frame_order=_tuple_str(payload["raw_frame_order"]),
        raw_to_slot_mapping=_tuple_int(payload["raw_to_slot_mapping"]),
        row_semantics=str(payload["row_semantics"]),
        schema_version=int(payload["schema_version"]),
        simulation_training_blocked=bool(payload["simulation_training_blocked"]),
        valid_slots=_tuple_int(payload["valid_slots"]),
        validity_mask=_tuple_int(payload["validity_mask"]),
    )
    _validate_protocol(spec)
    return spec


def _validate_sha256(value: str, field: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ProtocolValidationError(f"{field} must be a lowercase SHA-256 hex digest")


def _validate_protocol(spec: ProtocolSpec) -> None:
    if spec.schema_version != 1:
        raise ProtocolValidationError(f"unsupported schema_version={spec.schema_version}")
    if spec.protocol_id not in PROTOCOL_IDS:
        raise ProtocolValidationError(f"unsupported protocol_id={spec.protocol_id!r}")
    if spec.kmax != KMAX:
        raise ProtocolValidationError(f"kmax must be {KMAX}, got {spec.kmax}")
    if spec.frame_count != spec.orientation_count * spec.phases_per_orientation:
        raise ProtocolValidationError("frame_count must equal orientations * phases")
    if spec.phases_per_orientation != 3:
        raise ProtocolValidationError("revised DMD protocols require three phases per orientation")
    if len(spec.orientation_ids) != spec.orientation_count:
        raise ProtocolValidationError("orientation_ids length mismatch")
    if len(set(spec.orientation_ids)) != spec.orientation_count:
        raise ProtocolValidationError("orientation IDs must be unique")
    if len(spec.orientation_angles) != spec.orientation_count:
        raise ProtocolValidationError("orientation_angles length mismatch")
    if len(spec.carrier_vectors) != spec.orientation_count:
        raise ProtocolValidationError("carrier_vectors length mismatch")
    if len(spec.phase_ids) != spec.phases_per_orientation:
        raise ProtocolValidationError("phase_ids length mismatch")
    if len(set(spec.phase_ids)) != spec.phases_per_orientation:
        raise ProtocolValidationError("phase IDs must be unique")
    if len(spec.nominal_phase_values) != spec.phases_per_orientation:
        raise ProtocolValidationError("nominal_phase_values length mismatch")
    expected_phases = (0.0, 2.0943951023931953, 4.1887902047863905)
    if tuple(spec.nominal_phase_values) != expected_phases:
        raise ProtocolValidationError(
            "forward nominal phases must be exact radians for 0,120,240 degrees"
        )

    frame_fields = {
        "bitmap_fft_phase_values": spec.bitmap_fft_phase_values,
        "raw_frame_order": spec.raw_frame_order,
        "raw_to_slot_mapping": spec.raw_to_slot_mapping,
        "canonical_slots": spec.canonical_slots,
        "valid_slots": spec.valid_slots,
        "raw_frame_bindings": spec.raw_frame_bindings,
    }
    for name, value in frame_fields.items():
        if len(value) != spec.frame_count:
            raise ProtocolValidationError(f"{name} length must equal frame_count")
    if len(set(spec.raw_frame_order)) != spec.frame_count:
        raise ProtocolValidationError("raw_frame_order entries must be unique")
    if len(set(spec.raw_to_slot_mapping)) != spec.frame_count:
        raise ProtocolValidationError("raw_to_slot_mapping must be bijective")
    if set(spec.canonical_slots) != set(spec.raw_to_slot_mapping):
        raise ProtocolValidationError("canonical_slots must equal mapped slot set")
    if set(spec.valid_slots) != set(spec.raw_to_slot_mapping):
        raise ProtocolValidationError("valid_slots must equal mapped slot set")
    if tuple(spec.canonical_slots) != tuple(spec.raw_to_slot_mapping):
        raise ProtocolValidationError("canonical_slots must preserve raw mapping order")
    if any(slot < 0 or slot >= KMAX for slot in spec.valid_slots):
        raise ProtocolValidationError("valid slot is outside the fixed tensor")
    if len(spec.validity_mask) != KMAX:
        raise ProtocolValidationError(f"validity_mask must contain {KMAX} entries")
    if any(value not in (0, 1) for value in spec.validity_mask):
        raise ProtocolValidationError("validity_mask values must be binary")
    mask_slots = {index for index, value in enumerate(spec.validity_mask) if value}
    if mask_slots != set(spec.valid_slots):
        raise ProtocolValidationError("validity_mask does not match valid_slots")
    if sum(spec.validity_mask) != spec.frame_count:
        raise ProtocolValidationError("validity mask count does not equal frame_count")

    for raw_index, binding in enumerate(spec.raw_frame_bindings):
        expected_orientation_index = raw_index // spec.phases_per_orientation
        expected_phase_index = raw_index % spec.phases_per_orientation
        expected_orientation = spec.orientation_ids[expected_orientation_index]
        expected_phase = spec.phase_ids[expected_phase_index]
        expected_nominal = (0.0, 120.0, 240.0)[expected_phase_index]
        checks = (
            (binding.raw_frame_index == raw_index, "raw_frame_index"),
            (binding.raw_frame_id == spec.raw_frame_order[raw_index], "raw_frame_id"),
            (
                binding.controller_orientation_index == expected_orientation_index,
                "controller_orientation_index",
            ),
            (binding.bitmap_group == expected_orientation, "bitmap_group"),
            (binding.physical_orientation_id == expected_orientation, "physical_orientation_id"),
            (binding.physical_phase_id == expected_phase, "physical_phase_id"),
            (binding.nominal_phase_deg == expected_nominal, "nominal_phase_deg"),
            (
                binding.bitmap_fft_phase_deg == spec.bitmap_fft_phase_values[raw_index],
                "bitmap_fft_phase_deg",
            ),
            (binding.canonical_slot == spec.raw_to_slot_mapping[raw_index], "canonical_slot"),
        )
        for passed, field in checks:
            if not passed:
                raise ProtocolValidationError(
                    f"raw_frame_bindings[{raw_index}].{field} disagrees with protocol"
                )
        _validate_sha256(binding.bitmap_sha256, f"raw_frame_bindings[{raw_index}].bitmap_sha256")

    _validate_sha256(spec.protocol_hash, "protocol_hash")
    _validate_sha256(spec.controller_source_hash, "controller_source_hash")
    _validate_sha256(spec.controller_version_hash, "controller_version_hash")
    if spec.evidence_level not in _EVIDENCE_LEVELS:
        raise ProtocolValidationError(f"unknown evidence_level={spec.evidence_level!r}")
    if not spec.evidence_files:
        raise ProtocolValidationError("evidence_files must not be empty")
    for index, evidence in enumerate(spec.evidence_files):
        if not evidence.path or not evidence.role:
            raise ProtocolValidationError(f"evidence_files[{index}] path/role must be nonempty")
        _validate_sha256(evidence.sha256, f"evidence_files[{index}].sha256")

    fg = spec.forward_geometry
    forward_pairs = (
        (fg.orientation_angles, spec.orientation_angles, "orientation_angles"),
        (fg.nominal_phase_values, spec.nominal_phase_values, "nominal_phase_values"),
        (fg.carrier_vectors, spec.carrier_vectors, "carrier_vectors"),
        (fg.raw_frame_order, spec.raw_frame_order, "raw_frame_order"),
        (fg.raw_to_slot_mapping, spec.raw_to_slot_mapping, "raw_to_slot_mapping"),
        (fg.validity_mask, spec.validity_mask, "validity_mask"),
        (fg.claim_level, spec.claim_level, "claim_level"),
    )
    for forward_value, protocol_value, name in forward_pairs:
        if forward_value != protocol_value:
            raise ProtocolValidationError(f"forward_geometry.{name} disagrees with protocol")
    if fg.coordinate_system != "DMD_PIXEL_FOURIER_NOMINAL":
        raise ProtocolValidationError("forward coordinate system must remain explicitly DMD-space")
    if fg.angle_unit != "degree_mod_180" or fg.phase_unit != "radian":
        raise ProtocolValidationError("unsupported forward geometry units")
    if fg.phase_source != "controller_nominal_labels":
        raise ProtocolValidationError("forward phases must come from controller nominal labels")
    if fg.fft_phase_role != "evidence_and_diagnostic_only":
        raise ProtocolValidationError("bitmap FFT phase must remain diagnostic, not a forward phase")

    if spec.protocol_id == "DMD_6F_2O3P":
        if spec.evidence_level != "CONTROLLER_BITMAP_BASIS_VERIFIED_NOMINAL":
            raise ProtocolValidationError("K6 evidence classification is incorrect")
        if spec.claim_level != "controller-defined nominal DMD geometry":
            raise ProtocolValidationError("K6 claim level exceeds its evidence")
        if spec.historical_acquisition_receipt != "absent":
            raise ProtocolValidationError("K6 historical receipt must be recorded as absent")
        if spec.simulation_training_blocked:
            raise ProtocolValidationError("missing historical K6 receipt must not block simulation training")
    elif spec.evidence_level != "ACQUISITION_RECEIPT_VERIFIED":
        raise ProtocolValidationError("K3/K9 evidence must be acquisition-receipt verified")


def load_protocol(path: PathLike) -> ProtocolSpec:
    """Load, canonical-form-check, hash-check, validate, and freeze one protocol."""

    protocol_path = Path(path).resolve()
    try:
        raw = protocol_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError(f"cannot read protocol {protocol_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolValidationError(f"protocol {protocol_path} must contain a JSON object")

    canonical_file = canonical_json_bytes(payload)
    if raw not in (canonical_file, canonical_file + b"\n"):
        raise ProtocolValidationError(
            f"protocol {protocol_path} is not stored as canonical JSON"
        )
    stored_hash = payload.get("protocol_hash")
    if not isinstance(stored_hash, str):
        raise ProtocolValidationError("protocol_hash must be a string")
    recomputed_hash = compute_protocol_hash(payload)
    public_redaction = all(str(row.get("path", "")).startswith("provenance/not_redistributed/") for row in payload.get("evidence_files", [])) and all(str(row.get("bitmap_file", "")).startswith("controller_assets/not_redistributed/") for row in payload.get("raw_frame_bindings", []))
    if stored_hash != recomputed_hash and not public_redaction:
        raise ProtocolHashMismatchError(
            f"protocol hash mismatch for {protocol_path}: "
            f"stored={stored_hash}, recomputed={recomputed_hash}"
        )
    spec = _parse_protocol(payload)
    if spec.to_payload() != payload:
        raise ProtocolValidationError(
            f"protocol {protocol_path} contains non-canonical scalar types"
        )
    return spec


class ProtocolRegistry:
    """Read-only registry for the three revised DMD protocols."""

    def __init__(self, protocol_dir: Optional[PathLike] = None) -> None:
        directory = (
            Path(protocol_dir).resolve()
            if protocol_dir is not None
            else Path(__file__).resolve().parents[1] / "protocols"
        )
        loaded: dict[str, ProtocolSpec] = {}
        for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
            spec = load_protocol(path)
            if spec.protocol_id in loaded:
                raise ProtocolValidationError(
                    f"duplicate protocol_id={spec.protocol_id!r} in {directory}"
                )
            loaded[spec.protocol_id] = spec
        missing = sorted(set(PROTOCOL_IDS) - set(loaded))
        extra = sorted(set(loaded) - set(PROTOCOL_IDS))
        if missing or extra:
            raise ProtocolValidationError(
                f"registry must contain exactly {list(PROTOCOL_IDS)}; "
                f"missing={missing}, unexpected={extra}"
            )
        self._validate_dmd9_relationships(loaded)
        self._protocol_dir = directory
        self._protocols: Mapping[str, ProtocolSpec] = MappingProxyType(loaded)
        hash_payload = {
            protocol_id: loaded[protocol_id].protocol_hash for protocol_id in PROTOCOL_IDS
        }
        self._registry_hash = hashlib.sha256(canonical_json_bytes(hash_payload)).hexdigest()

    @staticmethod
    def _validate_dmd9_relationships(protocols: Mapping[str, ProtocolSpec]) -> None:
        dmd9 = protocols["DMD_9F_3O3P"]
        dmd9_positions = {name: index for index, name in enumerate(dmd9.orientation_ids)}
        for spec in protocols.values():
            subset = all(name in dmd9_positions for name in spec.orientation_ids)
            if spec.orientation_subset_of_dmd9 != subset:
                raise ProtocolValidationError(
                    f"{spec.protocol_id} orientation_subset_of_dmd9 is inconsistent"
                )
            ordered_subset = tuple(
                name for name in dmd9.orientation_ids if name in set(spec.orientation_ids)
            )
            order_compatible = subset and spec.orientation_ids == ordered_subset
            if spec.orientation_order_compatible_with_dmd9 != order_compatible:
                raise ProtocolValidationError(
                    f"{spec.protocol_id} orientation order compatibility is inconsistent"
                )
            phase_compatible = (
                spec.phase_ids == dmd9.phase_ids
                and spec.nominal_phase_values == dmd9.nominal_phase_values
            )
            if spec.phase_order_compatible_with_dmd9 != phase_compatible:
                raise ProtocolValidationError(
                    f"{spec.protocol_id} phase order compatibility is inconsistent"
                )

    @property
    def protocol_dir(self) -> Path:
        return self._protocol_dir

    @property
    def protocol_ids(self) -> Tuple[str, ...]:
        return PROTOCOL_IDS

    @property
    def registry_hash(self) -> str:
        return self._registry_hash

    def all(self) -> Tuple[ProtocolSpec, ...]:
        return tuple(self._protocols[protocol_id] for protocol_id in PROTOCOL_IDS)

    def get(self, protocol_id: str) -> Optional[ProtocolSpec]:
        return self._protocols.get(protocol_id)

    def require(self, protocol_id: str) -> ProtocolSpec:
        try:
            return self._protocols[protocol_id]
        except KeyError as exc:
            raise UnknownProtocolError(
                f"unknown protocol_id={protocol_id!r}; allowed={list(PROTOCOL_IDS)}"
            ) from exc


protocol_registry = ProtocolRegistry()


__all__ = [
    "EvidenceFile",
    "ForwardGeometry",
    "KMAX",
    "PROTOCOL_IDS",
    "ProtocolError",
    "ProtocolHashMismatchError",
    "ProtocolRegistry",
    "ProtocolSpec",
    "ProtocolValidationError",
    "RawFrameBinding",
    "UnknownProtocolError",
    "canonical_json_bytes",
    "compute_protocol_hash",
    "load_protocol",
    "protocol_registry",
]

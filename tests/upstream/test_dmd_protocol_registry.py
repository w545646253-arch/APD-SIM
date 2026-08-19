"""Focused contracts for the immutable controller-defined DMD registry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from unisim.protocols import (
    KMAX,
    PROTOCOL_IDS,
    ProtocolHashMismatchError,
    ProtocolRegistry,
    UnknownProtocolError,
    canonical_json_bytes,
    compute_protocol_hash,
    load_protocol,
    protocol_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = PROJECT_ROOT / "protocols"


class TestDmdProtocolRegistry(unittest.TestCase):
    def test_registry_contains_exactly_the_three_revised_protocols(self) -> None:
        self.assertEqual(protocol_registry.protocol_ids, PROTOCOL_IDS)
        self.assertEqual(
            tuple(spec.protocol_id for spec in protocol_registry.all()), PROTOCOL_IDS
        )
        self.assertRegex(protocol_registry.registry_hash, r"^[0-9a-f]{64}$")

    def test_required_1o3p_2o3p_3o3p_topologies(self) -> None:
        expected = {
            "DMD_3F_1O3P": (3, 1, 3),
            "DMD_6F_2O3P": (6, 2, 3),
            "DMD_9F_3O3P": (9, 3, 3),
        }
        for protocol_id, topology in expected.items():
            spec = protocol_registry.require(protocol_id)
            self.assertEqual(
                (spec.frame_count, spec.orientation_count, spec.phases_per_orientation),
                topology,
            )
            self.assertEqual(sum(spec.validity_mask), spec.frame_count)

    def test_controller_raw_orders_and_pattern_ids_are_explicit(self) -> None:
        expected = {
            "DMD_3F_1O3P": (
                ("X0", "X120", "X240"),
                (3, 4, 5),
            ),
            "DMD_6F_2O3P": (
                ("H0", "H120", "H240", "V0", "V120", "V240"),
                (3, 4, 5, 6, 7, 8),
            ),
            "DMD_9F_3O3P": (
                (
                    "X0",
                    "X120",
                    "X240",
                    "Y0",
                    "Y120",
                    "Y240",
                    "Z0",
                    "Z120",
                    "Z240",
                ),
                (3, 4, 5, 6, 7, 8, 9, 10, 11),
            ),
        }
        for protocol_id, (raw_order, pattern_ids) in expected.items():
            spec = protocol_registry.require(protocol_id)
            self.assertEqual(spec.raw_frame_order, raw_order)
            self.assertEqual(
                tuple(binding.controller_pattern_id for binding in spec.raw_frame_bindings),
                pattern_ids,
            )
            self.assertEqual(
                tuple(binding.raw_frame_id for binding in spec.raw_frame_bindings), raw_order
            )

    def test_mapping_is_bijective_and_validity_mask_matches_slots(self) -> None:
        for spec in protocol_registry.all():
            mapping = spec.raw_to_slot_mapping
            self.assertEqual(len(mapping), len(set(mapping)))
            self.assertEqual(set(mapping), set(spec.valid_slots))
            self.assertEqual(mapping, spec.canonical_slots)
            self.assertEqual(len(spec.validity_mask), KMAX)
            self.assertEqual(
                {index for index, value in enumerate(spec.validity_mask) if value},
                set(spec.valid_slots),
            )
            self.assertEqual(
                tuple(binding.canonical_slot for binding in spec.raw_frame_bindings),
                mapping,
            )

    def test_each_orientation_has_unique_three_phase_sequence(self) -> None:
        expected_phase_ids = (
            "PHASE_0_DEG",
            "PHASE_120_DEG",
            "PHASE_240_DEG",
        )
        expected_radians = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
        for spec in protocol_registry.all():
            self.assertEqual(spec.phase_ids, expected_phase_ids)
            for actual, expected in zip(spec.nominal_phase_values, expected_radians):
                self.assertAlmostEqual(actual, expected, places=15)
            for orientation_index, orientation_id in enumerate(spec.orientation_ids):
                start = orientation_index * spec.phases_per_orientation
                group = spec.raw_frame_bindings[start : start + spec.phases_per_orientation]
                self.assertEqual(
                    tuple(binding.physical_orientation_id for binding in group),
                    (orientation_id,) * 3,
                )
                self.assertEqual(
                    tuple(binding.physical_phase_id for binding in group), expected_phase_ids
                )
                self.assertEqual(len({binding.physical_phase_id for binding in group}), 3)

    def test_known_controller_orientations_and_carriers(self) -> None:
        dmd3 = protocol_registry.require("DMD_3F_1O3P")
        dmd6 = protocol_registry.require("DMD_6F_2O3P")
        dmd9 = protocol_registry.require("DMD_9F_3O3P")
        self.assertEqual(dmd3.orientation_ids, ("X",))
        self.assertEqual(dmd6.orientation_ids, ("H", "V"))
        self.assertEqual(dmd9.orientation_ids, ("X", "Y", "Z"))
        self.assertEqual(dmd6.orientation_angles, (90.0, 0.0))
        self.assertEqual(dmd6.carrier_vectors, ((0.0, 0.166666666667), (0.1669921875, 0.0)))
        self.assertEqual(dmd3.carrier_vectors[0], dmd9.carrier_vectors[0])
        self.assertEqual(dmd3.orientation_angles[0], dmd9.orientation_angles[0])

    def test_dmd9_subset_and_order_compatibility_are_evidence_based(self) -> None:
        dmd3 = protocol_registry.require("DMD_3F_1O3P")
        dmd6 = protocol_registry.require("DMD_6F_2O3P")
        dmd9 = protocol_registry.require("DMD_9F_3O3P")
        self.assertTrue(dmd3.orientation_subset_of_dmd9)
        self.assertTrue(dmd3.orientation_order_compatible_with_dmd9)
        self.assertTrue(dmd3.phase_order_compatible_with_dmd9)
        self.assertFalse(dmd6.orientation_subset_of_dmd9)
        self.assertFalse(dmd6.orientation_order_compatible_with_dmd9)
        self.assertTrue(dmd6.phase_order_compatible_with_dmd9)
        self.assertTrue(dmd9.orientation_subset_of_dmd9)
        self.assertTrue(dmd9.orientation_order_compatible_with_dmd9)

    def test_k6_evidence_and_claim_boundary_are_exact(self) -> None:
        dmd6 = protocol_registry.require("DMD_6F_2O3P")
        self.assertEqual(
            dmd6.evidence_level, "CONTROLLER_BITMAP_BASIS_VERIFIED_NOMINAL"
        )
        self.assertEqual(dmd6.claim_level, "controller-defined nominal DMD geometry")
        self.assertEqual(dmd6.historical_acquisition_receipt, "absent")
        self.assertFalse(dmd6.simulation_training_blocked)
        self.assertNotIn("fully calibrated", dmd6.claim_level)
        self.assertNotIn("fully acquisition-receipt-verified", dmd6.claim_level)

    def test_k3_and_k9_are_acquisition_receipt_verified(self) -> None:
        for protocol_id in ("DMD_3F_1O3P", "DMD_9F_3O3P"):
            spec = protocol_registry.require(protocol_id)
            self.assertEqual(spec.evidence_level, "ACQUISITION_RECEIPT_VERIFIED")
            self.assertEqual(spec.historical_acquisition_receipt, "present")

    def test_forward_geometry_is_an_exact_protocol_binding(self) -> None:
        for spec in protocol_registry.all():
            geometry = spec.forward_geometry
            self.assertEqual(geometry.orientation_angles, spec.orientation_angles)
            self.assertEqual(geometry.nominal_phase_values, spec.nominal_phase_values)
            self.assertEqual(geometry.carrier_vectors, spec.carrier_vectors)
            self.assertEqual(geometry.raw_frame_order, spec.raw_frame_order)
            self.assertEqual(geometry.raw_to_slot_mapping, spec.raw_to_slot_mapping)
            self.assertEqual(geometry.validity_mask, spec.validity_mask)
            self.assertEqual(geometry.phase_unit, "radian")
            self.assertEqual(geometry.phase_source, "controller_nominal_labels")
            self.assertEqual(geometry.fft_phase_role, "evidence_and_diagnostic_only")

    def test_protocol_json_is_canonical_and_self_hashed(self) -> None:
        paths = sorted(PROTOCOL_DIR.glob("*.json"), key=lambda path: path.name)
        self.assertEqual(len(paths), 3)
        for path in paths:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            self.assertIn(raw, (canonical_json_bytes(payload), canonical_json_bytes(payload) + b"\n"))
            self.assertEqual(payload["protocol_hash"], compute_protocol_hash(payload))
            self.assertEqual(load_protocol(path).protocol_hash, payload["protocol_hash"])

    def test_raw_order_change_changes_hash_and_stale_hash_is_rejected(self) -> None:
        source = PROTOCOL_DIR / "dmd_6f_2o3p.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        original_hash = payload["protocol_hash"]
        payload["raw_frame_order"][0], payload["raw_frame_order"][1] = (
            payload["raw_frame_order"][1],
            payload["raw_frame_order"][0],
        )
        self.assertNotEqual(compute_protocol_hash(payload), original_hash)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tampered.json"
            path.write_bytes(canonical_json_bytes(payload))
            with self.assertRaises(ProtocolHashMismatchError):
                load_protocol(path)

    def test_protocol_objects_and_nested_values_are_immutable(self) -> None:
        spec = protocol_registry.require("DMD_3F_1O3P")
        with self.assertRaises(FrozenInstanceError):
            spec.frame_count = 99  # type: ignore[misc]
        with self.assertRaises(TypeError):
            spec.validity_mask[0] = 0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            spec.raw_frame_bindings[0].canonical_slot = 4  # type: ignore[misc]
        detached = spec.to_payload()
        detached["raw_frame_order"][0] = "MUTATED_COPY"
        self.assertEqual(spec.raw_frame_order[0], "X0")

    def test_registry_require_fails_closed_for_unknown_and_legacy_ids(self) -> None:
        for protocol_id in ("DMD_6F_3O2P", "LEGACY_3F_3O1P", "", "DMD_6F"):
            with self.assertRaises(UnknownProtocolError):
                protocol_registry.require(protocol_id)

    def test_registry_hash_is_hash_of_ordered_id_to_protocol_hash_map(self) -> None:
        payload = {
            protocol_id: protocol_registry.require(protocol_id).protocol_hash
            for protocol_id in PROTOCOL_IDS
        }
        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self.assertEqual(protocol_registry.registry_hash, expected)
        self.assertEqual(ProtocolRegistry(PROTOCOL_DIR).registry_hash, expected)


if __name__ == "__main__":
    unittest.main()

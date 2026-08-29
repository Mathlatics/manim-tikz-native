from __future__ import annotations

import copy
import unittest

from tikz_native import compile_document
from tikz_native.planar_curves_3d import (
    PlanarTikz3DError,
    restore_planar_curve_geometry,
    restore_planar_frame_geometry,
)


SOURCE = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (O) at (1,2,3);
  \coordinate (U) at (4,2,3);
  \coordinate (V) at (1,6,3);
  \DeclareSpacePlane{plane-a}{O/U/V};
  \DrawSpaceCircle{circle-a}{plane-a}{0.25,-0.5}{2};
\end{tikzpicture}
"""


class TikzPlanarCurves3DContractTests(unittest.TestCase):
    def setUp(self) -> None:
        picture = compile_document(source_text=SOURCE).pictures[0]
        self.assertFalse(picture.unsupported)
        self.frame_payload = copy.deepcopy(picture.planar_frames_3d["plane-a"])
        self.curve_payload = copy.deepcopy(picture.objects[0].geometry)

    def test_registry_and_curve_payloads_round_trip_canonically(self) -> None:
        frame = restore_planar_frame_geometry(
            self.frame_payload,
            expected_plane_id="plane-a",
        )
        curve = restore_planar_curve_geometry(
            self.curve_payload,
            expected_curve_id="circle-a",
        )

        self.assertEqual(frame.to_dict(), self.frame_payload)
        self.assertEqual(curve.to_dict(), self.curve_payload)
        self.assertEqual(curve.frame, frame.frame)

    def test_missing_extra_and_noncanonical_fields_fail_closed(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        missing = copy.deepcopy(self.curve_payload)
        missing.pop("static")
        cases.append((missing, "missing static"))

        extra = copy.deepcopy(self.curve_payload)
        extra["guessed_plane"] = True
        cases.append((extra, "unexpected"))

        not_static = copy.deepcopy(self.curve_payload)
        not_static["static"] = False
        cases.append((not_static, "static=true"))

        tuple_names = copy.deepcopy(self.curve_payload)
        tuple_names["plane_point_names"] = ("O", "U", "V")
        cases.append((tuple_names, "JSON array"))

        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PlanarTikz3DError, message):
                    restore_planar_curve_geometry(payload)

    def test_nested_frame_curve_and_identity_tampering_fails_closed(self) -> None:
        cases: list[tuple[dict[str, object], str]] = []

        seed = copy.deepcopy(self.curve_payload)
        seed["frame"]["normalSeed"][0] = 0.5
        cases.append((seed, "seed"))

        plane_identity = copy.deepcopy(self.curve_payload)
        plane_identity["plane_id"] = "plane-b"
        cases.append((plane_identity, "identity"))

        curve_identity = copy.deepcopy(self.curve_payload)
        curve_identity["curve"]["curveId"] = "circle-b"
        cases.append((curve_identity, "compiler object identity"))

        curve_kind = copy.deepcopy(self.curve_payload)
        curve_kind["curve"]["kind"] = "parabola"
        cases.append((curve_kind, "kind"))

        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PlanarTikz3DError, message):
                    restore_planar_curve_geometry(
                        payload,
                        expected_curve_id="circle-a",
                    )

    def test_registry_identity_mismatch_and_tampering_fail_closed(self) -> None:
        with self.assertRaisesRegex(PlanarTikz3DError, "registry identity"):
            restore_planar_frame_geometry(
                self.frame_payload,
                expected_plane_id="plane-b",
            )

        tampered = copy.deepcopy(self.frame_payload)
        tampered["frame"]["uAxisSeed"] = [0.0, 1.0, 0.0]
        with self.assertRaisesRegex(PlanarTikz3DError, "canonical"):
            restore_planar_frame_geometry(tampered)


if __name__ == "__main__":
    unittest.main()

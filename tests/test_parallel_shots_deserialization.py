from __future__ import annotations

import json
import unittest

import numpy as np

from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
    parallel_camera_shot_sequence_from_dict,
    parallel_camera_shot_sequence_from_json,
)


def _sequence() -> ParallelCameraShotSequence:
    return ParallelCameraShotSequence(
        (
            ParallelCameraShot(
                "overview",
                ParallelCameraState.from_view_direction(
                    (1.0, 1.5, 0.8),
                    target=(0.25, -0.5, 1.0),
                    screen_anchor=(-0.75, 0.3),
                    zoom=1.2,
                ),
                duration=1.5,
                hold=0.25,
                transition="orbit",
                arc_height=0.7,
                cue="caf\N{LATIN SMALL LETTER E WITH ACUTE}",
            ),
            ParallelCameraShot(
                "section",
                ParallelCameraState(np.identity(3), zoom=0.9),
                duration=0.8,
                transition="shortest",
                arc_height=0.0,
            ),
        )
    )


class ParallelCameraShotDeserializationTests(unittest.TestCase):
    def test_dict_and_json_round_trip_to_the_same_canonical_sequence(self) -> None:
        original = _sequence()
        canonical = canonical_parallel_camera_shot_sequence_json(original)

        from_dict = parallel_camera_shot_sequence_from_dict(original.to_dict())
        from_text = parallel_camera_shot_sequence_from_json(canonical)
        from_bytes = parallel_camera_shot_sequence_from_json(canonical.encode("utf-8"))

        for rebuilt in (from_dict, from_text, from_bytes):
            with self.subTest(rebuilt=rebuilt):
                self.assertEqual(
                    canonical_parallel_camera_shot_sequence_json(rebuilt),
                    canonical,
                )
                self.assertTrue(
                    np.array_equal(
                        rebuilt.shots[0].state.matrix,
                        original.shots[0].state.matrix,
                    )
                )

    def test_objects_require_exact_fields(self) -> None:
        payload = _sequence().to_dict()
        cases = []
        missing_root = dict(payload)
        missing_root.pop("schema")
        cases.append((missing_root, "missing required fields: schema"))
        extra_root = {**payload, "generated": True}
        cases.append((extra_root, "unsupported fields: generated"))
        missing_shot = json.loads(json.dumps(payload))
        missing_shot["shots"][0].pop("duration")
        cases.append((missing_shot, "missing required fields: duration"))
        extra_state = json.loads(json.dumps(payload))
        extra_state["shots"][0]["state"]["perspective"] = 0.0
        cases.append((extra_state, "unsupported fields: perspective"))

        for candidate, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    parallel_camera_shot_sequence_from_dict(candidate)

    def test_json_shape_and_numeric_values_fail_closed(self) -> None:
        payload = _sequence().to_dict()
        cases = []
        tuple_target = json.loads(json.dumps(payload))
        tuple_target["shots"][0]["state"]["target"] = (0.0, 0.0, 0.0)
        cases.append((tuple_target, "exactly 3 finite numbers"))
        boolean_matrix = json.loads(json.dumps(payload))
        boolean_matrix["shots"][0]["state"]["matrix"][0][0] = True
        cases.append((boolean_matrix, "finite number"))
        singular_matrix = json.loads(json.dumps(payload))
        singular_matrix["shots"][0]["state"]["matrix"] = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        cases.append((singular_matrix, "invertible and right-handed"))
        bad_zoom = json.loads(json.dumps(payload))
        bad_zoom["shots"][0]["state"]["zoom"] = 0.0
        cases.append((bad_zoom, "positive"))
        huge_duration = json.loads(json.dumps(payload))
        huge_duration["shots"][0]["duration"] = 10**400
        cases.append((huge_duration, "finite number"))
        duplicate_id = json.loads(json.dumps(payload))
        duplicate_id["shots"][1]["id"] = duplicate_id["shots"][0]["id"]
        cases.append((duplicate_id, "unique"))

        for candidate, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    parallel_camera_shot_sequence_from_dict(candidate)

    def test_strict_json_rejects_duplicates_nonfinite_and_invalid_unicode(self) -> None:
        canonical = canonical_parallel_camera_shot_sequence_json(_sequence())
        duplicate = canonical.replace(
            '"schema":"parallel-shot-sequence/v1"',
            '"schema":"parallel-shot-sequence/v1",'
            '"schema":"parallel-shot-sequence/v1"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            parallel_camera_shot_sequence_from_json(duplicate)

        overflowing = canonical.replace('"zoom":1.2', '"zoom":1e999', 1)
        with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
            parallel_camera_shot_sequence_from_json(overflowing)

        with self.assertRaisesRegex(ValueError, "UTF-8"):
            parallel_camera_shot_sequence_from_json(b"\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            parallel_camera_shot_sequence_from_json(canonical.encode("utf-16"))

        invalid_unicode = json.loads(canonical)
        invalid_unicode["shots"][0]["cue"] = "\ud800"
        with self.assertRaisesRegex(ValueError, "valid Unicode"):
            parallel_camera_shot_sequence_from_dict(invalid_unicode)

    def test_unicode_is_normalized_before_identity_validation(self) -> None:
        payload = _sequence().to_dict()
        payload["shots"][0]["id"] = "cafe\N{COMBINING ACUTE ACCENT}"
        rebuilt = parallel_camera_shot_sequence_from_dict(payload)
        self.assertEqual(rebuilt.shots[0].id, "caf\N{LATIN SMALL LETTER E WITH ACUTE}")


if __name__ == "__main__":
    unittest.main()

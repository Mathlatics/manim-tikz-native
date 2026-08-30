from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
from jsonschema import Draft202012Validator

_SYNTHETIC_PACKAGE = False
if importlib.util.find_spec("manim") is None:
    # The production modules are renderer-neutral, but tikz_native/__init__.py
    # also exports Manim renderers.  Load this isolated contract without running
    # that package facade so the test proves the new module itself has no Manim
    # dependency in the lightweight geometry test environment.
    package = types.ModuleType("tikz_native")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "tikz_native")]
    sys.modules["tikz_native"] = package
    _SYNTHETIC_PACKAGE = True

from tikz_native.parallel_camera import CameraPlane, ParallelCameraState, ProjectionRank
from tikz_native.parallel_shots import (
    PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA,
    ParallelCameraSafeFrame,
    ParallelCameraShot,
    ParallelCameraShotSequence,
    canonical_parallel_camera_shot_sequence_json,
    fit_points_to_parallel_camera_state,
)

if _SYNTHETIC_PACKAGE:
    sys.modules.pop("tikz_native", None)


class ParallelCameraShotAuthoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = CameraPlane(
            (1.0, -2.0, 0.5),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        )

    def test_semantic_shot_constructors_match_parallel_camera_states(self) -> None:
        normal = ParallelCameraShot.normal_to_plane(
            "normal",
            self.plane,
            screen_anchor=(-1.5, 0.75),
            zoom=1.8,
            duration=1.2,
            hold=0.3,
            cue="show_section",
        )
        along = ParallelCameraShot.along_plane(
            "along",
            self.plane,
            azimuth_degrees=28.0,
            transition="shortest",
            arc_height=0.0,
        )
        relative = ParallelCameraShot.relative_to_plane(
            "relative",
            self.plane,
            inclination_degrees=42.0,
            azimuth_degrees=18.0,
        )

        expected = ParallelCameraState.normal_to_plane(
            self.plane,
            screen_anchor=(-1.5, 0.75),
            zoom=1.8,
        )
        np.testing.assert_allclose(normal.state.matrix, expected.matrix)
        np.testing.assert_allclose(normal.state.target, self.plane.point)
        np.testing.assert_allclose(normal.state.screen_anchor, (-1.5, 0.75))
        self.assertEqual(normal.duration, 1.2)
        self.assertEqual(normal.hold, 0.3)
        self.assertEqual(normal.cue, "show_section")
        self.assertIs(
            along.state.plane_projection_rank(self.plane),
            ProjectionRank.LINE,
        )
        self.assertIs(
            relative.state.plane_projection_rank(self.plane),
            ProjectionRank.AREA,
        )

    def test_look_at_keeps_target_on_requested_screen_anchor(self) -> None:
        shot = ParallelCameraShot.look_at(
            "look",
            (2.0, -1.0, 0.25),
            view_direction=(1.0, 2.0, 3.0),
            up_hint=(0.0, 0.0, 1.0),
            screen_anchor=(2.5, -0.75),
            zoom=2.2,
        )

        np.testing.assert_allclose(
            shot.state.project_point(shot.state.target)[:2],
            (2.5, -0.75),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            shot.state.view_direction,
            np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0),
            atol=1.0e-12,
        )

    def test_shot_metadata_fails_closed(self) -> None:
        state = ParallelCameraState(np.identity(3))
        invalid = (
            (("", state), ValueError, "shot id"),
            (("bad", object()), TypeError, "ParallelCameraState"),
        )
        for arguments, error_type, pattern in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(error_type, pattern):
                    ParallelCameraShot(*arguments)
        for field, value, pattern in (
            ("duration", 0.0, "positive"),
            ("duration", np.inf, "finite"),
            ("hold", -0.1, "non-negative"),
            ("transition", "linear", "orbit.*shortest"),
            ("arc_height", 0.0, "non-zero"),
            ("cue", "  ", "non-empty"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, pattern):
                    ParallelCameraShot("bad", state, **{field: value})

    def test_sequence_requires_strict_schema_unique_ids_and_nonempty_shots(
        self,
    ) -> None:
        state = ParallelCameraState(np.identity(3))
        first = ParallelCameraShot("first", state)
        second = ParallelCameraShot("second", state, duration=2.0, hold=0.5)
        sequence = ParallelCameraShotSequence((first, second))

        self.assertEqual(sequence.schema, PARALLEL_CAMERA_SHOT_SEQUENCE_SCHEMA)
        self.assertIs(sequence.shot("second"), second)
        self.assertAlmostEqual(sequence.total_duration, 3.5)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ParallelCameraShotSequence(())
        with self.assertRaisesRegex(ValueError, "unique"):
            ParallelCameraShotSequence((first, replace(first, duration=2.0)))
        with self.assertRaisesRegex(ValueError, "schema"):
            ParallelCameraShotSequence((first,), schema="parallel-shot-sequence/v2")
        with self.assertRaisesRegex(KeyError, "unknown"):
            sequence.shot("missing")

    def test_canonical_json_is_stable_and_contains_complete_state(self) -> None:
        first = ParallelCameraShot.look_at(
            "overview",
            (1.0, 2.0, 3.0),
            view_direction=(1.0, 1.0, 1.0),
            screen_anchor=(-2.0, 0.5),
            zoom=1.25,
            duration=1.4,
            hold=0.2,
            cue="overview_ready",
        )
        second = ParallelCameraShot.normal_to_plane(
            "normal",
            self.plane,
            transition="shortest",
            arc_height=0.0,
        )
        sequence = ParallelCameraShotSequence((first, second))

        encoded = canonical_parallel_camera_shot_sequence_json(sequence)
        self.assertEqual(
            encoded,
            canonical_parallel_camera_shot_sequence_json(sequence),
        )
        payload = json.loads(encoded)
        self.assertEqual(payload["schema"], "parallel-shot-sequence/v1")
        self.assertEqual(
            [item["id"] for item in payload["shots"]],
            ["overview", "normal"],
        )
        self.assertEqual(payload["shots"][0]["state"]["target"], [1.0, 2.0, 3.0])
        self.assertEqual(
            payload["shots"][0]["state"]["screenAnchor"],
            [-2.0, 0.5],
        )
        self.assertEqual(payload["shots"][0]["state"]["zoom"], 1.25)
        self.assertEqual(payload["shots"][0]["cue"], "overview_ready")

    def test_canonical_payload_matches_the_public_json_schema(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "tikz_native"
            / "schemas"
            / "parallel-shot-sequence-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot.look_at(
                    "schema-shot",
                    (0.0, 0.0, 0.0),
                    view_direction=(1.0, 1.0, 1.0),
                    cue="schema evidence",
                ),
            )
        )
        Draft202012Validator(schema).validate(sequence.to_dict())


class ParallelCameraFitTests(unittest.TestCase):
    def test_area_points_fit_asymmetrically_without_moving_target_or_anchor(
        self,
    ) -> None:
        state = ParallelCameraState(
            np.identity(3),
            target=(1.0, 2.0, 3.0),
            screen_anchor=(1.0, -0.5),
            zoom=7.0,
        )
        safe = ParallelCameraSafeFrame(-3.0, 5.0, -2.5, 3.5)
        points = np.asarray(
            (
                (-1.0, 1.0, 3.0),
                (4.0, 1.0, 3.0),
                (4.0, 4.0, 3.0),
                (-1.0, 4.0, 3.0),
            )
        )

        fitted = fit_points_to_parallel_camera_state(
            state,
            points,
            safe_frame=safe,
        )

        self.assertAlmostEqual(fitted.zoom, 4.0 / 3.0)
        np.testing.assert_allclose(fitted.matrix, state.matrix)
        np.testing.assert_allclose(fitted.target, state.target)
        np.testing.assert_allclose(fitted.screen_anchor, state.screen_anchor)
        np.testing.assert_allclose(
            fitted.project_point(fitted.target)[:2],
            state.screen_anchor,
        )
        screen = fitted.project_points(points)[:, :2]
        self.assertTrue(
            all(safe.contains(point, tolerance=1.0e-12) for point in screen)
        )
        self.assertAlmostEqual(float(np.max(screen[:, 0])), safe.right)

    def test_line_point_set_uses_only_its_retained_screen_extent(self) -> None:
        state = ParallelCameraState(
            np.identity(3),
            screen_anchor=(0.5, 1.0),
        )
        safe = ParallelCameraSafeFrame(-3.5, 4.5, -2.0, 4.0)
        points = np.asarray(((-2.0, 0.0, 0.0), (1.0, 0.0, 4.0), (4.0, 0.0, -2.0)))

        fitted = fit_points_to_parallel_camera_state(
            state,
            points,
            safe_frame=safe,
        )

        self.assertAlmostEqual(fitted.zoom, 1.0)
        screen = fitted.project_points(points)[:, :2]
        self.assertAlmostEqual(float(np.min(screen[:, 0])), -1.5)
        self.assertAlmostEqual(float(np.max(screen[:, 0])), 4.5)
        self.assertTrue(np.allclose(screen[:, 1], 1.0))

    def test_coincident_projection_requires_valid_fallback_zoom(self) -> None:
        state = ParallelCameraState(np.identity(3), screen_anchor=(0.0, 0.0))
        safe = ParallelCameraSafeFrame(-2.0, 2.0, -1.0, 1.0)
        points = np.asarray(((0.0, 0.0, -3.0), (0.0, 0.0, 4.0)))

        with self.assertRaisesRegex(ValueError, "coincident.*fallback_zoom"):
            fit_points_to_parallel_camera_state(
                state,
                points,
                safe_frame=safe,
            )
        fitted = fit_points_to_parallel_camera_state(
            state,
            points,
            safe_frame=safe,
            fallback_zoom=2.5,
        )
        self.assertEqual(fitted.zoom, 2.5)
        np.testing.assert_allclose(fitted.project_points(points)[:, :2], 0.0)

    def test_fallback_and_safe_frame_constraints_fail_explicitly(self) -> None:
        state = ParallelCameraState(np.identity(3), screen_anchor=(0.0, 0.0))
        safe = ParallelCameraSafeFrame(-1.0, 1.0, -1.0, 1.0)
        coincident = np.asarray(((2.0, 0.0, -1.0), (2.0, 0.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "inside the explicit safe frame"):
            fit_points_to_parallel_camera_state(
                state,
                coincident,
                safe_frame=safe,
                fallback_zoom=1.0,
            )
        with self.assertRaisesRegex(ValueError, "screen_anchor"):
            fit_points_to_parallel_camera_state(
                state.with_screen_anchor((2.0, 0.0)),
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                safe_frame=safe,
            )

    def test_invalid_points_and_safe_frames_fail_closed(self) -> None:
        state = ParallelCameraState(np.identity(3))
        safe = ParallelCameraSafeFrame(-1.0, 1.0, -1.0, 1.0)
        for points in ((), ((0.0, 1.0),), ((0.0, np.nan, 1.0),)):
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "Nx3"):
                    fit_points_to_parallel_camera_state(
                        state,
                        points,
                        safe_frame=safe,
                    )
        with self.assertRaisesRegex(ValueError, "left < right"):
            ParallelCameraSafeFrame(1.0, 1.0, -1.0, 1.0)
        with self.assertRaisesRegex(TypeError, "ParallelCameraSafeFrame"):
            fit_points_to_parallel_camera_state(
                state,
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                safe_frame=(-1.0, 1.0, -1.0, 1.0),  # type: ignore[arg-type]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

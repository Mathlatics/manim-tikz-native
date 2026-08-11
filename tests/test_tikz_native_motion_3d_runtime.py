from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
from manim import Scene, tempconfig
from manim.animation.animation import prepare_animation

import tikz_native.motion_3d_runtime as motion_3d_runtime
from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import (
    analyze_geometry_rig_3d,
    semantic_model_3d_hash,
)
from tikz_native.motion_3d_runtime import (
    EMBEDDED_MOTION_3D_RUNTIME_CONTRACT,
    EmbeddedMotion3DError,
    play_motion_3d_on_native_shape,
)
from tikz_native.manim_renderer_3d import NativeManim3DRenderer
from tikz_native.provider import instantiate_picture
from tikz_native.version import provider_revision


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


class _InstantScene(Scene):
    def __init__(self, *, fail_after_plays: int | None = None) -> None:
        super().__init__()
        self.fail_after_plays = fail_after_plays
        self.play_count = 0
        self.observed_centers: list[np.ndarray] = []
        self.observed_updater_counts: list[int] = []
        self.observed_object = None

    def play(self, *builders, **_kwargs) -> None:  # type: ignore[override]
        self.play_count += 1
        animations = [prepare_animation(builder) for builder in builders]
        for animation in animations:
            animation.begin()
        for alpha in (0.5, 1.0):
            for animation in animations:
                animation.interpolate(alpha)
            for mobject in self.mobjects:
                mobject.update(0.0)
            if self.observed_object is not None:
                self.observed_centers.append(
                    np.asarray(self.observed_object.get_center(), dtype=float).copy()
                )
                self.observed_updater_counts.append(
                    len(self.observed_object.updaters)
                )
        for animation in animations:
            animation.finish()
        if self.fail_after_plays == self.play_count:
            raise RuntimeError("intentional scene failure")

    def wait(self, *_args, **_kwargs) -> None:  # type: ignore[override]
        for mobject in self.mobjects:
            mobject.update(0.0)


class EmbeddedMotion3DRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_text = SOURCE.read_text(encoding="utf-8")
        cls.picture = compile_document(source_text=cls.source_text).pictures[0]
        cls.analysis = analyze_geometry_rig_3d(
            cls.picture,
            selection={
                "candidate_id": "hinge_fold:fold-angle",
                "range": [0.3141592653589793, 1.9547687622336491],
                "include_object_ids": [],
                "exclude_object_ids": [],
            },
        )

    def fixture(self, *, analysis=None, timeline=None):
        selected_analysis = self.analysis if analysis is None else analysis
        revision = provider_revision()
        semantic_hash = semantic_model_3d_hash(self.picture)
        figure = instantiate_picture(
            self.picture,
            scene_unit_per_cm=1.0,
        )
        shape = figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        shape._codex_tikz_native_objects = figure.objects  # type: ignore[attr-defined]
        shape._codex_tikz_native_picture = self.picture  # type: ignore[attr-defined]
        motion = {
            "schema": "tikz-native-motion-3d/v1",
            "picture_index": self.picture.index,
            **selected_analysis["motionSpecCore"],
            "timeline": (
                [
                    {"type": "driver", "to": 0.5, "duration": 0.1},
                    {
                        "type": "camera",
                        "mode": "side",
                        "transition": "linear",
                        "duration": 0.1,
                    },
                ]
                if timeline is None
                else timeline
            ),
        }
        definition = {
            "dimension": 3,
            "status": "ready",
            "authorConfirmed": True,
            "revisionMatch": True,
            "currentRigProviderRevision": revision,
            "expectedAssetProviderRevision": revision,
            "semanticModelHash": semantic_hash,
        }
        semantic_manifest = {
            "dimension": 3,
            "pictureIndex": self.picture.index,
            "providerRevision": revision,
            "semanticModelHash": semantic_hash,
            "bindings": selected_analysis["bindings"],
        }
        return figure, shape, motion, definition, semantic_manifest, revision

    @staticmethod
    def snapshot(figure, shape):
        return {
            "children": list(shape.submobjects),
            "objects": {
                object_id: mobject.copy()
                for object_id, mobject in figure.objects.items()
            },
            "family": [
                {
                    "member": member,
                    "submobjects": list(member.submobjects),
                    "points": member.get_all_points().copy(),
                    "style": repr(member.get_style()),
                    "updaters": list(member.updaters),
                    "updatingSuspended": member.updating_suspended,
                    "zIndex": member.z_index,
                }
                for member in shape.get_family()
            ],
        }

    def assert_restored(self, figure, shape, snapshot) -> None:
        self.assertEqual(len(shape.submobjects), len(snapshot["children"]))
        self.assertTrue(
            all(
                current is original
                for current, original in zip(
                    shape.submobjects, snapshot["children"], strict=True
                )
            )
        )
        for object_id, mobject in figure.objects.items():
            original = snapshot["objects"][object_id]
            self.assertEqual(
                mobject.get_all_points().shape,
                original.get_all_points().shape,
                object_id,
            )
            np.testing.assert_allclose(
                mobject.get_all_points(),
                original.get_all_points(),
                atol=1e-9,
                rtol=0.0,
                err_msg=object_id,
            )
        for state in snapshot["family"]:
            member = state["member"]
            self.assertEqual(member.submobjects, state["submobjects"])
            np.testing.assert_allclose(
                member.get_all_points(),
                state["points"],
                atol=1e-9,
                rtol=0.0,
            )
            self.assertEqual(repr(member.get_style()), state["style"])
            self.assertEqual(member.updaters, state["updaters"])
            self.assertEqual(
                member.updating_suspended, state["updatingSuspended"]
            )
            self.assertEqual(member.z_index, state["zIndex"])

    def test_uses_local_projection_and_restores_exact_input(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture()
        scene = _InstantScene()
        scene.add(shape)
        scene.observed_object = figure.objects["dot.M"]
        entry_center = np.asarray(scene.observed_object.get_center()).copy()
        snapshot = self.snapshot(figure, shape)
        global_camera = scene.camera

        result = play_motion_3d_on_native_shape(
            scene,
            shape,
            motion,
            definition=definition,
            semantic_manifest=manifest,
            expected_provider_revision=revision,
            runtime_contract=EMBEDDED_MOTION_3D_RUNTIME_CONTRACT,
        )

        self.assertIs(result, shape)
        self.assertIs(scene.camera, global_camera)
        self.assertGreaterEqual(scene.play_count, 3)
        self.assertTrue(
            any(
                not np.allclose(center, entry_center, atol=1e-5, rtol=0.0)
                for center in scene.observed_centers
            )
        )
        self.assert_restored(figure, shape, snapshot)

    def test_scene_failure_still_restores_input_and_camera(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture()
        scene = _InstantScene(fail_after_plays=1)
        scene.add(shape)
        snapshot = self.snapshot(figure, shape)
        global_camera = scene.camera

        with self.assertRaisesRegex(RuntimeError, "intentional scene failure"):
            play_motion_3d_on_native_shape(
                scene,
                shape,
                motion,
                definition=definition,
                semantic_manifest=manifest,
                expected_provider_revision=revision,
            )

        self.assertIs(scene.camera, global_camera)
        self.assert_restored(figure, shape, snapshot)

    def test_revision_mismatch_fails_before_mutating_shape(self) -> None:
        figure, shape, motion, definition, manifest, _revision = self.fixture()
        scene = _InstantScene()
        scene.add(shape)
        snapshot = self.snapshot(figure, shape)
        wrong_revision = "source-sha256:" + "0" * 64
        definition["currentRigProviderRevision"] = wrong_revision
        definition["expectedAssetProviderRevision"] = wrong_revision
        manifest["providerRevision"] = wrong_revision

        with self.assertRaisesRegex(EmbeddedMotion3DError, "Provider revision"):
            play_motion_3d_on_native_shape(
                scene,
                shape,
                motion,
                definition=definition,
                semantic_manifest=manifest,
                expected_provider_revision=wrong_revision,
            )

        self.assert_restored(figure, shape, snapshot)

    def test_recursive_updaters_and_custom_z_index_are_restored(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture()
        scene = _InstantScene()
        scene.add(shape)
        active = figure.objects["label.M.M"]
        descendant = active.get_family()[1]
        updater = lambda item, _dt=0.0: item
        descendant.add_updater(updater)
        line = figure.objects["line.M.N"]
        line.set_z_index(123, family=False)
        snapshot = self.snapshot(figure, shape)

        play_motion_3d_on_native_shape(
            scene,
            shape,
            motion,
            definition=definition,
            semantic_manifest=manifest,
            expected_provider_revision=revision,
        )

        self.assertEqual(descendant.updaters, [updater])
        self.assertEqual(line.z_index, 123)
        self.assert_restored(figure, shape, snapshot)

    def test_nonuniform_shape_state_is_rejected_instead_of_jumping(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture()
        shape.stretch(1.2, dim=0)
        scene = _InstantScene()
        scene.add(shape)
        snapshot = self.snapshot(figure, shape)

        with self.assertRaisesRegex(
            EmbeddedMotion3DError, "does not align with semantic object"
        ):
            play_motion_3d_on_native_shape(
                scene,
                shape,
                motion,
                definition=definition,
                semantic_manifest=manifest,
                expected_provider_revision=revision,
            )

        self.assert_restored(figure, shape, snapshot)

    def test_default_driver_only_timeline_has_safe_occlusion_capacity(self) -> None:
        lower, upper = self.analysis["motionSpecCore"]["driver"]["range"]
        timeline = [
            {"type": "driver", "to": lower, "duration": 0.1},
            {"type": "driver", "to": upper, "duration": 0.1},
        ]
        figure, shape, motion, definition, manifest, revision = self.fixture(
            timeline=timeline
        )
        scene = _InstantScene()
        scene.add(shape)
        snapshot = self.snapshot(figure, shape)

        play_motion_3d_on_native_shape(
            scene,
            shape,
            motion,
            definition=definition,
            semantic_manifest=manifest,
            expected_provider_revision=revision,
        )

        self.assert_restored(figure, shape, snapshot)

    def test_entry_wait_calibrates_occlusion_strokes_from_real_leaf_lines(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture(
            timeline=[{"type": "wait", "duration": 0.4, "cue": "entry_state"}]
        )
        scene = _InstantScene()
        scene.add(shape)

        relation = self.picture.occlusion_relations[0]
        relation_object = figure.objects[relation.object_ids[0]]
        relation_spec = next(
            item for item in self.picture.objects if item.id == relation.object_ids[0]
        )
        leaf_stroke_widths = []
        for member in relation_object.get_family():
            if not member.has_points():
                continue
            widths = np.asarray(member.get_stroke_width(), dtype=float).reshape(-1)
            opacities = np.asarray(member.get_stroke_opacity(), dtype=float).reshape(-1)
            if not np.any(np.isfinite(opacities) & (opacities > 0)):
                continue
            leaf_stroke_widths.extend(
                float(value)
                for value in widths
                if np.isfinite(value) and value > 0
            )
        self.assertTrue(leaf_stroke_widths)
        expected_stroke_width_per_pt = float(np.median(leaf_stroke_widths)) / float(
            relation_spec.style.line_width_pt
        )

        observed_stroke_width_per_pt: list[float] = []
        original_make_slots = NativeManim3DRenderer._make_occlusion_slots

        def record_make_slots(renderer, *args, **kwargs):
            observed_stroke_width_per_pt.append(renderer.stroke_width_per_pt)
            return original_make_slots(renderer, *args, **kwargs)

        with patch.object(
            NativeManim3DRenderer,
            "_make_occlusion_slots",
            new=record_make_slots,
        ):
            play_motion_3d_on_native_shape(
                scene,
                shape,
                motion,
                definition=definition,
                semantic_manifest=manifest,
                expected_provider_revision=revision,
            )

        self.assertTrue(observed_stroke_width_per_pt)
        for actual in observed_stroke_width_per_pt:
            self.assertAlmostEqual(actual, expected_stroke_width_per_pt, places=9)

    def test_stroke_calibration_is_not_controlled_by_the_first_drawable(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture(
            timeline=[{"type": "wait", "duration": 0.1}]
        )
        scene = _InstantScene()
        scene.add(shape)

        eligible = [
            (object_id, object_spec)
            for object_id, object_spec in (
                (item.id, item) for item in self.picture.objects
            )
            if object_spec.kind in {"line", "arrow", "polygon"}
            and object_spec.style.line_width_pt > 0
        ]
        self.assertGreaterEqual(len(eligible), 3)
        first_object_id = eligible[0][0]
        style_widths = {
            object_id: object_spec.style.line_width_pt
            for object_id, object_spec in eligible
        }
        object_ids_by_identity = {
            id(figure.objects[object_id]): object_id
            for object_id in style_widths
        }

        def synthetic_stroke_width(mobject):
            object_id = object_ids_by_identity.get(id(mobject))
            if object_id is None:
                return None
            ratio = 10.0 if object_id == first_object_id else 3.8
            return style_widths[object_id] * ratio

        observed_stroke_width_per_pt: list[float] = []
        original_make_slots = NativeManim3DRenderer._make_occlusion_slots

        def record_make_slots(renderer, *args, **kwargs):
            observed_stroke_width_per_pt.append(renderer.stroke_width_per_pt)
            return original_make_slots(renderer, *args, **kwargs)

        with (
            patch.object(
                motion_3d_runtime,
                "_mobject_stroke_width",
                side_effect=synthetic_stroke_width,
            ),
            patch.object(
                NativeManim3DRenderer,
                "_make_occlusion_slots",
                new=record_make_slots,
            ),
        ):
            play_motion_3d_on_native_shape(
                scene,
                shape,
                motion,
                definition=definition,
                semantic_manifest=manifest,
                expected_provider_revision=revision,
            )

        self.assertTrue(observed_stroke_width_per_pt)
        for actual in observed_stroke_width_per_pt:
            self.assertAlmostEqual(actual, 3.8, places=9)

    def test_real_manim_render_accepts_the_host_default_timeline(self) -> None:
        lower, upper = self.analysis["motionSpecCore"]["driver"]["range"]
        timeline = [
            {"type": "wait", "duration": 0.4, "cue": "entry_state"},
            {
                "type": "driver",
                "to": lower,
                "duration": 1.5,
                "hold": 0.2,
                "cue": "driver_minimum",
            },
            {
                "type": "driver",
                "to": upper,
                "duration": 2.2,
                "hold": 0.3,
                "cue": "driver_maximum",
            },
        ]
        figure, shape, motion, definition, manifest, revision = self.fixture(
            timeline=timeline
        )
        snapshot = self.snapshot(figure, shape)
        test_case = self

        class EmbeddedDefaultPreview(Scene):
            def construct(render_scene) -> None:
                global_camera = render_scene.camera
                render_scene.add(shape)
                play_motion_3d_on_native_shape(
                    render_scene,
                    shape,
                    motion,
                    definition=definition,
                    semantic_manifest=manifest,
                    expected_provider_revision=revision,
                )
                test_case.assertIs(render_scene.camera, global_camera)
                test_case.assert_restored(figure, shape, snapshot)

        with TemporaryDirectory() as directory, tempconfig(
            {
                "media_dir": directory,
                "pixel_width": 320,
                "pixel_height": 180,
                "frame_rate": 5,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
                "format": "mp4",
                "verbosity": "ERROR",
            }
        ):
            EmbeddedDefaultPreview().render()
            videos = [
                path
                for path in Path(directory).rglob("EmbeddedDefaultPreview.mp4")
                if "partial_movie_files" not in path.parts
            ]
            self.assertEqual(len(videos), 1)
            self.assertGreater(videos[0].stat().st_size, 0)

    def test_real_manim_entry_wait_has_no_pixel_jump(self) -> None:
        figure, shape, motion, definition, manifest, revision = self.fixture(
            timeline=[{"type": "wait", "duration": 0.4, "cue": "entry_state"}]
        )
        snapshot = self.snapshot(figure, shape)
        test_case = self

        class EmbeddedEntryBoundaryPreview(Scene):
            def construct(render_scene) -> None:
                render_scene.camera.background_color = "#FFFFFF"
                render_scene.add(shape)
                # Record the untouched ShapeState, then enter the embedded
                # runtime while its authored timeline still says to wait.  A
                # temporary representation must not change the entry pixels.
                render_scene.wait(0.4)
                play_motion_3d_on_native_shape(
                    render_scene,
                    shape,
                    motion,
                    definition=definition,
                    semantic_manifest=manifest,
                    expected_provider_revision=revision,
                )
                test_case.assert_restored(figure, shape, snapshot)

        with TemporaryDirectory() as directory, tempconfig(
            {
                "media_dir": directory,
                "pixel_width": 640,
                "pixel_height": 360,
                "frame_rate": 10,
                "disable_caching": True,
                "write_to_movie": True,
                "save_last_frame": False,
                "format": "mp4",
                "verbosity": "ERROR",
            }
        ):
            EmbeddedEntryBoundaryPreview().render()
            videos = [
                path
                for path in Path(directory).rglob(
                    "EmbeddedEntryBoundaryPreview.mp4"
                )
                if "partial_movie_files" not in path.parts
            ]
            self.assertEqual(len(videos), 1)
            frames = []
            for name, timestamp in (("entry", "0.20"), ("activated", "0.60")):
                frame = Path(directory) / f"{name}.png"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        timestamp,
                        "-i",
                        str(videos[0]),
                        "-frames:v",
                        "1",
                        str(frame),
                    ],
                    check=True,
                )
                frames.append(
                    np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
                )
            difference = np.abs(frames[0] - frames[1])
            changed_pixels = np.max(difference, axis=2) > 8
            entry_ink = np.min(frames[0], axis=2) < 235
            activated_ink = np.min(frames[1], axis=2) < 235
            self.assertLess(float(difference.mean()), 0.1)
            self.assertLessEqual(int(np.count_nonzero(changed_pixels)), 20)
            self.assertLessEqual(
                abs(
                    int(np.count_nonzero(entry_ink))
                    - int(np.count_nonzero(activated_ink))
                ),
                10,
            )

    def test_excluded_binding_stays_fixed_during_driver_motion(self) -> None:
        excluded_analysis = analyze_geometry_rig_3d(
            self.picture,
            selection={
                "candidate_id": "hinge_fold:fold-angle",
                "range": [0.3141592653589793, 1.9547687622336491],
                "include_object_ids": [],
                "exclude_object_ids": ["dot.M"],
            },
        )
        figure, shape, motion, definition, manifest, revision = self.fixture(
            analysis=excluded_analysis,
            timeline=[{"type": "driver", "to": 0.5, "duration": 0.1}],
        )
        scene = _InstantScene()
        scene.add(shape)
        excluded_dot = figure.objects["dot.M"]
        scene.observed_object = excluded_dot
        entry_center = np.asarray(excluded_dot.get_center(), dtype=float).copy()

        play_motion_3d_on_native_shape(
            scene,
            shape,
            motion,
            definition=definition,
            semantic_manifest=manifest,
            expected_provider_revision=revision,
        )

        self.assertTrue(scene.observed_centers)
        for center in scene.observed_centers:
            np.testing.assert_allclose(center, entry_center, atol=1e-9, rtol=0.0)
        self.assertEqual(scene.observed_updater_counts, [0] * len(scene.observed_centers))


if __name__ == "__main__":
    unittest.main()

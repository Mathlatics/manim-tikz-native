from __future__ import annotations

import json
import unittest

import numpy as np
from manim import Scene, ValueTracker, linear, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.boundary_compositing import (
    canonical_quadric_boundary_compositing_json,
)
from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.manim import QuadricManimLimits
from polyhedron_visibility.quadrics.rig import QuadricSectionRig
from polyhedron_visibility.quadrics.section_compositing import (
    canonical_quadric_section_compositing_json,
)


try:
    import cairo as _cairo  # noqa: F401
    from manim.renderer.cairo_renderer import CairoRenderer as _CairoRenderer  # noqa: F401
except (ImportError, OSError):
    CAIRO_AVAILABLE = False
else:
    CAIRO_AVAILABLE = True


VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)
SECTION_ID = "equivalence-section"
SURFACE = SphereSpec("equivalence-sphere", (0.0, 0.0, 0.0), 1.0)
SHIFT_DISTANCE = 0.4
MANUAL_CURVE_ID = f"{SECTION_ID}:component:circle"
RIG_CURVE_ID = f"{SECTION_ID}:rig:slot:0:interval:0"
SEMANTIC_CURVE_ID = f"{SECTION_ID}:semantic:component:0"
RGB_ERROR_THRESHOLD = 8.0
INK_ERROR_THRESHOLD = 12.0
BACKGROUND_RGB = np.asarray((16.0, 24.0, 32.0), dtype=float)


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=8,
        max_fragments_per_curve=8,
        max_segments_per_fragment=64,
        max_surface_segments=128,
        max_dashes_per_fragment=32,
        max_projected_length=18.0,
        max_total_mobjects=10000,
        max_boundary_sources=16,
    )


def _plane_at(progress: float) -> SectionPlane:
    return SectionPlane(
        "equivalence-plane",
        (0.0, 0.0, SHIFT_DISTANCE * float(progress)),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )


def _controller_options() -> dict[str, object]:
    return {
        "projection": VIEW,
        "limits": _limits(),
        "max_chord_error": 0.08,
        "section_max_screen_error": 0.08,
        "include_surface_boundaries": False,
        # Separate Scenes intentionally use the same exact band so painter z
        # values, not merely relative order, remain directly comparable.
        "painter_z_band": (20.0, 30.0),
    }


def _replace_source_id(value: object, source_id: str) -> object:
    """Replace exactly one known authoring-path identity, preserving all else.

    Fragment and painter identities embed their source identity as a prefix.
    Replacing that explicit prefix lets the old raw conic component and the
    Rig's stable slot describe the same semantic curve without sorting,
    rounding, dropping fields, or otherwise weakening the comparison.
    """

    if isinstance(value, str):
        return value.replace(source_id, SEMANTIC_CURVE_ID)
    if isinstance(value, list):
        return [_replace_source_id(item, source_id) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_source_id(item, source_id) for item in value)
    if isinstance(value, dict):
        return {
            _replace_source_id(key, source_id): _replace_source_id(item, source_id)
            for key, item in value.items()
        }
    return value


def _semantic_canonical_json(payload: object, source_id: str) -> str:
    return json.dumps(
        _replace_source_id(payload, source_id),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _section_source(frame, expected_id: str):
    sources = tuple(
        source
        for source in frame.sources
        if source.source_id.startswith(f"{SECTION_ID}:")
    )
    if len(sources) != 1:
        raise AssertionError(
            f"expected one active finite section source, received {len(sources)}"
        )
    source = sources[0]
    if source.source_id != expected_id:
        raise AssertionError(
            f"expected finite section source {expected_id!r}, "
            f"received {source.source_id!r}"
        )
    return source


def _capture_pixels(scene: Scene) -> np.ndarray:
    scene.camera.reset()
    scene.camera.capture_mobjects(scene.mobjects)
    return scene.camera.pixel_array[:, :, :3].copy()


@unittest.skipUnless(CAIRO_AVAILABLE, "Manim Cairo renderer is unavailable")
class QuadricSectionRigEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig(
            {
                "renderer": "cairo",
                "pixel_width": 240,
                "pixel_height": 135,
                "frame_rate": 8,
                "write_to_movie": False,
                "save_last_frame": False,
                "disable_caching": True,
            }
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def test_value_tracker_and_rig_frames_are_canonically_equivalent(self) -> None:
        progress = ValueTracker(0.0)
        manual_scene = Scene()
        rig_scene = Scene()
        manual = QuadricSection3D(
            manual_scene,
            surface=SURFACE,
            section_id=SECTION_ID,
            plane=lambda: _plane_at(progress.get_value()),
            **_controller_options(),
        ).attach()
        rig = QuadricSectionRig(
            rig_scene,
            surface=SURFACE,
            section_id=SECTION_ID,
            plane=_plane_at(0.0),
            **_controller_options(),
        ).attach()
        action = rig.animate_plane_shift(SHIFT_DISTANCE, rate_func=linear)
        action.begin()
        try:
            for value in (0.0, 0.2, 0.5, 0.8, 1.0):
                with self.subTest(progress=value):
                    progress.set_value(value)
                    manual.update()
                    action.interpolate_mobject(value)
                    rig.update()

                    manual_section = manual.last_section_frame
                    rig_section = rig.last_section_frame
                    manual_boundary = manual.last_boundary_frame
                    rig_boundary = rig.last_boundary_frame
                    assert manual_section is not None and rig_section is not None
                    assert manual_boundary is not None and rig_boundary is not None

                    # The mathematical plane and complete finite source curve
                    # must agree before comparing any visibility or paint data.
                    self.assertEqual(manual_section.plane, rig_section.plane)
                    self.assertEqual(rig.state.plane, rig_section.plane)
                    self.assertEqual(rig_section.plane, _plane_at(value))
                    manual_source = _section_source(
                        manual_boundary,
                        MANUAL_CURVE_ID,
                    )
                    rig_source = _section_source(rig_boundary, RIG_CURVE_ID)
                    self.assertEqual(
                        _semantic_canonical_json(
                            manual_source.to_dict(),
                            MANUAL_CURVE_ID,
                        ),
                        _semantic_canonical_json(
                            rig_source.to_dict(),
                            RIG_CURVE_ID,
                        ),
                    )

                    # These are complete canonical payloads: exact surface and
                    # plane geometry, visibility intervals, semantic boundary
                    # fragments, relations, and painter order are all retained.
                    self.assertEqual(
                        _semantic_canonical_json(
                            json.loads(
                                canonical_quadric_section_compositing_json(
                                    manual_section
                                )
                            ),
                            MANUAL_CURVE_ID,
                        ),
                        _semantic_canonical_json(
                            json.loads(
                                canonical_quadric_section_compositing_json(
                                    rig_section
                                )
                            ),
                            RIG_CURVE_ID,
                        ),
                    )
                    self.assertEqual(
                        _semantic_canonical_json(
                            json.loads(
                                canonical_quadric_boundary_compositing_json(
                                    manual_boundary
                                )
                            ),
                            MANUAL_CURVE_ID,
                        ),
                        _semantic_canonical_json(
                            json.loads(
                                canonical_quadric_boundary_compositing_json(
                                    rig_boundary
                                )
                            ),
                            RIG_CURVE_ID,
                        ),
                    )

                    # Keep painter-order evidence explicit even though it is
                    # also present in both canonical frame comparisons above.
                    self.assertEqual(
                        _replace_source_id(
                            manual_boundary.draw_order,
                            MANUAL_CURVE_ID,
                        ),
                        _replace_source_id(
                            rig_boundary.draw_order,
                            RIG_CURVE_ID,
                        ),
                    )
                    self.assertEqual(
                        _replace_source_id(
                            manual.active_painter_z_indices,
                            MANUAL_CURVE_ID,
                        ),
                        _replace_source_id(
                            rig.active_painter_z_indices,
                            RIG_CURVE_ID,
                        ),
                    )
            action.finish()
            self.assertEqual(rig.state, action.target_state)
        finally:
            rig.restore()
            manual.restore()

    def test_manual_and_rig_cairo_keyframes_match_semantic_pixel_tolerance(
        self,
    ) -> None:
        keyframe = 0.65
        manual_progress = ValueTracker(keyframe)
        manual_scene = Scene()
        rig_scene = Scene()
        manual_scene.camera.background_color = "#101820"
        rig_scene.camera.background_color = "#101820"
        manual = QuadricSection3D(
            manual_scene,
            surface=SURFACE,
            section_id=SECTION_ID,
            plane=lambda: _plane_at(manual_progress.get_value()),
            **_controller_options(),
        ).attach()
        rig = QuadricSectionRig(
            rig_scene,
            surface=SURFACE,
            section_id=SECTION_ID,
            plane=_plane_at(0.0),
            **_controller_options(),
        ).attach()
        action = rig.animate_plane_shift(SHIFT_DISTANCE, rate_func=linear)
        action.begin()
        try:
            action.interpolate_mobject(keyframe)
            rig.update()
            manual_pixels = _capture_pixels(manual_scene).astype(float)
            rig_pixels = _capture_pixels(rig_scene).astype(float)

            self.assertEqual(manual_pixels.shape, (135, 240, 3))
            self.assertEqual(rig_pixels.shape, manual_pixels.shape)
            manual_ink = (
                np.linalg.norm(manual_pixels - BACKGROUND_RGB, axis=2)
                > INK_ERROR_THRESHOLD
            )
            rig_ink = (
                np.linalg.norm(rig_pixels - BACKGROUND_RGB, axis=2)
                > INK_ERROR_THRESHOLD
            )
            self.assertGreater(int(np.count_nonzero(manual_ink)), 500)
            self.assertGreater(int(np.count_nonzero(rig_ink)), 500)

            # Match the project's established Cairo semantic RGB tolerance:
            # sub-threshold antialias differences are legal, while no pixel may
            # move outside that color-distance envelope.
            rgb_error = np.linalg.norm(manual_pixels - rig_pixels, axis=2)
            self.assertEqual(
                int(np.count_nonzero(rgb_error > RGB_ERROR_THRESHOLD)),
                0,
                "manual and Rig keyframes differ beyond Cairo AA tolerance "
                f"(maximum RGB distance {float(np.max(rgb_error)):.6g})",
            )
            self.assertEqual(
                int(np.count_nonzero(manual_ink ^ rig_ink)),
                0,
                "manual and Rig keyframes disagree on semantic ink coverage",
            )
        finally:
            rig.restore()
            manual.restore()


if __name__ == "__main__":
    unittest.main()

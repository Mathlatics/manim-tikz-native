"""Three-act classroom introduction to Dandelin spheres.

The geometry in every act is authored through :class:`DandelinSection3D`.
That facade keeps the finite cone section on the production compositor and
adds the Dandelin spheres, contact circles, directrices, and foci as an
explicitly *diagrammatic* teaching overlay.  The overlay is not advertised as
physical cone--sphere depth ordering.

Preview with::

    manim --renderer cairo --disable_caching -ql --fps 12 \
      examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py \
      DandelinThreeConicsLesson
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin, sqrt
from typing import Sequence

from manim import (
    DOWN,
    UP,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    VGroup,
    config,
)

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinSection3D,
    QuadricManimLimits,
    SectionPlane,
)
from polyhedron_visibility.quadrics.dandelin_authoring import (
    DEFAULT_DANDELIN_OVERLAY_STYLE,
    DEFAULT_DANDELIN_SECTION_STYLE,
)
from polyhedron_visibility.quadrics.manim import DEFAULT_QUADRIC_VIEW


BACKGROUND_COLOR = "#0D1722"
HALF_ANGLE = pi / 6.0


def _scaled_view(factor: float) -> ParallelView:
    matrix = DEFAULT_QUADRIC_VIEW.matrix
    matrix[:2] *= float(factor)
    return ParallelView.from_matrix(matrix)


LESSON_VIEW = _scaled_view(0.50)

DOUBLE_CONE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


LESSON_LIMITS = QuadricManimLimits(
    max_surfaces=4,
    max_curves=8,
    max_fragments_per_curve=20,
    max_segments_per_fragment=384,
    max_surface_segments=512,
    max_dashes_per_fragment=96,
    max_projected_length=32.0,
    max_total_mobjects=10000,
    max_boundary_sources=32,
    max_boundary_styles=16,
)


def _normal_with_axis_dot(value: float) -> tuple[float, float, float]:
    """Return a unit normal in the xz-plane with the requested z component."""

    return (sqrt(max(0.0, 1.0 - value * value)), 0.0, value)


@dataclass(frozen=True, slots=True)
class DandelinActSpec:
    act_id: str
    heading: str
    explanation: str
    model: ConeModel
    apex: tuple[float, float, float]
    axial_range: tuple[float, float]
    axis_normal_dot: float
    plane_axial_offset: float


ACTS: tuple[DandelinActSpec, ...] = (
    DandelinActSpec(
        act_id="ellipse",
        heading="ELLIPSE  ·  TWO SPHERES, ONE NAPPE",
        explanation="The two plane-contact points are the two foci.",
        model=ConeModel.OPEN_SINGLE,
        apex=(0.0, 0.0, -4.2),
        axial_range=(0.0, 7.0),
        axis_normal_dot=0.80,
        plane_axial_offset=1.5,
    ),
    DandelinActSpec(
        act_id="parabola",
        heading="PARABOLA  ·  THE CRITICAL ANGLE",
        explanation="One finite tangent sphere remains when the plane is parallel to a generator.",
        model=ConeModel.OPEN_SINGLE,
        apex=(0.0, 0.0, -4.2),
        axial_range=(0.0, 7.0),
        axis_normal_dot=sin(HALF_ANGLE),
        plane_axial_offset=3.0,
    ),
    DandelinActSpec(
        act_id="hyperbola",
        heading="HYPERBOLA  ·  ONE SPHERE ON EACH NAPPE",
        explanation="The complete two-branch construction needs the finite double cone.",
        model=ConeModel.OPEN_DOUBLE,
        apex=(0.0, 0.0, 0.0),
        axial_range=(-2.5, 2.5),
        axis_normal_dot=0.16,
        plane_axial_offset=0.0,
    ),
)


def build_dandelin_act(scene: Scene, act: DandelinActSpec) -> DandelinSection3D:
    """Build one reviewed, still-detached classroom construction."""

    if act.model is ConeModel.OPEN_DOUBLE:
        # This certified side view keeps the two nappe proxies disjoint except
        # at their shared apex, as required by the existing composite facade.
        plane_point = (0.0, 0.5, 0.0)
        plane_normal = (
            0.0,
            sqrt(1.0 - act.axis_normal_dot * act.axis_normal_dot),
            act.axis_normal_dot,
        )
        plane_u_axis = (1.0, 0.0, 0.0)
        projection = DOUBLE_CONE_VIEW
    else:
        plane_point = (
            act.apex[0],
            act.apex[1],
            act.apex[2] + act.plane_axial_offset,
        )
        plane_normal = _normal_with_axis_dot(act.axis_normal_dot)
        plane_u_axis = (0.0, 1.0, 0.0)
        projection = LESSON_VIEW
    facade = DandelinSection3D(
        scene,
        cone=ConeSpec(
            f"classroom:dandelin:{act.act_id}:cone",
            act.apex,
            (0.0, 0.0, 1.0),
            HALF_ANGLE,
            act.axial_range,
            radial_axis=(1.0, 0.0, 0.0),
            model=act.model,
        ),
        plane=SectionPlane(
            f"classroom:dandelin:{act.act_id}:plane",
            plane_point,
            plane_normal,
            u_axis=plane_u_axis,
        ),
        construction_id=f"classroom:dandelin:{act.act_id}",
        projection=projection,
        section_style=DEFAULT_DANDELIN_SECTION_STYLE,
        overlay_style=DEFAULT_DANDELIN_OVERLAY_STYLE,
        limits=LESSON_LIMITS,
        max_chord_error=0.02,
        section_max_screen_error=0.12,
        show_contact_circles=True,
        # In the certified exact side view used by the open-double compositor,
        # one directrix and one edge-on contact circle have coincident screen
        # support.  Omitting those optional teaching lines keeps the v1 scene
        # fail-closed instead of inventing an ordering for coincident ink.
        show_directrices=act.model is not ConeModel.OPEN_DOUBLE,
        show_foci=True,
    )
    # Keep the lesson honest if the facade's public semantic contract changes.
    if facade.visibility_authoritative or facade.overlay_mode != "diagrammatic":
        raise RuntimeError(
            "this lesson requires the non-authoritative diagrammatic "
            "Dandelin teaching overlay"
        )
    return facade


def _header() -> VGroup:
    title = Text(
        "Dandelin spheres: one idea, three conics",
        font_size=39,
        color="#F3F7FA",
        weight="SEMIBOLD",
    ).to_edge(UP, buff=0.20)
    disclaimer = Text(
        "orange = teaching overlay (not physical cone–sphere occlusion)",
        font_size=20,
        color="#F6B28E",
    ).next_to(title, DOWN, buff=0.10)
    return VGroup(title, disclaimer)


def _act_caption(facade: DandelinSection3D, act: DandelinActSpec) -> VGroup:
    sphere_count = len(facade.construction.spheres)
    heading = Text(
        act.heading,
        font_size=27,
        color="#FFD166",
        weight="SEMIBOLD",
    )
    explanation = Text(
        act.explanation,
        font_size=19,
        color="#D7E3EC",
    )
    evidence = Text(
        f"certified finite spheres: {sphere_count}   ·   eccentricity: "
        f"{facade.construction.eccentricity:.3f}",
        font_size=17,
        color="#9EC8DA",
    )
    group = VGroup(heading, explanation, evidence).arrange(DOWN, buff=0.08)
    group.to_edge(DOWN, buff=0.16)
    return group


class DandelinThreeConicsLesson(Scene):
    """A 16:9 Cairo lesson with separate ellipse, parabola, and hyperbola acts."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR
        header = _header()
        self.play(FadeIn(header), run_time=0.6)

        for index, act in enumerate(ACTS):
            facade = build_dandelin_act(self, act).attach()
            caption = _act_caption(facade, act)
            self.play(FadeIn(caption), run_time=0.45)
            self.wait(1.65 if index < len(ACTS) - 1 else 2.0)
            self.play(FadeOut(caption), run_time=0.35)
            facade.restore()

        self.play(FadeOut(header), run_time=0.45)


__all__: Sequence[str] = (
    "ACTS",
    "DandelinActSpec",
    "DandelinThreeConicsLesson",
    "build_dandelin_act",
)


# The project support contract is Cairo-only.  This assignment is deliberately
# passive: the command line still owns output resolution, frame rate, and
# quality, while the default Manim frame remains 16:9.
config.background_color = BACKGROUND_COLOR

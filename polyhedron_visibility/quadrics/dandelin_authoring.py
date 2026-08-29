"""Static Cairo authoring facade for diagrammatic Dandelin constructions.

``DandelinSection3D`` composes the existing finite-cone section controller with
one sphere-only teaching overlay.  The construction is immutable for the
lifetime of the facade.  It deliberately does not claim physical depth between
the contained spheres and their tangent cone; callers can inspect
``visibility_authoritative`` and receive ``False``.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import isfinite
from typing import Iterator, Mapping, Sequence

import numpy as np
from manim import Dot, Mobject, VGroup

from ..geometry import GeometryContext, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from .authoring import QuadricSection3D
from .composite_authoring import CompositeQuadricSection3D
from .contract import ConeModel, ConeSpec, PlaneDisplayPatchSpec, SectionPlane
from .dandelin import DandelinConstruction3D, compute_dandelin_construction
from .dandelin_overlay import (
    DandelinTeachingOverlay3D,
    DandelinTeachingOverlayError,
    build_dandelin_teaching_overlay,
)
from .manim import (
    DEFAULT_QUADRIC_VIEW,
    QUADRIC_MANIM_LIMITS,
    QuadricBoundaryStyle,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch


class DandelinSectionAuthoringError(DandelinTeachingOverlayError):
    """A static diagrammatic Dandelin facade was configured ambiguously."""


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise DandelinSectionAuthoringError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinSectionAuthoringError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise DandelinSectionAuthoringError(f"{label} must be finite and positive")
    return result


def _view(value: ParallelView | Sequence[Sequence[float]] | None) -> ParallelView:
    if value is None:
        return DEFAULT_QUADRIC_VIEW
    if isinstance(value, ParallelView):
        return value
    if callable(value):
        raise DandelinSectionAuthoringError(
            "DandelinSection3D v1 requires one immutable parallel projection"
        )
    try:
        return ParallelView.from_matrix(value)
    except (TypeError, ValueError) as exc:
        raise DandelinSectionAuthoringError(
            "projection must be a finite parallel-view matrix"
        ) from exc


def _directrix_patch(
    construction: DandelinConstruction3D,
    margin_ratio: float,
) -> PlaneDisplayPatchSpec:
    try:
        base = fit_plane_display_patch(
            f"{construction.plane.plane_id}:dandelin-base-patch",
            construction.plane,
            construction.cone.render_components,
            margin_ratio=margin_ratio,
        ).patch
    except PlanePatchFitError as exc:
        raise DandelinSectionAuthoringError(
            f"Dandelin directrix patch fitting failed: {exc}"
        ) from exc
    center = np.asarray(base.center_coordinates, dtype=float)
    lower = center - np.asarray((base.half_width, base.half_height), dtype=float)
    upper = center + np.asarray((base.half_width, base.half_height), dtype=float)
    padding = max(base.half_width, base.half_height) * 0.18
    for directrix in construction.directrices:
        coordinates = np.asarray(directrix.point.coordinates, dtype=float)
        lower = np.minimum(lower, coordinates - padding)
        upper = np.maximum(upper, coordinates + padding)
    expanded_center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return PlaneDisplayPatchSpec(
        f"{construction.plane.plane_id}:dandelin-directrix-patch",
        construction.plane.plane_id,
        float(half[0]),
        float(half[1]),
        tuple(float(item) for item in expanded_center),
    )


DEFAULT_DANDELIN_SECTION_STYLE = QuadricManimStyle(
    surface_fill_color="#2878A5",
    surface_fill_opacity=0.58,
    surface_stroke_color="#67D8EE",
    surface_stroke_opacity=0.0,
    visible_curve_color="#FFD166",
    visible_curve_width=4.0,
    hidden_curve_color="#FFD166",
    hidden_curve_width=2.8,
    hidden_curve_opacity=0.48,
    section_plane_fill_color="#2CB9A4",
    section_plane_fill_opacity=0.16,
    section_plane_stroke_color="#7EE5D5",
    section_plane_stroke_opacity=0.55,
    cone_lateral_fill_colors=("#173753", "#4F9AC1", "#1D4368"),
)

DEFAULT_DANDELIN_OVERLAY_STYLE = QuadricManimStyle(
    surface_fill_color="#F59E7A",
    surface_fill_opacity=0.38,
    surface_stroke_color="#FFD0B8",
    surface_stroke_width=1.5,
    surface_stroke_opacity=0.72,
    visible_curve_color="#FF8A5B",
    visible_curve_width=3.2,
    hidden_curve_color="#FF8A5B",
    hidden_curve_width=2.2,
    hidden_curve_opacity=0.42,
    dash_length=0.10,
    dash_gap=0.08,
)


class DandelinSection3D:
    """Author one immutable, diagrammatic Dandelin classroom construction.

    All geometry is derived before any Manim object is attached.  The cone
    section keeps using ``QuadricSection3D`` (or the existing open-double
    coordinator), while a sphere-only ``QuadricOcclusion3D`` displays the
    certified auxiliary geometry in a separate top teaching band.
    """

    visibility_authoritative = False
    overlay_mode = "diagrammatic"

    def __init__(
        self,
        scene: object,
        *,
        cone: ConeSpec,
        plane: SectionPlane,
        construction_id: str,
        projection: ParallelView | Sequence[Sequence[float]] | None = None,
        section_style: QuadricManimStyle = DEFAULT_DANDELIN_SECTION_STYLE,
        overlay_style: QuadricManimStyle = DEFAULT_DANDELIN_OVERLAY_STYLE,
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        context: GeometryContext | ResolvedGeometryContext | None = None,
        coefficient_tolerance: float | None = None,
        max_chord_error: float = 0.008,
        section_max_screen_error: float = 0.08,
        patch_margin: float = 0.14,
        show_contact_circles: bool = True,
        show_directrices: bool = True,
        show_foci: bool = True,
        focus_color: object = "#FFF4A3",
        focus_radius: float = 0.065,
        section_painter_z_band: tuple[float, float] = (10.0, 20.0),
        overlay_painter_z_band: tuple[float, float] = (21.0, 31.0),
    ) -> None:
        if not isinstance(cone, ConeSpec):
            raise TypeError("cone must be a ConeSpec")
        if not isinstance(plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(section_style, QuadricManimStyle):
            raise TypeError("section_style must be a QuadricManimStyle")
        if not isinstance(overlay_style, QuadricManimStyle):
            raise TypeError("overlay_style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        for name, value in (
            ("show_contact_circles", show_contact_circles),
            ("show_directrices", show_directrices),
            ("show_foci", show_foci),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        self.scene = scene
        self.cone = cone
        self.plane = plane
        self.view = _view(projection)
        self.context = context
        self.focus_radius = _positive(focus_radius, "focus_radius")
        resolved_patch_margin = _positive(patch_margin, "patch_margin")
        self.construction = compute_dandelin_construction(
            construction_id,
            cone,
            plane,
            context=context,
            coefficient_tolerance=coefficient_tolerance,
        )
        patch = _directrix_patch(
            self.construction,
            resolved_patch_margin,
        )
        self.teaching_overlay: DandelinTeachingOverlay3D = (
            build_dandelin_teaching_overlay(
                self.construction,
                patch,
                context=context,
            )
        )
        common_section = {
            "section_id": f"{self.construction.construction_id}:section",
            "plane": plane,
            "projection": self.view,
            "paint_policy": "depth_aware_diagrammatic",
            "style": section_style,
            "boundary_styles": boundary_styles,
            "limits": limits,
            "max_chord_error": max_chord_error,
            "painter_z_band": section_painter_z_band,
            "context": context,
            "coefficient_tolerance": coefficient_tolerance,
            "section_max_screen_error": section_max_screen_error,
            "plane_patch_margin": resolved_patch_margin,
            "include_surface_boundaries": True,
        }
        if cone.model is ConeModel.OPEN_DOUBLE:
            self.section_controller = CompositeQuadricSection3D(
                scene,
                surface=cone,
                **common_section,
            )
        else:
            self.section_controller = QuadricSection3D(
                scene,
                surface=cone,
                show_plane=True,
                **common_section,
            )
        overlay_curves = (
            *(
                self.teaching_overlay.contact_curves
                if show_contact_circles
                else ()
            ),
            *(
                self.teaching_overlay.directrix_curves
                if show_directrices
                else ()
            ),
        )
        self.overlay_controller = QuadricOcclusion3D(
            scene,
            surfaces=self.teaching_overlay.sphere_surfaces,
            curves=overlay_curves,
            projection=self.view,
            paint_policy="diagrammatic",
            style=overlay_style,
            limits=limits,
            max_chord_error=max_chord_error,
            painter_z_band=overlay_painter_z_band,
            surface_order_mode="explicit",
            boundary_visibility_mode="legacy",
            include_surface_boundaries=False,
        )
        matrix = self.view.matrix
        focus_dots = []
        if show_foci:
            for sphere in self.construction.spheres:
                screen = matrix[:2] @ np.asarray(sphere.focus.world_point, dtype=float)
                dot = Dot(
                    (float(screen[0]), float(screen[1]), 0.0),
                    radius=self.focus_radius,
                    color=focus_color,
                )
                dot.set_z_index(overlay_painter_z_band[1] + 1.0)
                focus_dots.append(dot)
        self.focus_group = VGroup(*focus_dots)
        self._display_group = VGroup(
            self.section_controller.display_mobject,
            self.overlay_controller.display_mobject,
            self.focus_group,
        )
        self._attached = False

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def display_mobject(self) -> Mobject:
        return self._display_group

    def attach(self) -> "DandelinSection3D":
        if self._attached:
            return self
        section_attached = False
        overlay_attached = False
        try:
            self.section_controller.attach()
            section_attached = True
            self.overlay_controller.attach()
            overlay_attached = True
            if self.focus_group.submobjects:
                self.scene.add(self.focus_group)
            self._attached = True
        except Exception:
            if self.focus_group.submobjects:
                self.scene.remove(self.focus_group)
            if overlay_attached:
                self.overlay_controller.restore()
            if section_attached:
                self.section_controller.restore()
            raise
        return self

    def restore(self) -> "DandelinSection3D":
        if not self._attached:
            return self
        if self.focus_group.submobjects:
            self.scene.remove(self.focus_group)
        self.overlay_controller.restore()
        self.section_controller.restore()
        self._attached = False
        return self

    def detach(self) -> "DandelinSection3D":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["DandelinSection3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return (
            *self.section_controller.slot_identities(),
            *self.overlay_controller.slot_identities(),
            *(id(item) for item in self.focus_group.get_family()),
        )


__all__ = [
    "DEFAULT_DANDELIN_OVERLAY_STYLE",
    "DEFAULT_DANDELIN_SECTION_STYLE",
    "DandelinSection3D",
    "DandelinSectionAuthoringError",
]

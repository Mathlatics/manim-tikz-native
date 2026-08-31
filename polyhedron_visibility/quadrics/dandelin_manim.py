"""Live Cairo binding for certified Dandelin hidden-line frames.

The class in this module is intentionally a thin specialization of
``QuadricOcclusion3D``.  Dandelin geometry contributes only stable semantic
boundary sources and tangent-contact evidence.  The existing quadric
visibility kernel, projected crossing solver, fragment painter graph, fixed
slot allocator, and transactional Manim binding remain the implementation.

The cone, sphere, and plane *strokes* are geometry-depth authoritative.  The
translucent quadric fills retain one explicit teaching order, so the aggregate
view and surface compositing remain non-authoritative.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from .boundary_compositing import QuadricBoundarySource
from .contract import PlaneDisplayPatchSpec
from .curves import PointMarker3D
from .dandelin import DandelinConstruction3D
from .dandelin_visibility import (
    DandelinVisibilityError,
    DandelinVisibilityFrame,
    DandelinVisibilitySource,
    _surface_items,
    _visibility_frame_from_compositing,
    build_dandelin_visibility_sources,
    fit_dandelin_visibility_patch,
)
from .manim import (
    DEFAULT_QUADRIC_VIEW,
    QUADRIC_MANIM_LIMITS,
    ProjectionInput,
    QuadricBoundaryStyle,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)


class DandelinManimError(QuadricManimError):
    """One live Dandelin frame cannot be certified or committed."""


DEFAULT_DANDELIN_OCCLUSION_LIMITS = replace(
    QUADRIC_MANIM_LIMITS,
    max_fragments_per_curve=96,
)

DEFAULT_DANDELIN_OCCLUSION_STYLE = QuadricManimStyle(
    surface_fill_color="#173753",
    surface_fill_opacity=0.22,
    surface_stroke_color="#67D8EE",
    surface_stroke_opacity=0.0,
    point_color="#FFF4A3",
    point_radius=0.065,
    point_opacity=1.0,
    cone_lateral_fill_colors=("#173753", "#4F9AC1", "#1D4368"),
    dash_length=0.10,
    dash_gap=0.08,
)


def _stroke_style(
    color: object,
    width: float,
    *,
    hidden_opacity: float = 0.48,
) -> QuadricBoundaryStyle:
    return QuadricBoundaryStyle(
        visible_color=color,
        visible_width=width,
        visible_opacity=1.0,
        hidden_color=color,
        hidden_width=max(0.8, 0.72 * width),
        hidden_opacity=hidden_opacity,
        dash_length=0.10,
        dash_gap=0.08,
    )


DEFAULT_DANDELIN_BOUNDARY_STYLES: Mapping[str, QuadricBoundaryStyle] = {
    "style:dandelin-cone-wire": _stroke_style("#67D8EE", 1.25),
    "style:dandelin-sphere-silhouette": _stroke_style("#FFD0B8", 1.8),
    "style:dandelin-contact": _stroke_style("#FF8A5B", 2.0),
    "style:dandelin-section": _stroke_style("#FFD166", 3.0),
    "style:dandelin-directrix": _stroke_style("#C4B5FD", 1.6),
    "style:dandelin-plane-outline": _stroke_style("#7EE5D5", 1.4),
}


class DandelinOcclusion3D(QuadricOcclusion3D):
    """Render one immutable construction under a live parallel camera.

    Every successful update stores a ``DandelinVisibilityFrame`` whose
    ``compositing_frame`` is the exact object committed by the inherited Manim
    binding.  Camera changes therefore recompute both evidence and display in
    one transaction without allocating a second painter system.
    """

    visibility_authoritative = False
    curve_visibility_authoritative = True
    surface_visibility_authoritative = False
    mode = "depth_aware_diagrammatic"

    def __init__(
        self,
        scene: object,
        *,
        construction: DandelinConstruction3D,
        projection: ProjectionInput | None = None,
        display_patch: PlaneDisplayPatchSpec | None = None,
        show_contact_circles: bool = True,
        show_directrices: bool = True,
        show_plane_boundary: bool = True,
        show_foci: bool = True,
        generator_count: int = 8,
        style: QuadricManimStyle = DEFAULT_DANDELIN_OCCLUSION_STYLE,
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = DEFAULT_DANDELIN_OCCLUSION_LIMITS,
        max_chord_error: float = 1.0e-3,
        painter_z_band: tuple[float, float] = (20.0, 40.0),
        patch_margin: float = 0.14,
    ) -> None:
        if not isinstance(construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        for name, value in (
            ("show_contact_circles", show_contact_circles),
            ("show_directrices", show_directrices),
            ("show_plane_boundary", show_plane_boundary),
            ("show_foci", show_foci),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not isinstance(style, QuadricManimStyle):
            raise TypeError("style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")

        self.construction = construction
        self.display_patch = (
            fit_dandelin_visibility_patch(
                construction,
                include_directrices=show_directrices,
                margin_ratio=patch_margin,
            )
            if display_patch is None
            else display_patch
        )
        if not isinstance(self.display_patch, PlaneDisplayPatchSpec):
            raise TypeError("display_patch must be a PlaneDisplayPatchSpec")
        if self.display_patch.plane_id != construction.plane.plane_id:
            raise DandelinManimError(
                "display_patch does not belong to the Dandelin cutting plane"
            )
        self._show_contact_circles = show_contact_circles
        self._show_directrices = show_directrices
        self._show_plane_boundary = show_plane_boundary
        self._generator_count = generator_count
        self._pending_sources: tuple[DandelinVisibilitySource, ...] | None = None
        self._pending_view = None
        self._last_visibility_frame: DandelinVisibilityFrame | None = None

        initial_sources = self._source_records(DEFAULT_QUADRIC_VIEW)
        allocated_boundary_ids = tuple(
            item.source_id for item in initial_sources
        )
        surfaces = _surface_items(construction)
        surface_ids = tuple(item.surface_id for item in surfaces)
        surface_constraints = tuple(zip(surface_ids, surface_ids[1:]))
        sphere_ids = {item.sphere_id for item in construction.spheres}
        surface_opacities = {
            item.surface_id: (0.22 if item.surface_id in sphere_ids else 0.13)
            for item in surfaces
        }
        surface_stroke_opacities = {
            item.surface_id: 0.0 for item in surfaces
        }
        points = (
            tuple(
                PointMarker3D(record.focus_id, record.focus.world_point)
                for record in construction.spheres
            )
            if show_foci
            else ()
        )
        merged_styles = dict(DEFAULT_DANDELIN_BOUNDARY_STYLES)
        if boundary_styles is not None:
            if not isinstance(boundary_styles, Mapping):
                raise TypeError("boundary_styles must be a mapping")
            merged_styles.update(boundary_styles)

        super().__init__(
            scene,
            surfaces=surfaces,
            surface_opacities=surface_opacities,
            surface_stroke_opacities=surface_stroke_opacities,
            curves=(),
            points=points,
            projection=(DEFAULT_QUADRIC_VIEW if projection is None else projection),
            paint_policy="depth_aware_diagrammatic",
            style=style,
            boundary_styles=merged_styles,
            limits=limits,
            max_chord_error=max_chord_error,
            context=construction.certification_context,
            painter_z_band=painter_z_band,
            surface_constraints=surface_constraints,
            surface_order_mode="explicit",
            boundary_visibility_mode="unified",
            include_surface_boundaries=False,
            boundary_source_factory=self._resolve_source_records,
            allocated_boundary_ids=allocated_boundary_ids,
        )

    def _source_records(
        self,
        view,
    ) -> tuple[DandelinVisibilitySource, ...]:
        return build_dandelin_visibility_sources(
            self.construction,
            view,
            display_patch=self.display_patch,
            include_contact_circles=self._show_contact_circles,
            include_directrices=self._show_directrices,
            include_plane_boundary=self._show_plane_boundary,
            generator_count=self._generator_count,
        )

    def _resolve_source_records(
        self,
        view,
    ) -> Sequence[QuadricBoundarySource]:
        records = self._source_records(view)
        self._pending_sources = records
        self._pending_view = view
        return tuple(item.source for item in records)

    def _commit_visibility_frame(self) -> None:
        boundary = self.last_boundary_frame
        sources = self._pending_sources
        view = self._pending_view
        if boundary is None or sources is None or view is None:
            raise DandelinManimError(
                "live Dandelin update committed without boundary evidence"
            )
        try:
            self._last_visibility_frame = _visibility_frame_from_compositing(
                self.construction,
                view,
                sources,
                boundary,
            )
        except DandelinVisibilityError as exc:
            raise DandelinManimError(
                f"live Dandelin evidence cannot be committed: {exc}"
            ) from exc

    @property
    def last_visibility_frame(self) -> DandelinVisibilityFrame | None:
        return self._last_visibility_frame

    def attach(self) -> "DandelinOcclusion3D":
        if self.attached:
            return self
        old_sources = self._pending_sources
        old_view = self._pending_view
        try:
            super().attach()
            self._commit_visibility_frame()
        except Exception:
            if self.attached:
                super().restore()
            self._pending_sources = old_sources
            self._pending_view = old_view
            self._last_visibility_frame = None
            raise
        return self

    def update(self, dt: float = 0.0) -> "DandelinOcclusion3D":
        if not self.attached:
            raise DandelinManimError(
                "Dandelin occlusion controller is not attached"
            )
        snapshot = self.snapshot_transaction_state()
        old_sources = self._pending_sources
        old_view = self._pending_view
        old_frame = self._last_visibility_frame
        try:
            super().update(dt)
            self._commit_visibility_frame()
        except Exception:
            try:
                self.restore_transaction_state(snapshot)
            finally:
                self._pending_sources = old_sources
                self._pending_view = old_view
                self._last_visibility_frame = old_frame
            raise
        return self

    def restore(self) -> "DandelinOcclusion3D":
        super().restore()
        self._pending_sources = None
        self._pending_view = None
        self._last_visibility_frame = None
        return self


__all__ = [
    "DEFAULT_DANDELIN_BOUNDARY_STYLES",
    "DEFAULT_DANDELIN_OCCLUSION_LIMITS",
    "DEFAULT_DANDELIN_OCCLUSION_STYLE",
    "DandelinManimError",
    "DandelinOcclusion3D",
]

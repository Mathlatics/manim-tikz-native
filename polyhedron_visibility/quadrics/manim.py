"""Public Cairo binding with depth-aware cutting-plane outline slots."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Mapping, Sequence

import numpy as np
from manim import VGroup, VMobject

from ..painter_band import ManagedPainterBand
from . import _manim_impl as _impl
from ._manim_impl import *  # noqa: F401,F403
from .section_compositing import PlaneDepthRole, QuadricSectionCompositingFrame


# Keep the public dependency name patchable. The inherited implementation reads
# its own module globals, so each preparation synchronizes this public hook into
# the private implementation module before any fallible production work begins.
compute_global_quadric_frame = _impl.compute_global_quadric_frame


def _set_open_subpaths(value: VMobject, paths: Sequence[np.ndarray]) -> None:
    """Replace one fixed VMobject with independently open polyline subpaths."""

    value.clear_points()
    for raw in paths:
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
            raise QuadricManimError(
                "section outline paths must contain finite 3D polylines"
            )
        if not np.all(np.isfinite(points)):
            raise QuadricManimError(
                "section outline paths must contain finite 3D polylines"
            )
        value.start_new_path(points[0])
        value.add_points_as_corners(points[1:])


class QuadricOcclusion3D(_impl.QuadricOcclusion3D):
    """Quadric controller whose plane border follows the fill depth roles."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        painter_z_band = kwargs.get("painter_z_band", (20.0, 30.0))
        super().__init__(*args, **kwargs)
        if not self._section_enabled:
            return

        estimated_mobjects = (
            len(self._surface_ids)
            + 10
            + 1
            + len(self._curve_ids)
            * (
                1
                + self.limits.max_fragments_per_curve
                * (self.limits.max_dashes_per_fragment + 3)
            )
            + 4
        )
        if estimated_mobjects > self.limits.max_total_mobjects:
            raise QuadricManimCapacityError(
                f"preallocated Mobject count {estimated_mobjects} exceeds fixed "
                f"limit {self.limits.max_total_mobjects}"
            )

        self._section_slots = tuple(VMobject() for _index in range(10))
        surface_root = VGroup(*self._surface_slots)
        section_root = VGroup(*self._section_slots)
        curve_root = VGroup(
            *(self._curve_slots[key].root for key in self._curve_ids)
        )
        self.root = _impl._ManagedQuadricDisplayGroup(
            surface_root,
            section_root,
            curve_root,
            opacity_sentinel=self._opacity_sentinel,
        )
        self._band = ManagedPainterBand(
            z_band=painter_z_band,  # type: ignore[arg-type]
            managed_roots=(self.root,),
        )

    def _prepare_numeric(self) -> object:
        _impl.compute_global_quadric_frame = compute_global_quadric_frame
        numeric = super()._prepare_numeric()
        prepared = numeric.section_layers
        if prepared is None:
            return numeric
        frame = prepared.frame
        if not isinstance(frame, QuadricSectionCompositingFrame):
            raise QuadricManimError(
                "section compositor did not return depth-aware outline geometry"
            )
        outline_paths: Mapping[PlaneDepthRole, tuple[np.ndarray, ...]] = {
            role: tuple(
                np.asarray(
                    (
                        (*fragment.screen_start, 0.0),
                        (*fragment.screen_end, 0.0),
                    ),
                    dtype=float,
                )
                for fragment in frame.outline_fragments_by_role[role]
            )
            for role in PlaneDepthRole
        }
        section_layers = _impl._PreparedSectionLayers(
            frame,
            prepared.surface_points,
            prepared.plane_polygons,
            outline_paths,  # type: ignore[arg-type]
        )
        return replace(numeric, section_layers=section_layers)

    def _apply_section_layers(self, prepared: object, opacity: float) -> None:
        frame = prepared.frame  # type: ignore[attr-defined]
        if not isinstance(frame, QuadricSectionCompositingFrame):
            raise QuadricManimError(
                "prepared section frame is not depth-aware"
            )
        if len(self._section_slots) != len(frame.paint_items.ordered):
            raise QuadricManimCapacityError(
                "section painter slots were not allocated"
            )
        slots = dict(zip(frame.paint_items.ordered, self._section_slots))
        surface_back = slots[frame.paint_items.surface_back]
        surface_front = slots[frame.paint_items.surface_front]

        surface_back.set_points_as_corners(prepared.surface_points)  # type: ignore[attr-defined]
        surface_front.set_points_as_corners(prepared.surface_points)  # type: ignore[attr-defined]
        combined_surface_opacity = min(
            1.0,
            self.style.surface_fill_opacity * opacity,
        )
        sheet_opacity = 1.0 - sqrt(max(0.0, 1.0 - combined_surface_opacity))
        surface_back.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
        )
        surface_back.set_stroke(opacity=0.0)
        surface_front.set_fill(
            color=self.style.surface_fill_color,
            opacity=sheet_opacity,
        )
        surface_front.set_stroke(
            color=self.style.surface_stroke_color,
            width=self.style.surface_stroke_width,
            opacity=self.style.surface_stroke_opacity * opacity,
        )

        fill_item_by_role = {
            PlaneDepthRole.BEHIND_SURFACE: frame.paint_items.plane_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: frame.paint_items.plane_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: frame.paint_items.plane_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: frame.paint_items.plane_front,
        }
        for role, item_id in fill_item_by_role.items():
            slot = slots[item_id]
            _impl._set_closed_subpaths(
                slot,
                prepared.plane_polygons[role],  # type: ignore[attr-defined]
            )
            slot.set_fill(
                color=self.style.section_plane_fill_color,
                opacity=self.style.section_plane_fill_opacity * opacity,
            )
            slot.set_stroke(opacity=0.0)

        outline_paths = prepared.plane_outline_points  # type: ignore[attr-defined]
        if not isinstance(outline_paths, Mapping):
            raise QuadricManimError(
                "prepared plane outline paths must be grouped by depth role"
            )
        for role, item_id in frame.paint_items.outline_by_role.items():
            slot = slots[item_id]
            _set_open_subpaths(slot, outline_paths[role])
            slot.set_fill(opacity=0.0)
            slot.set_stroke(
                color=self.style.section_plane_stroke_color,
                width=self.style.section_plane_stroke_width,
                opacity=self.style.section_plane_stroke_opacity * opacity,
            )


def __getattr__(name: str) -> object:
    return getattr(_impl, name)


__all__ = _impl.__all__

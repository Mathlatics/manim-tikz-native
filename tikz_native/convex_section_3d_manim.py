"""Bind a compiled TikZ convex solid to the dynamic section module.

The existing TikZ visibility adapter remains the topology authority.  This
module adds one infinite cutting plane, an automatically fitted display patch,
derived section geometry, and solid intersection traces without modifying the
compiler or legacy 3D runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ContextManager, Mapping, Sequence

from manim import Mobject, Polygon

from polyhedron_visibility import OcclusionStyle, ParallelProjection
from polyhedron_visibility.binding import DisplayPointProvider
from polyhedron_visibility.sections import (
    ConvexSection3D,
    ConvexSectionStyle,
    FaceDepthCueStyle,
    PlaneProvider,
    SectionedVisibilityFrame,
    TransparentSectionCompositingFrame,
)

from .compiler import PictureSpec
from .manim_renderer import NativeFigure
from .polyhedron_visibility_3d_adapter import (
    TikzNativeVisibility3DAdapterResult,
    adapt_picture_visibility_3d,
)
from .polyhedron_visibility_3d_manim import (
    CoordinateProvider,
    TikzNativeVisibility3DManimError,
    _canonical_position_provider,
    _fit_entry_display_mapper,
    _single_object_stroke_bindings,
)


@dataclass(frozen=True)
class TikzNativeConvexSection3D:
    """A proven TikZ closed solid plus its reversible section controller."""

    analysis: TikzNativeVisibility3DAdapterResult
    controller: ConvexSection3D

    @property
    def last_frame(self) -> SectionedVisibilityFrame | None:
        return self.controller.last_sectioned_frame

    @property
    def last_transparent_compositing(
        self,
    ) -> TransparentSectionCompositingFrame | None:
        return self.controller.last_transparent_compositing

    def attach(self) -> "TikzNativeConvexSection3D":
        self.controller.attach()
        return self

    def update(self, dt: float = 0.0) -> "TikzNativeConvexSection3D":
        self.controller.update(dt)
        return self

    def restore(self) -> "TikzNativeConvexSection3D":
        self.controller.restore()
        return self

    def session(self) -> ContextManager[ConvexSection3D]:
        return self.controller.session()


def bind_picture_convex_section_3d(
    scene: object,
    picture: PictureSpec,
    figure: NativeFigure,
    *,
    plane_provider: PlaneProvider,
    source_style: OcclusionStyle,
    section_style: ConvexSectionStyle | None = None,
    face_depth_style: FaceDepthCueStyle | None = None,
    accurate_transparency: bool = False,
    transparent_coplanar_policy: str = "section_over_solid",
    plane_patch_mode: str = "auto",
    plane_patch_margin: float = 0.15,
    section_id: str = "section",
    coordinate_provider: CoordinateProvider | None = None,
    projection: ParallelProjection | None = None,
    display_point_provider: DisplayPointProvider | None = None,
) -> TikzNativeConvexSection3D:
    """Bind one compiled closed convex solid and one moving infinite plane.

    Every complete named straight TikZ line is managed globally.  Surface
    edges are protected from their incident faces, while an independent line
    is clipped against all solid faces and the automatically fitted display
    patch.  Accurate transparency is opt-in and requires one compiler-proven
    native Polygon for every semantic face.  Set ``plane_patch_mode='strict'``
    only when authored dimensions intentionally describe a literal finite panel.
    """

    analysis = adapt_picture_visibility_3d(
        picture,
        validation_mode="closed_convex_polyhedron",
    )
    positions = _canonical_position_provider(analysis, coordinate_provider)
    stroke_bindings = _single_object_stroke_bindings(analysis, figure)
    face_fill_bindings: Mapping[str, Mobject] | None = None
    if accurate_transparency or face_depth_style is not None:
        resolved_faces: dict[str, Mobject] = {}
        for face in analysis.face_bindings:
            if len(face.object_ids) != 1:
                raise TikzNativeVisibility3DManimError(
                    "accurate TikZ transparency requires one complete Polygon "
                    f"per semantic face; {face.face_id} owns {len(face.object_ids)} objects"
                )
            object_id = face.object_ids[0]
            source = figure.objects.get(object_id)
            if (
                not isinstance(source, Polygon)
                or tuple(source.get_family()) != (source,)
            ):
                raise TikzNativeVisibility3DManimError(
                    "accurate TikZ transparency requires one native Manim Polygon "
                    f"per semantic face; {face.face_id} is compound or missing"
                )
            resolved_faces[face.face_id] = source
        face_fill_bindings = resolved_faces
    current_projection = projection or ParallelProjection(
        analysis.entry_projection
    )
    if display_point_provider is None:
        mapper = _fit_entry_display_mapper(picture, figure, analysis)

        def fitted_display_point(world: Sequence[float]) -> Sequence[float]:
            return mapper(world, current_projection.current_matrix(scene))

        display_point_provider = fitted_display_point

    controller = ConvexSection3D(
        scene,
        analysis.model,
        position_provider=positions,
        stroke_bindings=stroke_bindings,
        plane_provider=plane_provider,
        projection=current_projection,
        source_style=source_style,
        section_style=section_style,
        face_fill_bindings=face_fill_bindings,
        face_depth_style=face_depth_style,
        accurate_transparency=accurate_transparency,
        transparent_coplanar_policy=transparent_coplanar_policy,
        plane_patch_mode=plane_patch_mode,
        plane_patch_margin=plane_patch_margin,
        display_point_provider=display_point_provider,
        source_coordinate_mode="display",
        section_id=section_id,
    )
    return TikzNativeConvexSection3D(analysis, controller)


__all__ = [
    "TikzNativeConvexSection3D",
    "bind_picture_convex_section_3d",
]

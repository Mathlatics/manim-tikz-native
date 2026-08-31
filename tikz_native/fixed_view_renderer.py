from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from manim import Circle, Line, Mobject, ORIGIN, RIGHT, UP

from polyhedron_visibility.quadrics.planar_curves import (
    Circle3DSpec,
    Ellipse3DSpec,
)

from .compiler import ObjectSpec, PictureSpec
from .dandelin_contract import (
    TikzDandelinContractError,
    restore_dandelin_construction_contract,
    restore_dandelin_static_diagram_contract,
    restore_space_right_cone_contract,
)
from .dandelin_fixed_view import build_dandelin_fixed_view
from .manim_renderer import NativeFigure, NativeManimRenderer
from .planar_curve_projection import project_planar_curve_2d
from .planar_curve_style import (
    certify_planar_curve_affine_display,
    certify_planar_curve_display_segment,
    certify_planar_curve_display_scale,
    validate_planar_curve_stroke_style,
)
from .planar_curves_3d import (
    PlanarTikz3DError,
    restore_planar_frame_geometry,
    restore_registered_planar_curve_geometry,
)
from .projection_3d import project_point


class NativeFixedViewRenderer(NativeManimRenderer):
    """Instantiate 2D and authored-view 3D pictures as ordinary 2D Mobjects.

    A 3D ``PictureSpec`` retains its original world coordinates and TikZ view
    matrix in the manifest.  This renderer only projects the runtime Mobjects
    into the fixed authored view, which lets the PPT editor use the same normal
    ``Scene`` and timeline path for 2D diagrams and simple polyhedra.

    It deliberately does not implement camera motion.  Consumers that need to
    orbit a figure must use ``NativeManim3DRenderer`` instead.
    """

    def render(self, picture: PictureSpec) -> NativeFigure:
        if picture.dimension not in {2, 3}:
            raise ValueError(f"unsupported TikZ picture dimension: {picture.dimension}")
        if picture.dimension == 3 and picture.projection_3d is None:
            raise ValueError("fixed-view 3D picture has no TikZ projection")
        if picture.dimension == 3:
            plane_less_curves = [
                item.id
                for item in picture.objects
                if item.kind in {"circle", "ellipse"}
            ]
            if plane_less_curves:
                raise ValueError(
                    "fixed-view 3D renderer refuses plane-less circle/ellipse "
                    "objects: " + ", ".join(plane_less_curves)
                )
            unsupported_canvas_nodes = [
                item.id
                for item in picture.objects
                if item.style.transform_shape
                and item.style.native_canvas_plane is not None
            ]
            if unsupported_canvas_nodes:
                raise ValueError(
                    "fixed-view 3D renderer does not yet support transform-shape "
                    "canvas nodes: " + ", ".join(unsupported_canvas_nodes)
                )
        return super().render(picture)

    def _build(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        if spec.kind == "dandelin_diagram":
            return self._build_dandelin_diagram(spec, picture)
        if spec.kind in {"planar_circle_3d", "planar_ellipse_3d"}:
            return self._build_planar_curve_3d(spec, picture)
        return super()._build(spec, picture)

    @staticmethod
    def _dandelin_ref(
        payload: dict[str, object],
        key: str,
        label: str,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} has no valid {key}")
        return value.strip()

    def _build_dandelin_diagram(
        self,
        spec: ObjectSpec,
        picture: PictureSpec,
    ) -> Mobject:
        payload = spec.geometry
        if picture.dandelin_diagrams.get(spec.id) != payload:
            raise ValueError(
                "Dandelin object payload disagrees with its picture registry"
            )
        construction_ref = self._dandelin_ref(
            payload,
            "constructionRef",
            "Dandelin diagram",
        )
        cone_ref = self._dandelin_ref(payload, "coneRef", "Dandelin diagram")
        plane_ref = self._dandelin_ref(payload, "planeRef", "Dandelin diagram")
        construction_payload = picture.dandelin_constructions_3d.get(
            construction_ref
        )
        cone_payload = picture.space_right_cones_3d.get(cone_ref)
        plane_payload = picture.planar_frames_3d.get(plane_ref)
        if construction_payload is None or cone_payload is None or plane_payload is None:
            raise ValueError(
                "Dandelin diagram references unavailable cone, plane, or construction data"
            )
        try:
            cone_contract = restore_space_right_cone_contract(
                cone_payload,
                picture.coordinates,
                expected_cone_ref=cone_ref,
            )
            plane_frame = restore_planar_frame_geometry(
                plane_payload,
                coordinates=picture.coordinates,
                expected_plane_id=plane_ref,
            ).frame
            construction_contract = restore_dandelin_construction_contract(
                construction_payload,
                cone=cone_contract.cone,
                plane_frame=plane_frame,
                expected_construction_ref=construction_ref,
                expected_cone_ref=cone_ref,
                expected_plane_ref=plane_ref,
            )
            diagram = restore_dandelin_static_diagram_contract(
                payload,
                construction_contract.construction,
                expected_diagram_id=spec.id,
            )
            projection = (
                picture.projection_3d.matrix
                if diagram.view == "spatial" and picture.projection_3d is not None
                else None
            )
            group = build_dandelin_fixed_view(
                construction_contract.construction,
                view=diagram.view,
                projection_matrix=projection,
                preset=diagram.preset,
                mode=diagram.mode,
                show_contact_circles=diagram.show_contact_circles,
                show_directrices=diagram.show_directrices,
                show_foci=diagram.show_foci,
            )
        except (TikzDandelinContractError, PlanarTikz3DError) as exc:
            raise ValueError(str(exc)) from exc
        display_scale = float(self.unit * picture.scale)
        if not np.isfinite(display_scale) or display_scale <= 0.0:
            raise ValueError("Dandelin diagram display scale must be finite and positive")
        group.scale(display_scale)
        group.dandelin_metadata["sceneScale"] = display_scale
        return group

    def _build_planar_curve_3d(
        self,
        spec: ObjectSpec,
        picture: PictureSpec,
    ) -> Mobject:
        if picture.projection_3d is None:
            raise ValueError("explicit 3D planar curve has no projection matrix")
        validate_planar_curve_stroke_style(spec.style)
        geometry = restore_registered_planar_curve_geometry(
            spec.geometry,
            picture.planar_frames_3d,
            expected_curve_id=spec.id,
        )
        expected_type = (
            Circle3DSpec
            if spec.kind == "planar_circle_3d"
            else Ellipse3DSpec
        )
        if not isinstance(geometry.curve, expected_type):
            raise ValueError(
                f"object kind {spec.kind!r} disagrees with its planar curve payload"
            )
        analytic = geometry.curve.lower_to_analytic_curve()
        if not analytic.closed:
            raise ValueError(
                "explicit 3D planar curve v1 requires one full revolution"
            )
        projected = project_planar_curve_2d(
            geometry.curve,
            picture.projection_3d.matrix,
        )
        display_scale = certify_planar_curve_display_scale(
            self.unit,
            picture.scale,
        )
        if projected.rank == 1:
            assert projected.segment_start_offset is not None
            assert projected.segment_end_offset is not None
            with np.errstate(over="ignore", invalid="ignore"):
                screen_center = display_scale * np.asarray(
                    projected.center,
                    dtype=float,
                )
                start_offset = display_scale * np.asarray(
                    projected.segment_start_offset,
                    dtype=float,
                )
                end_offset = display_scale * np.asarray(
                    projected.segment_end_offset,
                    dtype=float,
                )
            screen_start, screen_end = certify_planar_curve_display_segment(
                screen_center,
                start_offset,
                end_offset,
            )
            start = np.asarray((*screen_start, 0.0), dtype=float)
            end = np.asarray((*screen_end, 0.0), dtype=float)
            return Line(start, end, **self._line_kwargs(spec.style))

        with np.errstate(over="ignore", invalid="ignore"):
            basis = display_scale * projected.screen_basis
            screen_center = display_scale * np.asarray(
                projected.center,
                dtype=float,
            )
        certify_planar_curve_affine_display(screen_center, basis)
        center = np.asarray((*screen_center, 0.0), dtype=float)
        transform = np.array(
            (
                (basis[0, 0], basis[0, 1], 0.0),
                (basis[1, 0], basis[1, 1], 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        curve = Circle(
            radius=1.0,
            fill_opacity=0.0,
            **self._line_kwargs(spec.style),
        )
        curve.apply_matrix(transform, about_point=ORIGIN)
        curve.shift(center)
        return curve

    def point(
        self,
        value: Sequence[float],
        picture: PictureSpec,
    ) -> np.ndarray:
        values = tuple(float(component) for component in value)
        if len(values) == 2:
            return super().point(values, picture)
        if len(values) != 3:
            raise ValueError(
                f"fixed-view renderer received {len(values)}D point: {values}"
            )
        if picture.projection_3d is None:
            raise ValueError("3D point has no TikZ projection matrix")
        screen_x, screen_y, _depth = project_point(
            picture.projection_3d.matrix,
            values,
        )
        return self.unit * picture.scale * (
            screen_x * RIGHT + screen_y * UP
        )


__all__ = ["NativeFixedViewRenderer"]

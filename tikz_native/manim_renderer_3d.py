from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import atan2, ceil, cos, sin

import numpy as np
from manim import Circle, Line, Mobject, ORIGIN, ParametricFunction, Dot3D, VGroup

from polyhedron_visibility.quadrics.planar_curves import (
    Circle3DSpec,
    Ellipse3DSpec,
)

from .compiler import ObjectSpec, OcclusionRelationSpec, PictureSpec, StyleSpec
from .manim_renderer import ANCHOR_TO_EDGE, NativeManimRenderer
from .occlusion_3d import parallel_occlusion_interval, parallel_view_direction
from .planar_curve_style import (
    certify_planar_curve_affine_display,
    certify_planar_curve_display_scale,
    validate_planar_curve_stroke_style,
)
from .planar_curves_3d import restore_registered_planar_curve_geometry
from .projection_3d import Matrix3, project_point, screen_delta_to_world


@dataclass
class Native3DFigure:
    picture: PictureSpec
    objects: dict[str, Mobject]
    world_group: VGroup
    fixed_orientation_labels: list[Mobject]
    view_center: np.ndarray
    warnings: list[str]
    occlusion_groups: dict[str, Mobject] = field(default_factory=dict)


@dataclass(frozen=True)
class _StableStrokeSlot:
    """A fixed family of native Lines used for one styled segment."""

    lines: tuple[Line, ...]
    dash_pattern: tuple[float, float] | None
    opacity: float


@dataclass(frozen=True)
class _OcclusionLineSlots:
    """Stable native children for the three possible occlusion intervals."""

    visible_before: _StableStrokeSlot
    hidden: _StableStrokeSlot
    visible_after: _StableStrokeSlot

    @property
    def lines(self) -> tuple[Line, ...]:
        return (
            *self.visible_before.lines,
            *self.hidden.lines,
            *self.visible_after.lines,
        )


class NativeManim3DRenderer(NativeManimRenderer):
    """Build native Manim objects while retaining TikZ world coordinates."""

    SUPPORTED_KINDS = {
        "line",
        "arrow",
        "polygon",
        "dot",
        "label",
        "path_label",
        "angle",
        "angle_label",
        "right_angle",
        "planar_circle_3d",
        "planar_ellipse_3d",
    }

    def render(self, picture: PictureSpec) -> Native3DFigure:
        if picture.dimension != 3 or picture.projection_3d is None:
            raise ValueError("NativeManim3DRenderer requires a 3D TikZ picture")

        objects: dict[str, Mobject] = {}
        labels: list[Mobject] = []
        world_objects: list[Mobject] = []
        warnings = list(picture.warnings)
        for spec in picture.objects:
            if spec.kind not in self.SUPPORTED_KINDS:
                raise ValueError(
                    f"No native 3D builder for picture {picture.index} "
                    f"object {spec.id} ({spec.kind})"
                )
            try:
                mobject = self._build(spec, picture)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to build 3D picture {picture.index} object "
                    f"{spec.id} ({spec.kind}): {error}"
                ) from error
            mobject.set_z_index(spec.z_index)
            objects[spec.id] = mobject
            if spec.kind in {"label", "path_label", "angle_label"}:
                labels.append(mobject)
            else:
                world_objects.append(mobject)

        return Native3DFigure(
            picture=picture,
            objects=objects,
            world_group=VGroup(*world_objects),
            fixed_orientation_labels=labels,
            view_center=self._view_center(picture),
            warnings=warnings,
        )

    def point(self, value: tuple[float, ...], picture: PictureSpec) -> np.ndarray:
        if len(value) != 3:
            raise ValueError(f"3D renderer received {len(value)}D point: {value}")
        return self.unit * picture.scale * np.asarray(value, dtype=float)

    def _build(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        if spec.kind in {"planar_circle_3d", "planar_ellipse_3d"}:
            return self._build_planar_curve_3d(spec, picture)
        return super()._build(spec, picture)

    def _build_planar_curve_3d(
        self,
        spec: ObjectSpec,
        picture: PictureSpec,
    ) -> Mobject:
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
        display_scale = certify_planar_curve_display_scale(
            self.unit,
            picture.scale,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            display_basis = display_scale * np.column_stack(
                (
                    np.asarray(analytic.first_axis, dtype=float),
                    np.asarray(analytic.second_axis, dtype=float),
                )
            )
            display_normal = display_scale * np.asarray(
                analytic.normal,
                dtype=float,
            )
            center = display_scale * np.asarray(analytic.center, dtype=float)
        certify_planar_curve_affine_display(center, display_basis)
        transform = np.column_stack((display_basis, display_normal))
        if not np.all(np.isfinite(transform)):
            raise ValueError(
                "explicit 3D planar curve lies outside the finite Manim range"
            )
        curve = Circle(
            radius=1.0,
            fill_opacity=0.0,
            **self._line_kwargs(spec.style),
        )
        curve.apply_matrix(transform, about_point=ORIGIN)
        curve.shift(center)
        return curve

    def _screen_delta_world(
        self,
        picture: PictureSpec,
        delta_u: float,
        delta_v: float,
        matrix: Matrix3 | None = None,
    ) -> np.ndarray:
        assert picture.projection_3d is not None
        return np.asarray(
            screen_delta_to_world(
                picture.projection_3d.matrix if matrix is None else matrix,
                delta_u,
                delta_v,
            ),
            dtype=float,
        )

    @staticmethod
    def _canvas_screen_matrix(
        picture: PictureSpec,
        plane: str,
        matrix: Matrix3,
    ) -> np.ndarray:
        """Return the TikZ local-canvas affine map in screen coordinates."""

        values = np.asarray(matrix, dtype=float)
        if plane == "yz":
            columns = (1, 2)
        elif plane == "xy":
            columns = (0, 1)
        else:
            raise ValueError(f"unsupported native canvas plane: {plane}")
        first, second = columns
        scale = float(picture.scale)
        return scale * np.array(
            [
                [values[0, first], values[0, second]],
                [values[1, first], values[1, second]],
            ],
            dtype=float,
        )

    def _apply_node_shape_transform(
        self,
        label: Mobject,
        spec: ObjectSpec,
        picture: PictureSpec,
    ) -> Mobject:
        plane = spec.style.native_canvas_plane
        if not spec.style.transform_shape or plane is None:
            return super()._apply_node_shape_transform(label, spec, picture)

        assert picture.projection_3d is not None
        local_width = float(label.width)
        local_height = float(label.height)
        local_pad = self._outer_node_padding(label, spec.style)
        screen_matrix = self._canvas_screen_matrix(
            picture,
            plane,
            picture.projection_3d.matrix,
        )
        label.apply_matrix(
            np.array(
                [
                    [screen_matrix[0, 0], screen_matrix[0, 1], 0.0],
                    [screen_matrix[1, 0], screen_matrix[1, 1], 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        label._tikz_local_anchor_box = (  # type: ignore[attr-defined]
            local_width,
            local_height,
            float(local_pad[0]),
            float(local_pad[1]),
        )
        label._tikz_canvas_screen_matrix = screen_matrix  # type: ignore[attr-defined]
        return label

    def _build_dot(self, spec: ObjectSpec, picture: PictureSpec) -> Dot3D:
        radius = spec.geometry.get("radius_pt", 1.0) * self.pt
        color = spec.style.fill_color or spec.style.draw_color or "#20242A"
        return Dot3D(
            self.point(tuple(spec.geometry["center"]), picture),
            radius=radius,
            color=color,
            fill_opacity=self._opacity(spec.style, "fill"),
            resolution=(6, 12),
        )

    def _build_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        label = self._apply_node_shape_transform(
            self._make_label(spec.label or "", spec.style),
            spec,
            picture,
        )
        assert picture.projection_3d is not None
        return label.move_to(
            self._label_center(
                label,
                spec,
                picture,
                picture.projection_3d.matrix,
            )
        )

    def _label_center(
        self,
        label: Mobject,
        spec: ObjectSpec,
        picture: PictureSpec,
        matrix: Matrix3,
        coordinate_provider: Callable[[str], Sequence[float] | np.ndarray]
        | None = None,
    ) -> np.ndarray:
        coordinate_name = spec.geometry.get("at_name")
        coordinate = (
            coordinate_provider(str(coordinate_name))
            if coordinate_provider is not None and coordinate_name
            else spec.geometry["at"]
        )
        target = self.point(tuple(float(value) for value in coordinate), picture)
        placement = spec.placement
        if placement is None:
            return target

        edge = ANCHOR_TO_EDGE[placement.anchor]
        local_box = getattr(label, "_tikz_local_anchor_box", None)
        screen_matrix = getattr(label, "_tikz_canvas_screen_matrix", None)
        if local_box is not None and screen_matrix is not None:
            local_width, local_height, pad_x, pad_y = local_box
            local_anchor = np.array(
                [
                    edge[0] * (local_width / 2 + pad_x),
                    edge[1] * (local_height / 2 + pad_y),
                ],
                dtype=float,
            )
            anchor_offset = np.asarray(screen_matrix, dtype=float) @ local_anchor
        else:
            pad_x, pad_y = self._outer_node_padding(label, spec.style)
            anchor_offset = np.array(
                [
                    edge[0] * (label.width / 2 + pad_x),
                    edge[1] * (label.height / 2 + pad_y),
                ],
                dtype=float,
            )
        screen_offset = np.array(
            [placement.dx_pt * self.pt, placement.dy_pt * self.pt],
            dtype=float,
        ) - anchor_offset
        return target + self._screen_delta_world(
            picture,
            float(screen_offset[0]),
            float(screen_offset[1]),
            matrix,
        )

    def _build_path_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        label = self._apply_node_shape_transform(
            self._make_label(spec.label or "", spec.style),
            spec,
            picture,
        )
        placement = spec.placement
        if placement is None:
            raise ValueError("path label placement missing")
        assert picture.projection_3d is not None
        center, display_angle = self._path_label_state(
            label,
            spec,
            picture,
            picture.projection_3d.matrix,
        )
        if display_angle is not None:
            label.rotate(display_angle, about_point=ORIGIN)
        return label.move_to(center)

    def _projected_angle_state(
        self,
        spec: ObjectSpec,
        picture: PictureSpec,
        matrix: Matrix3,
    ) -> tuple[np.ndarray, float, float, float]:
        first = self.point(tuple(spec.geometry["first"]), picture)
        vertex = self.point(tuple(spec.geometry["vertex"]), picture)
        third = self.point(tuple(spec.geometry["third"]), picture)
        projected_first = np.asarray(project_point(matrix, first), dtype=float)
        projected_vertex = np.asarray(project_point(matrix, vertex), dtype=float)
        projected_third = np.asarray(project_point(matrix, third), dtype=float)
        first_vector = projected_first[:2] - projected_vertex[:2]
        third_vector = projected_third[:2] - projected_vertex[:2]
        start = atan2(float(first_vector[1]), float(first_vector[0]))
        end = atan2(float(third_vector[1]), float(third_vector[0]))
        while end < start:
            end += 2 * np.pi
        radius = float(spec.geometry["radius_pt"]) * self.pt
        return vertex, start, end, radius

    def _build_angle(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        assert picture.projection_3d is not None
        matrix = picture.projection_3d.matrix
        vertex, start, end, radius = self._projected_angle_state(
            spec,
            picture,
            matrix,
        )

        def point_on_arc(parameter: float) -> np.ndarray:
            angle = start + parameter * (end - start)
            return vertex + self._screen_delta_world(
                picture,
                radius * cos(angle),
                radius * sin(angle),
                matrix,
            )

        if spec.style.fill_color is not None:
            return self.native_polygon_from_points(
                [vertex]
                + [point_on_arc(index / 32.0) for index in range(33)],
                spec.style,
            )

        return ParametricFunction(
            point_on_arc,
            t_range=(0.0, 1.0, 1.0 / 32.0),
            **self._line_kwargs(spec.style),
        )

    def _build_right_angle(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        assert picture.projection_3d is not None
        matrix = picture.projection_3d.matrix
        vertex, start, end, radius = self._projected_angle_state(
            spec,
            picture,
            matrix,
        )
        first_delta = self._screen_delta_world(
            picture,
            radius * cos(start),
            radius * sin(start),
            matrix,
        )
        third_delta = self._screen_delta_world(
            picture,
            radius * cos(end),
            radius * sin(end),
            matrix,
        )
        corner = vertex + first_delta + third_delta
        kwargs = self._line_kwargs(spec.style)
        return VGroup(
            Line(vertex + first_delta, corner, **kwargs),
            Line(corner, vertex + third_delta, **kwargs),
        )

    def _build_angle_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        assert picture.projection_3d is not None
        matrix = picture.projection_3d.matrix
        label = self._make_label(spec.label or "", spec.style)
        vertex, start, end, radius = self._projected_angle_state(
            spec,
            picture,
            matrix,
        )
        midpoint = 0.5 * (start + end)
        target = vertex + self._screen_delta_world(
            picture,
            float(spec.geometry.get("eccentricity", 1.0)) * radius * cos(midpoint),
            float(spec.geometry.get("eccentricity", 1.0)) * radius * sin(midpoint),
            matrix,
        )
        placement = spec.placement
        if placement is None:
            return label.move_to(target)
        edge = ANCHOR_TO_EDGE[placement.anchor]
        pad_x, pad_y = self._outer_node_padding(label, spec.style)
        screen_offset = np.array(
            [
                placement.dx_pt * self.pt
                - edge[0] * (label.width / 2 + pad_x),
                placement.dy_pt * self.pt
                - edge[1] * (label.height / 2 + pad_y),
            ]
        )
        return label.move_to(
            target
            + self._screen_delta_world(
                picture,
                float(screen_offset[0]),
                float(screen_offset[1]),
                matrix,
            )
        )

    def _path_label_state(
        self,
        label: Mobject,
        spec: ObjectSpec,
        picture: PictureSpec,
        matrix: Matrix3,
        coordinate_provider: Callable[[str], Sequence[float] | np.ndarray]
        | None = None,
    ) -> tuple[np.ndarray, float | None]:
        placement = spec.placement
        if placement is None:
            raise ValueError("path label placement missing")
        start_name = spec.geometry.get("start_name")
        end_name = spec.geometry.get("end_name")
        start_value = (
            coordinate_provider(str(start_name))
            if coordinate_provider is not None and start_name
            else spec.geometry["start"]
        )
        end_value = (
            coordinate_provider(str(end_name))
            if coordinate_provider is not None and end_name
            else spec.geometry["end"]
        )
        start = self.point(tuple(float(value) for value in start_value), picture)
        end = self.point(tuple(float(value) for value in end_value), picture)
        vector = end - start
        base = start + float(spec.geometry["pos"]) * vector
        projected_start = np.asarray(project_point(matrix, start), dtype=float)
        projected_end = np.asarray(project_point(matrix, end), dtype=float)
        screen_vector = projected_end[:2] - projected_start[:2]
        length = float(np.linalg.norm(screen_vector))
        if length <= 1e-12:
            raise ValueError("path label projects to a zero-length segment")
        tangent = screen_vector / length
        normal = np.array([-tangent[1], tangent[0]], dtype=float)

        edge = ANCHOR_TO_EDGE[placement.anchor]
        pad_x, pad_y = self._outer_node_padding(label, spec.style)
        anchor_offset = np.array(
            [
                edge[0] * (label.width / 2 + pad_x),
                edge[1] * (label.height / 2 + pad_y),
            ],
            dtype=float,
        )
        if placement.sloped:
            desired = self.pt * (
                placement.dx_pt * tangent + placement.dy_pt * normal
            ) - anchor_offset
            display_angle = atan2(tangent[1], tangent[0])
            if display_angle > np.pi / 2 or display_angle < -np.pi / 2:
                display_angle += np.pi
            rotation = np.array(
                [
                    [cos(display_angle), -sin(display_angle)],
                    [sin(display_angle), cos(display_angle)],
                ]
            )
            desired += anchor_offset - rotation @ anchor_offset
        else:
            desired = self.pt * np.array(
                [placement.dx_pt, placement.dy_pt],
                dtype=float,
            ) - anchor_offset

            display_angle = None

        return (
            base
            + self._screen_delta_world(
                picture,
                float(desired[0]),
                float(desired[1]),
                matrix,
            ),
            display_angle,
        )

    @staticmethod
    def _matrix_tuple(matrix: np.ndarray) -> Matrix3:
        values = np.asarray(matrix, dtype=float)
        if values.shape != (3, 3):
            raise ValueError("camera projection matrix must be 3x3")
        return tuple(
            tuple(float(value) for value in row) for row in values
        )  # type: ignore[return-value]

    def bind_labels_to_camera(
        self,
        figure: Native3DFigure,
        camera,
        *,
        coordinate_provider: Callable[[str], Sequence[float] | np.ndarray]
        | None = None,
    ) -> None:
        """Keep TikZ screen anchors stable while camera or coordinates move.

        ``coordinate_provider`` is optional so the established fixed-geometry
        behavior remains byte-for-byte equivalent at the API boundary.  A 3D
        motion runtime can provide current logical TikZ coordinates without
        mutating ``PictureSpec`` or rebuilding label objects.
        """

        for spec in figure.picture.objects:
            if spec.kind not in {"label", "path_label", "angle_label"}:
                continue
            label = figure.objects[spec.id]
            if spec.kind == "angle_label":
                # The initial TikZ screen placement is exact.  A later dynamic
                # pass can rebuild this label from the moving angle marker.
                continue
            if spec.kind == "label":

                def update_label(
                    mobject: Mobject,
                    _dt: float = 0.0,
                    *,
                    object_spec: ObjectSpec = spec,
                ) -> None:
                    matrix = self._matrix_tuple(camera.get_projection_matrix())
                    mobject.move_to(
                        self._label_center(
                            mobject,
                            object_spec,
                            figure.picture,
                            matrix,
                            coordinate_provider,
                        )
                    )

                label.add_updater(update_label)
                continue

            initial_state = self._path_label_state(
                label,
                spec,
                figure.picture,
                self._matrix_tuple(camera.get_projection_matrix()),
                coordinate_provider,
            )
            previous_angle = [initial_state[1] or 0.0]

            def update_path_label(
                mobject: Mobject,
                _dt: float = 0.0,
                *,
                object_spec: ObjectSpec = spec,
                angle_state: list[float] = previous_angle,
            ) -> None:
                matrix = self._matrix_tuple(camera.get_projection_matrix())
                center, angle = self._path_label_state(
                    mobject,
                    object_spec,
                    figure.picture,
                    matrix,
                    coordinate_provider,
                )
                if angle is not None:
                    mobject.rotate(
                        angle - angle_state[0],
                        about_point=mobject.get_center(),
                    )
                    angle_state[0] = angle
                mobject.move_to(center)

            label.add_updater(update_path_label)

    def _occlusion_world_state(
        self,
        relation: OcclusionRelationSpec,
        picture: PictureSpec,
        camera,
        coordinate_provider: Callable[[str], Sequence[float] | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None]:
        """Return current endpoints and the part hidden by the finite face."""

        perspective_getter = getattr(camera, "get_perspective_strength", None)
        perspective = float(perspective_getter()) if perspective_getter else 0.0
        if perspective > 1e-9:
            raise ValueError(
                "dynamic TikZ line-face occlusion currently supports "
                "parallel projection only"
            )

        def world_point(name: str) -> np.ndarray:
            value = tuple(float(component) for component in coordinate_provider(name))
            return self.point(value, picture)

        start = world_point(relation.start_name)
        end = world_point(relation.end_name)
        face = [world_point(name) for name in relation.face_names]
        view = parallel_view_direction(camera.get_projection_matrix())
        interval = parallel_occlusion_interval(start, end, face, view)
        return start, end, interval

    def _make_stable_stroke_slot(
        self,
        start: np.ndarray,
        end: np.ndarray,
        style: StyleSpec,
        z_index: int,
    ) -> _StableStrokeSlot:
        """Preallocate every native Line a styled interval can ever need."""

        dash_pattern: tuple[float, float] | None = None
        capacity = 1
        if style.dash_pattern_pt:
            on_pt, off_pt = style.dash_pattern_pt
            on_length = max(float(on_pt) * self.pt, 1e-6)
            off_length = max(float(off_pt) * self.pt, 0.0)
            dash_pattern = (on_length, off_length)
            full_length = float(np.linalg.norm(end - start))
            capacity = max(
                1,
                ceil(full_length / max(on_length + off_length, 1e-6)) + 1,
            )

        lines = tuple(
            Line(start, end, **self._line_kwargs(style))
            for _ in range(capacity)
        )
        for line in lines:
            line.set_z_index(z_index)
            line.set_stroke(opacity=0.0)
        return _StableStrokeSlot(
            lines=lines,
            dash_pattern=dash_pattern,
            opacity=self._opacity(style, "draw"),
        )

    def _make_occlusion_slots(
        self,
        relation: OcclusionRelationSpec,
        start: np.ndarray,
        end: np.ndarray,
    ) -> _OcclusionLineSlots:
        return _OcclusionLineSlots(
            visible_before=self._make_stable_stroke_slot(
                start,
                end,
                relation.visible_style,
                relation.z_index,
            ),
            hidden=self._make_stable_stroke_slot(
                start,
                end,
                relation.hidden_style,
                relation.z_index,
            ),
            visible_after=self._make_stable_stroke_slot(
                start,
                end,
                relation.visible_style,
                relation.z_index,
            ),
        )

    @staticmethod
    def _hide_stroke_slot(slot: _StableStrokeSlot, first_unused: int = 0) -> None:
        for line in slot.lines[first_unused:]:
            line.set_stroke(opacity=0.0)

    def _update_stroke_slot(
        self,
        slot: _StableStrokeSlot,
        start: np.ndarray,
        end: np.ndarray,
        *,
        active: bool,
    ) -> None:
        """Mutate a fixed Line pool without changing its object topology."""

        vector = end - start
        length = float(np.linalg.norm(vector))
        if not active or length <= 1e-10:
            self._hide_stroke_slot(slot)
            return

        if slot.dash_pattern is None:
            slot.lines[0].put_start_and_end_on(start, end)
            slot.lines[0].set_stroke(opacity=slot.opacity)
            self._hide_stroke_slot(slot, 1)
            return

        on_length, off_length = slot.dash_pattern
        direction = vector / length
        cursor = 0.0
        used = 0
        while cursor < length - 1e-9:
            if used >= len(slot.lines):
                raise ValueError(
                    "animated occlusion line exceeded its preallocated dash capacity"
                )
            dash_end = min(cursor + on_length, length)
            slot.lines[used].put_start_and_end_on(
                start + cursor * direction,
                start + dash_end * direction,
            )
            slot.lines[used].set_stroke(opacity=slot.opacity)
            used += 1
            cursor += on_length + off_length
        self._hide_stroke_slot(slot, used)

    def _update_occlusion_slots(
        self,
        slots: _OcclusionLineSlots,
        start: np.ndarray,
        end: np.ndarray,
        interval: tuple[float, float] | None,
    ) -> None:
        vector = end - start
        if interval is None:
            self._update_stroke_slot(
                slots.visible_before,
                start,
                end,
                active=True,
            )
            self._hide_stroke_slot(slots.hidden)
            self._hide_stroke_slot(slots.visible_after)
            return

        hidden_start, hidden_end = interval
        self._update_stroke_slot(
            slots.visible_before,
            start,
            start + hidden_start * vector,
            active=hidden_start > 1e-7,
        )
        self._update_stroke_slot(
            slots.hidden,
            start + hidden_start * vector,
            start + hidden_end * vector,
            active=hidden_end - hidden_start > 1e-7,
        )
        self._update_stroke_slot(
            slots.visible_after,
            start + hidden_end * vector,
            end,
            active=hidden_end < 1.0 - 1e-7,
        )

    def bind_occlusions_to_camera(
        self,
        figure: Native3DFigure,
        camera,
        *,
        coordinate_provider: Callable[[str], Sequence[float] | np.ndarray]
        | None = None,
    ) -> dict[str, Mobject]:
        """Recompute semantic TikZ line-face occlusion on every camera frame.

        Call this before adding ``figure.world_group`` to a scene.  Each source
        occlusion command becomes one stable container containing a fixed pool
        of native ``Line`` objects.  Camera and coordinate changes only mutate
        endpoints and opacity; no child is added, removed, or replaced during
        a Cairo ``play``.  A custom ``coordinate_provider`` can later expose
        animated source coordinates; by default the authored TikZ coordinates
        remain fixed.
        """

        if not figure.picture.occlusion_relations:
            return figure.occlusion_groups
        if figure.occlusion_groups:
            return figure.occlusion_groups

        provider = coordinate_provider or figure.picture.coordinates.__getitem__
        for relation in figure.picture.occlusion_relations:
            members = [figure.objects[object_id] for object_id in relation.object_ids]
            member_identity = {id(item) for item in members}
            children = list(figure.world_group.submobjects)
            positions = [
                index
                for index, child in enumerate(children)
                if id(child) in member_identity
            ]
            if len(positions) != len(members):
                raise ValueError(
                    f"occlusion relation {relation.id!r} is not a direct "
                    "member of figure.world_group"
                )
            insert_at = min(positions)
            start, end, interval = self._occlusion_world_state(
                relation,
                figure.picture,
                camera,
                provider,
            )
            slots = self._make_occlusion_slots(relation, start, end)
            container = VGroup(*slots.lines)
            container.set_z_index(relation.z_index)
            self._update_occlusion_slots(slots, start, end, interval)
            rebuilt: list[Mobject] = []
            inserted = False
            for index, child in enumerate(children):
                if index == insert_at:
                    rebuilt.append(container)
                    inserted = True
                if id(child) not in member_identity:
                    rebuilt.append(child)
            if not inserted:
                rebuilt.append(container)
            figure.world_group.submobjects[:] = rebuilt

            def update_occlusion(
                _item: Mobject,
                _dt: float = 0.0,
                *,
                relation_spec: OcclusionRelationSpec = relation,
                stable_slots: _OcclusionLineSlots = slots,
            ) -> None:
                current_start, current_end, current_interval = (
                    self._occlusion_world_state(
                        relation_spec,
                        figure.picture,
                        camera,
                        provider,
                    )
                )
                self._update_occlusion_slots(
                    stable_slots,
                    current_start,
                    current_end,
                    current_interval,
                )

            container.add_updater(update_occlusion)
            figure.occlusion_groups[relation.id] = container
            figure.objects[relation.id] = container
        return figure.occlusion_groups

    def _view_center(self, picture: PictureSpec) -> np.ndarray:
        assert picture.projection_3d is not None
        authored_points = [
            tuple(value) for value in picture.coordinates.values()
        ]
        for spec in picture.objects:
            if spec.kind not in {"planar_circle_3d", "planar_ellipse_3d"}:
                continue
            geometry = restore_registered_planar_curve_geometry(
                spec.geometry,
                picture.planar_frames_3d,
                expected_curve_id=spec.id,
            )
            analytic = geometry.curve.lower_to_analytic_curve()
            center = np.asarray(analytic.center, dtype=float)
            first = np.asarray(analytic.first_axis, dtype=float)
            second = np.asarray(analytic.second_axis, dtype=float)
            matrix = np.asarray(picture.projection_3d.matrix, dtype=float)
            for row in matrix:
                first_coefficient = float(np.dot(row, first))
                second_coefficient = float(np.dot(row, second))
                parameter = atan2(second_coefficient, first_coefficient)
                radial = first * cos(parameter) + second * sin(parameter)
                authored_points.extend(
                    tuple(float(item) for item in point)
                    for point in (center - radial, center + radial)
                )
        if not authored_points:
            return np.zeros(3)
        points = np.array(
            [self.point(tuple(value), picture) for value in authored_points],
            dtype=float,
        )
        projected = np.array(
            [project_point(picture.projection_3d.matrix, point) for point in points],
            dtype=float,
        )
        screen_center = 0.5 * (
            projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)
        )
        depth_center = 0.5 * (
            float(projected[:, 2].min()) + float(projected[:, 2].max())
        )
        center = self._screen_delta_world(
            picture,
            float(screen_center[0]),
            float(screen_center[1]),
        )
        center += depth_center * np.asarray(
            picture.projection_3d.matrix[2],
            dtype=float,
        )
        return center


__all__ = ["Native3DFigure", "NativeManim3DRenderer"]

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from typing import Callable, Sequence

import numpy as np
from manim import Mobject

from .compiler import ObjectSpec, PictureSpec
from .manim_renderer import NativeFigure, NativeManimRenderer


Point2 = tuple[float, float]
PointProvider = Callable[[], Sequence[float] | np.ndarray]


@dataclass(frozen=True)
class EllipseChordState:
    """One exact state of a line cutting an ellipse through an interior pivot."""

    center: Point2
    focus: Point2
    line_start: Point2
    line_end: Point2
    p: Point2
    q: Point2
    r: Point2


class EllipseChordDriver:
    """Cache a parameter-driven chord state once per animation frame value."""

    def __init__(
        self,
        angle: Callable[[], float],
        *,
        semi_major: float,
        semi_minor: float,
        center: Point2 = (0.0, 0.0),
        focus: Point2,
        backward_length: float,
        forward_length: float,
    ) -> None:
        self.angle = angle
        self.parameters = {
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "center": center,
            "focus": focus,
            "backward_length": backward_length,
            "forward_length": forward_length,
        }
        self._cached_angle: float | None = None
        self._cached_state: EllipseChordState | None = None

    @classmethod
    def from_named_intersection(
        cls,
        angle: Callable[[], float],
        picture: PictureSpec,
        *,
        relation_index: int = 0,
        pivot_name: str,
    ) -> "EllipseChordDriver":
        """Build the driver from a native TikZ named-path intersection relation."""

        try:
            relation = picture.intersections[relation_index]
        except IndexError as error:
            raise ValueError(
                f"Picture {picture.index} has no intersection relation {relation_index}"
            ) from error
        line = picture.named_paths[relation.sort_by]
        if line.kind != "line":
            raise ValueError("The intersection sort path must be an oriented line")
        ellipse_name = (
            relation.path_b
            if relation.path_a == relation.sort_by
            else relation.path_a
        )
        ellipse = picture.named_paths[ellipse_name]
        if ellipse.kind != "ellipse":
            raise ValueError("EllipseChordDriver requires one line and one ellipse")
        if tuple(relation.coordinate_names) != ("Q", "P"):
            raise ValueError(
                "EllipseChordDriver expects line-sorted intersection names ('Q', 'P')"
            )
        if pivot_name not in picture.coordinates:
            raise ValueError(f"Unknown pivot coordinate: {pivot_name}")

        pivot = picture.coordinates[pivot_name]
        line_start = line.geometry["start"]
        line_end = line.geometry["end"]
        direction = (
            line_end[0] - line_start[0],
            line_end[1] - line_start[1],
        )
        direction_length = sqrt(direction[0] ** 2 + direction[1] ** 2)
        if direction_length <= 1e-12:
            raise ValueError("The named line path has zero length")
        unit = (
            direction[0] / direction_length,
            direction[1] / direction_length,
        )

        def signed_distance(point: Point2) -> float:
            relative = (point[0] - pivot[0], point[1] - pivot[1])
            perpendicular = relative[0] * unit[1] - relative[1] * unit[0]
            if abs(perpendicular) > 1e-9:
                raise ValueError(
                    f"Pivot {pivot_name} is not on the named line path"
                )
            return relative[0] * unit[0] + relative[1] * unit[1]

        start_distance = signed_distance(line_start)
        end_distance = signed_distance(line_end)
        if start_distance >= 0 or end_distance <= 0:
            raise ValueError(
                "The oriented named line must run from the negative side of the "
                "pivot to the positive side"
            )
        return cls(
            angle,
            semi_major=ellipse.geometry["rx"],
            semi_minor=ellipse.geometry["ry"],
            center=ellipse.geometry["center"],
            focus=pivot,
            backward_length=-start_distance,
            forward_length=end_distance,
        )

    def state(self) -> EllipseChordState:
        angle = self.angle()
        if self._cached_state is None or angle != self._cached_angle:
            self._cached_angle = angle
            self._cached_state = ellipse_chord_state(angle, **self.parameters)
        return self._cached_state


def ellipse_chord_state(
    angle: float,
    *,
    semi_major: float,
    semi_minor: float,
    center: Point2 = (0.0, 0.0),
    focus: Point2,
    backward_length: float,
    forward_length: float,
) -> EllipseChordState:
    """Intersect a directed line with an axis-aligned ellipse analytically."""

    if semi_major <= 0 or semi_minor <= 0:
        raise ValueError("Ellipse semiaxes must be positive")
    if backward_length < 0 or forward_length < 0:
        raise ValueError("Displayed line lengths must be nonnegative")

    direction = (cos(angle), sin(angle))
    relative_focus = (focus[0] - center[0], focus[1] - center[1])
    quadratic_a = (
        direction[0] ** 2 / semi_major**2
        + direction[1] ** 2 / semi_minor**2
    )
    quadratic_b = 2 * (
        relative_focus[0] * direction[0] / semi_major**2
        + relative_focus[1] * direction[1] / semi_minor**2
    )
    quadratic_c = (
        relative_focus[0] ** 2 / semi_major**2
        + relative_focus[1] ** 2 / semi_minor**2
        - 1
    )
    discriminant = quadratic_b**2 - 4 * quadratic_a * quadratic_c
    if discriminant < -1e-12:
        raise ValueError("The driven line does not intersect the ellipse")
    root = sqrt(max(discriminant, 0.0))
    parameters = sorted(
        (
            (-quadratic_b - root) / (2 * quadratic_a),
            (-quadratic_b + root) / (2 * quadratic_a),
        )
    )

    def on_line(parameter: float) -> Point2:
        return (
            focus[0] + parameter * direction[0],
            focus[1] + parameter * direction[1],
        )

    q = on_line(parameters[0])
    p = on_line(parameters[1])
    r = (2 * center[0] - p[0], 2 * center[1] - p[1])
    return EllipseChordState(
        center=center,
        focus=focus,
        line_start=on_line(-backward_length),
        line_end=on_line(forward_length),
        p=p,
        q=q,
        r=r,
    )


def project_point_to_line(
    point: Point2,
    line_start: Point2,
    line_end: Point2,
) -> Point2:
    """Return the orthogonal projection of a point onto an infinite line."""

    direction = (
        line_end[0] - line_start[0],
        line_end[1] - line_start[1],
    )
    squared_length = direction[0] ** 2 + direction[1] ** 2
    if squared_length <= 1e-18:
        raise ValueError("Cannot project onto a zero-length line")
    relative = (point[0] - line_start[0], point[1] - line_start[1])
    parameter = (
        relative[0] * direction[0] + relative[1] * direction[1]
    ) / squared_length
    return (
        line_start[0] + parameter * direction[0],
        line_start[1] + parameter * direction[1],
    )


def _point3(value: Sequence[float] | np.ndarray) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape == (2,):
        return np.array([point[0], point[1], 0.0])
    if point.shape != (3,):
        raise ValueError(f"Expected a 2D or 3D point, got shape {point.shape}")
    return point


class NativeMotionBinder:
    """Attach dependency-driven updaters to already converted native objects."""

    def __init__(
        self,
        figure: NativeFigure,
        renderer: NativeManimRenderer,
    ) -> None:
        self.figure = figure
        self.renderer = renderer
        self.specs = {spec.id: spec for spec in figure.picture.objects}

    def _get(self, object_id: str, expected: set[str]) -> tuple[ObjectSpec, Mobject]:
        if object_id not in self.specs:
            raise KeyError(f"Picture {self.figure.picture.index} has no {object_id!r}")
        spec = self.specs[object_id]
        if spec.kind not in expected:
            raise TypeError(
                f"Object {object_id!r} is {spec.kind!r}, expected one of {expected}"
            )
        return spec, self.figure.objects[object_id]

    def bind_line(
        self,
        object_id: str,
        start: PointProvider,
        end: PointProvider,
    ) -> Mobject:
        spec, mobject = self._get(object_id, {"line", "arrow"})
        if spec.kind == "line" and spec.style.dash_pattern_pt:

            def update_dashes(item: Mobject) -> None:
                updated = self.renderer.native_line_from_points(
                    _point3(start()),
                    _point3(end()),
                    spec.style,
                )
                updated.set_z_index(spec.z_index)
                item.become(updated)

            mobject.add_updater(update_dashes)
        else:
            mobject.add_updater(
                lambda item: item.put_start_and_end_on(
                    _point3(start()),
                    _point3(end()),
                )
            )
        return mobject

    def bind_dot(self, object_id: str, center: PointProvider) -> Mobject:
        _, mobject = self._get(object_id, {"dot"})
        mobject.add_updater(lambda item: item.move_to(_point3(center())))
        return mobject

    def bind_polygon(
        self,
        object_id: str,
        points: Callable[[], Sequence[Sequence[float] | np.ndarray]],
    ) -> Mobject:
        spec, mobject = self._get(object_id, {"polygon"})

        def update_polygon(item: Mobject) -> None:
            updated = self.renderer.native_polygon_from_points(
                [_point3(point) for point in points()],
                spec.style,
            )
            updated.set_z_index(spec.z_index)
            item.become(updated)

        mobject.add_updater(update_polygon)
        return mobject

    def bind_label(
        self,
        object_id: str,
        anchor: PointProvider,
    ) -> Mobject:
        _, mobject = self._get(object_id, {"label", "path_label", "angle_label"})
        initial_anchor = _point3(anchor())
        offset = mobject.get_center() - initial_anchor
        mobject.add_updater(
            lambda item: item.move_to(_point3(anchor()) + offset)
        )
        return mobject

    @staticmethod
    def _display_angle(start: np.ndarray, end: np.ndarray) -> float:
        vector = end - start
        angle = atan2(vector[1], vector[0])
        if angle > pi / 2 or angle < -pi / 2:
            angle += pi
        return angle

    def bind_path_label(
        self,
        object_id: str,
        start: PointProvider,
        end: PointProvider,
    ) -> Mobject:
        spec, mobject = self._get(object_id, {"path_label"})
        placement = spec.placement
        if placement is None:
            raise ValueError(f"Path label {object_id!r} has no placement")

        initial_start = _point3(start())
        initial_end = _point3(end())
        initial_vector = initial_end - initial_start
        initial_length = float(np.linalg.norm(initial_vector))
        if initial_length <= 1e-12:
            raise ValueError(f"Path label {object_id!r} uses a zero-length path")
        initial_tangent = initial_vector / initial_length
        initial_normal = np.array(
            [-initial_tangent[1], initial_tangent[0], 0.0]
        )
        path_pos = float(spec.geometry["pos"])
        initial_base = initial_start + path_pos * initial_vector
        initial_offset = mobject.get_center() - initial_base
        local_tangent_offset = float(np.dot(initial_offset, initial_tangent))
        local_normal_offset = float(np.dot(initial_offset, initial_normal))
        initial_display_angle = self._display_angle(initial_start, initial_end)
        template = mobject.copy()

        def update_path_label(item: Mobject) -> None:
            current_start = _point3(start())
            current_end = _point3(end())
            vector = current_end - current_start
            length = float(np.linalg.norm(vector))
            if length <= 1e-12:
                return
            tangent = vector / length
            normal = np.array([-tangent[1], tangent[0], 0.0])
            base = current_start + path_pos * vector
            if placement.sloped:
                center = (
                    base
                    + local_tangent_offset * tangent
                    + local_normal_offset * normal
                )
            else:
                center = base + initial_offset

            updated = template.copy()
            if placement.sloped:
                updated.rotate(
                    self._display_angle(current_start, current_end)
                    - initial_display_angle,
                    about_point=updated.get_center(),
                )
            updated.move_to(center)
            updated.set_z_index(spec.z_index)
            item.become(updated)

        mobject.add_updater(update_path_label)
        return mobject

    def bind_right_angle(
        self,
        object_id: str,
        first: PointProvider,
        vertex: PointProvider,
        third: PointProvider,
    ) -> Mobject:
        spec, mobject = self._get(object_id, {"right_angle"})

        def update_right_angle(item: Mobject) -> None:
            updated = self.renderer.native_right_angle_from_points(
                _point3(first()),
                _point3(vertex()),
                _point3(third()),
                length=spec.geometry["radius_pt"] * self.renderer.pt,
                style=spec.style,
            )
            updated.set_z_index(spec.z_index)
            item.become(updated)

        mobject.add_updater(update_right_angle)
        return mobject

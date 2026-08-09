from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import Mobject, Scene, ValueTracker, smooth

from .compiler import PictureSpec
from .dynamic_geometry import EllipseChordDriver, NativeMotionBinder
from .manim_renderer import NativeFigure, NativeManimRenderer


MOTION_SCHEMA = "tikz-native-motion/v1"
Point2 = tuple[float, float]
ScenePointMapper = Callable[[Point2], Sequence[float] | np.ndarray]


class MotionConfigError(ValueError):
    """Raised before any updater is attached when a motion config is invalid."""


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionConfigError(f"{field} must be a number")
    result = float(value)
    if not isfinite(result):
        raise MotionConfigError(f"{field} must be finite")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MotionConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MotionConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise MotionConfigError(f"{field} must be an object")
    return value


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MotionConfigError(
            f"{field} contains unsupported fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True)
class MotionDriverSpec:
    id: str
    type: str
    active_path: str
    pivot: str
    intersection_index: int
    initial: float
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True)
class MotionBindingSpec:
    object_id: str
    type: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class MotionTimelineStep:
    to: float
    duration: float
    hold: float = 0.0
    cue: str | None = None


@dataclass(frozen=True)
class MotionSpec:
    schema: str
    picture_index: int
    driver: MotionDriverSpec
    bindings: tuple[MotionBindingSpec, ...]
    timeline: tuple[MotionTimelineStep, ...]

    @classmethod
    def load(cls, path: Path | str) -> "MotionSpec":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MotionConfigError(f"Could not read motion config {source}: {error}") from error
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: object) -> "MotionSpec":
        root = _mapping(payload, "motion config")
        _reject_unknown(
            root,
            {"schema", "picture_index", "driver", "bindings", "timeline"},
            "motion config",
        )
        schema = _string(root.get("schema"), "schema")
        if schema != MOTION_SCHEMA:
            raise MotionConfigError(
                f"schema must be {MOTION_SCHEMA!r}, got {schema!r}"
            )
        picture_index = _integer(root.get("picture_index"), "picture_index", minimum=1)

        raw_driver = _mapping(root.get("driver"), "driver")
        _reject_unknown(
            raw_driver,
            {
                "id",
                "type",
                "active_path",
                "pivot",
                "intersection_index",
                "initial",
                "range",
                "unit",
            },
            "driver",
        )
        driver_type = _string(raw_driver.get("type"), "driver.type")
        if driver_type != "rotate_named_line":
            raise MotionConfigError(
                "driver.type must be 'rotate_named_line' in motion/v1"
            )
        unit = _string(raw_driver.get("unit"), "driver.unit")
        if unit != "radians":
            raise MotionConfigError("driver.unit must be 'radians' in motion/v1")
        raw_range = raw_driver.get("range")
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise MotionConfigError("driver.range must contain [minimum, maximum]")
        minimum = _number(raw_range[0], "driver.range[0]")
        maximum = _number(raw_range[1], "driver.range[1]")
        initial = _number(raw_driver.get("initial"), "driver.initial")
        if not minimum < maximum:
            raise MotionConfigError("driver.range minimum must be less than maximum")
        if not minimum <= initial <= maximum:
            raise MotionConfigError("driver.initial must be inside driver.range")
        driver = MotionDriverSpec(
            id=_string(raw_driver.get("id"), "driver.id"),
            type=driver_type,
            active_path=_string(raw_driver.get("active_path"), "driver.active_path"),
            pivot=_string(raw_driver.get("pivot"), "driver.pivot"),
            intersection_index=_integer(
                raw_driver.get("intersection_index"),
                "driver.intersection_index",
            ),
            initial=initial,
            minimum=minimum,
            maximum=maximum,
            unit=unit,
        )

        raw_bindings = root.get("bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise MotionConfigError("bindings must be a non-empty array")
        expected_counts = {
            "line": (2, 2),
            "dot": (1, 1),
            "polygon": (3, None),
            "label": (1, 1),
            "path_label": (2, 2),
            "angle": (3, 3),
            "angle_label": (3, 3),
            "right_angle": (3, 3),
        }
        bindings: list[MotionBindingSpec] = []
        seen_objects: set[str] = set()
        for index, raw_binding_value in enumerate(raw_bindings):
            raw_binding = _mapping(raw_binding_value, f"bindings[{index}]")
            _reject_unknown(
                raw_binding,
                {"object_id", "type", "points"},
                f"bindings[{index}]",
            )
            binding_type = _string(
                raw_binding.get("type"), f"bindings[{index}].type"
            )
            if binding_type not in expected_counts:
                raise MotionConfigError(
                    f"bindings[{index}].type is unsupported: {binding_type!r}"
                )
            object_id = _string(
                raw_binding.get("object_id"), f"bindings[{index}].object_id"
            )
            if object_id in seen_objects:
                raise MotionConfigError(f"object {object_id!r} is bound more than once")
            seen_objects.add(object_id)
            raw_points = raw_binding.get("points")
            if not isinstance(raw_points, list):
                raise MotionConfigError(f"bindings[{index}].points must be an array")
            points = tuple(
                _string(value, f"bindings[{index}].points[{point_index}]")
                for point_index, value in enumerate(raw_points)
            )
            minimum_count, maximum_count = expected_counts[binding_type]
            if len(points) < minimum_count or (
                maximum_count is not None and len(points) > maximum_count
            ):
                expectation = (
                    f"exactly {minimum_count}"
                    if minimum_count == maximum_count
                    else f"at least {minimum_count}"
                )
                raise MotionConfigError(
                    f"bindings[{index}].points must contain {expectation} names"
                )
            bindings.append(MotionBindingSpec(object_id, binding_type, points))

        raw_timeline = root.get("timeline")
        if not isinstance(raw_timeline, list):
            raise MotionConfigError("timeline must be an array")
        timeline: list[MotionTimelineStep] = []
        for index, raw_step_value in enumerate(raw_timeline):
            raw_step = _mapping(raw_step_value, f"timeline[{index}]")
            _reject_unknown(
                raw_step,
                {"to", "duration", "hold", "cue"},
                f"timeline[{index}]",
            )
            target = _number(raw_step.get("to"), f"timeline[{index}].to")
            duration = _number(
                raw_step.get("duration"), f"timeline[{index}].duration"
            )
            hold = _number(raw_step.get("hold", 0.0), f"timeline[{index}].hold")
            if not minimum <= target <= maximum:
                raise MotionConfigError(
                    f"timeline[{index}].to must be inside driver.range"
                )
            if duration <= 0 or hold < 0:
                raise MotionConfigError(
                    f"timeline[{index}] duration must be positive and hold nonnegative"
                )
            cue_value = raw_step.get("cue")
            cue = None if cue_value is None else _string(cue_value, f"timeline[{index}].cue")
            timeline.append(MotionTimelineStep(target, duration, hold, cue))

        return cls(
            schema=schema,
            picture_index=picture_index,
            driver=driver,
            bindings=tuple(bindings),
            timeline=tuple(timeline),
        )

    def validate_cues(self, allowed: set[str] | frozenset[str]) -> None:
        unknown = sorted(
            {
                step.cue
                for step in self.timeline
                if step.cue is not None and step.cue not in allowed
            }
        )
        if unknown:
            raise MotionConfigError(
                "timeline contains unsupported cues: " + ", ".join(unknown)
            )

    def validate_picture(self, picture: PictureSpec, *, tolerance: float = 1e-9) -> None:
        if picture.index != self.picture_index:
            raise MotionConfigError(
                f"config selects picture {self.picture_index}, got picture {picture.index}"
            )
        if self.driver.active_path not in picture.named_paths:
            raise MotionConfigError(
                f"unknown active named path: {self.driver.active_path!r}"
            )
        active_path = picture.named_paths[self.driver.active_path]
        if active_path.kind != "line":
            raise MotionConfigError("active named path must be a line")
        if self.driver.pivot not in picture.coordinates:
            raise MotionConfigError(f"unknown pivot coordinate: {self.driver.pivot!r}")
        try:
            relation = picture.intersections[self.driver.intersection_index]
        except IndexError as error:
            raise MotionConfigError(
                f"picture has no intersection relation {self.driver.intersection_index}"
            ) from error
        if relation.sort_by != self.driver.active_path:
            raise MotionConfigError(
                "active_path must be the oriented sort path of the selected intersection"
            )
        if len(relation.coordinate_names) != 2:
            raise MotionConfigError(
                "rotate_named_line currently requires exactly two intersections"
            )
        start_name = active_path.geometry.get("start_name")
        end_name = active_path.geometry.get("end_name")
        if not start_name or not end_name:
            raise MotionConfigError(
                "active named line endpoints must be named TikZ coordinates"
            )

        objects = {item.id: item for item in picture.objects}
        accepted_kinds = {
            "line": {"line", "arrow"},
            "dot": {"dot"},
            "polygon": {"polygon"},
            "label": {"label"},
            "path_label": {"path_label"},
            "angle": {"angle"},
            "angle_label": {"angle_label"},
            "right_angle": {"right_angle"},
        }
        for binding in self.bindings:
            if binding.object_id not in objects:
                raise MotionConfigError(f"unknown object id: {binding.object_id!r}")
            actual_kind = objects[binding.object_id].kind
            if actual_kind not in accepted_kinds[binding.type]:
                raise MotionConfigError(
                    f"object {binding.object_id!r} is {actual_kind!r}, not {binding.type!r}"
                )
            for point_name in binding.points:
                if point_name not in picture.coordinates:
                    raise MotionConfigError(
                        f"binding {binding.object_id!r} uses unknown coordinate {point_name!r}"
                    )

        runtime = NativeMotionRuntime(
            self,
            picture,
            lambda: self.driver.initial,
        )
        current = runtime.coordinates()
        compared_names = {
            start_name,
            end_name,
            *relation.coordinate_names,
            *(point for binding in self.bindings for point in binding.points),
        }
        for name in compared_names:
            expected = picture.coordinates[name]
            actual = current[name]
            if len(expected) != 2 or len(actual) != 2 or any(
                not isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
                for left, right in zip(expected, actual)
            ):
                raise MotionConfigError(
                    f"driver.initial does not reproduce TikZ coordinate {name!r}: "
                    f"expected {expected}, got {actual}"
                )


@dataclass(frozen=True)
class EllipseChordMetrics:
    slope: float
    triangle_pqr_area: float
    triangle_pfo_area: float
    area_ratio: float
    angle_tangent: float


def ellipse_chord_metrics(
    coordinates: Mapping[str, Sequence[float]],
    *,
    p_name: str = "P",
    q_name: str = "Q",
    r_name: str = "R",
    focus_name: str = "F",
    center_name: str = "O",
) -> EllipseChordMetrics:
    p = coordinates[p_name]
    q = coordinates[q_name]
    r = coordinates[r_name]
    focus = coordinates[focus_name]
    center = coordinates[center_name]

    def cross(first: Sequence[float], second: Sequence[float]) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    line_dx = float(p[0] - focus[0])
    if abs(line_dx) <= 1e-12:
        raise ValueError("The driven line is vertical; slope is not finite")
    slope = float(p[1] - focus[1]) / line_dx
    qp = (float(p[0] - q[0]), float(p[1] - q[1]))
    qr = (float(r[0] - q[0]), float(r[1] - q[1]))
    pqr_area = abs(cross(qp, qr)) / 2
    pf = (float(focus[0] - p[0]), float(focus[1] - p[1]))
    po = (float(center[0] - p[0]), float(center[1] - p[1]))
    pfo_area = abs(cross(pf, po)) / 2
    if pfo_area <= 1e-12:
        raise ValueError("Triangle PFO is degenerate")
    dot = qp[0] * qr[0] + qp[1] * qr[1]
    if dot <= 1e-12:
        raise ValueError("Angle PQR is not acute in the configured interval")
    return EllipseChordMetrics(
        slope=slope,
        triangle_pqr_area=pqr_area,
        triangle_pfo_area=pfo_area,
        area_ratio=pqr_area / pfo_area,
        angle_tangent=abs(cross(qp, qr)) / dot,
    )


class NativeMotionRuntime:
    """Evaluate one declared geometry driver and bind it to native objects."""

    def __init__(
        self,
        spec: MotionSpec,
        picture: PictureSpec,
        parameter: Callable[[], float],
    ) -> None:
        self.spec = spec
        self.picture = picture
        self.parameter = parameter
        self.driver = EllipseChordDriver.from_named_intersection(
            parameter,
            picture,
            relation_index=spec.driver.intersection_index,
            pivot_name=spec.driver.pivot,
        )
        self._cached_parameter: float | None = None
        self._cached_coordinates: dict[str, Point2] | None = None

    def _resolve_coordinates(self) -> dict[str, Point2]:
        state = self.driver.state()
        relation = self.picture.intersections[self.spec.driver.intersection_index]
        active_path = self.picture.named_paths[self.spec.driver.active_path]
        start_name = str(active_path.geometry["start_name"])
        end_name = str(active_path.geometry["end_name"])
        overrides: dict[str, Point2] = {
            start_name: state.line_start,
            end_name: state.line_end,
            relation.coordinate_names[0]: state.q,
            relation.coordinate_names[1]: state.p,
        }
        cache: dict[str, Point2] = dict(overrides)
        resolving: list[str] = []

        def resolve(name: str) -> Point2:
            if name in cache:
                return cache[name]
            if name in resolving:
                cycle = " -> ".join([*resolving, name])
                raise MotionConfigError(f"coordinate dependency cycle: {cycle}")
            if name not in self.picture.coordinates:
                raise MotionConfigError(f"unknown coordinate dependency: {name!r}")
            resolving.append(name)
            dependency = self.picture.coordinate_dependencies.get(name)
            if dependency is None:
                value = tuple(float(item) for item in self.picture.coordinates[name])
            else:
                operation = dependency.get("operation")
                if operation == "intersection":
                    involved_paths = {
                        str(dependency.get("path_a", "")),
                        str(dependency.get("path_b", "")),
                        str(dependency.get("sort_by", "")),
                    }
                    if self.spec.driver.active_path in involved_paths:
                        raise MotionConfigError(
                            f"active path affects unselected intersection coordinate {name!r}"
                        )
                    value = tuple(
                        float(item) for item in self.picture.coordinates[name]
                    )
                elif operation == "reference":
                    value = resolve(str(dependency["coordinate"]))
                elif operation == "interpolation":
                    start = resolve(str(dependency["start"]))
                    end = resolve(str(dependency["end"]))
                    amount = float(dependency["parameter"])
                    value = (
                        start[0] + amount * (end[0] - start[0]),
                        start[1] + amount * (end[1] - start[1]),
                    )
                elif operation == "translation":
                    base = resolve(str(dependency["base"]))
                    offset = dependency["offset"]
                    value = (base[0] + float(offset[0]), base[1] + float(offset[1]))
                elif operation == "projection":
                    start = resolve(str(dependency["line_start"]))
                    point = resolve(str(dependency["point"]))
                    end = resolve(str(dependency["line_end"]))
                    direction = (end[0] - start[0], end[1] - start[1])
                    denominator = direction[0] ** 2 + direction[1] ** 2
                    if denominator <= 1e-18:
                        raise MotionConfigError("cannot project onto a zero-length line")
                    amount = (
                        (point[0] - start[0]) * direction[0]
                        + (point[1] - start[1]) * direction[1]
                    ) / denominator
                    value = (
                        start[0] + amount * direction[0],
                        start[1] + amount * direction[1],
                    )
                else:
                    raise MotionConfigError(
                        f"unsupported dynamic coordinate operation {operation!r} for {name!r}"
                    )
            resolving.pop()
            if len(value) != 2:
                raise MotionConfigError("motion/v1 only supports 2D coordinates")
            cache[name] = (float(value[0]), float(value[1]))
            return cache[name]

        for coordinate_name in self.picture.coordinates:
            resolve(coordinate_name)
        return cache

    def coordinates(self) -> Mapping[str, Point2]:
        current_parameter = float(self.parameter())
        if not isfinite(current_parameter):
            raise MotionConfigError("motion parameter must be finite")
        if self._cached_coordinates is None or current_parameter != self._cached_parameter:
            if not self.spec.driver.minimum <= current_parameter <= self.spec.driver.maximum:
                raise MotionConfigError(
                    f"parameter {current_parameter} is outside driver.range"
                )
            resolved = self._resolve_coordinates()
            self._cached_coordinates = resolved
            self._cached_parameter = current_parameter
        return self._cached_coordinates

    def bind(
        self,
        figure: NativeFigure,
        renderer: NativeManimRenderer,
        to_scene_point: ScenePointMapper,
    ) -> tuple[Mobject, ...]:
        self.spec.validate_picture(self.picture)
        binder = NativeMotionBinder(figure, renderer)

        def point_provider(name: str):
            return lambda: to_scene_point(self.coordinates()[name])

        bound: list[Mobject] = []
        for binding in self.spec.bindings:
            providers = [point_provider(name) for name in binding.points]
            if binding.type == "line":
                item = binder.bind_line(binding.object_id, providers[0], providers[1])
            elif binding.type == "dot":
                item = binder.bind_dot(binding.object_id, providers[0])
            elif binding.type == "polygon":
                item = binder.bind_polygon(
                    binding.object_id,
                    lambda names=binding.points: [
                        to_scene_point(self.coordinates()[name]) for name in names
                    ],
                )
            elif binding.type == "label":
                item = binder.bind_label(binding.object_id, providers[0])
            elif binding.type == "path_label":
                item = binder.bind_path_label(
                    binding.object_id, providers[0], providers[1]
                )
            elif binding.type == "angle":
                item = binder.bind_angle(
                    binding.object_id, providers[0], providers[1], providers[2]
                )
            elif binding.type == "angle_label":
                item = binder.bind_angle_label(
                    binding.object_id, providers[0], providers[1], providers[2]
                )
            elif binding.type == "right_angle":
                item = binder.bind_right_angle(
                    binding.object_id, providers[0], providers[1], providers[2]
                )
            else:  # pragma: no cover - MotionSpec validation owns this branch.
                raise MotionConfigError(f"unsupported binding type: {binding.type}")
            bound.append(item)
        return tuple(bound)

    def play_timeline(
        self,
        scene: Scene,
        tracker: ValueTracker,
        *,
        on_cue: Callable[[MotionTimelineStep], None] | None = None,
    ) -> None:
        for step in self.spec.timeline:
            scene.play(
                tracker.animate.set_value(step.to),
                run_time=step.duration,
                rate_func=smooth,
            )
            if on_cue is not None and step.cue is not None:
                on_cue(step)
            if step.hold:
                scene.wait(step.hold)


def load_motion_spec(path: Path | str) -> MotionSpec:
    return MotionSpec.load(path)

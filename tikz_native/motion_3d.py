"""Declarative, provider-owned motion runtime for native TikZ 3D figures.

The runtime keeps authored TikZ coordinates in logical space.  Conversion to
Manim scene coordinates happens only at object-binding and occlusion borders.
No ``PictureSpec`` coordinate is mutated while an animation is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import Mobject, Scene, ValueTracker, smooth

from .camera_3d import MultiProjectionCamera, ProjectionPreset
from .compiler import ObjectSpec, PictureSpec
from .manim_renderer_3d import Native3DFigure, NativeManim3DRenderer


MOTION_3D_SCHEMA = "tikz-native-motion-3d/v1"
Point3 = tuple[float, float, float]


class Motion3DConfigError(ValueError):
    """Raised before a malformed 3D motion can attach scene updaters."""


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise Motion3DConfigError(f"{field} must be an object")
    return value


def _reject_unknown(value: Mapping[str, object], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise Motion3DConfigError(
            f"{field} contains unsupported fields: {', '.join(unknown)}"
        )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Motion3DConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Motion3DConfigError(f"{field} must be a number")
    result = float(value)
    if not isfinite(result):
        raise Motion3DConfigError(f"{field} must be finite")
    return result


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Motion3DConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _string_tuple(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Motion3DConfigError(f"{field} must be an array")
    result = tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))
    if len(result) < minimum or (maximum is not None and len(result) > maximum):
        expected = f"exactly {minimum}" if minimum == maximum else f"at least {minimum}"
        raise Motion3DConfigError(f"{field} must contain {expected} names")
    return result


def _point3(value: Sequence[float] | np.ndarray, field: str = "point") -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise Motion3DConfigError(f"{field} must be a finite 3D point")
    return point


def rotate_point_about_axis(
    point: Sequence[float] | np.ndarray,
    axis_start: Sequence[float] | np.ndarray,
    axis_end: Sequence[float] | np.ndarray,
    angle: float,
) -> Point3:
    """Rotate one point around an arbitrary directed 3D axis using Rodrigues."""

    value = _point3(point)
    start = _point3(axis_start, "axis_start")
    end = _point3(axis_end, "axis_end")
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise Motion3DConfigError("hinge axis must not have zero length")
    direction /= length
    relative = value - start
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated = (
        relative * cosine
        + np.cross(direction, relative) * sine
        + direction * float(np.dot(direction, relative)) * (1.0 - cosine)
    )
    result = start + rotated
    return (float(result[0]), float(result[1]), float(result[2]))


def point_on_segment_3d(
    start: Sequence[float] | np.ndarray,
    end: Sequence[float] | np.ndarray,
    parameter: float,
) -> Point3:
    first = _point3(start, "segment start")
    second = _point3(end, "segment end")
    point = first + float(parameter) * (second - first)
    return (float(point[0]), float(point[1]), float(point[2]))


def project_point_to_line_3d(
    point: Sequence[float] | np.ndarray,
    line_start: Sequence[float] | np.ndarray,
    line_end: Sequence[float] | np.ndarray,
) -> Point3:
    value = _point3(point)
    start = _point3(line_start, "line_start")
    end = _point3(line_end, "line_end")
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-18:
        raise Motion3DConfigError("cannot project onto a zero-length line")
    amount = float(np.dot(value - start, direction)) / denominator
    result = start + amount * direction
    return (float(result[0]), float(result[1]), float(result[2]))


@dataclass(frozen=True)
class HingeFoldDriverSpec:
    id: str
    type: str
    axis: tuple[str, str]
    moving_points: tuple[str, ...]
    initial: float
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True)
class DerivedCoordinateSpec:
    name: str
    type: str
    start: str | None = None
    end: str | None = None
    parameter: float | None = None
    point: str | None = None
    line_start: str | None = None
    line_end: str | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        if self.type == "point_on_segment":
            assert self.start is not None and self.end is not None
            return (self.start, self.end)
        assert self.point is not None
        assert self.line_start is not None and self.line_end is not None
        return (self.point, self.line_start, self.line_end)


@dataclass(frozen=True)
class Motion3DBindingSpec:
    object_id: str
    type: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class Motion3DCameraSpec:
    entry_mode: str
    restore_transition: str
    restore_duration: float


@dataclass(frozen=True)
class Motion3DTimelineStep:
    type: str
    duration: float
    to: float | None = None
    mode: str | None = None
    transition: str = "linear"
    arc_height: float = 0.85
    hold: float = 0.0
    cue: str | None = None


@dataclass(frozen=True)
class Motion3DSpec:
    schema: str
    picture_index: int
    end_policy: str
    driver: HingeFoldDriverSpec
    derived_coordinates: tuple[DerivedCoordinateSpec, ...]
    bindings: tuple[Motion3DBindingSpec, ...]
    camera: Motion3DCameraSpec
    timeline: tuple[Motion3DTimelineStep, ...]

    @classmethod
    def load(cls, path: Path | str) -> "Motion3DSpec":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Motion3DConfigError(
                f"Could not read 3D motion config {source}: {error}"
            ) from error
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: object) -> "Motion3DSpec":
        root = _mapping(payload, "3D motion config")
        _reject_unknown(
            root,
            {
                "schema",
                "picture_index",
                "end_policy",
                "driver",
                "derived_coordinates",
                "bindings",
                "camera",
                "timeline",
            },
            "3D motion config",
        )
        schema = _string(root.get("schema"), "schema")
        if schema != MOTION_3D_SCHEMA:
            raise Motion3DConfigError(
                f"schema must be {MOTION_3D_SCHEMA!r}, got {schema!r}"
            )
        end_policy = _string(root.get("end_policy"), "end_policy")
        if end_policy != "restore_entry":
            raise Motion3DConfigError("motion-3d/v1 end_policy must be 'restore_entry'")

        raw_driver = _mapping(root.get("driver"), "driver")
        _reject_unknown(
            raw_driver,
            {"id", "type", "axis", "moving_points", "initial", "range", "unit"},
            "driver",
        )
        driver_type = _string(raw_driver.get("type"), "driver.type")
        if driver_type != "hinge_fold":
            raise Motion3DConfigError("driver.type must be 'hinge_fold'")
        unit = _string(raw_driver.get("unit"), "driver.unit")
        if unit != "radians":
            raise Motion3DConfigError("driver.unit must be 'radians'")
        axis_values = _string_tuple(raw_driver.get("axis"), "driver.axis", minimum=2, maximum=2)
        if axis_values[0] == axis_values[1]:
            raise Motion3DConfigError("driver.axis endpoints must be different names")
        moving_points = _string_tuple(
            raw_driver.get("moving_points"), "driver.moving_points", minimum=1
        )
        if len(set(moving_points)) != len(moving_points):
            raise Motion3DConfigError("driver.moving_points contains duplicate names")
        if any(name in axis_values for name in moving_points):
            raise Motion3DConfigError("hinge axis endpoints must remain fixed")
        raw_range = raw_driver.get("range")
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise Motion3DConfigError("driver.range must contain [minimum, maximum]")
        minimum = _number(raw_range[0], "driver.range[0]")
        maximum = _number(raw_range[1], "driver.range[1]")
        initial = _number(raw_driver.get("initial"), "driver.initial")
        if not minimum < maximum or not minimum <= initial <= maximum:
            raise Motion3DConfigError("driver.initial must be inside an increasing range")
        driver = HingeFoldDriverSpec(
            id=_string(raw_driver.get("id"), "driver.id"),
            type=driver_type,
            axis=(axis_values[0], axis_values[1]),
            moving_points=moving_points,
            initial=initial,
            minimum=minimum,
            maximum=maximum,
            unit=unit,
        )

        raw_derived = root.get("derived_coordinates", [])
        if not isinstance(raw_derived, list):
            raise Motion3DConfigError("derived_coordinates must be an array")
        derived: list[DerivedCoordinateSpec] = []
        seen_names: set[str] = set()
        for index, raw_value in enumerate(raw_derived):
            field = f"derived_coordinates[{index}]"
            item = _mapping(raw_value, field)
            kind = _string(item.get("type"), f"{field}.type")
            name = _string(item.get("name"), f"{field}.name")
            if name in seen_names:
                raise Motion3DConfigError(f"derived coordinate {name!r} is declared twice")
            seen_names.add(name)
            if kind == "point_on_segment":
                _reject_unknown(item, {"name", "type", "start", "end", "parameter"}, field)
                parameter = _number(item.get("parameter"), f"{field}.parameter")
                if not 0.0 <= parameter <= 1.0:
                    raise Motion3DConfigError(
                        f"{field}.parameter must be inside [0, 1]"
                    )
                derived.append(
                    DerivedCoordinateSpec(
                        name=name,
                        type=kind,
                        start=_string(item.get("start"), f"{field}.start"),
                        end=_string(item.get("end"), f"{field}.end"),
                        parameter=parameter,
                    )
                )
            elif kind == "project_point_to_line":
                _reject_unknown(
                    item,
                    {"name", "type", "point", "line_start", "line_end"},
                    field,
                )
                derived.append(
                    DerivedCoordinateSpec(
                        name=name,
                        type=kind,
                        point=_string(item.get("point"), f"{field}.point"),
                        line_start=_string(item.get("line_start"), f"{field}.line_start"),
                        line_end=_string(item.get("line_end"), f"{field}.line_end"),
                    )
                )
            else:
                raise Motion3DConfigError(
                    f"{field}.type must be 'point_on_segment' or 'project_point_to_line'"
                )

        raw_bindings = root.get("bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise Motion3DConfigError("bindings must be a non-empty array")
        point_counts = {
            "line": (2, 2),
            "dot": (1, 1),
            "polygon": (3, None),
            "label": (1, 1),
            "path_label": (2, 2),
        }
        bindings: list[Motion3DBindingSpec] = []
        seen_objects: set[str] = set()
        for index, raw_value in enumerate(raw_bindings):
            field = f"bindings[{index}]"
            item = _mapping(raw_value, field)
            _reject_unknown(item, {"object_id", "type", "points"}, field)
            kind = _string(item.get("type"), f"{field}.type")
            if kind not in point_counts:
                raise Motion3DConfigError(f"{field}.type is unsupported: {kind!r}")
            object_id = _string(item.get("object_id"), f"{field}.object_id")
            if object_id in seen_objects:
                raise Motion3DConfigError(f"object {object_id!r} is bound more than once")
            seen_objects.add(object_id)
            minimum_count, maximum_count = point_counts[kind]
            points = _string_tuple(
                item.get("points"),
                f"{field}.points",
                minimum=minimum_count,
                maximum=maximum_count,
            )
            bindings.append(Motion3DBindingSpec(object_id, kind, points))

        raw_camera = _mapping(root.get("camera"), "camera")
        _reject_unknown(
            raw_camera,
            {"entry_mode", "restore_transition", "restore_duration"},
            "camera",
        )
        restore_transition = _string(
            raw_camera.get("restore_transition"), "camera.restore_transition"
        )
        if restore_transition not in {"linear", "orbit"}:
            raise Motion3DConfigError(
                "camera.restore_transition must be 'linear' or 'orbit'"
            )
        restore_duration = _number(
            raw_camera.get("restore_duration"), "camera.restore_duration"
        )
        if restore_duration <= 0:
            raise Motion3DConfigError("camera.restore_duration must be positive")
        camera = Motion3DCameraSpec(
            entry_mode=_string(raw_camera.get("entry_mode"), "camera.entry_mode"),
            restore_transition=restore_transition,
            restore_duration=restore_duration,
        )

        raw_timeline = root.get("timeline")
        if not isinstance(raw_timeline, list) or not raw_timeline:
            raise Motion3DConfigError("timeline must be a non-empty array")
        timeline: list[Motion3DTimelineStep] = []
        for index, raw_value in enumerate(raw_timeline):
            field = f"timeline[{index}]"
            item = _mapping(raw_value, field)
            kind = _string(item.get("type"), f"{field}.type")
            cue_value = item.get("cue")
            cue = None if cue_value is None else _string(cue_value, f"{field}.cue")
            if kind == "driver":
                _reject_unknown(item, {"type", "to", "duration", "hold", "cue"}, field)
                target = _number(item.get("to"), f"{field}.to")
                if not minimum <= target <= maximum:
                    raise Motion3DConfigError(f"{field}.to must be inside driver.range")
                duration = _number(item.get("duration"), f"{field}.duration")
                hold = _number(item.get("hold", 0.0), f"{field}.hold")
                if duration <= 0 or hold < 0:
                    raise Motion3DConfigError(f"{field} duration must be positive and hold nonnegative")
                timeline.append(
                    Motion3DTimelineStep(kind, duration, to=target, hold=hold, cue=cue)
                )
            elif kind == "camera":
                _reject_unknown(
                    item,
                    {"type", "mode", "transition", "duration", "arc_height", "hold", "cue"},
                    field,
                )
                transition = _string(item.get("transition", "linear"), f"{field}.transition")
                if transition not in {"linear", "orbit"}:
                    raise Motion3DConfigError(f"{field}.transition must be 'linear' or 'orbit'")
                duration = _number(item.get("duration"), f"{field}.duration")
                hold = _number(item.get("hold", 0.0), f"{field}.hold")
                arc_height = _number(item.get("arc_height", 0.85), f"{field}.arc_height")
                if duration <= 0 or hold < 0:
                    raise Motion3DConfigError(f"{field} duration must be positive and hold nonnegative")
                timeline.append(
                    Motion3DTimelineStep(
                        kind,
                        duration,
                        mode=_string(item.get("mode"), f"{field}.mode"),
                        transition=transition,
                        arc_height=arc_height,
                        hold=hold,
                        cue=cue,
                    )
                )
            elif kind == "wait":
                _reject_unknown(item, {"type", "duration", "cue"}, field)
                duration = _number(item.get("duration"), f"{field}.duration")
                if duration <= 0:
                    raise Motion3DConfigError(f"{field}.duration must be positive")
                timeline.append(Motion3DTimelineStep(kind, duration, cue=cue))
            else:
                raise Motion3DConfigError(
                    f"{field}.type must be 'driver', 'camera', or 'wait'"
                )

        return cls(
            schema=schema,
            picture_index=_integer(root.get("picture_index"), "picture_index", 1),
            end_policy=end_policy,
            driver=driver,
            derived_coordinates=tuple(derived),
            bindings=tuple(bindings),
            camera=camera,
            timeline=tuple(timeline),
        )

    def validate_picture(self, picture: PictureSpec, tolerance: float = 1e-9) -> None:
        if picture.index != self.picture_index:
            raise Motion3DConfigError(
                f"config selects picture {self.picture_index}, got picture {picture.index}"
            )
        if picture.dimension != 3 or picture.projection_3d is None:
            raise Motion3DConfigError("motion-3d/v1 requires a compiled 3D TikZ picture")
        for name in (*self.driver.axis, *self.driver.moving_points):
            if name not in picture.coordinates:
                raise Motion3DConfigError(f"unknown driver coordinate: {name!r}")
            _point3(picture.coordinates[name], f"picture coordinate {name!r}")
        axis_start = _point3(picture.coordinates[self.driver.axis[0]])
        axis_end = _point3(picture.coordinates[self.driver.axis[1]])
        if np.linalg.norm(axis_end - axis_start) <= 1e-12:
            raise Motion3DConfigError("hinge axis must not have zero length")

        available = set(picture.coordinates)
        for derived in self.derived_coordinates:
            available.add(derived.name)
        for derived in self.derived_coordinates:
            for dependency in derived.dependencies:
                if dependency not in available:
                    raise Motion3DConfigError(
                        f"derived coordinate {derived.name!r} uses unknown {dependency!r}"
                    )

        objects = {item.id: item for item in picture.objects}
        accepted = {
            "line": {"line", "arrow"},
            "dot": {"dot"},
            "polygon": {"polygon"},
            "label": {"label", "angle_label"},
            "path_label": {"path_label"},
        }
        for binding in self.bindings:
            if binding.object_id not in objects:
                raise Motion3DConfigError(f"unknown object id: {binding.object_id!r}")
            actual = objects[binding.object_id].kind
            if actual not in accepted[binding.type]:
                raise Motion3DConfigError(
                    f"object {binding.object_id!r} is {actual!r}, not {binding.type!r}"
                )
            for name in binding.points:
                if name not in available:
                    raise Motion3DConfigError(
                        f"binding {binding.object_id!r} uses unknown coordinate {name!r}"
                    )

        runtime = NativeMotion3DRuntime(self, picture, lambda: self.driver.initial)
        initial = runtime.coordinates()
        for name in set(self.driver.moving_points) | {
            item.name for item in self.derived_coordinates
        }:
            if name not in picture.coordinates:
                continue
            expected = _point3(picture.coordinates[name])
            actual = _point3(initial[name])
            if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
                raise Motion3DConfigError(
                    f"driver.initial does not reproduce TikZ coordinate {name!r}: "
                    f"expected {tuple(expected)}, got {tuple(actual)}"
                )


class NativeMotion3DRuntime:
    """Evaluate one hinge driver and mutate an existing native figure in place."""

    _ENTRY_CAMERA_MODE = "__motion_3d_entry__"

    def __init__(
        self,
        spec: Motion3DSpec,
        picture: PictureSpec,
        parameter: Callable[[], float],
    ) -> None:
        self.spec = spec
        self.picture = picture
        self.parameter = parameter
        self._cached_parameter: float | None = None
        self._cached_coordinates: dict[str, Point3] | None = None
        self._bindings_attached = False
        self._labels_attached = False

    def _authored_coordinates(self) -> dict[str, Point3]:
        """Return logical coordinates without mutating the compiled picture.

        Semantic occlusion macros may name a displayed probe endpoint directly
        in an emitted line even when it was not declared with ``\\defPoint``.
        The compiler preserves that name on the object geometry.  Collecting it
        here keeps the motion/occlusion provider total while leaving the source
        ``PictureSpec.coordinates`` mapping untouched.
        """

        authored: dict[str, Point3] = {
            name: tuple(float(component) for component in value)  # type: ignore[misc]
            for name, value in self.picture.coordinates.items()
        }
        for spec in self.picture.objects:
            candidates = (
                ("start_name", "start"),
                ("end_name", "end"),
                ("at_name", "at"),
                ("first_name", "first"),
                ("vertex_name", "vertex"),
                ("third_name", "third"),
            )
            for name_field, value_field in candidates:
                name = spec.geometry.get(name_field)
                value = spec.geometry.get(value_field)
                if name and value is not None and str(name) not in authored:
                    point = _point3(value, f"object coordinate {name!r}")
                    authored[str(name)] = (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
        return authored

    def _resolve_coordinates(self, parameter: float) -> dict[str, Point3]:
        authored = self._authored_coordinates()
        for name, value in authored.items():
            _point3(value, f"picture coordinate {name!r}")
        axis_start = authored[self.spec.driver.axis[0]]
        axis_end = authored[self.spec.driver.axis[1]]
        delta = parameter - self.spec.driver.initial
        cache: dict[str, Point3] = dict(authored)
        for name in self.spec.driver.moving_points:
            cache[name] = rotate_point_about_axis(
                authored[name], axis_start, axis_end, delta
            )

        derived_by_name = {item.name: item for item in self.spec.derived_coordinates}
        resolving: list[str] = []

        def resolve(name: str) -> Point3:
            if name not in derived_by_name:
                if name not in cache:
                    raise Motion3DConfigError(f"unknown coordinate dependency: {name!r}")
                return cache[name]
            if name in resolving:
                cycle = " -> ".join([*resolving, name])
                raise Motion3DConfigError(f"coordinate dependency cycle: {cycle}")
            resolving.append(name)
            relation = derived_by_name[name]
            if relation.type == "point_on_segment":
                assert relation.start and relation.end and relation.parameter is not None
                value = point_on_segment_3d(
                    resolve(relation.start), resolve(relation.end), relation.parameter
                )
            else:
                assert relation.point and relation.line_start and relation.line_end
                value = project_point_to_line_3d(
                    resolve(relation.point),
                    resolve(relation.line_start),
                    resolve(relation.line_end),
                )
            resolving.pop()
            cache[name] = value
            return value

        for name in derived_by_name:
            resolve(name)
        return cache

    def coordinates(self) -> Mapping[str, Point3]:
        parameter = float(self.parameter())
        if not isfinite(parameter):
            raise Motion3DConfigError("motion parameter must be finite")
        if not self.spec.driver.minimum <= parameter <= self.spec.driver.maximum:
            raise Motion3DConfigError(
                f"parameter {parameter} is outside driver.range"
            )
        if self._cached_coordinates is None or parameter != self._cached_parameter:
            self._cached_coordinates = self._resolve_coordinates(parameter)
            self._cached_parameter = parameter
        return self._cached_coordinates

    def coordinate(self, name: str) -> Point3:
        return self.coordinates()[name]

    def prepare_camera(
        self,
        camera: MultiProjectionCamera,
        *,
        view_center: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(camera, MultiProjectionCamera):
            raise Motion3DConfigError("motion-3d/v1 requires MultiProjectionCamera")
        mode = self.spec.camera.entry_mode
        if mode == "tikz":
            assert self.picture.projection_3d is not None
            center = (
                np.zeros(3, dtype=float)
                if view_center is None
                else _point3(view_center, "camera view_center")
            )
            camera.register_mode(
                "tikz",
                np.asarray(self.picture.projection_3d.matrix, dtype=float),
                view_center=center,
                overwrite="tikz" in camera.presets,
            )
        if mode not in camera.presets:
            raise Motion3DConfigError(f"unknown entry camera mode: {mode!r}")
        camera.set_mode(mode)

    def bind(
        self,
        figure: Native3DFigure,
        renderer: NativeManim3DRenderer,
        *,
        camera: MultiProjectionCamera | None = None,
    ) -> tuple[Mobject, ...]:
        self.spec.validate_picture(self.picture)
        if figure.picture is not self.picture:
            raise Motion3DConfigError("figure and runtime must share the same PictureSpec")
        if self._bindings_attached:
            return tuple(figure.objects[item.object_id] for item in self.spec.bindings)
        object_specs: dict[str, ObjectSpec] = {
            item.id: item for item in self.picture.objects
        }

        def world(name: str) -> np.ndarray:
            return renderer.point(self.coordinate(name), self.picture)

        label_binding = False
        for binding in self.spec.bindings:
            spec = object_specs[binding.object_id]
            mobject = figure.objects[binding.object_id]
            if binding.type == "line":

                def update_line(
                    item: Mobject,
                    _dt: float = 0.0,
                    *,
                    names: tuple[str, ...] = binding.points,
                    object_spec: ObjectSpec = spec,
                ) -> None:
                    updated = renderer.native_line_from_points(
                        world(names[0]), world(names[1]), object_spec.style
                    )
                    updated.set_z_index(object_spec.z_index)
                    item.become(updated)

                mobject.add_updater(update_line)
            elif binding.type == "dot":
                mobject.add_updater(
                    lambda item, _dt=0.0, name=binding.points[0]: item.move_to(world(name))
                )
            elif binding.type == "polygon":

                def update_polygon(
                    item: Mobject,
                    _dt: float = 0.0,
                    *,
                    names: tuple[str, ...] = binding.points,
                    object_spec: ObjectSpec = spec,
                ) -> None:
                    updated = renderer.native_polygon_from_points(
                        [world(name) for name in names], object_spec.style
                    )
                    updated.set_z_index(object_spec.z_index)
                    item.become(updated)

                mobject.add_updater(update_polygon)
            else:
                label_binding = True

        if label_binding:
            if camera is None:
                raise Motion3DConfigError(
                    "dynamic 3D labels require a MultiProjectionCamera"
                )
            renderer.bind_labels_to_camera(
                figure,
                camera,
                coordinate_provider=self.coordinate,
            )
            self._labels_attached = True
        self._bindings_attached = True
        return tuple(figure.objects[item.object_id] for item in self.spec.bindings)

    def bind_occlusions(
        self,
        figure: Native3DFigure,
        renderer: NativeManim3DRenderer,
        camera: MultiProjectionCamera,
    ) -> dict[str, Mobject]:
        return renderer.bind_occlusions_to_camera(
            figure,
            camera,
            coordinate_provider=self.coordinate,
        )

    def play_timeline(
        self,
        scene: Scene,
        tracker: ValueTracker,
        camera: MultiProjectionCamera,
        *,
        on_cue: Callable[[Motion3DTimelineStep], None] | None = None,
    ) -> None:
        if not isinstance(scene, Scene):
            raise Motion3DConfigError("scene must be a Manim Scene")
        if not isinstance(tracker, ValueTracker):
            raise Motion3DConfigError("tracker must be a Manim ValueTracker")
        if not isinstance(camera, MultiProjectionCamera):
            raise Motion3DConfigError("camera must be a MultiProjectionCamera")

        entry_parameter = float(tracker.get_value())
        entry_camera = camera.snapshot()
        camera.presets[self._ENTRY_CAMERA_MODE] = ProjectionPreset(
            self._ENTRY_CAMERA_MODE,
            entry_camera.matrix,
            entry_camera.perspective_strength,
            entry_camera.focal_distance,
            entry_camera.view_center,
            entry_camera.principal_point,
        )
        for step in self.spec.timeline:
            if step.type == "driver":
                assert step.to is not None
                scene.play(
                    tracker.animate.set_value(step.to),
                    run_time=step.duration,
                    rate_func=smooth,
                )
            elif step.type == "camera":
                assert step.mode is not None
                if step.mode not in camera.presets:
                    raise Motion3DConfigError(
                        f"timeline uses unknown camera mode: {step.mode!r}"
                    )
                animation = (
                    camera.animate_orbit_to(step.mode, arc_height=step.arc_height)
                    if step.transition == "orbit"
                    else camera.animate_to(step.mode)
                )
                scene.play(animation, run_time=step.duration, rate_func=smooth)
            else:
                scene.wait(step.duration)
            if on_cue is not None and step.cue is not None:
                on_cue(step)
            if step.hold:
                scene.wait(step.hold)

        restore_animations = []
        if not isclose(
            float(tracker.get_value()), entry_parameter, abs_tol=1e-12, rel_tol=0.0
        ):
            restore_animations.append(tracker.animate.set_value(entry_parameter))
        camera_animation = (
            camera.animate_orbit_to(
                self._ENTRY_CAMERA_MODE,
                arc_height=0.85,
            )
            if self.spec.camera.restore_transition == "orbit"
            else camera.animate_to(self._ENTRY_CAMERA_MODE)
        )
        restore_animations.append(camera_animation)
        scene.play(
            *restore_animations,
            run_time=self.spec.camera.restore_duration,
            rate_func=smooth,
        )
        tracker.set_value(entry_parameter)
        camera.restore(entry_camera)


def load_motion_3d_spec(path: Path | str) -> Motion3DSpec:
    return Motion3DSpec.load(path)


__all__ = [
    "MOTION_3D_SCHEMA",
    "DerivedCoordinateSpec",
    "HingeFoldDriverSpec",
    "Motion3DBindingSpec",
    "Motion3DCameraSpec",
    "Motion3DConfigError",
    "Motion3DSpec",
    "Motion3DTimelineStep",
    "NativeMotion3DRuntime",
    "load_motion_3d_spec",
    "point_on_segment_3d",
    "project_point_to_line_3d",
    "rotate_point_about_axis",
]

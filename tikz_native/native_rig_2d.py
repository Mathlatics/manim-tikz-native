from __future__ import annotations

"""Author-facing Manim API for one semantic TikZ Native 2D rig.

``NativeGeometryRig2D`` deliberately exposes the Manim ``ValueTracker`` while
keeping the geometry solver, dependent-object updaters, ShapeState similarity
transform and exact entry restoration inside the Provider.  Generated Native
Clip source can therefore use ordinary ``scene.play`` calls without copying
the Provider's analytic-geometry implementation into author code.
"""

from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import Callable

import numpy as np
from manim import Mobject, Scene, ValueTracker

from .compiler import PictureSpec
from .manim_renderer import NativeFigure, NativeManimRenderer
from .motion_runtime import (
    MotionConfigError,
    MotionSpec,
    MotionTimelineStep,
    NativeMotionRuntime,
    _mobject_stroke_width,
    _similarity_mapper_from_line,
)


NATIVE_RIG_2D_API_SCHEMA = "tikz-native-rig-2d/v1"
Point2 = tuple[float, float]


def _semantic_payload(shape: Mobject) -> tuple[dict[str, Mobject], PictureSpec]:
    """Read either the editor runtime or direct Provider semantic metadata."""

    objects_value = getattr(shape, "_codex_tikz_native_objects", None)
    if objects_value is None:
        objects_value = getattr(shape, "_tikz_native_object_map", None)
    picture = getattr(shape, "_codex_tikz_native_picture", None)
    if picture is None:
        picture = getattr(shape, "_tikz_native_picture", None)
    if not isinstance(objects_value, Mapping) or not isinstance(picture, PictureSpec):
        raise MotionConfigError(
            "input shape does not expose TikZ Native semantic objects"
        )
    objects = dict(objects_value)
    invalid_ids = sorted(
        str(object_id)
        for object_id, mobject in objects.items()
        if not isinstance(object_id, str) or not isinstance(mobject, Mobject)
    )
    if invalid_ids:
        raise MotionConfigError(
            "input shape contains invalid TikZ Native semantic objects: "
            + ", ".join(invalid_ids)
        )
    return objects, picture


class NativeGeometryRig2D:
    """Bind one versioned 2D geometry motion to an existing ShapeState.

    The class is intentionally a context manager.  Entering installs only the
    Provider-declared dependent-object updaters and checks that their initial
    frame is pixel-aligned with the incoming ShapeState.  Exiting always
    removes all temporary updaters and restores the exact entry geometry and
    styles while retaining every semantic object's Python identity.

    Author code owns animation timing::

        with NativeGeometryRig2D(
            shape,
            motion_payload,
            active_object_id="line.Lstart.Lend",
        ) as rig:
            scene.play(
                rig.tracker.animate.set_value(rig.maximum),
                run_time=1.8,
            )
    """

    api_schema = NATIVE_RIG_2D_API_SCHEMA

    def __init__(
        self,
        shape: Mobject,
        motion_spec_payload: object,
        *,
        active_object_id: str,
    ) -> None:
        if not isinstance(shape, Mobject):
            raise MotionConfigError("shape must be a Manim Mobject")
        if not isinstance(active_object_id, str) or not active_object_id.strip():
            raise MotionConfigError("active_object_id must be a non-empty string")

        self.shape = shape
        self.active_object_id = active_object_id.strip()
        self.spec = (
            motion_spec_payload
            if isinstance(motion_spec_payload, MotionSpec)
            else MotionSpec.from_dict(motion_spec_payload)
        )
        self._objects, self.picture = _semantic_payload(shape)
        self.spec.validate_picture(self.picture)
        if self.active_object_id not in self._objects:
            raise MotionConfigError(
                f"unknown active object id: {self.active_object_id!r}"
            )

        active_binding = next(
            (
                item
                for item in self.spec.bindings
                if item.object_id == self.active_object_id
            ),
            None,
        )
        if active_binding is None or active_binding.type != "line":
            raise MotionConfigError(
                "active object must be a bound line in motion/v1"
            )
        self._active_mobject = self._objects[self.active_object_id]
        if not callable(getattr(self._active_mobject, "get_start", None)) or not callable(
            getattr(self._active_mobject, "get_end", None)
        ):
            raise MotionConfigError("active object does not expose line endpoints")

        active_path = self.picture.named_paths[self.spec.driver.active_path]
        start_name = str(active_path.geometry.get("start_name") or "")
        end_name = str(active_path.geometry.get("end_name") or "")
        if (
            start_name not in self.picture.coordinates
            or end_name not in self.picture.coordinates
        ):
            raise MotionConfigError("active named line endpoints are unavailable")

        self._to_scene_point, logical_scale = _similarity_mapper_from_line(
            self.picture.coordinates[start_name],
            self.picture.coordinates[end_name],
            self._active_mobject.get_start(),
            self._active_mobject.get_end(),
        )
        picture_scale = float(self.picture.scale)
        if not isfinite(picture_scale) or picture_scale <= 0:
            raise MotionConfigError("picture scale must be positive")
        scene_unit_per_cm = logical_scale / picture_scale

        object_specs = {item.id: item for item in self.picture.objects}
        active_spec = object_specs.get(self.active_object_id)
        stroke_width_per_pt = None
        if active_spec is not None and active_spec.style.line_width_pt > 0:
            current_stroke = _mobject_stroke_width(self._active_mobject)
            if current_stroke is not None:
                stroke_width_per_pt = (
                    current_stroke / active_spec.style.line_width_pt
                )
        renderer_arguments = {"scene_unit_per_cm": scene_unit_per_cm}
        if stroke_width_per_pt is not None:
            renderer_arguments["stroke_width_per_pt"] = stroke_width_per_pt
        self.renderer = NativeManimRenderer(**renderer_arguments)
        self.figure = NativeFigure(
            self.picture,
            self._objects,
            self.shape,
            [],
        )
        self.tracker = ValueTracker(self.spec.driver.initial)
        self.runtime = NativeMotionRuntime(
            self.spec,
            self.picture,
            self.tracker.get_value,
        )
        self._bound_objects = tuple(
            self._objects[item.object_id] for item in self.spec.bindings
        )
        # Snapshot each unique semantic object, not only the declared followers.
        # ``object(id)`` is deliberately public, so context exit must also undo
        # temporary style/geometry edits made through that API.
        self._entry_objects: list[
            tuple[Mobject, Mobject, tuple[Callable[..., object], ...]]
        ] = []
        self._entry_object_by_identity: dict[int, Mobject] = {}
        seen_objects: set[int] = set()
        for mobject in self._objects.values():
            identity = id(mobject)
            if identity in seen_objects:
                continue
            seen_objects.add(identity)
            original = mobject.copy()
            self._entry_objects.append(
                (mobject, original, tuple(getattr(mobject, "updaters", ())))
            )
            self._entry_object_by_identity[identity] = original

        self._owned_updaters: dict[Mobject, tuple[Callable[..., object], ...]] = {}
        self._attached = False

    @property
    def initial(self) -> float:
        return self.spec.driver.initial

    @property
    def minimum(self) -> float:
        return self.spec.driver.minimum

    @property
    def maximum(self) -> float:
        return self.spec.driver.maximum

    @property
    def range(self) -> tuple[float, float]:
        return (self.minimum, self.maximum)

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def objects(self) -> Mapping[str, Mobject]:
        return MappingProxyType(self._objects)

    def object(self, object_id: str) -> Mobject:
        """Return the existing semantic Mobject without rebuilding the figure."""

        if object_id not in self._objects:
            raise MotionConfigError(f"unknown semantic object id: {object_id!r}")
        return self._objects[object_id]

    def logical_coordinate(self, name: str) -> Point2:
        """Return the current coordinate in the Provider's logical TikZ space."""

        coordinates = self.runtime.coordinates()
        if name not in coordinates:
            raise MotionConfigError(f"unknown semantic coordinate: {name!r}")
        value = coordinates[name]
        return (float(value[0]), float(value[1]))

    def coordinate(self, name: str) -> np.ndarray:
        """Return the current coordinate in the incoming ShapeState scene space."""

        return np.asarray(
            self._to_scene_point(self.logical_coordinate(name)),
            dtype=float,
        ).copy()

    def attach(self) -> "NativeGeometryRig2D":
        """Install declared dependent-object updaters and verify the first frame."""

        if self._attached:
            return self
        for item in self._bound_objects:
            if list(getattr(item, "updaters", ())):
                raise MotionConfigError(
                    "TikZ Native motion input already has active updaters"
                )

        self.tracker.set_value(self.initial)
        before_updaters = {
            item: tuple(getattr(item, "updaters", ())) for item in self._bound_objects
        }
        try:
            self.runtime.bind(self.figure, self.renderer, self._to_scene_point)
            self._owned_updaters = {
                item: tuple(
                    updater
                    for updater in getattr(item, "updaters", ())
                    if updater not in before_updaters[item]
                )
                for item in self._bound_objects
            }
            for item in self._bound_objects:
                item.update(0)
            self._assert_bound_entry_alignment()
        except Exception:
            self._remove_owned_updaters()
            self._restore_entry_objects()
            raise
        self._attached = True
        return self

    @staticmethod
    def _numeric_attribute(mobject: Mobject, name: str) -> np.ndarray | None:
        value = getattr(mobject, name, None)
        if value is None:
            return None
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _entry_alignment_error(
        cls,
        current: Mobject,
        original: Mobject,
        *,
        geometry_tolerance: float = 1e-7,
        style_tolerance: float = 1e-9,
    ) -> str | None:
        current_family = current.get_family()
        original_family = original.get_family()
        if len(current_family) != len(original_family):
            return (
                "submobject family changed "
                f"from {len(original_family)} to {len(current_family)}"
            )
        style_attributes = (
            "stroke_rgbas",
            "fill_rgbas",
            "background_stroke_rgbas",
            "stroke_width",
            "background_stroke_width",
            "z_index",
        )
        for family_index, (current_item, original_item) in enumerate(
            zip(current_family, original_family)
        ):
            if type(current_item) is not type(original_item):
                return (
                    f"submobject {family_index} changed type from "
                    f"{type(original_item).__name__} to {type(current_item).__name__}"
                )
            current_points = np.asarray(current_item.points, dtype=float)
            original_points = np.asarray(original_item.points, dtype=float)
            if current_points.shape != original_points.shape:
                return (
                    f"submobject {family_index} point shape changed from "
                    f"{original_points.shape} to {current_points.shape}"
                )
            if current_points.size:
                point_delta = float(
                    np.max(np.linalg.norm(current_points - original_points, axis=1))
                )
                if point_delta > geometry_tolerance:
                    return (
                        f"submobject {family_index} moved by {point_delta:.12g} "
                        "scene unit"
                    )
            for attribute in style_attributes:
                current_value = cls._numeric_attribute(current_item, attribute)
                original_value = cls._numeric_attribute(original_item, attribute)
                if current_value is None and original_value is None:
                    continue
                if (
                    current_value is None
                    or original_value is None
                    or current_value.shape != original_value.shape
                    or not np.allclose(
                        current_value,
                        original_value,
                        atol=style_tolerance,
                        rtol=0.0,
                    )
                ):
                    return (
                        f"submobject {family_index} changed visual style {attribute}"
                    )
        return None

    def _assert_bound_entry_alignment(self) -> None:
        for binding in self.spec.bindings:
            current = self._objects[binding.object_id]
            original = self._entry_object_by_identity[id(current)]
            mismatch = self._entry_alignment_error(current, original)
            if mismatch is not None:
                raise MotionConfigError(
                    "motion driver initial state does not align with the input "
                    f"ShapeState object {binding.object_id!r}: {mismatch}"
                )

    def _remove_owned_updaters(self) -> None:
        for item, updaters in self._owned_updaters.items():
            for updater in updaters:
                item.remove_updater(updater)
        self._owned_updaters = {}

    def detach(self) -> "NativeGeometryRig2D":
        """Remove Provider updaters while leaving the current visible frame in place."""

        self._remove_owned_updaters()
        self._attached = False
        return self

    def _restore_entry_objects(self) -> None:
        for item, original, original_updaters in self._entry_objects:
            item.clear_updaters()
            item.become(original)
            item.clear_updaters()
            for updater in original_updaters:
                item.add_updater(updater)

    def restore_entry(self) -> Mobject:
        """Restore the exact incoming ShapeState and remove temporary updaters."""

        self.tracker.set_value(self.initial)
        update_error: Exception | None = None
        if self._attached:
            try:
                for item in self._bound_objects:
                    item.update(0)
            except Exception as error:  # Exact snapshots still restore the boundary.
                update_error = error
        self.detach()
        self._restore_entry_objects()
        if update_error is not None:
            raise update_error
        return self.shape

    def play_timeline(
        self,
        scene: Scene,
        *,
        on_cue: Callable[[MotionTimelineStep], None] | None = None,
    ) -> None:
        """Compatibility convenience; authored clips should use ``scene.play``."""

        if not isinstance(scene, Scene):
            raise MotionConfigError("scene must be a Manim Scene")
        if not self._attached:
            raise MotionConfigError("attach the 2D geometry rig before playing it")
        self.runtime.play_timeline(scene, self.tracker, on_cue=on_cue)

    def __enter__(self) -> "NativeGeometryRig2D":
        return self.attach()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.restore_entry()
        return False


# Short spelling for hand-authored clips; generated code uses the explicit name.
NativeRig2D = NativeGeometryRig2D


__all__ = [
    "NATIVE_RIG_2D_API_SCHEMA",
    "NativeGeometryRig2D",
    "NativeRig2D",
]

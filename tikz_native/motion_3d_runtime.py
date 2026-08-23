"""Embedded motion-3d runtime for an existing editor ShapeState.

The editor's unified canvas is an ordinary :class:`manim.Scene`.  Replacing
its camera with ``MultiProjectionCamera`` would re-project every unrelated
page object, so this module deliberately keeps the global camera untouched.
Only the semantic children of the input TikZ shape are projected from their
true 3D coordinates into the shape's existing 2D scene placement.

The helper is a zero-output ``restore_entry`` boundary.  Every object it
touches is snapshotted before the first updater is attached and is restored in
``finally``.  Temporary occlusion fragments are children of the input shape
only for the duration of the clip and are always removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite
from typing import Any, Mapping, Sequence

import numpy as np
from manim import Mobject, Scene, ValueTracker, VGroup, smooth

from .camera_3d import (
    DEFAULT_PRESETS,
    _is_rotation_matrix,
    _orbit_control_matrix,
    _spherical_bezier_matrix,
)
from .compiler import ObjectSpec, PictureSpec
from .geometry_rig_3d import semantic_model_3d_hash
from .manim_renderer import NativeManimRenderer
from .manim_renderer_3d import NativeManim3DRenderer
from .motion_3d import Motion3DConfigError, Motion3DSpec, NativeMotion3DRuntime
from .occlusion_3d import parallel_occlusion_interval, parallel_view_direction
from .projection_3d import project_point
from .version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_EMBEDDED_MOTION_3D,
    COMPONENT_GEOMETRY_RIG_3D,
    provider_component_revision,
    provider_component_revision_matches,
)


EMBEDDED_MOTION_3D_RUNTIME_CONTRACT = "tikz-native-embedded-motion-3d/v1"


class EmbeddedMotion3DError(Motion3DConfigError):
    """The current ShapeState cannot safely execute embedded motion-3d/v1."""


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbeddedMotion3DError(f"{field} must be an object")
    return dict(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbeddedMotion3DError(f"{field} must be a non-empty string")
    return value.strip()


def _point3(value: Sequence[float] | np.ndarray, field: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise EmbeddedMotion3DError(f"{field} must be a finite 3D point")
    return point


def _scene_point3(value: Sequence[float] | np.ndarray) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape == (2,):
        return np.array((point[0], point[1], 0.0), dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise EmbeddedMotion3DError("semantic object has an invalid scene point")
    return point


def _mobject_stroke_width(mobject: Mobject) -> float | None:
    """Return the physical stroke of the object's visible drawable leaves.

    ``VGroup.get_stroke_width()`` reports the container's default style even
    though the group has no points of its own.  TikZ dashed lines are VGroups
    of native ``Line`` children, so using the container value makes the
    temporary 3D occlusion slots jump to an unrelated stroke width as soon as
    their first updater runs.  Calibrate only from point-bearing family
    members whose stroke is actually visible in the input ShapeState.
    """

    widths: list[float] = []
    for member in mobject.get_family():
        has_points = getattr(member, "has_points", None)
        if not callable(has_points) or not has_points():
            continue
        opacity_getter = getattr(member, "get_stroke_opacity", None)
        if callable(opacity_getter):
            opacities = np.asarray(opacity_getter(), dtype=float).reshape(-1)
            if not np.any(np.isfinite(opacities) & (opacities > 0)):
                continue
        width_getter = getattr(member, "get_stroke_width", None)
        if not callable(width_getter):
            continue
        values = np.asarray(width_getter(), dtype=float).reshape(-1)
        widths.extend(
            float(value)
            for value in values
            if np.isfinite(value) and value > 0
        )
    return None if not widths else float(np.median(widths))


@dataclass
class _LocalProjectionState:
    """An object-local parallel camera represented by one ValueTracker."""

    entry_matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.entry_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise EmbeddedMotion3DError("TikZ entry projection must be finite and invertible")
        row_scales = np.max(np.abs(matrix), axis=1)
        if np.any(row_scales == 0.0) or not np.all(np.isfinite(row_scales)):
            raise EmbeddedMotion3DError("TikZ entry projection must be finite and invertible")
        normalized = matrix / row_scales[:, np.newaxis]
        row_norms = np.linalg.norm(normalized, axis=1)
        if np.any(row_norms == 0.0) or not np.all(np.isfinite(row_norms)):
            raise EmbeddedMotion3DError("TikZ entry projection must be finite and invertible")
        normalized /= row_norms[:, np.newaxis]
        determinant = float(np.linalg.det(normalized))
        if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
            raise EmbeddedMotion3DError("TikZ entry projection must be finite and invertible")
        self.entry_matrix = matrix.copy()
        self.source_matrix = matrix.copy()
        self.target_matrix = matrix.copy()
        self.control_matrix = matrix.copy()
        self.transition = "linear"
        self.tracker = ValueTracker(1.0)

    def matrix(self) -> np.ndarray:
        alpha = float(np.clip(self.tracker.get_value(), 0.0, 1.0))
        if self.transition == "orbit":
            return _spherical_bezier_matrix(
                self.source_matrix,
                self.control_matrix,
                self.target_matrix,
                alpha,
            )
        return (1.0 - alpha) * self.source_matrix + alpha * self.target_matrix

    def prepare(self, mode: str, transition: str, arc_height: float) -> None:
        if transition not in {"linear", "orbit"}:
            raise EmbeddedMotion3DError(
                "local camera transition must be 'linear' or 'orbit'"
            )
        if mode == "tikz":
            target = self.entry_matrix
        elif mode in DEFAULT_PRESETS:
            preset = DEFAULT_PRESETS[mode]
            if float(preset.perspective_strength) > 1e-9:
                raise EmbeddedMotion3DError(
                    "embedded motion-3d/v1 only supports parallel projection"
                )
            target = np.asarray(preset.matrix, dtype=float)
        else:
            available = ", ".join(["tikz", *sorted(DEFAULT_PRESETS)])
            raise EmbeddedMotion3DError(
                f"unknown local camera mode {mode!r}; available: {available}"
            )
        source = self.matrix().copy()
        if transition == "orbit" and (
            not _is_rotation_matrix(source) or not _is_rotation_matrix(target)
        ):
            raise EmbeddedMotion3DError(
                "local camera orbit endpoints must be right-handed orthogonal frames"
            )
        self.source_matrix = source
        self.target_matrix = target.copy()
        self.transition = transition
        self.control_matrix = (
            _orbit_control_matrix(source, target, float(arc_height))
            if transition == "orbit"
            else source.copy()
        )
        self.tracker.set_value(0.0)

    def restore_immediately(self) -> None:
        self.source_matrix = self.entry_matrix.copy()
        self.target_matrix = self.entry_matrix.copy()
        self.control_matrix = self.entry_matrix.copy()
        self.transition = "linear"
        self.tracker.set_value(1.0)


def _anchor_from_dot(
    coordinate_name: str,
    object_specs: Mapping[str, ObjectSpec],
    objects: Mapping[str, Mobject],
) -> np.ndarray | None:
    for object_id, spec in object_specs.items():
        if spec.kind != "dot" or spec.geometry.get("center_name") != coordinate_name:
            continue
        mobject = objects.get(object_id)
        if isinstance(mobject, Mobject):
            return _scene_point3(mobject.get_center()).copy()
    return None


def _anchor_from_full_line(
    coordinate_name: str,
    object_specs: Mapping[str, ObjectSpec],
    objects: Mapping[str, Mobject],
    picture: PictureSpec,
) -> np.ndarray | None:
    authored = picture.coordinates.get(coordinate_name)
    if authored is None:
        return None
    expected = _point3(authored, f"coordinate {coordinate_name!r}")
    for object_id, spec in object_specs.items():
        if spec.kind not in {"line", "arrow"}:
            continue
        mobject = objects.get(object_id)
        if not isinstance(mobject, Mobject):
            continue
        start_getter = getattr(mobject, "get_start", None)
        end_getter = getattr(mobject, "get_end", None)
        if not callable(start_getter) or not callable(end_getter):
            continue
        for name_field, value_field, getter in (
            ("start_name", "start", start_getter),
            ("end_name", "end", end_getter),
        ):
            if spec.geometry.get(name_field) != coordinate_name:
                continue
            value = spec.geometry.get(value_field)
            if value is not None and np.allclose(
                _point3(value, f"{object_id}.{value_field}"),
                expected,
                atol=1e-9,
                rtol=0.0,
            ):
                return _scene_point3(getter()).copy()
    return None


def _coordinate_anchor(
    coordinate_name: str,
    object_specs: Mapping[str, ObjectSpec],
    objects: Mapping[str, Mobject],
    picture: PictureSpec,
) -> np.ndarray:
    anchor = _anchor_from_dot(coordinate_name, object_specs, objects)
    if anchor is None:
        anchor = _anchor_from_full_line(
            coordinate_name, object_specs, objects, picture
        )
    if anchor is None:
        raise EmbeddedMotion3DError(
            f"ShapeState has no stable semantic anchor for {coordinate_name!r}"
        )
    return anchor


def _similarity_mapper(
    logical_start: np.ndarray,
    logical_end: np.ndarray,
    scene_start: np.ndarray,
    scene_end: np.ndarray,
):
    source = np.asarray(logical_end, dtype=float) - np.asarray(logical_start, dtype=float)
    target = _scene_point3(scene_end)[:2] - _scene_point3(scene_start)[:2]
    denominator = float(np.dot(source, source))
    if denominator <= 1e-18 or float(np.linalg.norm(target)) <= 1e-12:
        raise EmbeddedMotion3DError("3D hinge anchors must not collapse in the ShapeState")
    real = float(np.dot(target, source)) / denominator
    imaginary = float(target[1] * source[0] - target[0] * source[1]) / denominator
    matrix = np.array(((real, -imaginary), (imaginary, real)), dtype=float)
    scale = float(np.hypot(real, imaginary))
    scene_origin = _scene_point3(scene_start)

    def map_screen(value: Sequence[float] | np.ndarray) -> np.ndarray:
        point = np.asarray(value, dtype=float)
        if point.shape != (2,):
            raise EmbeddedMotion3DError("local projection must produce a 2D point")
        mapped = scene_origin[:2] + matrix @ (point - logical_start)
        return np.array((mapped[0], mapped[1], scene_origin[2]), dtype=float)

    return map_screen, scale


def _validate_frozen_context(
    *,
    definition: Mapping[str, Any],
    semantic_manifest: Mapping[str, Any],
    expected_provider_revision: str,
    expected_runtime_revision: str | None,
    picture: PictureSpec,
) -> None:
    expected_asset = _text(
        expected_provider_revision,
        "expected_provider_revision",
    )
    expected_runtime = _text(
        expected_runtime_revision or expected_asset,
        "expected_runtime_revision",
    )
    if not provider_component_revision_matches(
        COMPONENT_EMBEDDED_MOTION_3D,
        expected_runtime,
    ):
        raise EmbeddedMotion3DError(
            "embedded 3D runtime Provider revision component differs from the frozen draft"
        )
    if definition.get("dimension") != 3 or semantic_manifest.get("dimension") != 3:
        raise EmbeddedMotion3DError("frozen 3D definition and manifest must have dimension=3")
    if definition.get("status") != "ready" or definition.get("authorConfirmed") is not True:
        raise EmbeddedMotion3DError("frozen 3D definition is not author-confirmed and ready")
    if definition.get("revisionMatch") is not True:
        raise EmbeddedMotion3DError("frozen 3D definition records a Provider mismatch")
    asset_fields = (
        (
            "definition.expectedAssetProviderRevision",
            definition.get("expectedAssetProviderRevision"),
        ),
        (
            "semantic_manifest.providerRevision",
            semantic_manifest.get("providerRevision"),
        ),
        (
            "definition.currentRigProviderRevision",
            definition.get("currentRigProviderRevision"),
        ),
    )
    for field, value in asset_fields:
        if not provider_component_revision_matches(COMPONENT_ASSET_COMPILER, value):
            raise EmbeddedMotion3DError(
                f"{field} differs from the current asset_compiler component"
            )
    recorded_runtime = definition.get("embeddedMotion3dRevision")
    if recorded_runtime is not None and recorded_runtime != expected_runtime:
        raise EmbeddedMotion3DError(
            "definition.embeddedMotion3dRevision differs from expected_runtime_revision"
        )
    rig_fields = (
        ("definition.geometryRig3dRevision", definition.get("geometryRig3dRevision")),
        (
            "semantic_manifest.geometryRig3dRevision",
            semantic_manifest.get("geometryRig3dRevision"),
        ),
    )
    if any(value is not None for _field, value in rig_fields):
        if any(value is None for _field, value in rig_fields):
            raise EmbeddedMotion3DError(
                "frozen 3D definition and manifest disagree about geometry_rig_3d identity"
            )
        for field, value in rig_fields:
            if not provider_component_revision_matches(
                COMPONENT_GEOMETRY_RIG_3D,
                value,
            ):
                raise EmbeddedMotion3DError(
                    f"{field} differs from the current geometry_rig_3d component"
                )
    else:
        # Legacy v1 stored one global Provider identity in these two fields.
        # It remains valid only while that exact identity is also the reviewed
        # geometry-rig component identity.
        for field, value in (
            (
                "definition.currentRigProviderRevision",
                definition.get("currentRigProviderRevision"),
            ),
            (
                "semantic_manifest.providerRevision",
                semantic_manifest.get("providerRevision"),
            ),
        ):
            if not provider_component_revision_matches(
                COMPONENT_GEOMETRY_RIG_3D,
                value,
            ):
                raise EmbeddedMotion3DError(
                    f"{field} differs from the current geometry_rig_3d component"
                )
    if expected_asset != provider_component_revision(COMPONENT_ASSET_COMPILER):
        raise EmbeddedMotion3DError(
            "expected_provider_revision differs from the current asset compiler component"
        )
    manifest_picture = semantic_manifest.get("pictureIndex")
    if manifest_picture != picture.index:
        raise EmbeddedMotion3DError("semantic manifest selects a different TikZ picture")
    actual_semantic_hash = semantic_model_3d_hash(picture)
    for field, value in (
        ("definition.semanticModelHash", definition.get("semanticModelHash")),
        ("semantic_manifest.semanticModelHash", semantic_manifest.get("semanticModelHash")),
    ):
        if value != actual_semantic_hash:
            raise EmbeddedMotion3DError(f"{field} differs from the input ShapeState")


def _logical_value(
    coordinates: Mapping[str, Sequence[float] | np.ndarray],
    spec: ObjectSpec,
    name_field: str,
    value_field: str,
) -> np.ndarray:
    name = str(spec.geometry.get(name_field) or "").strip()
    if name:
        try:
            return _point3(coordinates[name], f"coordinate {name!r}")
        except KeyError as exc:
            raise EmbeddedMotion3DError(
                f"semantic object {spec.id!r} uses unknown coordinate {name!r}"
            ) from exc
    return _point3(spec.geometry.get(value_field), f"{spec.id}.{value_field}")


def _coordinate_norm_bounds(
    spec: Motion3DSpec,
    entry_coordinates: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, float]:
    """Return conservative absolute-coordinate bounds across the driver range."""

    axis_start = _point3(
        entry_coordinates[spec.driver.axis[0]], "driver axis start"
    )
    moving = set(spec.driver.moving_points)
    bounds: dict[str, float] = {}
    for name, value in entry_coordinates.items():
        point = _point3(value, f"coordinate {name!r}")
        bounds[name] = (
            float(np.linalg.norm(axis_start))
            + float(np.linalg.norm(point - axis_start))
            if name in moving
            else float(np.linalg.norm(point))
        )
    for derived in spec.derived_coordinates:
        if derived.type == "point_on_segment":
            assert derived.start is not None and derived.end is not None
            bounds[derived.name] = max(bounds[derived.start], bounds[derived.end])
        else:
            assert derived.point is not None and derived.line_start is not None
            # Orthogonal projection onto an infinite line is at most the
            # point-to-line-start distance away from line_start.
            bounds[derived.name] = (
                bounds[derived.point] + 2.0 * bounds[derived.line_start]
            )
    return bounds


def play_motion_3d_on_native_shape(
    scene: Scene,
    shape: Mobject,
    motion_spec_payload: object,
    *,
    definition: Mapping[str, Any],
    semantic_manifest: Mapping[str, Any],
    expected_provider_revision: str,
    expected_runtime_revision: str | None = None,
    runtime_contract: str = EMBEDDED_MOTION_3D_RUNTIME_CONTRACT,
) -> Mobject:
    """Play true-coordinate 3D motion locally and restore the input exactly.

    The returned object is the same input identity.  The helper intentionally
    creates no result ShapeState; Host v1 therefore declares zero outputs.
    """

    if runtime_contract != EMBEDDED_MOTION_3D_RUNTIME_CONTRACT:
        raise EmbeddedMotion3DError("unsupported embedded motion-3d runtime contract")
    if not isinstance(scene, Scene):
        raise EmbeddedMotion3DError("scene must be a Manim Scene")
    if not isinstance(shape, Mobject):
        raise EmbeddedMotion3DError("shape must be a Manim Mobject")
    frozen_definition = _mapping(definition, "definition")
    frozen_manifest = _mapping(semantic_manifest, "semantic_manifest")
    spec = (
        motion_spec_payload
        if isinstance(motion_spec_payload, Motion3DSpec)
        else Motion3DSpec.from_dict(motion_spec_payload)
    )
    if not spec.timeline:
        raise EmbeddedMotion3DError("embedded motion-3d timeline must not be empty")
    if spec.end_policy != "restore_entry":
        raise EmbeddedMotion3DError("embedded motion-3d requires restore_entry")

    objects_value = getattr(shape, "_codex_tikz_native_objects", None)
    picture = getattr(shape, "_codex_tikz_native_picture", None)
    if not isinstance(objects_value, Mapping) or not isinstance(picture, PictureSpec):
        raise EmbeddedMotion3DError(
            "input shape does not expose TikZ Native semantic objects"
        )
    objects = {
        str(object_id): mobject
        for object_id, mobject in objects_value.items()
        if isinstance(mobject, Mobject)
    }
    if set(objects) != {str(item.id) for item in picture.objects}:
        raise EmbeddedMotion3DError(
            "input ShapeState semantic objects differ from its compiled picture"
        )
    if picture.dimension != 3 or picture.projection_3d is None:
        raise EmbeddedMotion3DError("embedded motion-3d requires a 3D TikZ ShapeState")
    if spec.picture_index != picture.index:
        raise EmbeddedMotion3DError("motion spec selects a different TikZ picture")
    _validate_frozen_context(
        definition=frozen_definition,
        semantic_manifest=frozen_manifest,
        expected_provider_revision=expected_provider_revision,
        expected_runtime_revision=expected_runtime_revision,
        picture=picture,
    )
    spec.validate_picture(picture)

    manifest_bindings = frozen_manifest.get("bindings")
    if not isinstance(manifest_bindings, list):
        raise EmbeddedMotion3DError("semantic_manifest.bindings must be an array")
    manifest_binding_keys = {
        (
            str(item.get("objectId") or ""),
            str(item.get("bindingType") or ""),
            tuple(str(value) for value in item.get("pointNames", [])),
        )
        for item in manifest_bindings
        if isinstance(item, Mapping)
        and item.get("enabled") is True
        and isinstance(item.get("pointNames"), list)
    }
    for binding in spec.bindings:
        if (binding.object_id, binding.type, tuple(binding.points)) not in manifest_binding_keys:
            raise EmbeddedMotion3DError(
                f"motion binding {binding.object_id!r} is absent from the semantic manifest"
            )

    object_specs = {item.id: item for item in picture.objects}
    unsupported = sorted(
        item.id
        for item in picture.objects
        if item.kind not in {"line", "arrow", "polygon", "dot", "label", "path_label"}
    )
    if unsupported:
        raise EmbeddedMotion3DError(
            "embedded motion-3d/v1 cannot locally project these objects: "
            + ", ".join(unsupported)
        )
    unique_objects: list[Mobject] = []
    seen: set[int] = set()
    for item in objects.values():
        if id(item) in seen:
            continue
        seen.add(id(item))
        unique_objects.append(item)
    driver_tracker = ValueTracker(spec.driver.initial)
    runtime = NativeMotion3DRuntime(spec, picture, driver_tracker.get_value)
    entry_coordinates = {
        name: tuple(float(component) for component in value)
        for name, value in runtime.coordinates().items()
    }
    binding_ids = {binding.object_id for binding in spec.bindings}
    axis_start_name, axis_end_name = spec.driver.axis
    scene_start = _coordinate_anchor(
        axis_start_name, object_specs, objects, picture
    )
    scene_end = _coordinate_anchor(axis_end_name, object_specs, objects, picture)
    entry_matrix = np.asarray(picture.projection_3d.matrix, dtype=float)
    entry_start = np.asarray(
        project_point(entry_matrix, runtime.coordinate(axis_start_name)), dtype=float
    )[:2]
    entry_end = np.asarray(
        project_point(entry_matrix, runtime.coordinate(axis_end_name)), dtype=float
    )[:2]
    map_screen, logical_scale = _similarity_mapper(
        entry_start, entry_end, scene_start, scene_end
    )
    if not isfinite(logical_scale) or logical_scale <= 0:
        raise EmbeddedMotion3DError("ShapeState projection scale must be positive")
    picture_scale = float(picture.scale)
    if not isfinite(picture_scale) or picture_scale <= 0:
        raise EmbeddedMotion3DError("TikZ picture scale must be positive")
    scene_unit_per_cm = logical_scale / picture_scale

    # Keep the visible figure centered during an object-local camera change.
    authored_points = np.asarray(list(entry_coordinates.values()), dtype=float)
    pivot = 0.5 * (authored_points.min(axis=0) + authored_points.max(axis=0))
    entry_pivot = np.asarray(project_point(entry_matrix, pivot), dtype=float)[:2]
    local_camera = _LocalProjectionState(entry_matrix)
    coordinate_norm_bounds = _coordinate_norm_bounds(spec, entry_coordinates)
    projection_norm_bound = max(
        float(np.linalg.norm(matrix[:2], ord=2))
        for matrix in (
            entry_matrix,
            *(np.asarray(preset.matrix, dtype=float) for preset in DEFAULT_PRESETS.values()),
        )
    )

    def project_scene(value: Sequence[float] | np.ndarray) -> np.ndarray:
        point = _point3(value, "logical coordinate")
        matrix = local_camera.matrix()
        local = np.asarray(project_point(matrix, point - pivot), dtype=float)[:2]
        return map_screen(local + entry_pivot)

    stroke_width_ratios: list[float] = []
    for object_id, object_spec in object_specs.items():
        if (
            object_spec.kind not in {"line", "arrow", "polygon"}
            or object_spec.style.line_width_pt <= 0
        ):
            continue
        current = _mobject_stroke_width(objects[object_id])
        if current is not None:
            stroke_width_ratios.append(
                current / object_spec.style.line_width_pt
            )
    stroke_width_per_pt = (
        float(np.median(stroke_width_ratios))
        if stroke_width_ratios
        else None
    )
    renderer_arguments: dict[str, float] = {
        "scene_unit_per_cm": scene_unit_per_cm,
    }
    if stroke_width_per_pt is not None:
        renderer_arguments["stroke_width_per_pt"] = stroke_width_per_pt
    renderer = NativeManimRenderer(**renderer_arguments)
    occlusion_renderer = NativeManim3DRenderer(**renderer_arguments)

    relation_member_ids = {
        object_id
        for relation in picture.occlusion_relations
        for object_id in relation.object_ids
    }
    original_shape_children = list(shape.submobjects)
    originals = [(item, item.copy()) for item in unique_objects]
    original_by_object_id = {
        object_id: objects[object_id].copy() for object_id in object_specs
    }
    original_family_state = [
        (
            member,
            list(getattr(member, "updaters", ())),
            bool(getattr(member, "updating_suspended", False)),
            float(getattr(member, "z_index", 0.0)),
        )
        for member in shape.get_family()
    ]
    global_camera = scene.camera
    temporary_groups: list[Mobject] = []
    label_offsets: dict[str, np.ndarray] = {}
    path_label_states: dict[str, dict[str, Any]] = {}
    fixed_camera_callbacks: list[Any] = []

    def object_point(
        object_spec: ObjectSpec,
        name_field: str,
        value_field: str,
        *,
        dynamic: bool,
    ) -> np.ndarray:
        return project_scene(
            _logical_value(
                runtime.coordinates() if dynamic else entry_coordinates,
                object_spec,
                name_field,
                value_field,
            )
        )

    def register_projection_update(
        mobject: Mobject,
        updater,
        *,
        dynamic: bool,
    ) -> None:
        if dynamic:
            mobject.add_updater(updater)
        else:
            fixed_camera_callbacks.append(
                lambda callback=updater, item=mobject: callback(item, 0.0)
            )

    try:
        # Suspend the input's own update behavior while this exclusive clip is
        # active.  The complete recursive updater/z state is restored below.
        for member, _updaters, _suspended, _z_index in original_family_state:
            member.clear_updaters(recursive=False)
        for object_id, object_spec in object_specs.items():
            mobject = objects[object_id]
            dynamic = object_id in binding_ids
            if object_id in relation_member_ids:
                mobject.set_opacity(0.0)
                continue
            if object_spec.kind in {"line", "arrow"}:

                def update_line(
                    item: Mobject,
                    _dt: float = 0.0,
                    *,
                    current_spec: ObjectSpec = object_spec,
                    follows_driver: bool = dynamic,
                ) -> None:
                    start = object_point(
                        current_spec,
                        "start_name",
                        "start",
                        dynamic=follows_driver,
                    )
                    end = object_point(
                        current_spec,
                        "end_name",
                        "end",
                        dynamic=follows_driver,
                    )
                    endpoint_updater = getattr(item, "put_start_and_end_on", None)
                    if callable(endpoint_updater):
                        endpoint_updater(start, end)
                    else:
                        item.become(
                            renderer.native_line_from_points(
                                start, end, current_spec.style
                            )
                        )
                register_projection_update(
                    mobject, update_line, dynamic=dynamic
                )
                continue
            if object_spec.kind == "polygon":

                def update_polygon(
                    item: Mobject,
                    _dt: float = 0.0,
                    *,
                    current_spec: ObjectSpec = object_spec,
                    follows_driver: bool = dynamic,
                ) -> None:
                    names = current_spec.geometry.get("point_names")
                    values = current_spec.geometry.get("points")
                    if not isinstance(values, list):
                        values = list(values or [])
                    if isinstance(names, list) and len(names) == len(values):
                        points = [
                            project_scene(
                                (
                                    runtime.coordinates()
                                    if follows_driver
                                    else entry_coordinates
                                )[str(name)]
                            )
                            for name in names
                        ]
                    else:
                        points = [project_scene(value) for value in values]
                    if len(points) < 3:
                        raise EmbeddedMotion3DError(
                            f"polygon {current_spec.id!r} has fewer than three points"
                        )
                    item.set_points_as_corners([*points, points[0]])
                register_projection_update(
                    mobject, update_polygon, dynamic=dynamic
                )
                continue
            if object_spec.kind == "dot":
                update_dot = (
                    lambda item,
                    _dt=0.0,
                    current_spec=object_spec,
                    follows_driver=dynamic: item.move_to(
                        object_point(
                            current_spec,
                            "center_name",
                            "center",
                            dynamic=follows_driver,
                        )
                    )
                )
                register_projection_update(
                    mobject, update_dot, dynamic=dynamic
                )
                continue

            if object_spec.kind == "label":
                anchor = object_point(
                    object_spec, "at_name", "at", dynamic=dynamic
                )
            else:
                start = object_point(
                    object_spec, "start_name", "start", dynamic=dynamic
                )
                end = object_point(
                    object_spec, "end_name", "end", dynamic=dynamic
                )
                position = float(object_spec.geometry.get("pos", 0.5))
                anchor = start + position * (end - start)
            label_offsets[object_id] = _scene_point3(mobject.get_center()) - anchor
            if (
                object_spec.kind == "path_label"
                and object_spec.placement is not None
                and object_spec.placement.sloped
            ):
                vector = end[:2] - start[:2]
                length = float(np.linalg.norm(vector))
                if length <= 1e-12:
                    raise EmbeddedMotion3DError(
                        f"path label {object_id!r} has a collapsed entry path"
                    )
                tangent = vector / length
                normal = np.array((-tangent[1], tangent[0]), dtype=float)
                offset = label_offsets[object_id][:2]
                path_label_states[object_id] = {
                    "tangentOffset": float(np.dot(offset, tangent)),
                    "normalOffset": float(np.dot(offset, normal)),
                    "angle": atan2(float(tangent[1]), float(tangent[0])),
                }

            def update_label(
                item: Mobject,
                _dt: float = 0.0,
                *,
                current_spec: ObjectSpec = object_spec,
                current_id: str = object_id,
                follows_driver: bool = dynamic,
            ) -> None:
                if current_spec.kind == "label":
                    current_anchor = object_point(
                        current_spec,
                        "at_name",
                        "at",
                        dynamic=follows_driver,
                    )
                else:
                    start = object_point(
                        current_spec,
                        "start_name",
                        "start",
                        dynamic=follows_driver,
                    )
                    end = object_point(
                        current_spec,
                        "end_name",
                        "end",
                        dynamic=follows_driver,
                    )
                    position = float(current_spec.geometry.get("pos", 0.5))
                    current_anchor = start + position * (end - start)
                path_state = path_label_states.get(current_id)
                if path_state is None:
                    offset = label_offsets[current_id]
                else:
                    vector = end[:2] - start[:2]
                    length = float(np.linalg.norm(vector))
                    if length <= 1e-12:
                        raise EmbeddedMotion3DError(
                            f"path label {current_id!r} path collapsed during motion"
                        )
                    tangent = vector / length
                    normal = np.array((-tangent[1], tangent[0]), dtype=float)
                    angle = atan2(float(tangent[1]), float(tangent[0]))
                    item.rotate(
                        angle - float(path_state["angle"]),
                        about_point=item.get_center(),
                    )
                    path_state["angle"] = angle
                    offset_2d = (
                        float(path_state["tangentOffset"]) * tangent
                        + float(path_state["normalOffset"]) * normal
                    )
                    offset = np.array(
                        (offset_2d[0], offset_2d[1], label_offsets[current_id][2]),
                        dtype=float,
                    )
                item.move_to(current_anchor + offset)

            register_projection_update(
                mobject, update_label, dynamic=dynamic
            )

        if fixed_camera_callbacks:

            def update_fixed_projection(_item: Mobject, _dt: float = 0.0) -> None:
                for callback in fixed_camera_callbacks:
                    callback()

            shape.add_updater(update_fixed_projection)

        for relation in picture.occlusion_relations:
            start_world = _point3(runtime.coordinate(relation.start_name), relation.start_name)
            end_world = _point3(runtime.coordinate(relation.end_name), relation.end_name)
            start_scene = project_scene(start_world)
            end_scene = project_scene(end_world)
            safe_length = (
                1.05
                * logical_scale
                * projection_norm_bound
                * (
                    coordinate_norm_bounds[relation.start_name]
                    + coordinate_norm_bounds[relation.end_name]
                )
                + 1e-6
            )
            allocation_start = np.array((0.0, 0.0, start_scene[2]), dtype=float)
            allocation_end = np.array(
                (safe_length, 0.0, start_scene[2]), dtype=float
            )
            slots = occlusion_renderer._make_occlusion_slots(  # noqa: SLF001
                relation, allocation_start, allocation_end
            )
            container = VGroup(*slots.lines)
            container.set_z_index(relation.z_index)
            shape.add(container)
            temporary_groups.append(container)

            def update_occlusion(
                _item: Mobject,
                _dt: float = 0.0,
                *,
                relation_spec=relation,
                stable_slots=slots,
            ) -> None:
                current_start_world = _point3(
                    runtime.coordinate(relation_spec.start_name),
                    relation_spec.start_name,
                )
                current_end_world = _point3(
                    runtime.coordinate(relation_spec.end_name),
                    relation_spec.end_name,
                )
                face = [
                    _point3(runtime.coordinate(name), name)
                    for name in relation_spec.face_names
                ]
                interval = parallel_occlusion_interval(
                    current_start_world,
                    current_end_world,
                    face,
                    parallel_view_direction(local_camera.matrix()),
                )
                occlusion_renderer._update_occlusion_slots(  # noqa: SLF001
                    stable_slots,
                    project_scene(current_start_world),
                    project_scene(current_end_world),
                    interval,
                )

            container.add_updater(update_occlusion)

        shape.update(0.0)
        for object_id, mobject in objects.items():
            if object_id in relation_member_ids:
                continue
            original = original_by_object_id[object_id]
            current_points = mobject.get_all_points()
            original_points = original.get_all_points()
            if current_points.shape != original_points.shape or not np.allclose(
                current_points,
                original_points,
                atol=1e-7,
                rtol=0.0,
            ):
                raise EmbeddedMotion3DError(
                    "local 3D entry projection does not align with semantic object "
                    f"{object_id!r} in the input ShapeState"
                )
        if not np.allclose(
            _coordinate_anchor(axis_start_name, object_specs, objects, picture),
            scene_start,
            atol=1e-7,
            rtol=0.0,
        ) or not np.allclose(
            _coordinate_anchor(axis_end_name, object_specs, objects, picture),
            scene_end,
            atol=1e-7,
            rtol=0.0,
        ):
            raise EmbeddedMotion3DError(
                "motion driver initial state does not align with the input ShapeState"
            )

        for step in spec.timeline:
            if step.type == "driver":
                assert step.to is not None
                scene.play(
                    driver_tracker.animate.set_value(step.to),
                    run_time=step.duration,
                    rate_func=smooth,
                )
            elif step.type == "camera":
                assert step.mode is not None
                local_camera.prepare(step.mode, step.transition, step.arc_height)
                scene.play(
                    local_camera.tracker.animate.set_value(1.0),
                    run_time=step.duration,
                    rate_func=smooth,
                )
            else:
                scene.wait(step.duration)
            if step.hold:
                scene.wait(step.hold)

        restore_animations = []
        if abs(float(driver_tracker.get_value()) - spec.driver.initial) > 1e-12:
            restore_animations.append(
                driver_tracker.animate.set_value(spec.driver.initial)
            )
        if not np.allclose(
            local_camera.matrix(), entry_matrix, atol=1e-12, rtol=0.0
        ):
            local_camera.prepare(
                "tikz",
                spec.camera.restore_transition,
                0.85,
            )
            restore_animations.append(local_camera.tracker.animate.set_value(1.0))
        if restore_animations:
            scene.play(
                *restore_animations,
                run_time=spec.camera.restore_duration,
                rate_func=smooth,
            )
    finally:
        driver_tracker.set_value(spec.driver.initial)
        local_camera.restore_immediately()
        shape.clear_updaters(recursive=False)
        for item in unique_objects:
            item.clear_updaters(recursive=False)
        for group in temporary_groups:
            group.clear_updaters()
            shape.remove(group)
            scene.remove(group)
        for item, original in originals:
            item.become(original)
            item.clear_updaters(recursive=False)
        shape.submobjects[:] = original_shape_children
        for member, updaters, suspended, z_index in original_family_state:
            member.updaters[:] = updaters
            member.updating_suspended = suspended
            member.set_z_index(z_index, family=False)
        if scene.camera is not global_camera:
            raise EmbeddedMotion3DError(
                "embedded motion-3d changed the page Scene camera"
            )
    return shape


__all__ = [
    "EMBEDDED_MOTION_3D_RUNTIME_CONTRACT",
    "EmbeddedMotion3DError",
    "play_motion_3d_on_native_shape",
]

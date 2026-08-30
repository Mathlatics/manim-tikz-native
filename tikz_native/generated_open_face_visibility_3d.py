"""Product adapter for unified occlusion in generated Manim v3 sources.

The v3 generator deliberately publishes a self-contained legacy implementation.
Source-project builds keep that frozen source intact and append a small override
which routes the public ``install_open_face_visibility_3d`` name through this
module.  The adapter reconstructs the renderer-neutral model from the generated
constants while continuing to use the generated geometry rig as the live source
of coordinates, projection, semantic Mobjects, and static-entry lifecycle.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np
from manim import CapStyleType, Line, LineJointType, Mobject, Polygon, VGroup

from polyhedron_visibility import ParallelProjection
from polyhedron_visibility.open_faces.contract import (
    ARTICULATED_HINGE_POLICY,
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceVisibilityModel,
)
from polyhedron_visibility.open_faces.manim import OpenFaceOcclusion3D
from polyhedron_visibility.painter_band import (
    ScenePainterBandError,
    ScenePainterBandReservation,
    release_scene_painter_band,
    reserve_scene_painter_band,
)
from polyhedron_visibility.style import OcclusionStyle


_OVERRIDE_BEGIN = "# === tikz-native unified open-face override v1: begin ==="
_OVERRIDE_END = "# === tikz-native unified open-face override v1: end ==="
_OWNER_ATTRIBUTE = "_mathppt_open_face_visibility_owner"
_TEX_POINTS_PER_CM = 72.27 / 2.54


class GeneratedOpenFaceVisibility3DError(RuntimeError):
    """A generated v3 source cannot be attached to the unified runtime."""


def _normalise_band(value: object) -> tuple[float, float]:
    if hasattr(value, "as_list"):
        value = value.as_list()  # type: ignore[union-attr]
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise GeneratedOpenFaceVisibility3DError(
            "preferred_painter_z_band must contain two values"
        )
    low, high = (float(item) for item in value)
    if not isfinite(low) or not isfinite(high) or low >= high:
        raise GeneratedOpenFaceVisibility3DError(
            "preferred_painter_z_band must be finite and increasing"
        )
    return low, high


def _normalise_policy(value: str) -> str:
    aliases = {
        "diagrammatic": "diagrammatic",
        "physical": "physical",
        "source": "diagrammatic",
        "source-order": "diagrammatic",
        "faces-first": "diagrammatic",
    }
    try:
        return aliases[str(value).strip().lower()]
    except KeyError as exc:
        raise GeneratedOpenFaceVisibility3DError(
            "paint_policy must be 'diagrammatic' or 'physical'"
        ) from exc


def _reserve_band(
    scene: object,
    reservation: ScenePainterBandReservation,
) -> tuple[float, float]:
    try:
        return reserve_scene_painter_band(scene, reservation)
    except ScenePainterBandError as exc:
        raise GeneratedOpenFaceVisibility3DError(str(exc)) from exc


def _release_band(
    scene: object,
    reservation: ScenePainterBandReservation,
) -> None:
    try:
        release_scene_painter_band(scene, reservation)
    except ScenePainterBandError as exc:
        raise GeneratedOpenFaceVisibility3DError(str(exc)) from exc


def _point3(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise GeneratedOpenFaceVisibility3DError(
            f"{label} must be a finite three-component point"
        )
    return result


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (tuple, list)) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise GeneratedOpenFaceVisibility3DError(
            f"{label} must be an array of objects"
        )
    return tuple(value)  # type: ignore[return-value]


def _build_model(
    *,
    visibility_group_id: str,
    vertex_ids: object,
    faces: object,
    inclusive_edges: object,
    strokes: object,
    positions: Mapping[str, Sequence[float]],
) -> OpenFaceVisibilityModel:
    if not isinstance(vertex_ids, (tuple, list)):
        raise GeneratedOpenFaceVisibility3DError(
            "OPEN_FACE_VERTEX_IDS must be an array"
        )
    vertex_names = tuple(str(item) for item in vertex_ids)
    face_values = _mapping_sequence(faces, "OPEN_FACE_FACES")
    stroke_values = _mapping_sequence(strokes, "OPEN_FACE_STROKES")
    if not isinstance(inclusive_edges, Mapping):
        raise GeneratedOpenFaceVisibility3DError(
            "OPEN_FACE_INCLUSIVE_EDGES must be an object"
        )

    seam_faces: dict[tuple[str, str], set[str]] = {}
    for raw_face_id, raw_edges in inclusive_edges.items():
        face_id = str(raw_face_id)
        if not isinstance(raw_edges, (tuple, list)):
            raise GeneratedOpenFaceVisibility3DError(
                f"inclusive edges for face {face_id!r} must be an array"
            )
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 2:
                raise GeneratedOpenFaceVisibility3DError(
                    f"inclusive edge for face {face_id!r} must contain two vertices"
                )
            edge = tuple(sorted((str(raw_edge[0]), str(raw_edge[1]))))
            seam_faces.setdefault(edge, set()).add(face_id)
    seams = []
    for edge, owners in sorted(seam_faces.items()):
        if len(owners) != 2:
            raise GeneratedOpenFaceVisibility3DError(
                f"generated seam {edge!r} must belong to exactly two faces"
            )
        seams.append(
            {
                "seamId": "generated-seam:" + ":".join(edge),
                "policy": ARTICULATED_HINGE_POLICY,
                "faceIds": sorted(owners),
                "vertexIds": list(edge),
            }
        )

    payload = {
        "schema": OPEN_FACE_MODEL_SCHEMA,
        "topology": OPEN_FACE_TOPOLOGY,
        "visibilityGroupId": visibility_group_id,
        "vertices": [
            {
                "vertexId": vertex_id,
                "entryPosition": list(_point3(positions[vertex_id], vertex_id)),
            }
            for vertex_id in vertex_names
        ],
        "faces": [
            {
                "faceId": str(item["face_id"]),
                "logicalSurfaceId": "generated:" + str(item["face_id"]),
                "vertexIds": [str(value) for value in item["vertex_ids"]],
                "occludesStrokes": bool(item["occludes_strokes"]),
            }
            for item in face_values
        ],
        "seams": seams,
        "strokes": [
            {
                "sourceEdgeId": str(item["source_edge_id"]),
                "vertexIds": [str(value) for value in item["vertex_ids"]],
                "incidentFaceIds": [str(value) for value in item["incident_face_ids"]],
                "excludedOccluderFaceIds": [
                    str(value) for value in item["excluded_face_ids"]
                ],
                "visibilityMode": str(item["visibility_mode"]),
            }
            for item in stroke_values
        ],
    }
    model = OpenFaceVisibilityModel.from_dict(payload)
    model.validate(vertex_positions=positions)
    return model


def _cap_style(value: object) -> CapStyleType:
    return {
        "round": CapStyleType.ROUND,
        "butt": CapStyleType.BUTT,
        "square": CapStyleType.SQUARE,
    }.get(str(value).lower(), CapStyleType.AUTO)


def _joint_style(value: object) -> LineJointType:
    return {
        "round": LineJointType.ROUND,
        "bevel": LineJointType.BEVEL,
        "miter": LineJointType.MITER,
    }.get(str(value).lower(), LineJointType.AUTO)


def _edge_style(
    binding: Mapping[str, Any],
    *,
    max_projected_length: float,
    scene_unit_per_cm: float,
    stroke_width_per_pt: float,
) -> tuple[Line, OcclusionStyle]:
    visible = binding.get("visible_style")
    hidden = binding.get("hidden_style")
    if not isinstance(visible, Mapping) or not isinstance(hidden, Mapping):
        raise GeneratedOpenFaceVisibility3DError(
            "generated stroke binding lost its visible/hidden styles"
        )
    visible_width = float(visible["line_width_pt"]) * stroke_width_per_pt
    hidden_width = float(hidden["line_width_pt"]) * stroke_width_per_pt
    visible_opacity = float(visible.get("opacity", 1.0))
    hidden_opacity = float(hidden.get("opacity", 1.0))
    visible_cap_style = _cap_style(visible.get("line_cap"))
    visible_joint_type = _joint_style(visible.get("line_join"))
    hidden_cap_style = (
        visible_cap_style
        if hidden.get("line_cap") is None
        else _cap_style(hidden.get("line_cap"))
    )
    hidden_joint_type = (
        visible_joint_type
        if hidden.get("line_join") is None
        else _joint_style(hidden.get("line_join"))
    )
    pattern = hidden.get("dash_pattern_pt")
    if pattern is None:
        dash_length, dash_gap = max_projected_length, 0.0
    elif isinstance(pattern, (tuple, list)) and len(pattern) == 2:
        dash_length = max(
            float(pattern[0]) * scene_unit_per_cm / _TEX_POINTS_PER_CM,
            1.0e-6,
        )
        dash_gap = max(
            float(pattern[1]) * scene_unit_per_cm / _TEX_POINTS_PER_CM,
            0.0,
        )
    else:
        raise GeneratedOpenFaceVisibility3DError(
            "generated hidden stroke dash pattern is invalid"
        )
    proxy = Line(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        buff=0,
        cap_style=visible_cap_style,
        joint_type=visible_joint_type,
    )
    proxy.set_stroke(
        color=str(visible["draw_color"]),
        width=1.0,
        opacity=1.0,
    )
    proxy.set_z_index(float(binding.get("z_index", 0.0)))
    style = OcclusionStyle(
        max_projected_length=max_projected_length,
        dash_length=dash_length,
        dash_gap=dash_gap,
        visible_color=str(visible["draw_color"]),
        hidden_color=str(hidden["draw_color"]),
        visible_width_scale=visible_width,
        hidden_width_scale=hidden_width,
        visible_opacity_scale=visible_opacity,
        hidden_opacity_scale=hidden_opacity,
        hidden_cap_style=hidden_cap_style,
        hidden_joint_type=hidden_joint_type,
    )
    return proxy, style


def _family_members(roots: Mapping[str, Sequence[Mobject]]) -> tuple[Mobject, ...]:
    result: list[Mobject] = []
    seen: set[int] = set()
    for values in roots.values():
        for root in values:
            for member in root.get_family():
                if id(member) not in seen:
                    seen.add(id(member))
                    result.append(member)
    return tuple(result)


def _capture_original_sources(
    roots: Mapping[str, Sequence[Mobject]],
) -> tuple[tuple[Mobject, dict[str, object]], ...]:
    result = []
    for member in _family_members(roots):
        attributes: dict[str, object] = {}
        for name in (
            "stroke_rgbas",
            "background_stroke_rgbas",
            "stroke_opacity",
            "background_stroke_opacity",
        ):
            if hasattr(member, name):
                value = getattr(member, name)
                attributes[name] = value.copy() if isinstance(value, np.ndarray) else value
        result.append((member, attributes))
    return tuple(result)


def _hide_original_sources(
    snapshots: Sequence[tuple[Mobject, Mapping[str, object]]],
) -> None:
    for member, attributes in snapshots:
        for name in ("stroke_rgbas", "background_stroke_rgbas"):
            if name not in attributes:
                continue
            value = np.asarray(attributes[name], dtype=float).copy()
            if value.ndim >= 1 and value.shape[-1] >= 4:
                value[..., 3] = 0.0
            setattr(member, name, value)
        for name in ("stroke_opacity", "background_stroke_opacity"):
            if name in attributes:
                setattr(member, name, 0.0)


def _restore_original_sources(
    snapshots: Sequence[tuple[Mobject, Mapping[str, object]]],
) -> None:
    for member, attributes in snapshots:
        for name, value in attributes.items():
            setattr(member, name, value.copy() if isinstance(value, np.ndarray) else value)


class GeneratedOpenFaceVisibility3D(OpenFaceOcclusion3D):
    """Unified controller with the released generated-source lifecycle."""

    def __init__(
        self,
        *args: Any,
        shape: Mobject,
        proxy_root: VGroup,
        proxy_bindings: Mapping[str, Line],
        original_sources: Mapping[str, Sequence[Mobject]],
        geometry_state: Mapping[str, Any],
        detach_static_entry: Callable[[Mobject], object],
        restore_static_entry: Callable[[Mobject, object], None],
        preferred_painter_z_band: tuple[float, float],
        reservation_token: ScenePainterBandReservation,
        **kwargs: Any,
    ) -> None:
        self.shape = shape
        self._proxy_root = proxy_root
        self._proxy_bindings = dict(proxy_bindings)
        self._original_sources = {
            key: tuple(values) for key, values in original_sources.items()
        }
        self._geometry_state = geometry_state
        self._detach_static_entry = detach_static_entry
        self._restore_static_entry = restore_static_entry
        self.preferred_painter_z_band = preferred_painter_z_band
        self._reservation_token = reservation_token
        self._reservation_active = True
        self._static_entry: object | None = None
        self._original_snapshots: tuple[tuple[Mobject, dict[str, object]], ...] = ()
        super().__init__(*args, **kwargs)

    def _sync_proxy_geometry(self) -> None:
        positions = self._geometry_state["coordinates"]()
        display = self._geometry_state["project_scene"]
        for stroke in self.model.strokes:
            self._proxy_bindings[stroke.source_edge_id].put_start_and_end_on(
                _point3(display(positions[stroke.vertex_ids[0]]), stroke.vertex_ids[0]),
                _point3(display(positions[stroke.vertex_ids[1]]), stroke.vertex_ids[1]),
            )

    def _ensure_proxy_owned(self) -> bool:
        containers = getattr(self.scene, "mobjects", None)
        if not isinstance(containers, list):
            raise GeneratedOpenFaceVisibility3DError("Scene.mobjects must be a list")
        if not any(item is self._proxy_root for item in containers):
            containers.append(self._proxy_root)
            return True
        return False

    def _remove_proxy(self) -> None:
        for name in ("mobjects", "foreground_mobjects", "moving_mobjects", "static_mobjects"):
            values = getattr(self.scene, name, None)
            if isinstance(values, list):
                values[:] = [item for item in values if item is not self._proxy_root]

    def attach(self) -> "GeneratedOpenFaceVisibility3D":
        if self.attached:
            return self
        owner = getattr(self.shape, _OWNER_ATTRIBUTE, None)
        if owner is not None and owner is not self:
            raise GeneratedOpenFaceVisibility3DError(
                "TikZ ShapeState already has an open-face visibility owner"
            )
        if not self._reservation_active:
            actual = _reserve_band(self.scene, self._reservation_token)
            try:
                self.set_painter_z_band(actual)
            except Exception:
                _release_band(self.scene, self._reservation_token)
                raise
            self._reservation_active = True
        proxy_added = False
        static_entry_detached = False
        try:
            proxy_added = self._ensure_proxy_owned()
            self._sync_proxy_geometry()
            self._static_entry = self._detach_static_entry(self.shape)
            static_entry_detached = True
            self._original_snapshots = _capture_original_sources(
                self._original_sources
            )
            super().attach()
            _hide_original_sources(self._original_snapshots)
            setattr(self.shape, _OWNER_ATTRIBUTE, self)
            return self
        except Exception:
            # Preserve the original failure while undoing every preparation
            # step that may already have changed the Scene or ShapeState.
            if self.attached:
                try:
                    super().restore()
                except Exception:
                    pass
            try:
                _restore_original_sources(self._original_snapshots)
            except Exception:
                pass
            self._original_snapshots = ()
            if static_entry_detached:
                try:
                    self._restore_static_entry(self.shape, self._static_entry)
                except Exception:
                    pass
            self._static_entry = None
            if proxy_added:
                self._remove_proxy()
            if getattr(self.shape, _OWNER_ATTRIBUTE, None) is self:
                delattr(self.shape, _OWNER_ATTRIBUTE)
            _release_band(self.scene, self._reservation_token)
            self._reservation_active = False
            raise

    def update(self, dt: float = 0.0) -> "GeneratedOpenFaceVisibility3D":
        self._sync_proxy_geometry()
        try:
            super().update(dt)
        finally:
            if self.attached:
                _hide_original_sources(self._original_snapshots)
        return self

    def restore(self) -> "GeneratedOpenFaceVisibility3D":
        try:
            super().restore()
        finally:
            _restore_original_sources(self._original_snapshots)
            self._original_snapshots = ()
            self._remove_proxy()
            self._restore_static_entry(self.shape, self._static_entry)
            self._static_entry = None
            if getattr(self.shape, _OWNER_ATTRIBUTE, None) is self:
                delattr(self.shape, _OWNER_ATTRIBUTE)
            if self._reservation_active:
                _release_band(self.scene, self._reservation_token)
                self._reservation_active = False
        return self


def install_generated_open_face_visibility_3d(
    scene: object,
    shape: Mobject,
    objects: Mapping[str, Mobject],
    geometry_state: Mapping[str, Any],
    *,
    open_face_vertex_ids: object,
    open_face_faces: object,
    open_face_face_bindings: object,
    open_face_inclusive_edges: object,
    open_face_strokes: object,
    open_face_bindings: object,
    source_resolver: Callable[[Mapping[str, Mobject], Mapping[str, Any]], Mapping[str, Sequence[Mobject]]],
    face_source_resolver: Callable[[Mapping[str, Mobject]], Mapping[str, Polygon]],
    detach_static_entry: Callable[[Mobject], object],
    restore_static_entry: Callable[[Mobject, object], None],
    safe_length: Callable[[Mapping[str, Any], Mapping[str, Any]], float],
    projection_matrix: Callable[[], Sequence[Sequence[float]]],
    paint_policy: str,
    preferred_painter_z_band: object,
    visibility_group_id: str,
) -> GeneratedOpenFaceVisibility3D:
    """Install the unified controller through the released v3 data contract."""

    if geometry_state.get("shape") is not shape:
        raise GeneratedOpenFaceVisibility3DError(
            "open-face visibility received a foreign Geometry Rig state"
        )
    if getattr(shape, _OWNER_ATTRIBUTE, None) is not None:
        raise GeneratedOpenFaceVisibility3DError(
            "TikZ ShapeState already has an open-face visibility owner"
        )
    positions = geometry_state["coordinates"]()
    model = _build_model(
        visibility_group_id=visibility_group_id,
        vertex_ids=open_face_vertex_ids,
        faces=open_face_faces,
        inclusive_edges=open_face_inclusive_edges,
        strokes=open_face_strokes,
        positions=positions,
    )
    face_sources = dict(face_source_resolver(objects))
    original_sources = {
        str(key): tuple(values)
        for key, values in source_resolver(objects, geometry_state).items()
    }
    stroke_values = _mapping_sequence(open_face_strokes, "OPEN_FACE_STROKES")
    bindings = {
        str(item["source_edge_id"]): item
        for item in _mapping_sequence(open_face_bindings, "OPEN_FACE_BINDINGS")
    }
    expected = set(model.stroke_map)
    if set(bindings) != expected or set(original_sources) != expected:
        raise GeneratedOpenFaceVisibility3DError(
            "generated stroke constants and source bindings disagree"
        )
    scene_scale = float(geometry_state["scene_unit_per_cm"])
    width_scale = float(geometry_state["stroke_width_per_pt"])
    proxies: dict[str, Line] = {}
    styles: dict[str, OcclusionStyle] = {}
    for stroke in stroke_values:
        edge_id = str(stroke["source_edge_id"])
        proxy, style = _edge_style(
            bindings[edge_id],
            max_projected_length=float(safe_length(stroke, geometry_state)),
            scene_unit_per_cm=scene_scale,
            stroke_width_per_pt=width_scale,
        )
        proxies[edge_id] = proxy
        styles[edge_id] = style
    proxy_root = VGroup(*(proxies[key] for key in sorted(proxies)))
    preferred = _normalise_band(preferred_painter_z_band)
    token = ScenePainterBandReservation(
        owner_key=(
            "tikz-generated-open-face",
            str(visibility_group_id),
            id(shape),
        ),
        preferred_z_band=preferred,
    )
    actual = _reserve_band(scene, token)
    try:
        default_style = (
            styles[min(styles)]
            if styles
            else OcclusionStyle(max_projected_length=1.0)
        )
        controller = GeneratedOpenFaceVisibility3D(
            scene,
            model,
            position_provider=geometry_state["coordinates"],
            stroke_bindings=proxies,
            face_fill_bindings=face_sources,
            projection=ParallelProjection(lambda _scene: projection_matrix()),
            display_point_provider=geometry_state["project_scene"],
            style=default_style,
            stroke_styles=styles,
            source_coordinate_mode="display",
            compositing_mode="unified",
            paint_policy=_normalise_policy(paint_policy),
            painter_z_band=actual,
            shape=shape,
            proxy_root=proxy_root,
            proxy_bindings=proxies,
            original_sources=original_sources,
            geometry_state=geometry_state,
            detach_static_entry=detach_static_entry,
            restore_static_entry=restore_static_entry,
            preferred_painter_z_band=preferred,
            reservation_token=token,
        )
        return controller.attach()
    except Exception:
        _release_band(scene, token)
        raise


def restore_generated_open_face_visibility_3d(state: object) -> None:
    """Restore a generated controller; non-controller legacy values are ignored."""

    if isinstance(state, GeneratedOpenFaceVisibility3D):
        state.restore()


def _strip_override(source: str) -> str:
    start = source.find(_OVERRIDE_BEGIN)
    if start < 0:
        return source
    end = source.find(_OVERRIDE_END, start)
    if end < 0:
        raise GeneratedOpenFaceVisibility3DError(
            "generated unified override marker is unterminated"
        )
    end += len(_OVERRIDE_END)
    while end < len(source) and source[end] in "\r\n":
        end += 1
    return source[:start].rstrip() + "\n" + source[end:]


def _controller_bindings(tree: ast.Module) -> tuple[int, tuple[str, ...]]:
    result: list[str] = []
    call_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "install_open_face_visibility_3d"
        ):
            call_count += 1
    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: Sequence[ast.AST] = ()
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign):
            value, targets = node.value, (node.target,)
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        if value.func.id != "install_open_face_visibility_3d":
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                result.append(target.id)
    return call_count, tuple(result)


def _rewrite_exact_fades(
    source: str,
    *,
    targets: Sequence[str],
    controller_names: Sequence[str],
    controller_call_count: int,
) -> str:
    if not targets:
        return source
    if controller_call_count != 1 or len(controller_names) != 1:
        raise GeneratedOpenFaceVisibility3DError(
            "whole-figure Fade rewrite requires exactly one directly assigned "
            "generated controller"
        )
    tree = ast.parse(source)
    replacements: list[tuple[int, int, bytes]] = []
    encoded = source.encode("utf-8")
    lines = source.splitlines(keepends=True)
    byte_starts = []
    current = 0
    for line in lines:
        byte_starts.append(current)
        current += len(line.encode("utf-8"))

    def offset(node: ast.AST, end: bool = False) -> int:
        line = (node.end_lineno if end else node.lineno) - 1  # type: ignore[attr-defined]
        column = node.end_col_offset if end else node.col_offset  # type: ignore[attr-defined]
        return byte_starts[line] + column

    target_set = set(targets)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        first = node.args[0]
        if name not in {"FadeIn", "FadeOut"} or not isinstance(first, ast.Name):
            continue
        if first.id not in target_set:
            continue
        replacement = f"{controller_names[0]}.display_mobject".encode("utf-8")
        replacements.append((offset(first), offset(first, True), replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode("utf-8")


def _override_block(
    *,
    paint_policy: str,
    painter_z_band: tuple[float, float],
) -> str:
    return f'''{_OVERRIDE_BEGIN}
from tikz_native.generated_open_face_visibility_3d import (
    install_generated_open_face_visibility_3d as _tikz_native_install_generated_open_face_visibility_3d,
    restore_generated_open_face_visibility_3d as _tikz_native_restore_generated_open_face_visibility_3d,
)

def install_open_face_visibility_3d(scene, shape, objects, geometry_state):
    return _tikz_native_install_generated_open_face_visibility_3d(
        scene,
        shape,
        objects,
        geometry_state,
        open_face_vertex_ids=OPEN_FACE_VERTEX_IDS,
        open_face_faces=OPEN_FACE_FACES,
        open_face_face_bindings=OPEN_FACE_FACE_BINDINGS,
        open_face_inclusive_edges=OPEN_FACE_INCLUSIVE_EDGES,
        open_face_strokes=OPEN_FACE_STROKES,
        open_face_bindings=OPEN_FACE_BINDINGS,
        source_resolver=_open_face_sources,
        face_source_resolver=_open_face_face_sources,
        detach_static_entry=_open_face_detach_static_entry,
        restore_static_entry=_open_face_restore_static_entry,
        safe_length=_open_face_safe_length,
        projection_matrix=lambda: local_camera_matrix(geometry_state),
        paint_policy={paint_policy!r},
        preferred_painter_z_band={painter_z_band!r},
        visibility_group_id="generated:" + OPEN_FACE_MODEL_SHA256,
    )

def restore_open_face_visibility_3d(state):
    _tikz_native_restore_generated_open_face_visibility_3d(state)

{_OVERRIDE_END}
'''


def rewrite_legacy_open_face_source(
    source: str,
    *,
    paint_policy: str,
    preferred_painter_z_band: object,
    whole_figure_targets: Sequence[str] = (),
) -> str:
    """Append an idempotent unified override to one real generated v3 source."""

    if not isinstance(source, str):
        raise GeneratedOpenFaceVisibility3DError("generated source must be text")
    base = _strip_override(source)
    try:
        tree = ast.parse(base)
    except SyntaxError as exc:
        raise GeneratedOpenFaceVisibility3DError(
            f"generated v3 source is invalid at line {exc.lineno}: {exc.msg}"
        ) from exc
    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            (*node.targets,) if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    required_functions = {
        "install_open_face_visibility_3d",
        "restore_open_face_visibility_3d",
        "_open_face_sources",
        "_open_face_face_sources",
        "_open_face_detach_static_entry",
        "_open_face_restore_static_entry",
        "_open_face_safe_length",
        "local_camera_matrix",
    }
    required_constants = {
        "OPEN_FACE_VERTEX_IDS",
        "OPEN_FACE_FACES",
        "OPEN_FACE_FACE_BINDINGS",
        "OPEN_FACE_INCLUSIVE_EDGES",
        "OPEN_FACE_STROKES",
        "OPEN_FACE_BINDINGS",
        "OPEN_FACE_MODEL_SHA256",
    }
    missing = sorted(
        (required_functions - function_names) | (required_constants - assigned_names)
    )
    if missing:
        raise GeneratedOpenFaceVisibility3DError(
            "generated source is not the real open-face v3 contract; missing: "
            + ", ".join(missing)
        )
    controller_call_count, controller_names = _controller_bindings(tree)
    base = _rewrite_exact_fades(
        base,
        targets=tuple(whole_figure_targets),
        controller_names=controller_names,
        controller_call_count=controller_call_count,
    )
    policy = _normalise_policy(paint_policy)
    band = _normalise_band(preferred_painter_z_band)
    rewritten = base.rstrip() + "\n\n" + _override_block(
        paint_policy=policy,
        painter_z_band=band,
    )
    try:
        compile(rewritten, "<generated-open-face-v3-unified>", "exec")
    except SyntaxError as exc:
        raise GeneratedOpenFaceVisibility3DError(
            "generated v3 source becomes invalid after unified adaptation at "
            f"line {exc.lineno}: {exc.msg}"
        ) from exc
    return rewritten


__all__ = [
    "GeneratedOpenFaceVisibility3D",
    "GeneratedOpenFaceVisibility3DError",
    "install_generated_open_face_visibility_3d",
    "restore_generated_open_face_visibility_3d",
    "rewrite_legacy_open_face_source",
]

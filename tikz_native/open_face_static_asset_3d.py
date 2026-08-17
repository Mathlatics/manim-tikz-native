from __future__ import annotations

"""Bake one proven open-face entry frame into a normal ``NativeFigure``.

The legacy asset compiler remains unchanged.  A Host that explicitly opts a
ShapeAsset into automatic occlusion first compiles the ordinary semantic
figure, then calls this independently versioned adapter.  The original
semantic Mobjects stay in ``figure.objects`` for Geometry Rig authoring while
one frozen overlay becomes a child of ``figure.group`` for static rendering.
"""

from hashlib import sha256
from math import isfinite
import re
from statistics import median
from typing import Any, Mapping

from manim import Scene, tempconfig

from polyhedron_visibility import OcclusionStyle
from polyhedron_visibility.open_faces import canonical_open_face_trace_json

from .manim_renderer import NativeFigure
from .open_face_visibility_3d_manim import (
    TikzNativeOpenFaceVisibility3DManimError,
    _release_figure_owner,
    bind_picture_open_face_visibility_3d,
)
from .version import (
    COMPONENT_ASSET_COMPILER,
    COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    COMPONENT_OPEN_FACE_VISIBILITY,
    COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
    provider_component_contract_revision,
    provider_component_revision,
)


OPEN_FACE_STATIC_ASSET_3D_SCHEMA_V1 = (
    "latex-ppt-tikz-native-open-face-static-asset/v1"
)
OPEN_FACE_STATIC_ASSET_3D_SCHEMA = (
    "latex-ppt-tikz-native-open-face-static-asset/v2"
)
OPEN_FACE_STATIC_ASSET_3D_COMPATIBILITY_SCHEMA = (
    "tikz-native-artifact-compatibility/v1"
)
OPEN_FACE_STATIC_ENTRY_3D_SCHEMA = "tikz-native-open-face-static-entry-3d/v2"
OPEN_FACE_STATIC_ASSET_3D_COMPONENT = "tikz_open_face_static_asset_3d"
OPEN_FACE_STATIC_ASSET_3D_CAPABILITY = "tikz_open_face_static_asset_3d_v1"
OPEN_FACE_STATIC_ASSET_3D_MODE = "open_convex_faces_parallel"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^(?:source|component)-sha256:[0-9a-f]{64}$")
_BASE_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "sourceSha256",
        "pictureIndex",
        "entryMacro",
        "modelSha256",
        "entryTraceSha256",
        "adapterResultSha256",
        "faceCount",
        "strokeCount",
        "seamCount",
    }
)
_V1_CONTRACT_FIELDS = _BASE_CONTRACT_FIELDS | {"componentRevisions"}
_V2_CONTRACT_FIELDS = _BASE_CONTRACT_FIELDS | {"compatibility"}
_LEGACY_COMPONENT_FIELDS = frozenset(
    {
        COMPONENT_ASSET_COMPILER,
        COMPONENT_OPEN_FACE_VISIBILITY,
        COMPONENT_TIKZ_OPEN_FACE_VISIBILITY_3D,
        OPEN_FACE_STATIC_ASSET_3D_COMPONENT,
        COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    }
)
_COMPATIBILITY_FIELDS = frozenset(
    {"schema", "contractRevisions", "renderRevisions", "buildRevision"}
)
_CONTRACT_COMPONENT_FIELDS = frozenset(
    {
        OPEN_FACE_STATIC_ASSET_3D_COMPONENT,
        COMPONENT_NATIVE_MANIM_SOURCE_3D_V3,
    }
)
_CONTRACT_REVISION_RE = re.compile(
    r"^tikz-native-contract:[a-z0-9_]+/v[1-9][0-9]*$"
)


class TikzNativeOpenFaceStaticAsset3DError(ValueError):
    """The explicit static automatic-occlusion contract is not current."""


def _digest(value: object, field: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise TikzNativeOpenFaceStaticAsset3DError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return result


def _revision(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not _REVISION_RE.fullmatch(result):
        raise TikzNativeOpenFaceStaticAsset3DError(
            f"{field} must be a Provider component revision"
        )
    return result


def _contract_revision(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not _CONTRACT_REVISION_RE.fullmatch(result):
        raise TikzNativeOpenFaceStaticAsset3DError(
            f"{field} must be a Provider component contract revision"
        )
    return result


def _validate_v2_compatibility(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMPATIBILITY_FIELDS:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion compatibility metadata is incomplete"
        )
    if value.get("schema") != OPEN_FACE_STATIC_ASSET_3D_COMPATIBILITY_SCHEMA:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion compatibility schema is unsupported"
        )
    raw_contracts = value.get("contractRevisions")
    raw_renders = value.get("renderRevisions")
    if (
        not isinstance(raw_contracts, Mapping)
        or set(raw_contracts) != _CONTRACT_COMPONENT_FIELDS
        or not isinstance(raw_renders, Mapping)
        or set(raw_renders) != _CONTRACT_COMPONENT_FIELDS
    ):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion compatibility component scope is incomplete"
        )
    contracts = {
        component: _contract_revision(raw_contracts.get(component), component)
        for component in sorted(_CONTRACT_COMPONENT_FIELDS)
    }
    renders = {
        component: _revision(raw_renders.get(component), component)
        for component in sorted(_CONTRACT_COMPONENT_FIELDS)
    }
    for component, recorded in contracts.items():
        current = provider_component_contract_revision(component)
        if recorded != current:
            raise TikzNativeOpenFaceStaticAsset3DError(
                f"automaticOcclusion contract {component!r} is incompatible"
            )
    return {
        "schema": OPEN_FACE_STATIC_ASSET_3D_COMPATIBILITY_SCHEMA,
        "contractRevisions": contracts,
        "renderRevisions": renders,
        "buildRevision": _revision(value.get("buildRevision"), "buildRevision"),
    }


def validate_open_face_static_asset_3d_contract(
    value: object,
    *,
    source_sha256: object | None = None,
    picture_index: object | None = None,
    entry_macro: object | None = None,
) -> dict[str, Any]:
    """Return one exact, portable contract or fail before mutating a figure."""

    if not isinstance(value, Mapping):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion is not a static open-face contract"
        )
    schema = value.get("schema")
    if schema == OPEN_FACE_STATIC_ASSET_3D_SCHEMA_V1:
        expected_fields = _V1_CONTRACT_FIELDS
    elif schema == OPEN_FACE_STATIC_ASSET_3D_SCHEMA:
        expected_fields = _V2_CONTRACT_FIELDS
    else:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion uses an unsupported schema"
        )
    if set(value) != expected_fields:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion fields do not match its declared schema"
        )
    if value.get("mode") != OPEN_FACE_STATIC_ASSET_3D_MODE:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion uses an unsupported mode"
        )
    declared_source = _digest(value.get("sourceSha256"), "sourceSha256")
    if source_sha256 is not None and declared_source != _digest(
        source_sha256, "current sourceSha256"
    ):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion source identity is stale"
        )
    index = value.get("pictureIndex")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion.pictureIndex must be a positive integer"
        )
    if picture_index is not None and index != int(picture_index):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion picture identity is stale"
        )
    macro = value.get("entryMacro")
    if not isinstance(macro, str):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion.entryMacro must be a string"
        )
    if entry_macro is not None and macro != str(entry_macro or ""):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "automaticOcclusion entry macro identity is stale"
        )
    counts: dict[str, int] = {}
    for field, minimum, maximum in (
        ("faceCount", 1, 64),
        ("strokeCount", 1, 128),
        ("seamCount", 0, 64),
    ):
        count = value.get(field)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not minimum <= count <= maximum
        ):
            raise TikzNativeOpenFaceStaticAsset3DError(
                f"automaticOcclusion.{field} is out of range"
            )
        counts[field] = count
    compatibility: dict[str, Any] | None = None
    components: dict[str, str] | None = None
    if schema == OPEN_FACE_STATIC_ASSET_3D_SCHEMA_V1:
        raw_components = value.get("componentRevisions")
        if (
            not isinstance(raw_components, Mapping)
            or set(raw_components) != _LEGACY_COMPONENT_FIELDS
        ):
            raise TikzNativeOpenFaceStaticAsset3DError(
                "automaticOcclusion component revisions are incomplete"
            )
        components = {
            component: _revision(raw_components.get(component), component)
            for component in sorted(_LEGACY_COMPONENT_FIELDS)
        }
        for component, recorded in components.items():
            current = provider_component_revision(component)
            if recorded != current:
                raise TikzNativeOpenFaceStaticAsset3DError(
                    f"automaticOcclusion component {component!r} is stale"
                )
    else:
        compatibility = _validate_v2_compatibility(value.get("compatibility"))
    result = {
        "schema": schema,
        "mode": OPEN_FACE_STATIC_ASSET_3D_MODE,
        "sourceSha256": declared_source,
        "pictureIndex": index,
        "entryMacro": macro,
        "modelSha256": _digest(value.get("modelSha256"), "modelSha256"),
        "entryTraceSha256": _digest(
            value.get("entryTraceSha256"), "entryTraceSha256"
        ),
        "adapterResultSha256": _digest(
            value.get("adapterResultSha256"), "adapterResultSha256"
        ),
        **counts,
    }
    if components is not None:
        result["componentRevisions"] = components
    if compatibility is not None:
        result["compatibility"] = compatibility
    return result


def _entry_trace_sha256(frame: object) -> str:
    return sha256(canonical_open_face_trace_json(frame).encode("utf-8")).hexdigest()


def _binding_stroke_width_per_pt(binding: object) -> float:
    """Freeze the real Manim stroke scale before the source lines are hidden."""

    controller = getattr(binding, "controller", None)
    analysis = getattr(binding, "analysis", None)
    resolved_styles = getattr(controller, "resolved_styles", None)
    stroke_bindings = getattr(analysis, "stroke_bindings", None)
    if not isinstance(resolved_styles, Mapping) or not isinstance(
        stroke_bindings, tuple
    ):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "static open-face binding did not expose its resolved stroke scale"
        )
    ratios: list[float] = []
    for stroke_binding in stroke_bindings:
        edge_id = str(getattr(stroke_binding, "source_edge_id", ""))
        visible_style = getattr(stroke_binding, "visible_style", None)
        resolved = resolved_styles.get(edge_id)
        if not isinstance(visible_style, Mapping) or resolved is None:
            raise TikzNativeOpenFaceStaticAsset3DError(
                f"static open-face stroke {edge_id!r} lost its resolved style"
            )
        try:
            width_pt = float(visible_style.get("lineWidthPt"))
            resolved_width = float(getattr(resolved, "visible_width"))
        except (TypeError, ValueError) as exc:
            raise TikzNativeOpenFaceStaticAsset3DError(
                f"static open-face stroke {edge_id!r} has an invalid width"
            ) from exc
        if (
            not isfinite(width_pt)
            or width_pt <= 0.0
            or not isfinite(resolved_width)
            or resolved_width <= 0.0
        ):
            raise TikzNativeOpenFaceStaticAsset3DError(
                f"static open-face stroke {edge_id!r} has no positive width"
            )
        ratios.append(resolved_width / width_pt)
    if not ratios:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "static open-face entry has no measurable semantic stroke"
        )
    reference = float(median(ratios))
    if any(
        abs(value - reference) > 1.0e-7 * max(reference, value)
        for value in ratios
    ):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "static open-face strokes do not share one Manim width scale"
        )
    return reference


def bake_open_face_static_entry_3d(
    figure: NativeFigure,
    contract: object,
    *,
    source_sha256: object,
    picture_index: object,
    entry_macro: object = "",
) -> NativeFigure:
    """Bake the entry overlay without replacing semantic object identities."""

    normalized = validate_open_face_static_asset_3d_contract(
        contract,
        source_sha256=source_sha256,
        picture_index=picture_index,
        entry_macro=entry_macro,
    )
    if not isinstance(figure, NativeFigure):
        raise TikzNativeOpenFaceStaticAsset3DError(
            "static open-face conversion requires one NativeFigure"
        )
    if getattr(figure.group, "_mathppt_open_face_static_entry", None) is not None:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "NativeFigure already contains a static open-face entry"
        )

    width = float(getattr(figure.group, "width", 0.0))
    height = float(getattr(figure.group, "height", 0.0))
    max_length = width + height
    if not isfinite(max_length) or max_length <= 0.0:
        raise TikzNativeOpenFaceStaticAsset3DError(
            "NativeFigure has no finite entry-frame extent"
        )

    with tempconfig({"renderer": "cairo"}):
        scene = Scene()
        scene.add(figure.group)
        binding = bind_picture_open_face_visibility_3d(
            scene,
            figure.picture,
            figure,
            style=OcclusionStyle(max_projected_length=max_length + 1.0e-6),
        )
        stroke_width_per_pt = _binding_stroke_width_per_pt(binding)
        overlay_root = binding.controller.overlay_root
        try:
            binding.attach()
            analysis = binding.analysis
            frame = binding.last_frame
            if frame is None:
                raise TikzNativeOpenFaceStaticAsset3DError(
                    "static open-face binding produced no entry trace"
                )
            if (
                analysis.model_sha256 != normalized["modelSha256"]
                or analysis.entry_trace_sha256
                != normalized["entryTraceSha256"]
                or analysis.result_sha256
                != normalized["adapterResultSha256"]
                or _entry_trace_sha256(frame)
                != normalized["entryTraceSha256"]
                or len(analysis.model.faces) != normalized["faceCount"]
                or len(analysis.model.strokes) != normalized["strokeCount"]
                or len(analysis.model.seams) != normalized["seamCount"]
            ):
                raise TikzNativeOpenFaceStaticAsset3DError(
                    "static open-face entry no longer matches its frozen trace"
                )

            controller = binding.controller
            overlay_root.clear_updaters(recursive=False)
            controller._attached = False
            controller._remove_fixed_frame()
            controller._remove_overlay_identity()
            figure.group.add(overlay_root)

            element_objects = dict(figure.objects)
            slot_roots = {
                edge_id: slots.root
                for edge_id, slots in controller._slots.items()
            }
            for stroke_binding in analysis.stroke_bindings:
                root = slot_roots[stroke_binding.source_edge_id]
                for object_id in stroke_binding.object_ids:
                    element_objects[object_id] = root
            figure.group._codex_tikz_native_element_objects = element_objects
            figure.group._mathppt_open_face_static_entry = {
                "schema": OPEN_FACE_STATIC_ENTRY_3D_SCHEMA,
                "contractSchema": normalized["schema"],
                "sourceSha256": normalized["sourceSha256"],
                "modelSha256": normalized["modelSha256"],
                "entryTraceSha256": normalized["entryTraceSha256"],
                "adapterResultSha256": normalized["adapterResultSha256"],
                "strokeWidthPerPt": stroke_width_per_pt,
                "overlayRoot": overlay_root,
            }
            controller._snapshots = {}
            if controller._owner_claimed:
                _release_figure_owner(controller)
        except Exception:
            figure.group.remove(overlay_root)
            if binding.controller.attached or binding.controller._owner_claimed:
                binding.restore()
            raise
    return figure


__all__ = [
    "OPEN_FACE_STATIC_ASSET_3D_CAPABILITY",
    "OPEN_FACE_STATIC_ASSET_3D_COMPATIBILITY_SCHEMA",
    "OPEN_FACE_STATIC_ASSET_3D_COMPONENT",
    "OPEN_FACE_STATIC_ASSET_3D_MODE",
    "OPEN_FACE_STATIC_ASSET_3D_SCHEMA",
    "OPEN_FACE_STATIC_ASSET_3D_SCHEMA_V1",
    "OPEN_FACE_STATIC_ENTRY_3D_SCHEMA",
    "TikzNativeOpenFaceStaticAsset3DError",
    "bake_open_face_static_entry_3d",
    "validate_open_face_static_asset_3d_contract",
]

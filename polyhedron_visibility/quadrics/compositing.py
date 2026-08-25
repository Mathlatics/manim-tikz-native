"""Renderer-neutral painter graph for opaque quadrics and curve spans.

The exact visibility kernel decides whether each analytic curve interval is
visible or hidden.  Projection proxies provide display-only opaque fills.  This
module combines those two results into one deterministic far-to-near painter
graph without importing Manim or using the proxy as geometric evidence.

Three policies are explicit:

``physical``
    Hidden curve spans remain in the trace but are not paint items.

``diagrammatic``
    Hidden curve spans are dashed teaching overlays painted above every
    surface, just like visible curve spans.

``depth_aware_diagrammatic``
    Hidden curve spans remain dashed, but are painted behind the surfaces that
    mathematically occlude them.  A section compositor can refine that depth
    relation by placing the dash between its coincident back/front sheets.

Surface-to-surface order is accepted only as explicit caller evidence.  The
first version deliberately does not split intersecting surface proxies; any
cycle therefore fails closed instead of choosing an unstable order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from typing import Mapping, Sequence

from ..compositor import (
    CompositorCycleError,
    PainterConstraint,
    stable_topological_sort,
)
from ..style import OcclusionStyle
from ..topology import ParameterInterval
from ..visibility import VisibilityKind
from .projection import OpaqueProjectionProxy
from .curve_intersections import ProjectedCurveCrossing
from .visibility import CurveVisibilityFrame, CurveVisibilityRecord


QUADRIC_COMPOSITING_FRAME_SCHEMA = "manim-quadric-compositing-frame/v1"


class QuadricCompositingError(ValueError):
    """A complete, deterministic quadric painter graph cannot be formed."""


class QuadricPaintPolicy(str, Enum):
    """How mathematically hidden curve spans are represented in the picture."""

    PHYSICAL = "physical"
    DIAGRAMMATIC = "diagrammatic"
    DEPTH_AWARE_DIAGRAMMATIC = "depth_aware_diagrammatic"


class QuadricPaintKind(str, Enum):
    SURFACE = "surface"
    VISIBLE_CURVE = "visible_curve"
    HIDDEN_CURVE = "hidden_curve"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricCompositingError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise QuadricCompositingError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuadricCompositingError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise QuadricCompositingError(f"{label} must be finite")
    return result


def _style_token(value: object, label: str) -> object:
    """Return a deterministic JSON value for one optional style override.

    Manim colors and enum-like cap/joint values commonly expose ``to_hex`` or
    ``value``.  Unknown mutable renderer objects are rejected rather than
    leaking process-specific ``repr`` text into a supposedly canonical trace.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value, label)
    if isinstance(value, (tuple, list)):
        return [_style_token(item, f"{label}[]") for item in value]
    to_hex = getattr(value, "to_hex", None)
    if callable(to_hex):
        token = to_hex()
        if isinstance(token, str) and token:
            return token
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _style_token(enum_value, label)
    raise QuadricCompositingError(
        f"{label} is not canonically serializable; use a string or numeric token"
    )


@dataclass(frozen=True, slots=True)
class QuadricStyleDescriptor:
    """Serializable renderer-neutral subset of :class:`OcclusionStyle`."""

    style_id: str
    max_projected_length: float
    dash_length: float
    dash_gap: float
    visible_color: object | None
    hidden_color: object | None
    visible_width_scale: float
    hidden_width_scale: float
    visible_opacity_scale: float
    hidden_opacity_scale: float
    hidden_cap_style: object | None
    hidden_joint_type: object | None

    @classmethod
    def from_style(
        cls,
        style_id: str,
        style: OcclusionStyle,
    ) -> "QuadricStyleDescriptor":
        if not isinstance(style, OcclusionStyle):
            raise TypeError("curve styles must be OcclusionStyle objects")
        return cls(
            _identity(style_id, "style_id"),
            _finite(style.max_projected_length, "max_projected_length"),
            _finite(style.dash_length, "dash_length"),
            _finite(style.dash_gap, "dash_gap"),
            _style_token(style.visible_color, "visible_color"),
            _style_token(style.hidden_color, "hidden_color"),
            _finite(style.visible_width_scale, "visible_width_scale"),
            _finite(style.hidden_width_scale, "hidden_width_scale"),
            _finite(style.visible_opacity_scale, "visible_opacity_scale"),
            _finite(style.hidden_opacity_scale, "hidden_opacity_scale"),
            _style_token(style.hidden_cap_style, "hidden_cap_style"),
            _style_token(style.hidden_joint_type, "hidden_joint_type"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "styleId": self.style_id,
            "maxProjectedLength": self.max_projected_length,
            "dashLength": self.dash_length,
            "dashGap": self.dash_gap,
            "visibleColor": self.visible_color,
            "hiddenColor": self.hidden_color,
            "visibleWidthScale": self.visible_width_scale,
            "hiddenWidthScale": self.hidden_width_scale,
            "visibleOpacityScale": self.visible_opacity_scale,
            "hiddenOpacityScale": self.hidden_opacity_scale,
            "hiddenCapStyle": self.hidden_cap_style,
            "hiddenJointType": self.hidden_joint_type,
        }


@dataclass(frozen=True, slots=True)
class QuadricSurfacePaintItem:
    """One opaque display proxy registered as a painter-graph item."""

    item_id: str
    proxy: OpaqueProjectionProxy
    kind: QuadricPaintKind = QuadricPaintKind.SURFACE

    def __post_init__(self) -> None:
        item_id = _identity(self.item_id, "surface item_id")
        if not isinstance(self.proxy, OpaqueProjectionProxy):
            raise TypeError("proxy must be an OpaqueProjectionProxy")
        if self.kind is not QuadricPaintKind.SURFACE:
            raise QuadricCompositingError("surface paint item has an invalid kind")
        object.__setattr__(self, "item_id", item_id)

    @property
    def surface_id(self) -> str:
        return self.proxy.surface_id

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "kind": self.kind.value,
            "surfaceId": self.surface_id,
            "proxy": self.proxy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class QuadricCurvePaintFragment:
    """One exact visibility span and its renderer intent."""

    item_id: str
    curve_id: str
    span_index: int
    interval: ParameterInterval
    kind: QuadricPaintKind
    occluder_surface_ids: tuple[str, ...]
    painted: bool
    render_intent: str
    style_id: str | None = None

    def __post_init__(self) -> None:
        item_id = _identity(self.item_id, "curve fragment item_id")
        curve_id = _identity(self.curve_id, "curve_id")
        if isinstance(self.span_index, bool) or not isinstance(self.span_index, int):
            raise QuadricCompositingError("span_index must be a non-negative integer")
        if self.span_index < 0:
            raise QuadricCompositingError("span_index must be a non-negative integer")
        if not isinstance(self.interval, ParameterInterval):
            raise TypeError("interval must be a ParameterInterval")
        if self.kind not in (
            QuadricPaintKind.VISIBLE_CURVE,
            QuadricPaintKind.HIDDEN_CURVE,
        ):
            raise QuadricCompositingError("curve fragment has an invalid kind")
        occluders = tuple(
            _identity(owner, "occluder surface identity")
            for owner in self.occluder_surface_ids
        )
        if occluders != tuple(sorted(set(occluders))):
            raise QuadricCompositingError(
                "occluder_surface_ids must be unique and sorted"
            )
        if not isinstance(self.painted, bool):
            raise TypeError("painted must be a bool")
        intent = _identity(self.render_intent, "render_intent")
        if self.kind is QuadricPaintKind.VISIBLE_CURVE:
            if occluders or not self.painted or intent != "solid":
                raise QuadricCompositingError(
                    "visible fragments must be painted solid without occluders"
                )
        elif self.painted:
            if not occluders or intent != "dashed":
                raise QuadricCompositingError(
                    "painted hidden fragments must be dashed and name occluders"
                )
        elif not occluders or intent != "omit":
            raise QuadricCompositingError(
                "omitted hidden fragments must name occluders and use omit intent"
            )
        style_id = (
            None if self.style_id is None else _identity(self.style_id, "style_id")
        )
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "occluder_surface_ids", occluders)
        object.__setattr__(self, "render_intent", intent)
        object.__setattr__(self, "style_id", style_id)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "itemId": self.item_id,
            "curveId": self.curve_id,
            "spanIndex": self.span_index,
            "interval": [self.interval.start, self.interval.end],
            "kind": self.kind.value,
            "occluderSurfaceIds": list(self.occluder_surface_ids),
            "painted": self.painted,
            "renderIntent": self.render_intent,
        }
        if self.style_id is not None:
            result["styleId"] = self.style_id
        return result


@dataclass(frozen=True, slots=True)
class QuadricPaintRelation:
    """One explicit far-to-near edge in the unified painter graph."""

    far_item_id: str
    near_item_id: str
    reason: str

    def __post_init__(self) -> None:
        far = _identity(self.far_item_id, "far_item_id")
        near = _identity(self.near_item_id, "near_item_id")
        reason = _identity(self.reason, "paint relation reason")
        if far == near:
            raise QuadricCompositingError(
                "a painter relation cannot order one item against itself"
            )
        object.__setattr__(self, "far_item_id", far)
        object.__setattr__(self, "near_item_id", near)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "farItemId": self.far_item_id,
            "nearItemId": self.near_item_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuadricCompositingFrame:
    """A complete deterministic painter trace for one visibility frame."""

    visibility: CurveVisibilityFrame
    paint_policy: QuadricPaintPolicy
    surface_items: tuple[QuadricSurfacePaintItem, ...]
    curve_fragments: tuple[QuadricCurvePaintFragment, ...]
    styles: tuple[QuadricStyleDescriptor, ...]
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    curve_crossings: tuple[ProjectedCurveCrossing, ...] = ()
    schema: str = QUADRIC_COMPOSITING_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_COMPOSITING_FRAME_SCHEMA:
            raise QuadricCompositingError("invalid quadric-compositing schema")
        if not isinstance(self.visibility, CurveVisibilityFrame):
            raise TypeError("visibility must be a CurveVisibilityFrame")
        if not isinstance(self.paint_policy, QuadricPaintPolicy):
            raise TypeError("paint_policy must be a QuadricPaintPolicy")
        if not all(
            isinstance(item, QuadricSurfacePaintItem) for item in self.surface_items
        ):
            raise TypeError("surface_items must contain QuadricSurfacePaintItem")
        if not all(
            isinstance(item, QuadricCurvePaintFragment)
            for item in self.curve_fragments
        ):
            raise TypeError(
                "curve_fragments must contain QuadricCurvePaintFragment"
            )
        if not all(isinstance(item, QuadricStyleDescriptor) for item in self.styles):
            raise TypeError("styles must contain QuadricStyleDescriptor")
        if not all(
            isinstance(item, QuadricPaintRelation) for item in self.order_relations
        ):
            raise TypeError("order_relations must contain QuadricPaintRelation")
        if not all(
            isinstance(item, ProjectedCurveCrossing)
            for item in self.curve_crossings
        ):
            raise TypeError(
                "curve_crossings must contain ProjectedCurveCrossing objects"
            )

        surface_keys = tuple(
            (item.surface_id, item.proxy.patch_id) for item in self.surface_items
        )
        if surface_keys != tuple(sorted(surface_keys)):
            raise QuadricCompositingError("surface_items must be sorted")
        surface_ids = tuple(item.surface_id for item in self.surface_items)
        if len(set(surface_ids)) != len(surface_ids):
            raise QuadricCompositingError(
                "exactly one opaque projection proxy is required per surface"
            )
        if surface_ids != self.visibility.surface_ids:
            raise QuadricCompositingError(
                "surface proxies must cover visibility.surface_ids exactly"
            )

        fragment_ids = tuple(item.item_id for item in self.curve_fragments)
        if fragment_ids != tuple(sorted(fragment_ids)):
            raise QuadricCompositingError("curve_fragments must be sorted")
        if len(set(fragment_ids)) != len(fragment_ids):
            raise QuadricCompositingError(
                "curve fragment identities must be unique"
            )
        expected_span_count = sum(
            len(record.spans) for record in self.visibility.records
        )
        if len(self.curve_fragments) != expected_span_count:
            raise QuadricCompositingError(
                "curve_fragments must preserve every visibility span"
            )
        crossing_ids = tuple(item.crossing_id for item in self.curve_crossings)
        if crossing_ids != tuple(sorted(crossing_ids)) or len(set(crossing_ids)) != len(
            crossing_ids
        ):
            raise QuadricCompositingError(
                "curve_crossings must have unique sorted identities"
            )
        record_map = self.visibility.record_map
        for crossing in self.curve_crossings:
            for curve_id, parameter in (
                (crossing.first_curve_id, crossing.first_parameter),
                (crossing.second_curve_id, crossing.second_parameter),
            ):
                record = record_map.get(curve_id)
                if record is None:
                    raise QuadricCompositingError(
                        f"curve crossing references unknown curve {curve_id!r}"
                    )
                if not record.domain.contains(
                    parameter, tolerance=record.parameter_tolerance
                ):
                    raise QuadricCompositingError(
                        f"curve crossing lies outside {curve_id!r} domain"
                    )

        style_ids = tuple(item.style_id for item in self.styles)
        if style_ids != tuple(sorted(style_ids)) or len(set(style_ids)) != len(
            style_ids
        ):
            raise QuadricCompositingError("styles must have unique sorted identities")
        known_styles = set(style_ids)
        if any(
            item.style_id is not None and item.style_id not in known_styles
            for item in self.curve_fragments
        ):
            raise QuadricCompositingError(
                "curve fragment references an unknown style identity"
            )

        active_ids = {
            *(item.item_id for item in self.surface_items),
            *(
                item.item_id
                for item in self.curve_fragments
                if item.painted
            ),
        }
        if len(active_ids) != len(self.surface_items) + sum(
            item.painted for item in self.curve_fragments
        ):
            raise QuadricCompositingError("active paint item identities must be unique")
        if len(self.draw_order) != len(set(self.draw_order)) or set(
            self.draw_order
        ) != active_ids:
            raise QuadricCompositingError(
                "draw_order must contain every active paint item exactly once"
            )

        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in self.order_relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise QuadricCompositingError("order_relations must be sorted")
        pairs: set[tuple[str, str]] = set()
        for relation in self.order_relations:
            if (
                relation.far_item_id not in active_ids
                or relation.near_item_id not in active_ids
            ):
                raise QuadricCompositingError(
                    "paint relation references an inactive or unknown item"
                )
            pair = (relation.far_item_id, relation.near_item_id)
            if pair in pairs:
                raise QuadricCompositingError("duplicate painter relation")
            if tuple(reversed(pair)) in pairs:
                raise QuadricCompositingError(
                    "paint items have contradictory direct relations"
                )
            pairs.add(pair)
        try:
            expected_order = stable_topological_sort(
                sorted(active_ids),
                (
                    PainterConstraint(item.far_item_id, item.near_item_id)
                    for item in self.order_relations
                ),
                key=lambda item_id: item_id,
            )
        except CompositorCycleError as exc:
            raise QuadricCompositingError(
                "quadric painter graph contains a cycle: "
                + ", ".join(sorted(str(item) for item in exc.unresolved))
            ) from exc
        if self.draw_order != expected_order:
            raise QuadricCompositingError(
                "draw_order is not the canonical order of its painter graph"
            )

    @property
    def item_ids(self) -> tuple[str, ...]:
        """All active paint identities, sorted independently of draw order."""

        return tuple(sorted(self.draw_order))

    @property
    def omitted_fragment_ids(self) -> tuple[str, ...]:
        return tuple(
            item.item_id for item in self.curve_fragments if not item.painted
        )

    @property
    def surface_item_map(self) -> dict[str, QuadricSurfacePaintItem]:
        return {item.item_id: item for item in self.surface_items}

    @property
    def curve_fragment_map(self) -> dict[str, QuadricCurvePaintFragment]:
        return {item.item_id: item for item in self.curve_fragments}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "paintPolicy": self.paint_policy.value,
            "visibility": self.visibility.to_dict(),
            "surfaceItems": [item.to_dict() for item in self.surface_items],
            "curveFragments": [item.to_dict() for item in self.curve_fragments],
            "styles": [item.to_dict() for item in self.styles],
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "curveCrossings": [item.to_dict() for item in self.curve_crossings],
            "drawOrder": list(self.draw_order),
            "omittedFragmentIds": list(self.omitted_fragment_ids),
        }


StyleInput = Mapping[str, OcclusionStyle] | OcclusionStyle | None
SurfaceConstraintInput = PainterConstraint[str] | tuple[str, str]


def _coerce_policy(value: QuadricPaintPolicy | str) -> QuadricPaintPolicy:
    try:
        return QuadricPaintPolicy(value)
    except (TypeError, ValueError) as exc:
        raise QuadricCompositingError(
            "paint_policy must be 'physical', 'diagrammatic', or "
            "'depth_aware_diagrammatic'"
        ) from exc


def _surface_item_id(proxy: OpaqueProjectionProxy) -> str:
    return f"surface:{proxy.patch_id}"


def _curve_fragment_id(
    record: CurveVisibilityRecord,
    span_index: int,
    kind: QuadricPaintKind,
) -> str:
    return f"curve:{record.curve_id}:span:{span_index}:{kind.value}"


def _style_descriptors(
    curve_ids: tuple[str, ...],
    styles: StyleInput,
) -> tuple[tuple[QuadricStyleDescriptor, ...], dict[str, str | None]]:
    if styles is None:
        return (), {curve_id: None for curve_id in curve_ids}
    if isinstance(styles, OcclusionStyle):
        descriptor = QuadricStyleDescriptor.from_style("style:default", styles)
        return (descriptor,), {curve_id: descriptor.style_id for curve_id in curve_ids}
    if not isinstance(styles, Mapping):
        raise TypeError(
            "curve_styles must be None, OcclusionStyle, or a curve/style mapping"
        )
    normalized: dict[str, OcclusionStyle] = {}
    for raw_curve_id, style in styles.items():
        curve_id = _identity(raw_curve_id, "curve style key")
        if curve_id in normalized:
            raise QuadricCompositingError("curve style identities must be unique")
        if not isinstance(style, OcclusionStyle):
            raise TypeError("curve styles must be OcclusionStyle objects")
        normalized[curve_id] = style
    if set(normalized) != set(curve_ids):
        missing = sorted(set(curve_ids) - set(normalized))
        unknown = sorted(set(normalized) - set(curve_ids))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise QuadricCompositingError(
            "curve_styles must cover visibility records exactly: "
            + "; ".join(details)
        )
    descriptors = tuple(
        QuadricStyleDescriptor.from_style(f"style:{curve_id}", normalized[curve_id])
        for curve_id in sorted(curve_ids)
    )
    return descriptors, {
        curve_id: f"style:{curve_id}" for curve_id in curve_ids
    }


def _surface_relations(
    surface_items: tuple[QuadricSurfacePaintItem, ...],
    constraints: Sequence[SurfaceConstraintInput],
) -> list[QuadricPaintRelation]:
    by_surface = {item.surface_id: item.item_id for item in surface_items}
    result: list[QuadricPaintRelation] = []
    for constraint in constraints:
        if isinstance(constraint, PainterConstraint):
            farther, nearer = constraint.farther, constraint.nearer
        else:
            if not isinstance(constraint, tuple) or len(constraint) != 2:
                raise TypeError(
                    "surface_constraints must contain PainterConstraint or pairs"
                )
            farther, nearer = constraint
        farther = _identity(farther, "farther surface identity")
        nearer = _identity(nearer, "nearer surface identity")
        unknown = sorted({farther, nearer} - set(by_surface))
        if unknown:
            raise QuadricCompositingError(
                "surface constraint references unknown surfaces: "
                + ", ".join(unknown)
            )
        if farther == nearer:
            raise QuadricCompositingError(
                "surface constraint cannot order one surface against itself"
            )
        result.append(
            QuadricPaintRelation(
                by_surface[farther],
                by_surface[nearer],
                "explicit_surface_order",
            )
        )
    return result


def _surface_predecessors(
    surface_items: tuple[QuadricSurfacePaintItem, ...],
    relations: Sequence[QuadricPaintRelation],
) -> dict[str, frozenset[str]]:
    """Return every surface known to paint before each surface.

    Only explicit surface-order evidence participates.  The lexicographic tie
    break used by the final topological sort is deterministic, but it is not
    geometric evidence and therefore must not decide which surface attenuates
    a depth-aware hidden stroke.
    """

    item_ids = {item.item_id for item in surface_items}
    direct: dict[str, set[str]] = {item_id: set() for item_id in item_ids}
    for relation in relations:
        if (
            relation.far_item_id not in item_ids
            or relation.near_item_id not in item_ids
        ):
            raise QuadricCompositingError(
                "surface predecessor evidence references a non-surface item"
            )
        direct[relation.near_item_id].add(relation.far_item_id)

    result: dict[str, frozenset[str]] = {}
    for target in sorted(item_ids):
        predecessors: set[str] = set()
        pending = list(direct[target])
        while pending:
            current = pending.pop()
            if current == target:
                raise QuadricCompositingError(
                    "surface constraints contain a contradictory painter cycle"
                )
            if current in predecessors:
                continue
            predecessors.add(current)
            pending.extend(direct[current])
        result[target] = frozenset(predecessors)
    return result


def _dedupe_relations(
    relations: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    by_pair: dict[tuple[str, str], set[str]] = {}
    for relation in relations:
        pair = (relation.far_item_id, relation.near_item_id)
        reverse = tuple(reversed(pair))
        if reverse in by_pair:
            raise QuadricCompositingError(
                "paint items have contradictory direct relations: "
                f"{pair[0]!r}, {pair[1]!r}"
            )
        by_pair.setdefault(pair, set()).add(relation.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(reasons)))
        for (far, near), reasons in sorted(by_pair.items())
    )


def _crossing_relations(
    crossings: Sequence[ProjectedCurveCrossing],
    records: tuple[CurveVisibilityRecord, ...],
    fragments: Sequence[QuadricCurvePaintFragment],
) -> list[QuadricPaintRelation]:
    record_map = {record.curve_id: record for record in records}
    by_curve: dict[str, list[QuadricCurvePaintFragment]] = {}
    for fragment in fragments:
        if fragment.painted:
            by_curve.setdefault(fragment.curve_id, []).append(fragment)

    result: list[QuadricPaintRelation] = []
    for crossing in crossings:
        if crossing.far_curve_id is None or crossing.near_curve_id is None:
            continue
        parameter_by_curve = {
            crossing.first_curve_id: crossing.first_parameter,
            crossing.second_curve_id: crossing.second_parameter,
        }
        active: dict[str, tuple[QuadricCurvePaintFragment, ...]] = {}
        for curve_id in (crossing.far_curve_id, crossing.near_curve_id):
            record = record_map.get(curve_id)
            if record is None:
                raise QuadricCompositingError(
                    f"curve crossing references unknown curve {curve_id!r}"
                )
            parameter = parameter_by_curve[curve_id]
            if not record.domain.contains(
                parameter, tolerance=record.parameter_tolerance
            ):
                raise QuadricCompositingError(
                    f"curve crossing lies outside {curve_id!r} domain"
                )
            active[curve_id] = tuple(
                fragment
                for fragment in by_curve.get(curve_id, ())
                if fragment.interval.contains(
                    parameter, tolerance=record.parameter_tolerance
                )
            )
        # In physical mode a mathematically hidden stroke has no paint item,
        # so there is deliberately no painter edge to emit at that crossing.
        for farther in active[crossing.far_curve_id]:
            for nearer in active[crossing.near_curve_id]:
                result.append(
                    QuadricPaintRelation(
                        farther.item_id,
                        nearer.item_id,
                        f"projected_curve_crossing:{crossing.crossing_id}",
                    )
                )
    return result


def compute_quadric_compositing(
    visibility: CurveVisibilityFrame,
    surface_proxies: Sequence[OpaqueProjectionProxy],
    *,
    paint_policy: QuadricPaintPolicy | str = QuadricPaintPolicy.DIAGRAMMATIC,
    curve_styles: StyleInput = None,
    surface_constraints: Sequence[SurfaceConstraintInput] = (),
    curve_crossings: Sequence[ProjectedCurveCrossing] = (),
) -> QuadricCompositingFrame:
    """Combine exact visibility and opaque fills into one painter graph.

    ``surface_constraints`` use semantic ``surface_id`` values and mean
    ``(farther, nearer)``.  They are the only supported source of
    surface-to-surface ordering in this stage.  Intersecting surfaces must be
    split by a later geometry stage; supplying cyclic whole-surface evidence
    raises :class:`QuadricCompositingError`.
    """

    if not isinstance(visibility, CurveVisibilityFrame):
        raise TypeError("visibility must be a CurveVisibilityFrame")
    policy = _coerce_policy(paint_policy)
    proxies = tuple(surface_proxies)
    if not all(isinstance(item, OpaqueProjectionProxy) for item in proxies):
        raise TypeError("surface_proxies must contain OpaqueProjectionProxy objects")
    proxies = tuple(sorted(proxies, key=lambda item: (item.surface_id, item.patch_id)))
    proxy_surface_ids = tuple(item.surface_id for item in proxies)
    if len(set(proxy_surface_ids)) != len(proxy_surface_ids):
        raise QuadricCompositingError(
            "exactly one opaque projection proxy is required per surface"
        )
    if proxy_surface_ids != visibility.surface_ids:
        raise QuadricCompositingError(
            "surface proxies must cover visibility.surface_ids exactly"
        )
    patch_ids = tuple(item.patch_id for item in proxies)
    if len(set(patch_ids)) != len(patch_ids):
        raise QuadricCompositingError("projection proxy patch identities must be unique")

    surface_items = tuple(
        QuadricSurfacePaintItem(_surface_item_id(proxy), proxy) for proxy in proxies
    )
    curve_ids = tuple(record.curve_id for record in visibility.records)
    style_descriptors, style_by_curve = _style_descriptors(curve_ids, curve_styles)

    fragments: list[QuadricCurvePaintFragment] = []
    for record in visibility.records:
        for span_index, span in enumerate(record.spans):
            if span.kind is VisibilityKind.VISIBLE:
                kind = QuadricPaintKind.VISIBLE_CURVE
                painted = True
                intent = "solid"
            else:
                kind = QuadricPaintKind.HIDDEN_CURVE
                painted = policy is not QuadricPaintPolicy.PHYSICAL
                intent = "dashed" if painted else "omit"
            fragments.append(
                QuadricCurvePaintFragment(
                    _curve_fragment_id(record, span_index, kind),
                    record.curve_id,
                    span_index,
                    span.interval,
                    kind,
                    tuple(span.occluders),
                    painted,
                    intent,
                    style_by_curve[record.curve_id],
                )
            )
    fragments.sort(key=lambda item: item.item_id)

    crossings = tuple(curve_crossings)
    if not all(isinstance(item, ProjectedCurveCrossing) for item in crossings):
        raise TypeError(
            "curve_crossings must contain ProjectedCurveCrossing objects"
        )
    crossing_ids = tuple(item.crossing_id for item in crossings)
    if len(set(crossing_ids)) != len(crossing_ids):
        raise QuadricCompositingError("curve crossing identities must be unique")
    crossings = tuple(sorted(crossings, key=lambda item: item.crossing_id))

    relations = _surface_relations(surface_items, tuple(surface_constraints))
    surface_predecessors = _surface_predecessors(surface_items, relations)
    surface_item_by_id = {
        item.surface_id: item.item_id for item in surface_items
    }
    for fragment in fragments:
        if not fragment.painted:
            continue
        if fragment.kind is QuadricPaintKind.VISIBLE_CURVE:
            relations.extend(
                QuadricPaintRelation(
                    surface.item_id,
                    fragment.item_id,
                    "visible_curve_overlay",
                )
                for surface in surface_items
            )
        elif policy is QuadricPaintPolicy.DIAGRAMMATIC:
            relations.extend(
                QuadricPaintRelation(
                    surface.item_id,
                    fragment.item_id,
                    "diagrammatic_hidden_overlay",
                )
                for surface in surface_items
            )
        else:
            occluder_items = {
                surface_item_by_id[surface_id]
                for surface_id in fragment.occluder_surface_ids
            }
            farther_surface_items = {
                farther_item
                for occluder_item in occluder_items
                for farther_item in surface_predecessors[occluder_item]
                if farther_item not in occluder_items
            }
            relations.extend(
                QuadricPaintRelation(
                    surface_item_id,
                    fragment.item_id,
                    "depth_aware_hidden_after_farther_surface",
                )
                for surface_item_id in sorted(farther_surface_items)
            )
            relations.extend(
                QuadricPaintRelation(
                    fragment.item_id,
                    surface.item_id,
                    "depth_aware_hidden_occlusion",
                )
                for surface in surface_items
                if surface.item_id in occluder_items
            )
    relations.extend(_crossing_relations(crossings, visibility.records, fragments))
    normalized_relations = _dedupe_relations(relations)
    active_ids = tuple(
        sorted(
            (
                *(item.item_id for item in surface_items),
                *(item.item_id for item in fragments if item.painted),
            )
        )
    )
    try:
        draw_order = stable_topological_sort(
            active_ids,
            (
                PainterConstraint(item.far_item_id, item.near_item_id)
                for item in normalized_relations
            ),
            key=lambda item_id: item_id,
        )
    except CompositorCycleError as exc:
        raise QuadricCompositingError(
            "quadric painter graph contains a cycle: "
            + ", ".join(sorted(str(item) for item in exc.unresolved))
        ) from exc
    return QuadricCompositingFrame(
        visibility=visibility,
        paint_policy=policy,
        surface_items=surface_items,
        curve_fragments=tuple(fragments),
        styles=style_descriptors,
        order_relations=normalized_relations,
        draw_order=draw_order,
        curve_crossings=crossings,
    )


def canonical_quadric_compositing_json(frame: QuadricCompositingFrame) -> str:
    if not isinstance(frame, QuadricCompositingFrame):
        raise TypeError("frame must be a QuadricCompositingFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "QUADRIC_COMPOSITING_FRAME_SCHEMA",
    "QuadricCompositingError",
    "QuadricCompositingFrame",
    "QuadricCurvePaintFragment",
    "QuadricPaintKind",
    "QuadricPaintPolicy",
    "QuadricPaintRelation",
    "QuadricStyleDescriptor",
    "QuadricSurfacePaintItem",
    "canonical_quadric_compositing_json",
    "compute_quadric_compositing",
]

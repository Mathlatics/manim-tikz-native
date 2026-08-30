"""Renderer-neutral dynamic compositing roles for one quadric section.

The semantic display catalog owns the complete, immutable renderer slot pool.
This module binds three independent per-frame decisions to that pool:

* ``display_opacity`` controls only how much authored ink is painted;
* ``occlusion_participation`` controls whether certified geometry participates
  in visibility/compositing work; and
* ``depth_presentation`` selects a reviewed depth-presentation policy.

No axis is inferred from another.  In particular, zero opacity does not turn
off occlusion participation, and paint-only geometry may remain fully opaque.
The compiled frame contains no renderer objects and is safe to prepare before
an atomic scene commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence

from .semantic_display import (
    SectionDisplayCatalog,
    SectionDisplayRole,
    SectionSemanticDisplayError,
    SectionSemanticSlot,
)


SECTION_COMPOSITING_INSTRUCTION_SCHEMA = (
    "quadric-section-compositing-instruction/v1"
)
SECTION_COMPOSITING_FRAME_SCHEMA = "quadric-section-compositing-frame/v1"


class SectionSemanticCompositingError(ValueError):
    """A semantic compositing contract is invalid or ambiguous."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionSemanticCompositingError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _sha256_digest(value: object, label: str) -> str:
    digest = _identity(value, label)
    prefix = "sha256:"
    payload = digest[len(prefix) :] if digest.startswith(prefix) else ""
    if len(payload) != 64 or any(
        character not in "0123456789abcdef" for character in payload
    ):
        raise SectionSemanticCompositingError(
            f"{label} must be a lowercase sha256 digest"
        )
    return digest


def _opacity(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionSemanticCompositingError(
            f"{label} must be a finite number between 0 and 1"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionSemanticCompositingError(
            f"{label} must be a finite number between 0 and 1"
        ) from exc
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise SectionSemanticCompositingError(
            f"{label} must be a finite number between 0 and 1"
        )
    return result


def _strict_keys(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SectionSemanticCompositingError(f"{label} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise SectionSemanticCompositingError(
            f"{label} is missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise SectionSemanticCompositingError(
            f"{label} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SectionSemanticCompositingError(
            f"compositing value is not canonical JSON: {exc}"
        ) from exc


def _strict_json_object(value: str, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")

    def reject_constant(token: str) -> object:
        raise SectionSemanticCompositingError(
            f"{label} contains non-finite number {token}"
        )

    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise SectionSemanticCompositingError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except SectionSemanticCompositingError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SectionSemanticCompositingError(f"invalid {label}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SectionSemanticCompositingError(f"{label} must contain an object")
    return parsed


class SectionOcclusionParticipation(str, Enum):
    """Whether one semantic item contributes to certified compositing.

    ``CERTIFIED`` does not mean that every item is itself an opaque surface.
    It means the downstream visibility solver may use the item's certified
    geometric semantics.  ``PAINT_ONLY`` excludes it from that work while
    leaving the display axis untouched.
    """

    CERTIFIED = "certified"
    PAINT_ONLY = "paint-only"


class SectionDepthPresentationPolicy(str, Enum):
    """Reviewed author intent for presenting certified depth evidence."""

    PHYSICAL = "physical"
    DIAGRAMMATIC = "diagrammatic"
    DEPTH_AWARE_DIAGRAMMATIC = "depth-aware-diagrammatic"


class SectionCompositingTargetKind(str, Enum):
    """The fixed catalog identity selected by an instruction override."""

    SLOT = "slot"
    HANDLE = "handle"


@dataclass(frozen=True, slots=True)
class SectionCompositingAxes:
    """The three independent decisions resolved for one semantic item."""

    display_opacity: float = 1.0
    occlusion_participation: SectionOcclusionParticipation = (
        SectionOcclusionParticipation.CERTIFIED
    )
    depth_presentation: SectionDepthPresentationPolicy = (
        SectionDepthPresentationPolicy.PHYSICAL
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "display_opacity",
            _opacity(self.display_opacity, "display_opacity"),
        )
        try:
            participation = SectionOcclusionParticipation(
                self.occlusion_participation
            )
        except (TypeError, ValueError) as exc:
            raise SectionSemanticCompositingError(
                "occlusion_participation must be a SectionOcclusionParticipation"
            ) from exc
        try:
            presentation = SectionDepthPresentationPolicy(self.depth_presentation)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticCompositingError(
                "depth_presentation must be a SectionDepthPresentationPolicy"
            ) from exc
        object.__setattr__(self, "occlusion_participation", participation)
        object.__setattr__(self, "depth_presentation", presentation)

    def to_dict(self) -> dict[str, object]:
        return {
            "displayOpacity": self.display_opacity,
            "occlusionParticipation": self.occlusion_participation.value,
            "depthPresentation": self.depth_presentation.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SectionCompositingAxes":
        raw = _strict_keys(
            value,
            required=frozenset(
                {
                    "displayOpacity",
                    "occlusionParticipation",
                    "depthPresentation",
                }
            ),
            label="compositing axes",
        )
        return cls(
            display_opacity=raw["displayOpacity"],  # type: ignore[arg-type]
            occlusion_participation=raw[  # type: ignore[arg-type]
                "occlusionParticipation"
            ],
            depth_presentation=raw["depthPresentation"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SectionCompositingOverride:
    """An explicit change to one or more axes for a slot or stable handle."""

    target_kind: SectionCompositingTargetKind
    target_id: str
    display_opacity: float | None = None
    occlusion_participation: SectionOcclusionParticipation | None = None
    depth_presentation: SectionDepthPresentationPolicy | None = None

    def __post_init__(self) -> None:
        try:
            target_kind = SectionCompositingTargetKind(self.target_kind)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticCompositingError(
                "target_kind must be a SectionCompositingTargetKind"
            ) from exc
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(
            self,
            "target_id",
            _identity(self.target_id, "target_id"),
        )
        if self.display_opacity is not None:
            object.__setattr__(
                self,
                "display_opacity",
                _opacity(self.display_opacity, "display_opacity"),
            )
        if self.occlusion_participation is not None:
            try:
                participation = SectionOcclusionParticipation(
                    self.occlusion_participation
                )
            except (TypeError, ValueError) as exc:
                raise SectionSemanticCompositingError(
                    "occlusion_participation must be a "
                    "SectionOcclusionParticipation"
                ) from exc
            object.__setattr__(self, "occlusion_participation", participation)
        if self.depth_presentation is not None:
            try:
                presentation = SectionDepthPresentationPolicy(
                    self.depth_presentation
                )
            except (TypeError, ValueError) as exc:
                raise SectionSemanticCompositingError(
                    "depth_presentation must be a "
                    "SectionDepthPresentationPolicy"
                ) from exc
            object.__setattr__(self, "depth_presentation", presentation)
        if (
            self.display_opacity is None
            and self.occlusion_participation is None
            and self.depth_presentation is None
        ):
            raise SectionSemanticCompositingError(
                "compositing override must change at least one axis"
            )

    @classmethod
    def for_slot(
        cls,
        slot_id: str,
        *,
        display_opacity: float | None = None,
        occlusion_participation: SectionOcclusionParticipation | str | None = None,
        depth_presentation: SectionDepthPresentationPolicy | str | None = None,
    ) -> "SectionCompositingOverride":
        return cls(
            SectionCompositingTargetKind.SLOT,
            slot_id,
            display_opacity,
            occlusion_participation,  # type: ignore[arg-type]
            depth_presentation,  # type: ignore[arg-type]
        )

    @classmethod
    def for_handle(
        cls,
        handle_id: str,
        *,
        display_opacity: float | None = None,
        occlusion_participation: SectionOcclusionParticipation | str | None = None,
        depth_presentation: SectionDepthPresentationPolicy | str | None = None,
    ) -> "SectionCompositingOverride":
        return cls(
            SectionCompositingTargetKind.HANDLE,
            handle_id,
            display_opacity,
            occlusion_participation,  # type: ignore[arg-type]
            depth_presentation,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "targetKind": self.target_kind.value,
            "targetId": self.target_id,
        }
        if self.display_opacity is not None:
            result["displayOpacity"] = self.display_opacity
        if self.occlusion_participation is not None:
            result["occlusionParticipation"] = (
                self.occlusion_participation.value
            )
        if self.depth_presentation is not None:
            result["depthPresentation"] = self.depth_presentation.value
        return result

    @classmethod
    def from_dict(cls, value: object) -> "SectionCompositingOverride":
        raw = _strict_keys(
            value,
            required=frozenset({"targetKind", "targetId"}),
            optional=frozenset(
                {
                    "displayOpacity",
                    "occlusionParticipation",
                    "depthPresentation",
                }
            ),
            label="compositing override",
        )
        return cls(
            target_kind=raw["targetKind"],  # type: ignore[arg-type]
            target_id=raw["targetId"],  # type: ignore[arg-type]
            display_opacity=raw.get("displayOpacity"),  # type: ignore[arg-type]
            occlusion_participation=raw.get(  # type: ignore[arg-type]
                "occlusionParticipation"
            ),
            depth_presentation=raw.get("depthPresentation"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SectionCompositingInstruction:
    """One catalog-bound, renderer-neutral instruction for a single frame."""

    section_id: str
    catalog_digest: str
    defaults: SectionCompositingAxes = field(
        default_factory=SectionCompositingAxes
    )
    overrides: tuple[SectionCompositingOverride, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_id",
            _identity(self.section_id, "section_id"),
        )
        object.__setattr__(
            self,
            "catalog_digest",
            _sha256_digest(self.catalog_digest, "catalog_digest"),
        )
        if not isinstance(self.defaults, SectionCompositingAxes):
            raise TypeError("defaults must be SectionCompositingAxes")
        if not all(
            isinstance(item, SectionCompositingOverride)
            for item in self.overrides
        ):
            raise TypeError(
                "overrides must contain SectionCompositingOverride values"
            )
        overrides = tuple(
            sorted(
                self.overrides,
                key=lambda item: (item.target_kind.value, item.target_id),
            )
        )
        targets = tuple(
            (item.target_kind, item.target_id) for item in overrides
        )
        if len(set(targets)) != len(targets):
            raise SectionSemanticCompositingError(
                "compositing override targets must be unique"
            )
        object.__setattr__(self, "overrides", overrides)

    @classmethod
    def for_catalog(
        cls,
        catalog: SectionDisplayCatalog,
        *,
        defaults: SectionCompositingAxes | None = None,
        overrides: Sequence[SectionCompositingOverride] = (),
    ) -> "SectionCompositingInstruction":
        if not isinstance(catalog, SectionDisplayCatalog):
            raise TypeError("catalog must be a SectionDisplayCatalog")
        return cls(
            catalog.section_id,
            catalog.digest,
            SectionCompositingAxes() if defaults is None else defaults,
            tuple(overrides),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_COMPOSITING_INSTRUCTION_SCHEMA,
            "sectionId": self.section_id,
            "catalogDigest": self.catalog_digest,
            "defaults": self.defaults.to_dict(),
            "overrides": [item.to_dict() for item in self.overrides],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            self.to_json().encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "SectionCompositingInstruction":
        raw = _strict_keys(
            value,
            required=frozenset(
                {
                    "schema",
                    "sectionId",
                    "catalogDigest",
                    "defaults",
                    "overrides",
                }
            ),
            label="compositing instruction",
        )
        if raw["schema"] != SECTION_COMPOSITING_INSTRUCTION_SCHEMA:
            raise SectionSemanticCompositingError(
                "unsupported compositing instruction schema "
                f"{raw['schema']!r}"
            )
        raw_overrides = raw["overrides"]
        if not isinstance(raw_overrides, list):
            raise SectionSemanticCompositingError(
                "compositing instruction overrides must be an array"
            )
        return cls(
            section_id=raw["sectionId"],  # type: ignore[arg-type]
            catalog_digest=raw["catalogDigest"],  # type: ignore[arg-type]
            defaults=SectionCompositingAxes.from_dict(raw["defaults"]),
            overrides=tuple(
                SectionCompositingOverride.from_dict(item)
                for item in raw_overrides
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "SectionCompositingInstruction":
        return cls.from_dict(
            _strict_json_object(value, "compositing instruction JSON")
        )


@dataclass(frozen=True, slots=True)
class SectionCompositingSlotState:
    """The resolved axes bound to one immutable catalog slot/source."""

    slot_id: str
    role: SectionDisplayRole
    source_id: str | None
    topology_bank: str | None
    axes: SectionCompositingAxes

    def __post_init__(self) -> None:
        try:
            semantic_slot = SectionSemanticSlot(
                self.slot_id,
                self.role,
                self.source_id,
                self.topology_bank,
            )
        except SectionSemanticDisplayError as exc:
            raise SectionSemanticCompositingError(
                f"invalid compositing slot identity: {exc}"
            ) from exc
        object.__setattr__(self, "slot_id", semantic_slot.slot_id)
        object.__setattr__(self, "role", semantic_slot.role)
        object.__setattr__(self, "source_id", semantic_slot.source_id)
        object.__setattr__(self, "topology_bank", semantic_slot.topology_bank)
        if not isinstance(self.axes, SectionCompositingAxes):
            raise TypeError("axes must be SectionCompositingAxes")

    @property
    def display_opacity(self) -> float:
        return self.axes.display_opacity

    @property
    def occlusion_participation(self) -> SectionOcclusionParticipation:
        return self.axes.occlusion_participation

    @property
    def depth_presentation(self) -> SectionDepthPresentationPolicy:
        return self.axes.depth_presentation

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "slotId": self.slot_id,
            "role": self.role.value,
            "axes": self.axes.to_dict(),
        }
        if self.source_id is not None:
            result["sourceId"] = self.source_id
        if self.topology_bank is not None:
            result["topologyBank"] = self.topology_bank
        return result

    @classmethod
    def from_dict(cls, value: object) -> "SectionCompositingSlotState":
        raw = _strict_keys(
            value,
            required=frozenset({"slotId", "role", "axes"}),
            optional=frozenset({"sourceId", "topologyBank"}),
            label="compositing slot state",
        )
        return cls(
            slot_id=raw["slotId"],  # type: ignore[arg-type]
            role=raw["role"],  # type: ignore[arg-type]
            source_id=raw.get("sourceId"),  # type: ignore[arg-type]
            topology_bank=raw.get("topologyBank"),  # type: ignore[arg-type]
            axes=SectionCompositingAxes.from_dict(raw["axes"]),
        )


@dataclass(frozen=True, slots=True)
class SectionCompositingFrame:
    """Canonical per-frame state ready for a later transactional adapter."""

    section_id: str
    catalog_digest: str
    slots: tuple[SectionCompositingSlotState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_id",
            _identity(self.section_id, "section_id"),
        )
        object.__setattr__(
            self,
            "catalog_digest",
            _sha256_digest(self.catalog_digest, "catalog_digest"),
        )
        if not all(
            isinstance(item, SectionCompositingSlotState) for item in self.slots
        ):
            raise TypeError(
                "slots must contain SectionCompositingSlotState values"
            )
        slots = tuple(sorted(self.slots, key=lambda item: item.slot_id))
        slot_ids = tuple(item.slot_id for item in slots)
        if not slots or len(set(slot_ids)) != len(slot_ids):
            raise SectionSemanticCompositingError(
                "compositing frame slots must be non-empty and unique"
            )
        object.__setattr__(self, "slots", slots)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_COMPOSITING_FRAME_SCHEMA,
            "sectionId": self.section_id,
            "catalogDigest": self.catalog_digest,
            "slots": [item.to_dict() for item in self.slots],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            self.to_json().encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "SectionCompositingFrame":
        raw = _strict_keys(
            value,
            required=frozenset(
                {"schema", "sectionId", "catalogDigest", "slots"}
            ),
            label="compositing frame",
        )
        if raw["schema"] != SECTION_COMPOSITING_FRAME_SCHEMA:
            raise SectionSemanticCompositingError(
                f"unsupported compositing frame schema {raw['schema']!r}"
            )
        raw_slots = raw["slots"]
        if not isinstance(raw_slots, list):
            raise SectionSemanticCompositingError(
                "compositing frame slots must be an array"
            )
        return cls(
            section_id=raw["sectionId"],  # type: ignore[arg-type]
            catalog_digest=raw["catalogDigest"],  # type: ignore[arg-type]
            slots=tuple(
                SectionCompositingSlotState.from_dict(item)
                for item in raw_slots
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "SectionCompositingFrame":
        return cls.from_dict(_strict_json_object(value, "compositing frame JSON"))

    def state_for_slot(self, slot_id: str) -> SectionCompositingSlotState:
        key = _identity(slot_id, "slot_id")
        for item in self.slots:
            if item.slot_id == key:
                return item
        raise SectionSemanticCompositingError(
            f"unknown compositing slot {key!r}"
        )

    def states_for_source(
        self,
        source_id: str,
    ) -> tuple[SectionCompositingSlotState, ...]:
        key = _identity(source_id, "source_id")
        result = tuple(item for item in self.slots if item.source_id == key)
        if not result:
            raise SectionSemanticCompositingError(
                f"unknown compositing source {key!r}"
            )
        return result

    def states_for_handle(
        self,
        catalog: SectionDisplayCatalog,
        handle_id: str,
    ) -> tuple[SectionCompositingSlotState, ...]:
        self.validate_catalog(catalog)
        try:
            selected = set(catalog.handle(handle_id).slot_ids)
        except SectionSemanticDisplayError as exc:
            raise SectionSemanticCompositingError(str(exc)) from exc
        return tuple(item for item in self.slots if item.slot_id in selected)

    def validate_catalog(self, catalog: SectionDisplayCatalog) -> None:
        """Fail closed if persisted slot/source identity drifted."""

        if not isinstance(catalog, SectionDisplayCatalog):
            raise TypeError("catalog must be a SectionDisplayCatalog")
        if self.section_id != catalog.section_id:
            raise SectionSemanticCompositingError(
                "compositing frame section_id does not match catalog"
            )
        if self.catalog_digest != catalog.digest:
            raise SectionSemanticCompositingError(
                "compositing frame catalog_digest does not match catalog"
            )
        expected = {
            item.slot_id: (item.role, item.source_id, item.topology_bank)
            for item in catalog.slots
        }
        observed = {
            item.slot_id: (item.role, item.source_id, item.topology_bank)
            for item in self.slots
        }
        if observed != expected:
            raise SectionSemanticCompositingError(
                "compositing frame slot/source identity does not match catalog"
            )


def _target_slot_ids(
    catalog: SectionDisplayCatalog,
    override: SectionCompositingOverride,
) -> tuple[str, ...]:
    if override.target_kind is SectionCompositingTargetKind.HANDLE:
        try:
            return catalog.handle(override.target_id).slot_ids
        except SectionSemanticDisplayError as exc:
            raise SectionSemanticCompositingError(str(exc)) from exc
    for slot in catalog.slots:
        if slot.slot_id == override.target_id:
            return (slot.slot_id,)
    raise SectionSemanticCompositingError(
        f"compositing slot {override.target_id!r} is unavailable"
    )


def compile_section_compositing(
    catalog: SectionDisplayCatalog,
    instruction: SectionCompositingInstruction,
) -> SectionCompositingFrame:
    """Compile one frame without allocating or mutating renderer objects."""

    if not isinstance(catalog, SectionDisplayCatalog):
        raise TypeError("catalog must be a SectionDisplayCatalog")
    if not isinstance(instruction, SectionCompositingInstruction):
        raise TypeError(
            "instruction must be a SectionCompositingInstruction"
        )
    if instruction.section_id != catalog.section_id:
        raise SectionSemanticCompositingError(
            "compositing instruction section_id does not match catalog"
        )
    if instruction.catalog_digest != catalog.digest:
        raise SectionSemanticCompositingError(
            "compositing instruction catalog_digest does not match catalog"
        )

    axes_by_slot = {
        slot.slot_id: instruction.defaults for slot in catalog.slots
    }
    assignments: dict[tuple[str, str], tuple[str, str]] = {}
    axis_values = (
        ("display_opacity", "display_opacity"),
        ("occlusion_participation", "occlusion_participation"),
        ("depth_presentation", "depth_presentation"),
    )
    for override in instruction.overrides:
        slot_ids = _target_slot_ids(catalog, override)
        for field_name, axis_name in axis_values:
            value = getattr(override, field_name)
            if value is None:
                continue
            for slot_id in slot_ids:
                assignment_key = (slot_id, axis_name)
                if assignment_key in assignments:
                    prior_kind, prior_id = assignments[assignment_key]
                    raise SectionSemanticCompositingError(
                        f"compositing slot {slot_id!r} axis {axis_name!r} is "
                        "assigned by overlapping targets "
                        f"{prior_kind}:{prior_id!r} and "
                        f"{override.target_kind.value}:{override.target_id!r}"
                    )
                assignments[assignment_key] = (
                    override.target_kind.value,
                    override.target_id,
                )
                axes_by_slot[slot_id] = replace(
                    axes_by_slot[slot_id],
                    **{field_name: value},
                )

    frame = SectionCompositingFrame(
        catalog.section_id,
        catalog.digest,
        tuple(
            SectionCompositingSlotState(
                slot.slot_id,
                slot.role,
                slot.source_id,
                slot.topology_bank,
                axes_by_slot[slot.slot_id],
            )
            for slot in catalog.slots
        ),
    )
    frame.validate_catalog(catalog)
    return frame


__all__ = [
    "SECTION_COMPOSITING_FRAME_SCHEMA",
    "SECTION_COMPOSITING_INSTRUCTION_SCHEMA",
    "SectionCompositingAxes",
    "SectionCompositingFrame",
    "SectionCompositingInstruction",
    "SectionCompositingOverride",
    "SectionCompositingSlotState",
    "SectionCompositingTargetKind",
    "SectionDepthPresentationPolicy",
    "SectionOcclusionParticipation",
    "SectionSemanticCompositingError",
    "compile_section_compositing",
]

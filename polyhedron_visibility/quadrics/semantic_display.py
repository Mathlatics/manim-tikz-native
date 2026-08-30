"""Renderer-neutral semantic display handles for one quadric section.

The low-level Cairo binding owns fixed Mobject slots.  Teaching code should
not need to know which slot currently paints an ellipse branch, a parabola, or
one half of a hyperbola.  This module gives those immutable slots stable
semantic names and compiles high-level display modes into per-slot opacity
multipliers.

No renderer objects live here.  A Manim adapter may bind each ``slot_id`` to
its already allocated Mobject and multiply the controller's authored opacity
by the compiled multiplier.  Inactive topology-bank slots remain inactive;
the display contract never turns geometry on by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence


SECTION_DISPLAY_CATALOG_SCHEMA = "quadric-section-display-catalog/v1"
SECTION_DISPLAY_INSTRUCTION_SCHEMA = "quadric-section-display-instruction/v1"
SECTION_DISPLAY_FRAME_SCHEMA = "quadric-section-display-frame/v1"


class SectionSemanticDisplayError(ValueError):
    """A semantic display catalog or instruction is ambiguous."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionSemanticDisplayError(f"{label} must be a non-empty string")
    return value.strip()


def _strict_keys(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SectionSemanticDisplayError(f"{label} must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise SectionSemanticDisplayError(
            f"{label} is missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise SectionSemanticDisplayError(
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
        raise SectionSemanticDisplayError(
            f"display value is not canonical JSON: {exc}"
        ) from exc


def _sha256_digest(value: object, label: str) -> str:
    digest = _identity(value, label)
    prefix = "sha256:"
    payload = digest[len(prefix) :] if digest.startswith(prefix) else ""
    if len(payload) != 64 or any(item not in "0123456789abcdef" for item in payload):
        raise SectionSemanticDisplayError(
            f"{label} must be a lowercase sha256 digest"
        )
    return digest


def _strict_json_object(value: str, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")

    def reject_constant(token: str) -> object:
        raise SectionSemanticDisplayError(
            f"{label} contains non-finite number {token}"
        )

    def reject_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise SectionSemanticDisplayError(
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
    except SectionSemanticDisplayError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SectionSemanticDisplayError(f"invalid {label}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SectionSemanticDisplayError(f"{label} must contain an object")
    return parsed


class SectionDisplayRole(str, Enum):
    """Atomic semantic roles owned by fixed renderer slots."""

    SURFACE_FILL = "surface-fill"
    SURFACE_OUTLINE = "surface-outline"
    PLANE_FILL = "plane-fill"
    PLANE_OUTLINE = "plane-outline"
    SECTION_CURVE = "section-curve"
    GENERATOR = "generator"
    CONTOUR = "contour"
    CAP_RIM = "cap-rim"
    CAP_CHORD = "cap-chord"


class SectionDisplayMode(str, Enum):
    """Reviewed teaching presets; each only suppresses existing ink."""

    PAINTED = "painted"
    OUTLINE_ONLY = "outline-only"
    SECTION_ONLY = "section-only"
    HIDDEN = "hidden"


_FILL_ROLES = frozenset(
    {SectionDisplayRole.SURFACE_FILL, SectionDisplayRole.PLANE_FILL}
)
_SECTION_ROLES = frozenset(
    {SectionDisplayRole.SECTION_CURVE, SectionDisplayRole.CAP_CHORD}
)
_SURFACE_HANDLE_ROLES = frozenset(
    {
        SectionDisplayRole.SURFACE_FILL,
        SectionDisplayRole.SURFACE_OUTLINE,
        SectionDisplayRole.GENERATOR,
        SectionDisplayRole.CONTOUR,
        SectionDisplayRole.CAP_RIM,
    }
)
_PLANE_HANDLE_ROLES = frozenset(
    {SectionDisplayRole.PLANE_FILL, SectionDisplayRole.PLANE_OUTLINE}
)
_BOUNDARY_ROLES = frozenset(
    {
        SectionDisplayRole.GENERATOR,
        SectionDisplayRole.CONTOUR,
        SectionDisplayRole.CAP_RIM,
        SectionDisplayRole.CAP_CHORD,
    }
)


@dataclass(frozen=True, slots=True)
class SectionSemanticSlot:
    """One immutable adapter-owned slot with a stable semantic role."""

    slot_id: str
    role: SectionDisplayRole
    source_id: str | None = None
    topology_bank: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _identity(self.slot_id, "slot_id"))
        try:
            role = SectionDisplayRole(self.role)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                "role must be a SectionDisplayRole"
            ) from exc
        object.__setattr__(self, "role", role)
        if self.source_id is not None:
            object.__setattr__(
                self,
                "source_id",
                _identity(self.source_id, "source_id"),
            )
        if self.topology_bank is not None:
            object.__setattr__(
                self,
                "topology_bank",
                _identity(self.topology_bank, "topology_bank"),
            )
        if role in _BOUNDARY_ROLES and self.source_id is None:
            raise SectionSemanticDisplayError(
                f"{role.value} slot requires source_id"
            )
        if (
            self.topology_bank is not None
            and role is not SectionDisplayRole.SECTION_CURVE
        ):
            raise SectionSemanticDisplayError(
                "topology_bank is valid only for section-curve slots"
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "slotId": self.slot_id,
            "role": self.role.value,
        }
        if self.source_id is not None:
            result["sourceId"] = self.source_id
        if self.topology_bank is not None:
            result["topologyBank"] = self.topology_bank
        return result

    @classmethod
    def from_dict(cls, value: object) -> "SectionSemanticSlot":
        raw = _strict_keys(
            value,
            required=frozenset({"slotId", "role"}),
            optional=frozenset({"sourceId", "topologyBank"}),
            label="semantic slot",
        )
        return cls(
            slot_id=raw["slotId"],  # type: ignore[arg-type]
            role=raw["role"],  # type: ignore[arg-type]
            source_id=raw.get("sourceId"),  # type: ignore[arg-type]
            topology_bank=raw.get("topologyBank"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SectionDisplayHandle:
    """A stable teaching identity spanning one or more fixed slots."""

    handle_id: str
    slot_ids: tuple[str, ...]
    roles: tuple[SectionDisplayRole, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "handle_id",
            _identity(self.handle_id, "handle_id"),
        )
        slots = tuple(
            sorted(_identity(item, "handle slot_id") for item in self.slot_ids)
        )
        if not slots or len(set(slots)) != len(slots):
            raise SectionSemanticDisplayError(
                "display handle slot_ids must be non-empty and unique"
            )
        try:
            roles = tuple(
                sorted(
                    {SectionDisplayRole(item) for item in self.roles},
                    key=lambda item: item.value,
                )
            )
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                "display handle contains an unknown role"
            ) from exc
        if not roles:
            raise SectionSemanticDisplayError(
                "display handle roles must be non-empty"
            )
        object.__setattr__(self, "slot_ids", slots)
        object.__setattr__(self, "roles", roles)


class SectionDisplayCatalog:
    """Canonical semantic index over an adapter's complete fixed slot pool."""

    __slots__ = ("section_id", "slots", "_slot_by_id", "_handles")

    def __init__(
        self,
        section_id: str,
        slots: Sequence[SectionSemanticSlot],
    ) -> None:
        self.section_id = _identity(section_id, "section_id")
        authored = tuple(slots)
        if not authored:
            raise SectionSemanticDisplayError(
                "display catalog requires at least one semantic slot"
            )
        if not all(isinstance(item, SectionSemanticSlot) for item in authored):
            raise TypeError("slots must contain SectionSemanticSlot values")
        ordered = tuple(sorted(authored, key=lambda item: item.slot_id))
        ids = tuple(item.slot_id for item in ordered)
        if len(set(ids)) != len(ids):
            raise SectionSemanticDisplayError("semantic slot ids must be unique")
        self.slots = ordered
        self._slot_by_id = MappingProxyType(
            {item.slot_id: item for item in ordered}
        )
        self._handles = MappingProxyType(self._build_handles())

    @property
    def handles(self) -> Mapping[str, SectionDisplayHandle]:
        return self._handles

    def _handle_id(self, suffix: str) -> str:
        return f"{self.section_id}:display:{suffix}"

    def _add_handle(
        self,
        result: dict[str, SectionDisplayHandle],
        suffix: str,
        slots: Sequence[SectionSemanticSlot],
    ) -> None:
        selected = tuple(slots)
        if not selected:
            return
        handle = SectionDisplayHandle(
            self._handle_id(suffix),
            tuple(item.slot_id for item in selected),
            tuple(item.role for item in selected),
        )
        result[handle.handle_id] = handle

    def _build_handles(self) -> dict[str, SectionDisplayHandle]:
        result: dict[str, SectionDisplayHandle] = {}
        self._add_handle(
            result,
            "surface",
            tuple(item for item in self.slots if item.role in _SURFACE_HANDLE_ROLES),
        )
        self._add_handle(
            result,
            "plane",
            tuple(item for item in self.slots if item.role in _PLANE_HANDLE_ROLES),
        )
        self._add_handle(
            result,
            "section-curve",
            tuple(item for item in self.slots if item.role in _SECTION_ROLES),
        )
        self._add_handle(
            result,
            "boundary",
            tuple(item for item in self.slots if item.role in _BOUNDARY_ROLES),
        )
        for role in SectionDisplayRole:
            selected = tuple(item for item in self.slots if item.role is role)
            self._add_handle(result, f"role:{role.value}", selected)
            if role not in _BOUNDARY_ROLES:
                continue
            source_ids = sorted(
                {item.source_id for item in selected if item.source_id is not None}
            )
            for source_id in source_ids:
                self._add_handle(
                    result,
                    f"boundary:{role.value}:{source_id}",
                    tuple(item for item in selected if item.source_id == source_id),
                )
        return dict(sorted(result.items()))

    def handle(self, handle_id: str) -> SectionDisplayHandle:
        key = _identity(handle_id, "handle_id")
        try:
            return self._handles[key]
        except KeyError as exc:
            raise SectionSemanticDisplayError(
                f"display handle {key!r} is unavailable"
            ) from exc

    @property
    def surface(self) -> SectionDisplayHandle:
        return self.handle(self._handle_id("surface"))

    @property
    def plane(self) -> SectionDisplayHandle:
        return self.handle(self._handle_id("plane"))

    @property
    def section_curve(self) -> SectionDisplayHandle:
        return self.handle(self._handle_id("section-curve"))

    def boundary(
        self,
        kind: SectionDisplayRole | str | None = None,
        *,
        source_id: str | None = None,
    ) -> SectionDisplayHandle:
        if kind is None:
            if source_id is not None:
                raise SectionSemanticDisplayError(
                    "source_id requires a boundary kind"
                )
            return self.handle(self._handle_id("boundary"))
        try:
            role = SectionDisplayRole(kind)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                f"unknown boundary kind {kind!r}"
            ) from exc
        if role not in _BOUNDARY_ROLES:
            raise SectionSemanticDisplayError(
                f"{role.value!r} is not a boundary kind"
            )
        suffix = f"role:{role.value}"
        if source_id is not None:
            suffix = f"boundary:{role.value}:{_identity(source_id, 'source_id')}"
        return self.handle(self._handle_id(suffix))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_DISPLAY_CATALOG_SCHEMA,
            "sectionId": self.section_id,
            "slots": [item.to_dict() for item in self.slots],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "SectionDisplayCatalog":
        raw = _strict_keys(
            value,
            required=frozenset({"schema", "sectionId", "slots"}),
            label="display catalog",
        )
        if raw["schema"] != SECTION_DISPLAY_CATALOG_SCHEMA:
            raise SectionSemanticDisplayError(
                f"unsupported display catalog schema {raw['schema']!r}"
            )
        raw_slots = raw["slots"]
        if not isinstance(raw_slots, list):
            raise SectionSemanticDisplayError("display catalog slots must be an array")
        return cls(
            raw["sectionId"],  # type: ignore[arg-type]
            tuple(SectionSemanticSlot.from_dict(item) for item in raw_slots),
        )

    @classmethod
    def from_json(cls, value: str) -> "SectionDisplayCatalog":
        return cls.from_dict(_strict_json_object(value, "display catalog JSON"))


@dataclass(frozen=True, slots=True)
class SectionDisplayPolicy:
    """One named opacity policy over all atomic semantic roles."""

    mode: SectionDisplayMode

    def __post_init__(self) -> None:
        try:
            mode = SectionDisplayMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                "mode must be a SectionDisplayMode"
            ) from exc
        object.__setattr__(self, "mode", mode)

    def opacity_for(self, role: SectionDisplayRole | str) -> float:
        semantic_role = SectionDisplayRole(role)
        if self.mode is SectionDisplayMode.HIDDEN:
            return 0.0
        if self.mode is SectionDisplayMode.OUTLINE_ONLY:
            return 0.0 if semantic_role in _FILL_ROLES else 1.0
        if self.mode is SectionDisplayMode.SECTION_ONLY:
            return 1.0 if semantic_role in _SECTION_ROLES else 0.0
        return 1.0


@dataclass(frozen=True, slots=True)
class SectionDisplayInstruction:
    """A display mode plus optional semantic emphasis for one frame."""

    policy: SectionDisplayPolicy
    emphasized_handles: tuple[str, ...] = ()
    dim_unemphasized: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SectionDisplayPolicy):
            raise TypeError("policy must be a SectionDisplayPolicy")
        handles = tuple(
            _identity(item, "emphasized handle")
            for item in self.emphasized_handles
        )
        if len(set(handles)) != len(handles):
            raise SectionSemanticDisplayError(
                "emphasized_handles must be unique"
            )
        object.__setattr__(self, "emphasized_handles", tuple(sorted(handles)))
        if isinstance(self.dim_unemphasized, bool):
            raise SectionSemanticDisplayError(
                "dim_unemphasized must be between 0 and 1"
            )
        try:
            dim = float(self.dim_unemphasized)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SectionSemanticDisplayError(
                "dim_unemphasized must be between 0 and 1"
            ) from exc
        if not isfinite(dim) or dim < 0.0 or dim > 1.0:
            raise SectionSemanticDisplayError(
                "dim_unemphasized must be between 0 and 1"
            )
        object.__setattr__(self, "dim_unemphasized", dim)

    @classmethod
    def for_mode(
        cls,
        mode: SectionDisplayMode | str,
        *,
        emphasized_handles: Sequence[str] = (),
        dim_unemphasized: float = 0.25,
    ) -> "SectionDisplayInstruction":
        return cls(
            SectionDisplayPolicy(SectionDisplayMode(mode)),
            tuple(emphasized_handles),
            dim_unemphasized,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_DISPLAY_INSTRUCTION_SCHEMA,
            "mode": self.policy.mode.value,
            "emphasizedHandles": list(self.emphasized_handles),
            "dimUnemphasized": self.dim_unemphasized,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "SectionDisplayInstruction":
        raw = _strict_keys(
            value,
            required=frozenset(
                {
                    "schema",
                    "mode",
                    "emphasizedHandles",
                    "dimUnemphasized",
                }
            ),
            label="display instruction",
        )
        if raw["schema"] != SECTION_DISPLAY_INSTRUCTION_SCHEMA:
            raise SectionSemanticDisplayError(
                f"unsupported display instruction schema {raw['schema']!r}"
            )
        handles = raw["emphasizedHandles"]
        if not isinstance(handles, list):
            raise SectionSemanticDisplayError(
                "emphasizedHandles must be an array"
            )
        try:
            mode = SectionDisplayMode(raw["mode"])
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                f"unsupported display mode {raw['mode']!r}"
            ) from exc
        return cls.for_mode(
            mode,
            emphasized_handles=handles,  # type: ignore[arg-type]
            dim_unemphasized=raw["dimUnemphasized"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, value: str) -> "SectionDisplayInstruction":
        return cls.from_dict(
            _strict_json_object(value, "display instruction JSON")
        )


@dataclass(frozen=True, slots=True)
class SectionDisplaySlotState:
    """Compiled opacity evidence for one immutable adapter slot."""

    slot_id: str
    role: SectionDisplayRole
    source_id: str | None
    topology_bank: str | None
    opacity_multiplier: float
    emphasized: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _identity(self.slot_id, "slot_id"))
        try:
            role = SectionDisplayRole(self.role)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                "role must be a SectionDisplayRole"
            ) from exc
        object.__setattr__(self, "role", role)
        if self.source_id is not None:
            object.__setattr__(
                self,
                "source_id",
                _identity(self.source_id, "source_id"),
            )
        if self.topology_bank is not None:
            object.__setattr__(
                self,
                "topology_bank",
                _identity(self.topology_bank, "topology_bank"),
            )
        if (
            self.topology_bank is not None
            and role is not SectionDisplayRole.SECTION_CURVE
        ):
            raise SectionSemanticDisplayError(
                "topology_bank is valid only for section-curve slots"
            )
        if isinstance(self.opacity_multiplier, bool):
            raise SectionSemanticDisplayError(
                "opacity_multiplier must be between 0 and 1"
            )
        try:
            opacity = float(self.opacity_multiplier)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SectionSemanticDisplayError(
                "opacity_multiplier must be between 0 and 1"
            ) from exc
        if not isfinite(opacity) or opacity < 0.0 or opacity > 1.0:
            raise SectionSemanticDisplayError(
                "opacity_multiplier must be between 0 and 1"
            )
        object.__setattr__(self, "opacity_multiplier", opacity)
        if not isinstance(self.emphasized, bool):
            raise TypeError("emphasized must be a bool")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "slotId": self.slot_id,
            "role": self.role.value,
            "opacityMultiplier": self.opacity_multiplier,
            "emphasized": self.emphasized,
        }
        if self.source_id is not None:
            result["sourceId"] = self.source_id
        if self.topology_bank is not None:
            result["topologyBank"] = self.topology_bank
        return result


@dataclass(frozen=True, slots=True)
class SectionDisplayFrame:
    """Canonical renderer-neutral result ready for transactional commit."""

    section_id: str
    catalog_digest: str
    mode: SectionDisplayMode
    slots: tuple[SectionDisplaySlotState, ...]
    emphasized_handles: tuple[str, ...]

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
        try:
            mode = SectionDisplayMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise SectionSemanticDisplayError(
                "mode must be a SectionDisplayMode"
            ) from exc
        object.__setattr__(self, "mode", mode)
        if not all(isinstance(item, SectionDisplaySlotState) for item in self.slots):
            raise TypeError("slots must contain SectionDisplaySlotState values")
        slots = tuple(sorted(self.slots, key=lambda item: item.slot_id))
        slot_ids = tuple(item.slot_id for item in slots)
        if not slots or len(set(slot_ids)) != len(slot_ids):
            raise SectionSemanticDisplayError(
                "display frame slots must be non-empty and unique"
            )
        object.__setattr__(self, "slots", slots)
        handles = tuple(
            _identity(item, "emphasized handle")
            for item in self.emphasized_handles
        )
        if len(set(handles)) != len(handles):
            raise SectionSemanticDisplayError(
                "emphasized_handles must be unique"
            )
        object.__setattr__(self, "emphasized_handles", tuple(sorted(handles)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_DISPLAY_FRAME_SCHEMA,
            "sectionId": self.section_id,
            "catalogDigest": self.catalog_digest,
            "mode": self.mode.value,
            "emphasizedHandles": list(self.emphasized_handles),
            "slots": [item.to_dict() for item in self.slots],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def opacity_for(self, slot_id: str) -> float:
        key = _identity(slot_id, "slot_id")
        for item in self.slots:
            if item.slot_id == key:
                return item.opacity_multiplier
        raise SectionSemanticDisplayError(f"unknown semantic slot {key!r}")


def compile_section_display(
    catalog: SectionDisplayCatalog,
    instruction: SectionDisplayInstruction,
) -> SectionDisplayFrame:
    """Compile a semantic instruction without activating inactive geometry."""

    if not isinstance(catalog, SectionDisplayCatalog):
        raise TypeError("catalog must be a SectionDisplayCatalog")
    if not isinstance(instruction, SectionDisplayInstruction):
        raise TypeError("instruction must be a SectionDisplayInstruction")
    emphasized_slots: set[str] = set()
    for handle_id in instruction.emphasized_handles:
        emphasized_slots.update(catalog.handle(handle_id).slot_ids)
    use_emphasis = bool(instruction.emphasized_handles)
    states = []
    for slot in catalog.slots:
        emphasized = slot.slot_id in emphasized_slots
        opacity = instruction.policy.opacity_for(slot.role)
        if use_emphasis and not emphasized:
            opacity *= instruction.dim_unemphasized
        states.append(
            SectionDisplaySlotState(
                slot.slot_id,
                slot.role,
                slot.source_id,
                slot.topology_bank,
                opacity,
                emphasized,
            )
        )
    return SectionDisplayFrame(
        catalog.section_id,
        catalog.digest,
        instruction.policy.mode,
        tuple(states),
        instruction.emphasized_handles,
    )


__all__ = [
    "SECTION_DISPLAY_CATALOG_SCHEMA",
    "SECTION_DISPLAY_FRAME_SCHEMA",
    "SECTION_DISPLAY_INSTRUCTION_SCHEMA",
    "SectionDisplayCatalog",
    "SectionDisplayFrame",
    "SectionDisplayHandle",
    "SectionDisplayInstruction",
    "SectionDisplayMode",
    "SectionDisplayPolicy",
    "SectionDisplayRole",
    "SectionDisplaySlotState",
    "SectionSemanticDisplayError",
    "SectionSemanticSlot",
    "compile_section_display",
]

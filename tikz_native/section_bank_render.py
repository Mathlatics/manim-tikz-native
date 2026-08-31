"""Immutable render-bank frames for certified quadric-section handoffs.

The topology timeline and semantic display catalog are renderer-neutral.  This
module is the small transaction payload between those contracts and a future
renderer binding: every layer identifies one preallocated semantic bank, the
certified reference geometry loaded into it, and its handoff opacity.

No renderer object is allocated here.  A binding owns the mutable bank slots
and implements one full-frame snapshot/apply/restore transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import unicodedata

from polyhedron_visibility.quadrics.section_timeline_transition import (
    SectionTimelineLayerRole,
)

from .parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)


SECTION_BANK_RENDER_FRAME_SCHEMA = "parallel-section-bank-render-frame/v1"
SECTION_BANK_RENDER_CHANNEL = "section-bank-render-frame"


class SectionBankRenderError(ValueError):
    """A semantic render-bank frame is malformed or ambiguous."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SectionBankRenderError(f"{label} must be a non-empty string")
    result = value.strip()
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SectionBankRenderError(
            f"{label} must contain valid Unicode"
        ) from exc
    return unicodedata.normalize("NFC", result)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SectionBankRenderError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SectionBankRenderError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise SectionBankRenderError(f"{label} must be finite")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SectionBankRenderError(
            f"{label} must be a non-negative integer"
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
        raise SectionBankRenderError(
            f"section bank render value is not canonical JSON: {exc}"
        ) from exc


def _sha256_digest(value: object, label: str) -> str:
    digest = _identity(value, label)
    payload = digest[7:] if digest.startswith("sha256:") else ""
    if len(payload) != 64 or any(
        item not in "0123456789abcdef" for item in payload
    ):
        raise SectionBankRenderError(
            f"{label} must be a lowercase sha256 digest"
        )
    return digest


@dataclass(frozen=True, slots=True)
class SectionBankRenderLayer:
    """One fixed semantic bank and the certified geometry loaded into it."""

    bank_index: int
    semantic_bank_id: str
    reference_frame_index: int
    geometry_time: float
    opacity: float
    branch_count: int
    isolated_point_count: int
    role: SectionTimelineLayerRole
    geometry_digest: str
    active_cap_chord_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bank_index = _nonnegative_integer(self.bank_index, "bank_index")
        if bank_index not in {0, 1}:
            raise SectionBankRenderError("bank_index must be 0 or 1")
        object.__setattr__(self, "bank_index", bank_index)
        object.__setattr__(
            self,
            "semantic_bank_id",
            _identity(self.semantic_bank_id, "semantic_bank_id"),
        )
        object.__setattr__(
            self,
            "reference_frame_index",
            _nonnegative_integer(
                self.reference_frame_index,
                "reference_frame_index",
            ),
        )
        object.__setattr__(
            self,
            "geometry_time",
            _finite(self.geometry_time, "geometry_time"),
        )
        opacity = _finite(self.opacity, "opacity")
        if opacity < 0.0 or opacity > 1.0:
            raise SectionBankRenderError("opacity must lie in [0, 1]")
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(
            self,
            "branch_count",
            _nonnegative_integer(self.branch_count, "branch_count"),
        )
        object.__setattr__(
            self,
            "isolated_point_count",
            _nonnegative_integer(
                self.isolated_point_count,
                "isolated_point_count",
            ),
        )
        try:
            role = SectionTimelineLayerRole(self.role)
        except (TypeError, ValueError) as exc:
            raise SectionBankRenderError(
                "role must be a SectionTimelineLayerRole"
            ) from exc
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "geometry_digest",
            _sha256_digest(self.geometry_digest, "geometry_digest"),
        )
        if isinstance(self.active_cap_chord_ids, (str, bytes, bytearray)):
            raise TypeError(
                "active_cap_chord_ids must be a sequence of identities"
            )
        cap_ids = tuple(
            sorted(
                _identity(item, "active cap chord id")
                for item in self.active_cap_chord_ids
            )
        )
        if len(set(cap_ids)) != len(cap_ids):
            raise SectionBankRenderError(
                "layer active_cap_chord_ids must be unique"
            )
        object.__setattr__(self, "active_cap_chord_ids", cap_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "bankIndex": self.bank_index,
            "semanticBankId": self.semantic_bank_id,
            "referenceFrameIndex": self.reference_frame_index,
            "geometryTime": self.geometry_time,
            "opacity": self.opacity,
            "branchCount": self.branch_count,
            "isolatedPointCount": self.isolated_point_count,
            "role": self.role.value,
            "geometryDigest": self.geometry_digest,
            "activeCapChordIds": list(self.active_cap_chord_ids),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            self.to_json().encode("utf-8")
        ).hexdigest()


_SINGLE_LAYER_ROLES = frozenset(
    {
        SectionTimelineLayerRole.LIVE,
        SectionTimelineLayerRole.EXACT_CRITICAL,
    }
)
_TWO_LAYER_ROLE_SETS = frozenset(
    {
        frozenset(
            {
                SectionTimelineLayerRole.LIVE_BEFORE,
                SectionTimelineLayerRole.EXACT_CRITICAL,
            }
        ),
        frozenset(
            {
                SectionTimelineLayerRole.EXACT_CRITICAL,
                SectionTimelineLayerRole.LIVE_AFTER,
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class SectionBankRenderFrame:
    """Canonical one- or two-bank commit payload at one global time."""

    time: float
    layers: tuple[SectionBankRenderLayer, ...]
    active_cap_chord_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite(self.time, "frame time"))
        authored_layers = tuple(self.layers)
        if not authored_layers or len(authored_layers) > 2:
            raise SectionBankRenderError(
                "render frame requires one or two bank layers"
            )
        if not all(
            isinstance(item, SectionBankRenderLayer)
            for item in authored_layers
        ):
            raise TypeError(
                "layers must contain SectionBankRenderLayer values"
            )
        layers = tuple(sorted(authored_layers, key=lambda item: item.bank_index))
        bank_indices = tuple(item.bank_index for item in layers)
        semantic_ids = tuple(item.semantic_bank_id for item in layers)
        if len(set(bank_indices)) != len(bank_indices):
            raise SectionBankRenderError(
                "render frame layers must use unique bank indices"
            )
        if len(set(semantic_ids)) != len(semantic_ids):
            raise SectionBankRenderError(
                "render frame layers must use unique semantic bank ids"
            )
        if any(item.opacity <= 0.0 for item in layers):
            raise SectionBankRenderError(
                "render frame layers must have positive active opacity"
            )
        if abs(sum(item.opacity for item in layers) - 1.0) > 1.0e-12:
            raise SectionBankRenderError(
                "render frame layer opacities must sum to one"
            )
        roles = frozenset(item.role for item in layers)
        if len(layers) == 1:
            if layers[0].role not in _SINGLE_LAYER_ROLES:
                raise SectionBankRenderError(
                    "a single bank layer must be live or exact-critical"
                )
        elif roles not in _TWO_LAYER_ROLE_SETS:
            raise SectionBankRenderError(
                "two bank layers must describe one certified crossfade side"
            )
        object.__setattr__(self, "layers", layers)

        if isinstance(self.active_cap_chord_ids, (str, bytes, bytearray)):
            raise TypeError(
                "active_cap_chord_ids must be a sequence of string ids"
            )
        authored_cap_ids = tuple(
            sorted(
                _identity(item, "active cap chord id")
                for item in self.active_cap_chord_ids
            )
        )
        if len(set(authored_cap_ids)) != len(authored_cap_ids):
            raise SectionBankRenderError(
                "active_cap_chord_ids must be unique"
            )
        derived_cap_ids = tuple(
            sorted(
                {
                    cap_id
                    for layer in layers
                    for cap_id in layer.active_cap_chord_ids
                }
            )
        )
        if authored_cap_ids and authored_cap_ids != derived_cap_ids:
            raise SectionBankRenderError(
                "frame cap-chord ids must equal the union of layer cap ids"
            )
        object.__setattr__(self, "active_cap_chord_ids", derived_cap_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SECTION_BANK_RENDER_FRAME_SCHEMA,
            "time": self.time,
            "layers": [item.to_dict() for item in self.layers],
            "activeCapChordIds": list(self.active_cap_chord_ids),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(
            self.to_json().encode("utf-8")
        ).hexdigest()


def section_bank_frame_participant(
    target: object,
    *,
    channel_name: str = SECTION_BANK_RENDER_CHANNEL,
    participant_id: str = "section-bank-render",
) -> ParallelFrameParticipant[ParallelFrameState]:
    """Bind one target through an atomic full-frame bank transaction."""

    channel = _identity(channel_name, "channel_name")
    snapshot = getattr(target, "snapshot_section_bank_render_state", None)
    apply = getattr(target, "apply_section_bank_render_frame", None)
    restore = getattr(target, "restore_section_bank_render_state", None)
    if not all(callable(item) for item in (snapshot, apply, restore)):
        raise TypeError(
            "bank render target must provide snapshot, apply, and restore methods"
        )

    def prepare(frame: ParallelFrameState) -> SectionBankRenderFrame:
        if not isinstance(frame, ParallelFrameState):
            raise TypeError("bank render participant requires ParallelFrameState")
        value = frame.channel(channel)
        if not isinstance(value, SectionBankRenderFrame):
            raise TypeError(
                "bank render channel must contain SectionBankRenderFrame"
            )
        return value

    def commit(value: object) -> None:
        if not isinstance(value, SectionBankRenderFrame):
            raise TypeError(
                "prepared bank render value must be SectionBankRenderFrame"
            )
        apply(value)

    return ParallelFrameParticipant(
        participant_id=participant_id,
        phase=ParallelFramePhase.GEOMETRY,
        prepare=prepare,
        snapshot=snapshot,
        commit=commit,
        rollback=restore,
        binding_kind=ParallelFrameBindingKind.SECTION_BANK,
    )


__all__ = [
    "SECTION_BANK_RENDER_CHANNEL",
    "SECTION_BANK_RENDER_FRAME_SCHEMA",
    "SectionBankRenderError",
    "SectionBankRenderFrame",
    "SectionBankRenderLayer",
    "section_bank_frame_participant",
]

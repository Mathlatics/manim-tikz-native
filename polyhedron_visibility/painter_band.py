from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from manim import Mobject
from manim.mobject.types.image_mobject import AbstractImageMobject
from manim.mobject.types.point_cloud_mobject import PMobject
from manim.mobject.types.vectorized_mobject import VMobject

from .binding import OcclusionBindingError


class ManagedPainterBandError(OcclusionBindingError):
    """Raised before a managed far-to-near z band is mutated."""


class ScenePainterBandError(ManagedPainterBandError):
    """Raised before a Scene-level painter-band reservation is mutated."""


@dataclass(frozen=True, slots=True, eq=False)
class ScenePainterBandReservation:
    """One identity-bearing request for a Scene-level painter z band.

    ``owner_key`` is the stable logical identity of the owner inside one
    Scene.  Reservation identity is deliberately *object identity*, not value
    equality: reserving the same instance twice is idempotent, while a second
    instance with the same owner key is diagnosed as a duplicate owner.

    When ``exact`` is false, allocation starts at ``preferred_z_band`` and
    moves upward to the first non-overlapping slot.  When it is true, the
    preferred band is the only acceptable slot.
    """

    owner_key: Hashable
    preferred_z_band: tuple[float, float]
    exact: bool = False

    def __post_init__(self) -> None:
        try:
            hash(self.owner_key)
        except (TypeError, ValueError) as exc:
            raise ScenePainterBandError(
                "scene painter-band owner_key must be hashable"
            ) from exc
        object.__setattr__(
            self,
            "preferred_z_band",
            _validate_scene_painter_band(self.preferred_z_band),
        )
        if not isinstance(self.exact, bool):
            raise TypeError("scene painter-band exact must be a bool")


@dataclass(frozen=True, slots=True)
class ScenePainterBandAllocation:
    """Read-only public description of one active Scene reservation."""

    owner_key: Hashable
    z_band: tuple[float, float]
    exact: bool


@dataclass(frozen=True, slots=True)
class _ScenePainterBandEntry:
    reservation: ScenePainterBandReservation
    z_band: tuple[float, float]


_SCENE_PAINTER_BAND_REGISTRY = (
    "_polyhedron_visibility_scene_painter_band_reservations"
)


def _validate_scene_painter_band(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ScenePainterBandError(
            "scene painter z band must be a two-value tuple"
        )
    low, high = (float(item) for item in value)
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or low >= high
        or not np.isfinite(high - low)
    ):
        raise ScenePainterBandError(
            "scene painter z band must contain two finite increasing values "
            "with a finite span"
        )
    return low, high


def _read_scene_painter_band_registry(
    scene: object,
) -> dict[Hashable, _ScenePainterBandEntry] | None:
    value = getattr(scene, _SCENE_PAINTER_BAND_REGISTRY, None)
    if value is None:
        return None
    if not isinstance(value, dict) or any(
        not isinstance(entry, _ScenePainterBandEntry)
        for entry in value.values()
    ):
        raise ScenePainterBandError(
            "Scene painter-band reservation registry is invalid"
        )
    return value


def _scene_painter_bands_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    # Endpoints are drawable z slots too, so touching closed bands conflict.
    return not (first[1] < second[0] or second[1] < first[0])


def reserve_scene_painter_band(
    scene: object,
    reservation: ScenePainterBandReservation,
) -> tuple[float, float]:
    """Reserve and return one non-overlapping z band for ``reservation``.

    All validation and conflict detection happen before the Scene registry is
    created or changed.  See :class:`ScenePainterBandReservation` for the
    duplicate-owner and idempotency rules.
    """

    if not isinstance(reservation, ScenePainterBandReservation):
        raise TypeError("reservation must be a ScenePainterBandReservation")
    registry = _read_scene_painter_band_registry(scene)
    existing = None if registry is None else registry.get(reservation.owner_key)
    if existing is not None:
        if existing.reservation is reservation:
            return existing.z_band
        raise ScenePainterBandError(
            f"duplicate Scene painter-band owner {reservation.owner_key!r}"
        )

    preferred = reservation.preferred_z_band
    occupied = () if registry is None else tuple(
        entry.z_band for entry in registry.values()
    )
    if reservation.exact:
        if any(
            _scene_painter_bands_overlap(preferred, other)
            for other in occupied
        ):
            raise ScenePainterBandError(
                "exact Scene painter z band conflicts with an active reservation"
            )
        actual = preferred
    else:
        width = preferred[1] - preferred[0]
        gap = max(1.0, width * 1.0e-6)
        actual = preferred
        while True:
            conflicts = tuple(
                other
                for other in occupied
                if _scene_painter_bands_overlap(actual, other)
            )
            if not conflicts:
                break
            blocking_high = max(other[1] for other in conflicts)
            low = blocking_high + gap
            high = low + width
            if (
                not np.isfinite(low)
                or not np.isfinite(high)
                or not blocking_high < low < high
            ):
                raise ScenePainterBandError(
                    "automatic Scene painter z band overflowed or could not "
                    "advance above active reservations"
                )
            actual = (low, high)

    entry = _ScenePainterBandEntry(reservation, actual)
    if registry is None:
        registry = {}
        setattr(scene, _SCENE_PAINTER_BAND_REGISTRY, registry)
    registry[reservation.owner_key] = entry
    return actual


def release_scene_painter_band(
    scene: object,
    reservation: ScenePainterBandReservation,
) -> bool:
    """Release ``reservation`` and return whether an active entry was removed.

    Releasing an already-released token is a no-op.  A different token cannot
    release the band belonging to the same logical owner.
    """

    if not isinstance(reservation, ScenePainterBandReservation):
        raise TypeError("reservation must be a ScenePainterBandReservation")
    registry = _read_scene_painter_band_registry(scene)
    if registry is None:
        return False
    existing = registry.get(reservation.owner_key)
    if existing is None:
        return False
    if existing.reservation is not reservation:
        raise ScenePainterBandError(
            f"Scene painter-band owner {reservation.owner_key!r} belongs to "
            "a different reservation token"
        )
    del registry[reservation.owner_key]
    if not registry:
        try:
            delattr(scene, _SCENE_PAINTER_BAND_REGISTRY)
        except AttributeError:
            pass
    return True


def scene_painter_band_allocations(
    scene: object,
) -> tuple[ScenePainterBandAllocation, ...]:
    """Return active Scene allocations in reservation order."""

    registry = _read_scene_painter_band_registry(scene)
    if registry is None:
        return ()
    return tuple(
        ScenePainterBandAllocation(
            owner_key=entry.reservation.owner_key,
            z_band=entry.z_band,
            exact=entry.reservation.exact,
        )
        for entry in registry.values()
    )


@dataclass(frozen=True, slots=True)
class PreparedPainterItem:
    item_id: str
    mobject: Mobject
    z_index: float


@dataclass(frozen=True, slots=True)
class PreparedPainterBand:
    items: tuple[PreparedPainterItem, ...]


class _PainterBandActiveState(dict[str, float]):
    """Rollback state with identity metadata and ordinary mapping behavior."""

    def __init__(
        self,
        z_indices: Mapping[str, float],
        mobject_ids: Mapping[str, int],
    ) -> None:
        super().__init__((str(key), float(value)) for key, value in z_indices.items())
        self.mobject_ids = {
            str(key): int(value) for key, value in mobject_ids.items()
        }


def _scene_family(containers: Sequence[list[object]]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[int] = set()
    for container in containers:
        for root in container:
            for member in root.get_family():
                if id(member) not in seen:
                    seen.add(id(member))
                    result.append(member)
    return tuple(result)


def _has_drawable_points(value: object) -> bool:
    if not isinstance(value, (VMobject, PMobject, AbstractImageMobject)):
        # Cairo deliberately treats a plain Mobject (including ValueTracker)
        # as a non-rendering scene participant even when it stores points.
        return False
    points = np.asarray(getattr(value, "points", np.empty((0, 3))), dtype=float)
    if points.size == 0:
        return False
    for attribute in ("fill_rgbas", "stroke_rgbas", "background_stroke_rgbas"):
        if not hasattr(value, attribute):
            continue
        rgba = np.asarray(getattr(value, attribute), dtype=float)
        if rgba.ndim >= 1 and rgba.shape[-1] >= 4 and np.any(rgba[..., 3] > 0.0):
            return True
    # Some custom Cairo mobjects do not expose style arrays.  Treat finite
    # point-bearing objects as drawable rather than silently sharing the band.
    return True


class ManagedPainterBand:
    """Map one complete painter order into one reserved Cairo z band.

    The class deliberately knows nothing about faces, paths, or visibility.  A
    domain binding supplies the source objects which own the reservation, the
    active item-to-Mobject mapping, and the complete far-to-near draw order.
    """

    def __init__(
        self,
        *,
        z_band: tuple[float, float] | None = None,
        require_distinct_source_z: bool = False,
        managed_roots: Sequence[Mobject] = (),
    ) -> None:
        self._requested_band = z_band
        self._require_distinct_source_z = bool(require_distinct_source_z)
        self._managed_family_ids = {
            id(member)
            for root in managed_roots
            for member in root.get_family()
        }
        self._z_low = 0.0
        self._z_high = 0.0
        self._configured = False
        self._active_z_indices: dict[str, float] = {}
        self._active_mobject_ids: dict[str, int] = {}

    @staticmethod
    def _validate_explicit_band(value: tuple[float, float]) -> tuple[float, float]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ManagedPainterBandError("painter_z_band must be a two-value tuple")
        low, high = (float(item) for item in value)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ManagedPainterBandError(
                "painter_z_band must contain two finite increasing values"
            )
        return low, high

    def configure(
        self,
        *,
        containers: Sequence[list[object]],
        sources: Mapping[str, Mobject],
    ) -> None:
        if not sources:
            raise ManagedPainterBandError("managed painter band requires source objects")
        family = _scene_family(containers)
        scene_ids = {id(item) for item in family}
        source_family_ids = {
            id(member)
            for source in sources.values()
            for member in source.get_family()
        }
        authored: dict[str, float] = {}
        for source_id, source in sorted(sources.items()):
            if id(source) not in scene_ids:
                raise ManagedPainterBandError(
                    f"managed painter source {source_id!r} is not owned by the Scene"
                )
            value = float(source.z_index)
            if not np.isfinite(value):
                raise ManagedPainterBandError(
                    f"managed painter source {source_id!r} has non-finite z_index"
                )
            authored[source_id] = value

        if self._configured:
            low, high = self._z_low, self._z_high
        else:
            if self._requested_band is None:
                if self._require_distinct_source_z and len(set(authored.values())) != len(authored):
                    raise ManagedPainterBandError(
                        "managed painter sources must occupy distinct authored z_index slots"
                    )
                if len(set(authored.values())) < 2:
                    raise ManagedPainterBandError(
                        "derived painter band requires at least two authored z_index values"
                    )
                low, high = min(authored.values()), max(authored.values())
            else:
                low, high = self._validate_explicit_band(self._requested_band)

        for member in family:
            if (
                id(member) in source_family_ids
                or id(member) in self._managed_family_ids
                or not _has_drawable_points(member)
            ):
                continue
            value = float(getattr(member, "z_index", float("nan")))
            if np.isfinite(value) and low <= value <= high:
                raise ManagedPainterBandError(
                    "an unrelated Scene drawable occupies the managed painter z band"
                )
        self._z_low = low
        self._z_high = high
        self._configured = True

    def prepare(
        self,
        *,
        draw_order: Sequence[str],
        item_mobjects: Mapping[str, Mobject],
    ) -> PreparedPainterBand:
        if not self._configured:
            raise ManagedPainterBandError("managed painter band is not configured")
        order = tuple(str(item) for item in draw_order)
        if len(set(order)) != len(order):
            raise ManagedPainterBandError("managed painter draw_order contains duplicates")
        if set(order) != set(item_mobjects):
            raise ManagedPainterBandError(
                "managed painter draw_order does not cover every active item"
            )
        object_ids = [id(item_mobjects[item_id]) for item_id in order]
        if len(set(object_ids)) != len(object_ids):
            raise ManagedPainterBandError(
                "one Mobject cannot represent multiple active painter items"
            )
        denominator = max(1, len(order) - 1)
        return PreparedPainterBand(
            tuple(
                PreparedPainterItem(
                    item_id,
                    item_mobjects[item_id],
                    self._z_low
                    + (self._z_high - self._z_low) * rank / denominator,
                )
                for rank, item_id in enumerate(order)
            )
        )

    def apply(self, prepared: PreparedPainterBand) -> None:
        changed = {item.item_id for item in self.changed_items(prepared)}
        active: dict[str, float] = {}
        active_mobjects: dict[str, int] = {}
        for item in prepared.items:
            if item.item_id in changed:
                item.mobject.set_z_index(item.z_index, family=True)
            active[item.item_id] = item.z_index
            active_mobjects[item.item_id] = id(item.mobject)
        self._active_z_indices = active
        self._active_mobject_ids = active_mobjects

    def changed_items(
        self,
        prepared: PreparedPainterBand,
    ) -> tuple[PreparedPainterItem, ...]:
        """Return only painter items whose family needs a z-index write."""

        if not isinstance(prepared, PreparedPainterBand):
            raise TypeError("prepared must be a PreparedPainterBand")
        return tuple(
            item
            for item in prepared.items
            if self._active_z_indices.get(item.item_id) != item.z_index
            or self._active_mobject_ids.get(item.item_id) != id(item.mobject)
        )

    @property
    def active_z_indices(self) -> dict[str, float]:
        return dict(self._active_z_indices)

    def capture_active_state(self) -> dict[str, float]:
        return _PainterBandActiveState(
            self._active_z_indices,
            self._active_mobject_ids,
        )

    def restore_active_state(self, state: Mapping[str, float]) -> None:
        self._active_z_indices = {
            str(item_id): float(value) for item_id, value in state.items()
        }
        self._active_mobject_ids = (
            dict(state.mobject_ids)
            if isinstance(state, _PainterBandActiveState)
            else {}
        )

    @property
    def z_band(self) -> tuple[float, float]:
        if not self._configured:
            raise ManagedPainterBandError("managed painter band is not configured")
        return self._z_low, self._z_high

    def restore(self) -> None:
        self._active_z_indices = {}
        self._active_mobject_ids = {}
        self._configured = False
        self._z_low = 0.0
        self._z_high = 0.0


__all__ = [
    "ManagedPainterBand",
    "ManagedPainterBandError",
    "PreparedPainterBand",
    "PreparedPainterItem",
    "ScenePainterBandAllocation",
    "ScenePainterBandError",
    "ScenePainterBandReservation",
    "release_scene_painter_band",
    "reserve_scene_painter_band",
    "scene_painter_band_allocations",
]

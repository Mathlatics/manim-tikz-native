"""Static Cairo authoring facade for diagrammatic Dandelin constructions.

``DandelinSection3D`` composes the existing finite-cone section controller with
one sphere-only teaching overlay.  The construction is immutable for the
lifetime of the facade.  It deliberately does not claim physical depth between
the contained spheres and their tangent cone; callers can inspect
``visibility_authoritative`` and receive ``False``.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import isfinite
from typing import Iterator, Mapping

import numpy as np
from manim import Dot, Mobject, ThreeDCamera, VGroup

from ..geometry import GeometryContext, ResolvedGeometryContext
from ..painter_band import (
    ScenePainterBandReservation,
    release_scene_painter_band,
    reserve_scene_painter_band,
)
from .authoring import QuadricSection3D
from .composite_authoring import CompositeQuadricSection3D
from .contract import ConeModel, ConeSpec, PlaneDisplayPatchSpec, SectionPlane
from .dandelin import DandelinConstruction3D, compute_dandelin_construction
from .dandelin_overlay import (
    DandelinTeachingOverlay3D,
    DandelinTeachingOverlayError,
    build_dandelin_teaching_overlay,
)
from .manim import (
    DEFAULT_QUADRIC_VIEW,
    ProjectionValue,
    QUADRIC_MANIM_LIMITS,
    QuadricBoundaryStyle,
    QuadricManimError,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from .manim_runtime import (
    _ResolvedParallelCameraFrame,
    _coerce_projection_frame,
    _projection_display_offset,
    _register_fixed_frame,
    _remove_fixed_frame,
    _scene_containers,
)
from .plane_patch import PlanePatchFitError, fit_plane_display_patch


class DandelinSectionAuthoringError(DandelinTeachingOverlayError):
    """A static diagrammatic Dandelin facade was configured ambiguously."""


_DEFAULT_PREFERRED_PAINTER_Z_BAND = (10.0, 32.0)
_PAINTER_BAND_PARTS = 22.0


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise DandelinSectionAuthoringError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinSectionAuthoringError(
            f"{label} must be finite and positive"
        ) from exc
    if not isfinite(result) or result <= 0.0:
        raise DandelinSectionAuthoringError(f"{label} must be finite and positive")
    return result


def _painter_band(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise DandelinSectionAuthoringError(
            f"{label} must be a two-value tuple"
        )
    if any(isinstance(item, (bool, np.bool_)) for item in value):
        raise DandelinSectionAuthoringError(
            f"{label} must contain two finite increasing values"
        )
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinSectionAuthoringError(
            f"{label} must contain two finite increasing values"
        ) from exc
    if (
        not isfinite(low)
        or not isfinite(high)
        or low >= high
        or not isfinite(high - low)
    ):
        raise DandelinSectionAuthoringError(
            f"{label} must contain two finite increasing values with a finite span"
        )
    return low, high


def _automatic_painter_subbands(
    aggregate: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    low, high = aggregate
    span = high - low
    section_high = low + span * (10.0 / _PAINTER_BAND_PARTS)
    overlay_low = low + span * (11.0 / _PAINTER_BAND_PARTS)
    overlay_high = low + span * (21.0 / _PAINTER_BAND_PARTS)
    values = (low, section_high, overlay_low, overlay_high, high)
    if not all(isfinite(value) for value in values) or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise DandelinSectionAuthoringError(
            "automatic Dandelin painter sub-band split lost strict ordering "
            "at floating-point precision"
        )
    return (low, section_high), (overlay_low, overlay_high), high


def _legacy_painter_configuration(
    section: object,
    overlay: object,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
] | None:
    if section is None and overlay is None:
        return None
    if section is None or overlay is None:
        raise DandelinSectionAuthoringError(
            "section_painter_z_band and overlay_painter_z_band must be "
            "provided together"
        )
    section_band = _painter_band(section, "section_painter_z_band")
    overlay_band = _painter_band(overlay, "overlay_painter_z_band")
    focus_z = overlay_band[1] + 1.0
    if (
        not isfinite(focus_z)
        or not section_band[1] < overlay_band[0]
        or not overlay_band[1] < focus_z
    ):
        raise DandelinSectionAuthoringError(
            "legacy Dandelin painter bands must satisfy section_high < "
            "overlay_low < overlay_high < overlay_high + 1"
        )
    aggregate = _painter_band(
        (section_band[0], focus_z),
        "aggregate Dandelin painter_z_band",
    )
    return aggregate, section_band, overlay_band, focus_z


def _static_projection_frame(
    scene: object,
    value: ProjectionValue | None,
) -> _ResolvedParallelCameraFrame:
    """Resolve and freeze one complete affine parallel-camera frame.

    The low-level renderer keeps viewport-relative semantic states live by
    adding the current Manim frame center during every update.  This facade is
    deliberately static, so it resolves that final translation once as well as
    freezing the effective view matrix.  Passing the resulting non-relative
    frame to every child guarantees that the section, overlay, and focus dots
    consume exactly the same affine projection for their whole lifetime.
    """

    if callable(value):
        raise DandelinSectionAuthoringError(
            "DandelinSection3D v1 requires one immutable parallel projection; "
            "callable projection is unsupported"
        )
    try:
        resolved = _coerce_projection_frame(
            DEFAULT_QUADRIC_VIEW if value is None else value,
            scene=scene,
        )
        final_offset = _projection_display_offset(scene, resolved)
        return _ResolvedParallelCameraFrame(
            resolved.view,
            final_offset,
            viewport_relative=False,
        )
    except (QuadricManimError, TypeError, ValueError) as exc:
        raise DandelinSectionAuthoringError(
            f"invalid static parallel projection: {exc}"
        ) from exc


def _directrix_patch(
    construction: DandelinConstruction3D,
    margin_ratio: float,
) -> PlaneDisplayPatchSpec:
    try:
        base = fit_plane_display_patch(
            f"{construction.plane.plane_id}:dandelin-base-patch",
            construction.plane,
            construction.cone.render_components,
            margin_ratio=margin_ratio,
        ).patch
    except PlanePatchFitError as exc:
        raise DandelinSectionAuthoringError(
            f"Dandelin directrix patch fitting failed: {exc}"
        ) from exc
    center = np.asarray(base.center_coordinates, dtype=float)
    lower = center - np.asarray((base.half_width, base.half_height), dtype=float)
    upper = center + np.asarray((base.half_width, base.half_height), dtype=float)
    padding = max(base.half_width, base.half_height) * 0.18
    for directrix in construction.directrices:
        coordinates = np.asarray(directrix.point.coordinates, dtype=float)
        lower = np.minimum(lower, coordinates - padding)
        upper = np.maximum(upper, coordinates + padding)
    expanded_center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return PlaneDisplayPatchSpec(
        f"{construction.plane.plane_id}:dandelin-directrix-patch",
        construction.plane.plane_id,
        float(half[0]),
        float(half[1]),
        tuple(float(item) for item in expanded_center),
    )


DEFAULT_DANDELIN_SECTION_STYLE = QuadricManimStyle(
    surface_fill_color="#2878A5",
    surface_fill_opacity=0.58,
    surface_stroke_color="#67D8EE",
    surface_stroke_opacity=0.0,
    visible_curve_color="#FFD166",
    visible_curve_width=4.0,
    hidden_curve_color="#FFD166",
    hidden_curve_width=2.8,
    hidden_curve_opacity=0.48,
    section_plane_fill_color="#2CB9A4",
    section_plane_fill_opacity=0.16,
    section_plane_stroke_color="#7EE5D5",
    section_plane_stroke_opacity=0.55,
    cone_lateral_fill_colors=("#173753", "#4F9AC1", "#1D4368"),
)

DEFAULT_DANDELIN_OVERLAY_STYLE = QuadricManimStyle(
    surface_fill_color="#F59E7A",
    surface_fill_opacity=0.38,
    surface_stroke_color="#FFD0B8",
    surface_stroke_width=1.5,
    surface_stroke_opacity=0.72,
    visible_curve_color="#FF8A5B",
    visible_curve_width=3.2,
    hidden_curve_color="#FF8A5B",
    hidden_curve_width=2.2,
    hidden_curve_opacity=0.42,
    dash_length=0.10,
    dash_gap=0.08,
)


class DandelinSection3D:
    """Author one immutable, diagrammatic Dandelin classroom construction.

    All geometry is derived before any Manim object is attached.  The cone
    section keeps using ``QuadricSection3D`` (or the existing open-double
    coordinator), while a sphere-only ``QuadricOcclusion3D`` displays the
    certified auxiliary geometry in a separate top teaching band.
    """

    visibility_authoritative = False
    overlay_mode = "diagrammatic"

    def __init__(
        self,
        scene: object,
        *,
        cone: ConeSpec,
        plane: SectionPlane,
        construction_id: str,
        projection: ProjectionValue | None = None,
        section_style: QuadricManimStyle = DEFAULT_DANDELIN_SECTION_STYLE,
        overlay_style: QuadricManimStyle = DEFAULT_DANDELIN_OVERLAY_STYLE,
        boundary_styles: Mapping[str, QuadricBoundaryStyle] | None = None,
        limits: QuadricManimLimits = QUADRIC_MANIM_LIMITS,
        context: GeometryContext | ResolvedGeometryContext | None = None,
        coefficient_tolerance: float | None = None,
        max_chord_error: float = 0.008,
        section_max_screen_error: float = 0.08,
        patch_margin: float = 0.14,
        show_contact_circles: bool = True,
        show_directrices: bool = True,
        show_foci: bool = True,
        focus_color: object = "#FFF4A3",
        focus_radius: float = 0.065,
        section_painter_z_band: tuple[float, float] | None = None,
        overlay_painter_z_band: tuple[float, float] | None = None,
        preferred_painter_z_band: tuple[float, float] = (
            _DEFAULT_PREFERRED_PAINTER_Z_BAND
        ),
    ) -> None:
        if not isinstance(cone, ConeSpec):
            raise TypeError("cone must be a ConeSpec")
        if not isinstance(plane, SectionPlane):
            raise TypeError("plane must be a SectionPlane")
        if not isinstance(section_style, QuadricManimStyle):
            raise TypeError("section_style must be a QuadricManimStyle")
        if not isinstance(overlay_style, QuadricManimStyle):
            raise TypeError("overlay_style must be a QuadricManimStyle")
        if not isinstance(limits, QuadricManimLimits):
            raise TypeError("limits must be a QuadricManimLimits")
        for name, value in (
            ("show_contact_circles", show_contact_circles),
            ("show_directrices", show_directrices),
            ("show_foci", show_foci),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        legacy_painter = _legacy_painter_configuration(
            section_painter_z_band,
            overlay_painter_z_band,
        )
        preferred_painter = _painter_band(
            preferred_painter_z_band,
            "preferred_painter_z_band",
        )
        self.scene = scene
        self.cone = cone
        self.plane = plane
        self._projection_frame = _static_projection_frame(scene, projection)
        self.view = self._projection_frame.view
        self.display_offset = self._projection_frame.screen_offset
        self.context = context
        self.focus_radius = _positive(focus_radius, "focus_radius")
        resolved_patch_margin = _positive(patch_margin, "patch_margin")
        self.construction = compute_dandelin_construction(
            construction_id,
            cone,
            plane,
            context=context,
            coefficient_tolerance=coefficient_tolerance,
        )
        self.resolved_context = self.construction.certification_context
        patch = _directrix_patch(
            self.construction,
            resolved_patch_margin,
        )
        self.teaching_overlay: DandelinTeachingOverlay3D = (
            build_dandelin_teaching_overlay(
                self.construction,
                patch,
                context=self.resolved_context,
            )
        )
        self._section_configuration = {
            "section_id": f"{self.construction.construction_id}:section",
            "plane": plane,
            "projection": self._projection_frame,
            "paint_policy": "depth_aware_diagrammatic",
            "style": section_style,
            "boundary_styles": boundary_styles,
            "limits": limits,
            "max_chord_error": max_chord_error,
            "context": self.resolved_context,
            "coefficient_tolerance": coefficient_tolerance,
            "section_max_screen_error": section_max_screen_error,
            "plane_patch_margin": resolved_patch_margin,
            "include_surface_boundaries": True,
        }
        self._overlay_curves = (
            *(
                self.teaching_overlay.contact_curves
                if show_contact_circles
                else ()
            ),
            *(
                self.teaching_overlay.directrix_curves
                if show_directrices
                else ()
            ),
        )
        self._overlay_configuration = {
            "surfaces": self.teaching_overlay.sphere_surfaces,
            "curves": self._overlay_curves,
            "projection": self._projection_frame,
            "paint_policy": "diagrammatic",
            "style": overlay_style,
            "limits": limits,
            "max_chord_error": max_chord_error,
            "context": self.resolved_context,
            "surface_order_mode": "explicit",
            "boundary_visibility_mode": "legacy",
            "include_surface_boundaries": False,
        }
        self._show_foci = show_foci
        self._focus_color = focus_color
        if legacy_painter is None:
            reservation_band = preferred_painter
            exact = False
            self._legacy_painter_subbands = None
        else:
            reservation_band, section_band, overlay_band, focus_z = legacy_painter
            exact = True
            self._legacy_painter_subbands = (
                section_band,
                overlay_band,
                focus_z,
            )
        self._band_reservation = ScenePainterBandReservation(
            ("dandelin-section-3d", self.construction.construction_id),
            reservation_band,
            exact=exact,
        )
        self._display_group = VGroup()
        self._section_controller: (
            QuadricSection3D | CompositeQuadricSection3D | None
        ) = None
        self._overlay_controller: QuadricOcclusion3D | None = None
        self._focus_group: VGroup | None = None
        self._attached = False
        self._reservation_active = False
        self._focus_fixed_frame_camera = None
        self._painter_z_band: tuple[float, float] | None = None
        self._section_painter_z_band: tuple[float, float] | None = None
        self._overlay_painter_z_band: tuple[float, float] | None = None
        self._focus_z: float | None = None

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def display_mobject(self) -> Mobject:
        return self._display_group

    @property
    def section_controller(
        self,
    ) -> QuadricSection3D | CompositeQuadricSection3D:
        if not self._attached or self._section_controller is None:
            raise DandelinSectionAuthoringError(
                "section_controller is available only while attached"
            )
        return self._section_controller

    @property
    def overlay_controller(self) -> QuadricOcclusion3D:
        if not self._attached or self._overlay_controller is None:
            raise DandelinSectionAuthoringError(
                "overlay_controller is available only while attached"
            )
        return self._overlay_controller

    @property
    def focus_group(self) -> VGroup:
        if not self._attached or self._focus_group is None:
            raise DandelinSectionAuthoringError(
                "focus_group is available only while attached"
            )
        return self._focus_group

    @property
    def painter_z_band(self) -> tuple[float, float] | None:
        return self._painter_z_band if self._attached else None

    @property
    def section_painter_z_band(self) -> tuple[float, float] | None:
        return self._section_painter_z_band if self._attached else None

    @property
    def overlay_painter_z_band(self) -> tuple[float, float] | None:
        return self._overlay_painter_z_band if self._attached else None

    @property
    def painter_subbands(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if (
            not self._attached
            or self._section_painter_z_band is None
            or self._overlay_painter_z_band is None
        ):
            return None
        return self._section_painter_z_band, self._overlay_painter_z_band

    @property
    def focus_z(self) -> float | None:
        return self._focus_z if self._attached else None

    def _build_section_controller(
        self,
        painter_z_band: tuple[float, float],
    ) -> QuadricSection3D | CompositeQuadricSection3D:
        options = dict(self._section_configuration)
        options["painter_z_band"] = painter_z_band
        if self.cone.model is ConeModel.OPEN_DOUBLE:
            return CompositeQuadricSection3D(
                self.scene,
                surface=self.cone,
                **options,
            )
        return QuadricSection3D(
            self.scene,
            surface=self.cone,
            show_plane=True,
            **options,
        )

    def _build_overlay_controller(
        self,
        painter_z_band: tuple[float, float],
    ) -> QuadricOcclusion3D:
        options = dict(self._overlay_configuration)
        options["painter_z_band"] = painter_z_band
        return QuadricOcclusion3D(self.scene, **options)

    def _build_focus_group(self, focus_z: float) -> VGroup:
        matrix = self._projection_frame.view.matrix
        display_offset = np.asarray(
            self._projection_frame.screen_offset,
            dtype=float,
        )
        focus_dots = []
        if self._show_foci:
            for sphere in self.construction.spheres:
                screen = (
                    matrix[:2]
                    @ np.asarray(sphere.focus.world_point, dtype=float)
                    + display_offset
                )
                dot = Dot(
                    (float(screen[0]), float(screen[1]), 0.0),
                    radius=self.focus_radius,
                    color=self._focus_color,
                )
                dot.set_z_index(focus_z)
                focus_dots.append(dot)
        return VGroup(*focus_dots)

    def _attach_section_layer(self) -> None:
        if self._section_controller is None:  # pragma: no cover - internal guard
            raise DandelinSectionAuthoringError(
                "section controller was not built before attachment"
            )
        self._section_controller.attach()

    def _attach_overlay_layer(self) -> None:
        if self._overlay_controller is None:  # pragma: no cover - internal guard
            raise DandelinSectionAuthoringError(
                "overlay controller was not built before attachment"
            )
        self._overlay_controller.attach()

    def _attach_focus_layer(self) -> None:
        if self._focus_group is None:  # pragma: no cover - internal guard
            raise DandelinSectionAuthoringError(
                "focus group was not built before attachment"
            )
        if self._focus_group.submobjects:
            self.scene.add(self._focus_group)
            self._focus_fixed_frame_camera = _register_fixed_frame(
                self.scene,
                self._focus_group,
            )

    def _commit_author_state(
        self,
        aggregate: tuple[float, float],
        section: tuple[float, float],
        overlay: tuple[float, float],
        focus_z: float,
    ) -> None:
        """Commit the facade-owned tail of one successful attachment.

        Keeping this tiny author-state commit distinct makes the complete
        section/overlay/focus transaction fault-injectable without weakening
        the child controllers' own display and cache transactions.
        """

        self._painter_z_band = aggregate
        self._section_painter_z_band = section
        self._overlay_painter_z_band = overlay
        self._focus_z = focus_z
        self._attached = True

    def _remove_focus_fixed_frame(self) -> None:
        camera = self._focus_fixed_frame_camera
        if camera is None:
            candidate = getattr(self.scene, "camera", None)
            if isinstance(candidate, ThreeDCamera):
                camera = candidate
        if self._focus_group is not None:
            try:
                _remove_fixed_frame(camera, self._focus_group)
            except BaseException:
                self._focus_fixed_frame_camera = camera
                raise
        self._focus_fixed_frame_camera = None

    def _remove_focus_scene_ownership(self) -> None:
        if self._focus_group is not None and self._focus_group.submobjects:
            owned = self._family_ids(self._focus_group)
            self.scene.remove(self._focus_group)
            # During a real Cairo ``play()``, Manim also caches family members
            # in ``moving_mobjects`` and ``static_mobjects``.  ``Scene.remove``
            # clears the public scene list but does not purge those transient
            # renderer containers, so the ownership audit must remove the
            # exact focus identities from every known scene container too.
            for container in _scene_containers(self.scene):
                container[:] = [
                    item for item in container if id(item) not in owned
                ]

    def _restore_overlay_controller(self) -> None:
        if self._overlay_controller is not None:
            self._overlay_controller.restore()

    def _restore_section_controller(self) -> None:
        if self._section_controller is not None:
            self._section_controller.restore()

    def _release_painter_band(self) -> None:
        if not self._reservation_active:
            return
        release_scene_painter_band(self.scene, self._band_reservation)
        self._reservation_active = False

    @staticmethod
    def _family_ids(*roots: object) -> set[int]:
        result: set[int] = set()
        for root in roots:
            if isinstance(root, Mobject):
                result.update(id(item) for item in root.get_family())
        return result

    def _scene_family_ids(self) -> set[int]:
        result: set[int] = set()
        for container in _scene_containers(self.scene):
            for root in container:
                if isinstance(root, Mobject):
                    result.update(id(item) for item in root.get_family())
        return result

    def _fixed_frame_family_ids(self) -> set[int]:
        cameras: list[ThreeDCamera] = []
        for candidate in (
            getattr(self.scene, "camera", None),
            self._focus_fixed_frame_camera,
        ):
            if isinstance(candidate, ThreeDCamera) and not any(
                candidate is existing for existing in cameras
            ):
                cameras.append(candidate)
        return {
            id(item)
            for camera in cameras
            for item in camera.fixed_in_frame_mobjects
        }

    def _controller_ownership_remains(self, controller: object | None) -> bool:
        if controller is None:
            return False
        if bool(getattr(controller, "attached")):
            return True
        low_level = getattr(controller, "controller", controller)
        owned = self._family_ids(
            getattr(low_level, "root", None),
            getattr(low_level, "_update_driver", None),
        )
        return bool(
            owned
            & (self._scene_family_ids() | self._fixed_frame_family_ids())
        )

    def _runtime_ownership_remains(self) -> bool:
        if self._controller_ownership_remains(self._section_controller):
            return True
        if self._controller_ownership_remains(self._overlay_controller):
            return True
        focus_ids = self._family_ids(self._focus_group)
        return bool(
            focus_ids
            & (self._scene_family_ids() | self._fixed_frame_family_ids())
        )

    def _clear_runtime_references(self) -> None:
        if self._display_group.submobjects:
            self._display_group.remove(*tuple(self._display_group.submobjects))
        self._section_controller = None
        self._overlay_controller = None
        self._focus_group = None
        self._focus_fixed_frame_camera = None
        self._painter_z_band = None
        self._section_painter_z_band = None
        self._overlay_painter_z_band = None
        self._focus_z = None
        self._attached = False

    @staticmethod
    def _record_cleanup_failure(
        primary: BaseException | None,
        label: str,
        cleanup_error: BaseException,
    ) -> BaseException:
        if primary is None:
            return cleanup_error
        primary.add_note(f"{label} also failed: {cleanup_error!r}")
        return primary

    def _cleanup_layers(
        self,
        primary: BaseException | None = None,
    ) -> BaseException | None:
        """Clean every layer, retaining ownership evidence for a safe retry."""

        self._attached = False
        for label, cleanup in (
            ("focus fixed-frame cleanup", self._remove_focus_fixed_frame),
            ("focus Scene cleanup", self._remove_focus_scene_ownership),
            ("overlay cleanup", self._restore_overlay_controller),
            ("section cleanup", self._restore_section_controller),
        ):
            try:
                cleanup()
            except BaseException as cleanup_error:
                primary = self._record_cleanup_failure(
                    primary,
                    label,
                    cleanup_error,
                )

        try:
            ownership_remains = self._runtime_ownership_remains()
        except BaseException as audit_error:
            primary = self._record_cleanup_failure(
                primary,
                "runtime ownership audit",
                audit_error,
            )
            ownership_remains = True
        if ownership_remains:
            if primary is None:
                primary = DandelinSectionAuthoringError(
                    "Dandelin runtime ownership remains after cleanup"
                )
            primary.add_note(
                "painter band and runtime references were retained for a "
                "later restore() retry because Scene or fixed-frame ownership "
                "remains"
            )
            return primary

        try:
            self._release_painter_band()
        except BaseException as release_error:
            primary = self._record_cleanup_failure(
                primary,
                "painter-band release",
                release_error,
            )
            primary.add_note(
                "runtime references were retained for a later restore() retry "
                "because the painter-band release did not complete"
            )
            return primary

        self._clear_runtime_references()
        return primary

    def attach(self) -> "DandelinSection3D":
        if self._attached:
            return self
        if self._reservation_active:
            raise DandelinSectionAuthoringError(
                "a previous Dandelin painter-band release did not complete"
            )
        try:
            aggregate = reserve_scene_painter_band(
                self.scene,
                self._band_reservation,
            )
            self._reservation_active = True
            if self._legacy_painter_subbands is None:
                section_band, overlay_band, focus_z = (
                    _automatic_painter_subbands(aggregate)
                )
            else:
                section_band, overlay_band, focus_z = (
                    self._legacy_painter_subbands
                )
            # Stage the actual allocation before constructing any Manim layer.
            # Public properties still expose it only after author commit, while
            # a failed cleanup can retain the exact evidence for restore retry.
            self._painter_z_band = aggregate
            self._section_painter_z_band = section_band
            self._overlay_painter_z_band = overlay_band
            self._focus_z = focus_z
            self._section_controller = self._build_section_controller(
                section_band
            )
            self._overlay_controller = self._build_overlay_controller(
                overlay_band
            )
            self._focus_group = self._build_focus_group(focus_z)
            self._display_group.add(
                self._section_controller.display_mobject,
                self._overlay_controller.display_mobject,
                self._focus_group,
            )
            self._attach_section_layer()
            self._attach_overlay_layer()
            self._attach_focus_layer()
            self._commit_author_state(
                aggregate,
                section_band,
                overlay_band,
                focus_z,
            )
        except BaseException as error:
            self._cleanup_layers(error)
            raise
        return self

    def restore(self) -> "DandelinSection3D":
        failure = self._cleanup_layers()
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)
        return self

    def detach(self) -> "DandelinSection3D":
        return self.restore()

    @contextmanager
    def session(self) -> Iterator["DandelinSection3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        if (
            not self._attached
            or self._section_controller is None
            or self._overlay_controller is None
            or self._focus_group is None
        ):
            raise DandelinSectionAuthoringError(
                "slot_identities are available only while attached"
            )
        return (
            *self._section_controller.slot_identities(),
            *self._overlay_controller.slot_identities(),
            *(id(item) for item in self._focus_group.get_family()),
        )


__all__ = [
    "DEFAULT_DANDELIN_OVERLAY_STYLE",
    "DEFAULT_DANDELIN_SECTION_STYLE",
    "DandelinSection3D",
    "DandelinSectionAuthoringError",
]

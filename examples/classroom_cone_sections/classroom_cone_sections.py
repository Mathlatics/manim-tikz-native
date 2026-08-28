"""Five classroom-ready finite-cone section lessons.

The scenes in this module use the same public authoring controllers as the
production examples and Cairo acceptance tests.  They add only fixed teaching
labels and pacing; no geometry, visibility, or painter-order calculation is
duplicated here.

Preview every lesson with::

    manim -ql --fps 8 \
      examples/classroom_cone_sections/classroom_cone_sections.py \
      ConicFamilyTransitionLesson ClosedVsOpenConeLesson \
      HiddenCurvePoliciesLesson ProjectionDegenerationLesson \
      CapChordTopologyLesson
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Callable, Sequence

import numpy as np
from manim import DOWN, UP, Scene, Text, VGroup, ValueTracker, smooth

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    DEFAULT_QUADRIC_VIEW,
    QuadricBoundaryStyle,
    QuadricGeometryPrototype,
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_motion import (
    AxisAnglePlaneMotion,
    track_scheduled_plane_section,
)


BACKGROUND_COLOR = "#101820"
SECTION_START = -0.48
SECTION_END = 0.48
TOPOLOGY_END_ANGLE = 1.40
PROJECTION_OBLIQUE_PROGRESS = 0.45


def _topology_schedule():
    cone = ConeSpec(
        "classroom:topology:cone",
        (0.0, 0.0, -1.5),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
    )
    motion = AxisAnglePlaneMotion(
        "classroom:topology:motion",
        SectionPlane(
            "classroom:topology:plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        ),
        (0.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        0.0,
        TOPOLOGY_END_ANGLE,
    )
    return track_scheduled_plane_section(
        "classroom:topology:section", cone, motion
    )


def _certified_parabola_progress() -> float:
    schedule = _topology_schedule().schedule
    event = min(
        schedule.critical_events,
        key=lambda item: abs(float(item.angle) - pi / 3.0),
    )
    return float(event.progress)


# Always use the schedule's certified event value.  Recomputing angle/end-angle
# directly can differ by a few ULPs and would no longer identify the same plane.
PARABOLA_PROGRESS = _certified_parabola_progress()


def _cap_contact_progress() -> float:
    """Return the first upper-cap contact for the shared moving plane."""

    cone_height = 4.0
    cone_radius = cone_height * np.tan(pi / 6.0)
    plane_normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
    normal_length = float(np.linalg.norm(plane_normal))
    contact_tracker = (1.95 - 0.82 * cone_radius) / normal_length
    return float(
        (contact_tracker - SECTION_START) / (SECTION_END - SECTION_START)
    )


CAP_CONTACT_PROGRESS = _cap_contact_progress()
# An exactly tangent plane has a zero-length chord, so there is no visible
# classroom stroke yet.  Pause three percent later at the first clearly
# readable post-contact chord while retaining the analytic contact separately.
CAP_FIRST_VISIBLE_PROGRESS = min(1.0, CAP_CONTACT_PROGRESS + 0.03)


def _scaled_view(view: ParallelView, factor: float) -> ParallelView:
    matrix = view.matrix
    matrix[:2] *= float(factor)
    return ParallelView.from_matrix(matrix)


SINGLE_VIEW = _scaled_view(DEFAULT_QUADRIC_VIEW, 1.02)
PAIR_VIEW = _scaled_view(DEFAULT_QUADRIC_VIEW, 0.82)
TRIPLE_VIEW = _scaled_view(DEFAULT_QUADRIC_VIEW, 0.64)
TOPOLOGY_VIEW = _scaled_view(DEFAULT_QUADRIC_VIEW, 0.86)


STYLE = QuadricManimStyle(
    surface_fill_color="#315A8A",
    surface_fill_opacity=0.76,
    surface_stroke_opacity=0.0,
    cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
    cone_cap_fill_colors=("#557A99", "#294B6B"),
    cone_lateral_sheen_direction=(1.0, 0.0, 0.0),
    cone_cap_sheen_direction=(-1.0, 1.0, 0.0),
    visible_curve_color="#FFD166",
    visible_curve_width=4.0,
    hidden_curve_color="#F59E0B",
    hidden_curve_width=3.0,
    hidden_curve_opacity=0.62,
    section_plane_fill_color="#43D9C0",
    section_plane_fill_opacity=0.34,
    section_plane_stroke_color="#B39DDB",
    section_plane_stroke_width=1.8,
    section_plane_stroke_opacity=0.9,
    dash_length=0.12,
    dash_gap=0.09,
)


BOUNDARY_STYLE = QuadricBoundaryStyle(
    visible_color="#5CE1E6",
    visible_width=4.4,
    visible_opacity=1.0,
    hidden_color="#5CE1E6",
    hidden_width=3.0,
    hidden_opacity=0.22,
    dash_length=0.12,
    dash_gap=0.09,
)


BOUNDARY_STYLES = {
    "style:surface-silhouette": BOUNDARY_STYLE,
    "style:surface-boundary": BOUNDARY_STYLE,
}


def classroom_limits(*, transition: bool = False) -> QuadricManimLimits:
    """Reviewed fixed capacities for the five classroom scenes."""

    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=32 if transition else 4,
        max_fragments_per_curve=32 if transition else 18,
        max_segments_per_fragment=384,
        max_surface_segments=768,
        max_dashes_per_fragment=100 if transition else 72,
        max_projected_length=18.0,
        max_total_mobjects=100000 if transition else 24000,
        max_boundary_sources=64 if transition else 28,
        max_boundary_styles=12,
    )


@dataclass(frozen=True, slots=True)
class LessonKeyframe:
    """One reviewed teaching stop in normalized lesson time."""

    label: str
    progress: float
    teaching_point: str


@dataclass(frozen=True, slots=True)
class ClassroomLessonSpec:
    """Human-facing metadata for one classroom lesson."""

    lesson_id: str
    scene_name: str
    title: str
    parameters: tuple[str, ...]
    conclusion: str
    teacher_prompts: tuple[str, ...]
    keyframes: tuple[LessonKeyframe, ...]


LESSON_SPECS = (
    ClassroomLessonSpec(
        lesson_id="conic_family_transition",
        scene_name="ConicFamilyTransitionLesson",
        title="Why ellipse, parabola, and hyperbola appear",
        parameters=(
            "cone half-angle = 30 degrees",
            "plane-normal rotation = 0 to 1.40 radians",
            "exact parabola when the plane is parallel to one generator",
        ),
        conclusion=(
            "同一个有限圆锥的截线会随平面倾角连续地从椭圆经过精确抛物线，"
            "再变成双曲线；临界条件是截平面平行于一条母线。"
        ),
        teacher_prompts=(
            "先让学生预测闭合曲线会在什么时刻打开。",
            "暂停在精确抛物线，追问平面与哪条母线平行。",
            "比较临界点前后的虚线位置，说明遮挡变化不等于曲线断裂。",
        ),
        keyframes=(
            LessonKeyframe("ellipse", 0.0, "闭合截线"),
            LessonKeyframe(
                "exact-parabola",
                PARABOLA_PROGRESS,
                "截平面与母线平行",
            ),
            LessonKeyframe("hyperbola", 1.0, "开放的双曲线支"),
        ),
    ),
    ClassroomLessonSpec(
        lesson_id="closed_vs_open",
        scene_name="ClosedVsOpenConeLesson",
        title="Closed cone versus open cone shell",
        parameters=(
            "same cone axis, half-angle, trim height, plane, and view",
            "left model = CLOSED_SINGLE",
            "right model = OPEN_SINGLE",
        ),
        conclusion=(
            "封闭圆锥体的完整截面可以包含侧面弧和真实底面弦；张口圆锥壳"
            "没有底面，因此只能保留侧面截线，不能凭空补弦。"
        ),
        teacher_prompts=(
            "让学生指出黄色直线段究竟来自侧面还是底面。",
            "遮住模型标签，让学生先凭截面边界判断左右模型。",
            "强调青色开口圆周是壳的边界，不等于存在底面。",
        ),
        keyframes=(
            LessonKeyframe("before-base", 0.0, "两者都只有侧面截线"),
            LessonKeyframe("mid-plane", 0.5, "同一平面继续上移"),
            LessonKeyframe("base-crossing", 1.0, "封闭模型出现底面弦"),
        ),
    ),
    ClassroomLessonSpec(
        lesson_id="hidden_curve_policies",
        scene_name="HiddenCurvePoliciesLesson",
        title="Three hidden-curve drawing policies",
        parameters=(
            "three identical closed cones and cutting planes",
            "physical / diagrammatic / depth_aware_diagrammatic",
            "same visible and hidden curve styles",
        ),
        conclusion=(
            "physical 表示物理不可见就不画；diagrammatic 把教学虚线放到顶层；"
            "depth-aware 仍画虚线，但让它位于真实遮挡面之后并受到透明度衰减。"
        ),
        teacher_prompts=(
            "先问哪一栏最接近真实视觉，再问哪一栏最适合讲解。",
            "比较中栏与右栏虚线亮度，解释半透明表面的前后层级。",
            "强调三种策略只改变绘制政策，不改变几何计算结果。",
        ),
        keyframes=(
            LessonKeyframe("lower-cut", 0.0, "隐藏弧较短"),
            LessonKeyframe("comparison", 0.5, "三种政策并排比较"),
            LessonKeyframe("upper-cut", 1.0, "遮挡区间随平面改变"),
        ),
    ),
    ClassroomLessonSpec(
        lesson_id="projection_degeneration",
        scene_name="ProjectionDegenerationLesson",
        title="Parallel projection and the edge-on limit",
        parameters=(
            "view direction moves from (1,1,1) to (0,1,0)",
            "one certified oblique parallel view at progress 0.45",
            "open trim circle lies in the xy-plane",
        ),
        conclusion=(
            "圆周在一般平行投影下是椭圆；观察方向逐渐进入圆周所在平面时，"
            "椭圆越来越扁，精确侧视时投影秩降为 1，成为有限线段。"
        ),
        teacher_prompts=(
            "追问线段是否意味着三维圆周本身退化。",
            "把椭圆变扁与二维线性变换的秩联系起来。",
            "指出侧视线段只占有限范围，不能延长成无限直线。",
        ),
        keyframes=(
            LessonKeyframe("orthographic", 0.0, "正投影下的椭圆"),
            LessonKeyframe(
                "oblique-parallel",
                PROJECTION_OBLIQUE_PROGRESS,
                "一般平行投影下的斜椭圆",
            ),
            LessonKeyframe("exact-side-view", 1.0, "秩一有限线段"),
        ),
    ),
    ClassroomLessonSpec(
        lesson_id="cap_chord_topology",
        scene_name="CapChordTopologyLesson",
        title="Why a base chord changes the finite section",
        parameters=(
            "closed cone height = 4 and half-angle = 30 degrees",
            "plane translates along normal (0.82,0,1)",
            "analytic first contact is followed by a 0.03 progress display offset",
        ),
        conclusion=(
            "平面未碰到底面时，有限截面只有侧面圆锥曲线；第一次接触底面后，"
            "完整截面边界变成侧面弧加真实底面弦。"
        ),
        teacher_prompts=(
            "在接触前暂停，让学生预测下一刻会新增什么边界。",
            "强调新增 chord 来自有限圆锥的底面，而不是圆锥曲线家族改变。",
            "比较接触瞬间的零长度弦与穿过底面后的有限弦。",
        ),
        keyframes=(
            LessonKeyframe("lateral-only", 0.0, "纯侧面圆锥曲线"),
            LessonKeyframe(
                "first-visible-chord",
                CAP_FIRST_VISIBLE_PROGRESS,
                "第一次接触后刚出现可辨认的底面弦",
            ),
            LessonKeyframe("arc-plus-chord", 1.0, "侧面弧加底面弦"),
        ),
    ),
)


_SPEC_BY_ID = {item.lesson_id: item for item in LESSON_SPECS}


def classroom_lesson_specs() -> tuple[ClassroomLessonSpec, ...]:
    """Return the fixed five-lesson classroom gallery contract."""

    return LESSON_SPECS


def _title(scene: Scene, heading: str, note: str) -> tuple[Text, Text]:
    title = Text(heading, font_size=27, color="#F4F7FB").to_edge(UP, buff=0.22)
    subtitle = Text(note, font_size=15, color="#B8C5D6").next_to(
        title, DOWN, buff=0.09
    )
    title.set_z_index(100)
    subtitle.set_z_index(100)
    scene.add_foreground_mobjects(title, subtitle)
    return title, subtitle


def _caption(scene: Scene, label: str, x: float, *, y: float = -3.25) -> Text:
    value = Text(label, font_size=17, color="#DCE6F2").move_to((x, y, 0.0))
    value.set_z_index(100)
    scene.add_foreground_mobjects(value)
    return value


def _stage_group(
    scene: Scene,
    labels: Sequence[tuple[str, float]],
    *,
    y: float = -3.25,
) -> VGroup:
    group = VGroup(
        *(
            Text(label, font_size=17, color="#FFD166").move_to((x, y, 0.0))
            for label, x in labels
        )
    )
    group.set_z_index(100)
    scene.add_foreground_mobjects(group)
    return group


def _highlight_nearest(group: VGroup, progress: float, stops: Sequence[float]) -> None:
    value = float(progress)
    distances = [abs(value - float(stop)) for stop in stops]
    active = min(range(len(distances)), key=distances.__getitem__)
    for index, label in enumerate(group):
        label.set_opacity(1.0 if index == active else 0.28)


@dataclass(slots=True)
class ClassroomState:
    """Fixed-identity state shared by lesson playback and keyframe capture."""

    scene: Scene
    lesson_id: str
    tracker: ValueTracker
    start_value: float
    end_value: float
    authorings: tuple[object, ...]
    controllers: tuple[tuple[str, QuadricOcclusion3D], ...]
    overlay: VGroup | None = None

    def value_at(self, progress: float) -> float:
        value = float(progress)
        if not 0.0 <= value <= 1.0:
            raise ValueError("classroom progress must lie in [0, 1]")
        return self.start_value + value * (self.end_value - self.start_value)

    def set_progress(self, progress: float) -> None:
        self.tracker.set_value(self.value_at(progress))
        for authoring in self.authorings:
            authoring.update(0.0)
        if self.overlay is not None:
            self.overlay.update(0.0)

    def restore(self) -> None:
        if self.overlay is not None:
            self.overlay.clear_updaters()
        for authoring in reversed(self.authorings):
            authoring.restore()


def _bind_stage_overlay(
    state: ClassroomState,
    group: VGroup,
    stops: Sequence[float],
) -> ClassroomState:
    # Manim detects time-aware updaters by the literal ``dt`` parameter name.
    def update_overlay(_group: VGroup, dt: float) -> None:
        del dt
        denominator = state.end_value - state.start_value
        progress = (
            0.0
            if denominator == 0.0
            else (state.tracker.get_value() - state.start_value) / denominator
        )
        _highlight_nearest(_group, progress, stops)

    group.add_updater(update_overlay)
    state.overlay = group
    group.update(0.0)
    return state


def _plane_interaction_authoring(
    scene: Scene,
    *,
    prefix: str,
    model: ConeModel,
    horizontal: float,
    tracker: ValueTracker,
    policy: QuadricPaintPolicy,
    view: ParallelView,
    painter_band: tuple[float, float],
    geometry_prototype: QuadricGeometryPrototype | None = None,
) -> QuadricSection3D:
    normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
    normal /= np.linalg.norm(normal)
    vertical_shift = -0.55 * np.asarray(view.matrix[1], dtype=float)
    horizontal_shift = horizontal * np.asarray(view.matrix[0], dtype=float)
    shift = vertical_shift if geometry_prototype is not None else (
        horizontal_shift + vertical_shift
    )
    display_offset = (
        tuple(float(item) for item in view.matrix[:2] @ horizontal_shift)
        if geometry_prototype is not None
        else (0.0, 0.0)
    )
    cone = ConeSpec(
        f"{prefix}:cone",
        tuple(shift + np.asarray((0.0, 0.0, -2.4))),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 4.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )

    def current_plane() -> SectionPlane:
        point = shift + np.asarray((0.0, 0.0, -0.35)) + tracker.get_value() * normal
        return SectionPlane(
            f"{prefix}:plane",
            tuple(float(item) for item in point),
            (0.82, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )

    return QuadricSection3D(
        scene,
        surface=cone,
        section_id=f"{prefix}:section",
        plane=current_plane,
        projection=view,
        paint_policy=policy,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=classroom_limits(),
        max_chord_error=0.008,
        painter_z_band=painter_band,
        include_surface_boundaries=True,
        geometry_prototype=geometry_prototype,
        display_offset=display_offset,
    ).attach()


def _build_topology(
    scene: Scene, progress: float, with_labels: bool
) -> ClassroomState:
    tracker = ValueTracker(0.0)
    facade = QuadricSection3D(
        scene,
        scheduled=_topology_schedule(),
        progress=tracker,
        projection=TOPOLOGY_VIEW,
        transition_fraction=0.055,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=classroom_limits(transition=True),
        max_chord_error=0.02,
        include_surface_boundaries=True,
    ).attach()
    state = ClassroomState(
        scene,
        "conic_family_transition",
        tracker,
        0.0,
        1.0,
        (facade,),
        (("transition", facade.controller),),
    )
    if with_labels:
        _title(
            scene,
            "Why ellipse, parabola, and hyperbola appear",
            "the plane tilt changes continuously; the true section keeps its occlusion",
        )
        group = _stage_group(
            scene,
            (("ELLIPSE", -2.7), ("EXACT PARABOLA", 0.0), ("HYPERBOLA", 2.8)),
        )
        _bind_stage_overlay(state, group, (0.0, PARABOLA_PROGRESS, 1.0))
    state.set_progress(progress)
    return state


def _build_closed_open(
    scene: Scene, progress: float, with_labels: bool
) -> ClassroomState:
    tracker = ValueTracker(SECTION_START)
    rows = (
        ("closed", "CLOSED SOLID", ConeModel.CLOSED_SINGLE, -3.35),
        ("open", "OPEN SHELL", ConeModel.OPEN_SINGLE, 3.35),
    )
    authorings = []
    controllers = []
    for index, (name, label, model, horizontal) in enumerate(rows):
        authoring = _plane_interaction_authoring(
            scene,
            prefix=f"classroom:closed-open:{name}",
            model=model,
            horizontal=horizontal,
            tracker=tracker,
            policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            view=PAIR_VIEW,
            painter_band=(20.0 + index * 20.0, 30.0 + index * 20.0),
        )
        authorings.append(authoring)
        controllers.append((name, authoring.controller))
        if with_labels:
            _caption(scene, label, horizontal)
    if with_labels:
        _title(
            scene,
            "Closed cone versus open cone shell",
            "same plane and lateral surface; only the closed model owns a base disk",
        )
    state = ClassroomState(
        scene,
        "closed_vs_open",
        tracker,
        SECTION_START,
        SECTION_END,
        tuple(authorings),
        tuple(controllers),
    )
    state.set_progress(progress)
    return state


def _build_policies(
    scene: Scene, progress: float, with_labels: bool
) -> ClassroomState:
    tracker = ValueTracker(SECTION_START)
    prototype = QuadricGeometryPrototype()
    rows = (
        ("physical", "PHYSICAL", QuadricPaintPolicy.PHYSICAL, -4.45),
        ("diagrammatic", "TEACHING DASH", QuadricPaintPolicy.DIAGRAMMATIC, 0.0),
        (
            "depth-aware",
            "DEPTH-AWARE",
            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            4.45,
        ),
    )
    authorings = []
    controllers = []
    for index, (name, label, policy, horizontal) in enumerate(rows):
        authoring = _plane_interaction_authoring(
            scene,
            prefix="classroom:policies:shared",
            model=ConeModel.CLOSED_SINGLE,
            horizontal=horizontal,
            tracker=tracker,
            policy=policy,
            view=TRIPLE_VIEW,
            painter_band=(20.0 + index * 18.0, 30.0 + index * 18.0),
            geometry_prototype=prototype,
        )
        authorings.append(authoring)
        controllers.append((name, authoring.controller))
        if with_labels:
            _caption(scene, label, horizontal)
    if with_labels:
        _title(
            scene,
            "Three hidden-curve drawing policies",
            "omit hidden ink  •  top teaching dash  •  dash below translucent surface",
        )
    state = ClassroomState(
        scene,
        "hidden_curve_policies",
        tracker,
        SECTION_START,
        SECTION_END,
        tuple(authorings),
        tuple(controllers),
    )
    state.set_progress(progress)
    return state


def classroom_projection_view(progress: float) -> ParallelView:
    """Move from isometric orthographic through oblique to exact side view."""

    value = min(1.0, max(0.0, float(progress)))
    start = np.asarray((1.0, 1.0, 1.0), dtype=float)
    middle = np.asarray((0.35, 1.0, 0.55), dtype=float)
    side = np.asarray((0.0, 1.0, 0.0), dtype=float)
    if value <= PROJECTION_OBLIQUE_PROGRESS:
        local = value / PROJECTION_OBLIQUE_PROGRESS
        direction = (1.0 - local) * start + local * middle
        shear = 0.34 * local
    else:
        local = (value - PROJECTION_OBLIQUE_PROGRESS) / (
            1.0 - PROJECTION_OBLIQUE_PROGRESS
        )
        direction = (1.0 - local) * middle + local * side
        shear = 0.34 * (1.0 - local)
    direction /= np.linalg.norm(direction)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    zoom = 1.12
    matrix = np.vstack((zoom * (right + shear * direction), zoom * up, direction))
    return ParallelView.from_matrix(matrix)


def _build_projection(
    scene: Scene, progress: float, with_labels: bool
) -> ClassroomState:
    tracker = ValueTracker(0.0)
    cone = ConeSpec(
        "classroom:projection:open-cone",
        (0.0, 0.0, -1.9),
        (0.0, 0.0, 1.0),
        pi / 5.5,
        (0.0, 3.8),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_SINGLE,
    )

    def projection(_scene: object) -> ParallelView:
        del _scene
        return classroom_projection_view(tracker.get_value())

    controller = QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=(),
        projection=projection,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=STYLE,
        boundary_styles=BOUNDARY_STYLES,
        limits=classroom_limits(),
        max_chord_error=0.008,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
    ).attach()
    state = ClassroomState(
        scene,
        "projection_degeneration",
        tracker,
        0.0,
        1.0,
        (controller,),
        (("open-shell", controller),),
    )
    if with_labels:
        _title(
            scene,
            "Parallel projection and the edge-on limit",
            "the trim circle projects as ellipse → flatter ellipse → finite segment",
        )
        group = _stage_group(
            scene,
            (("ORTHOGRAPHIC", -3.3), ("OBLIQUE", 0.0), ("SIDE VIEW", 3.3)),
        )
        _bind_stage_overlay(
            state,
            group,
            (0.0, PROJECTION_OBLIQUE_PROGRESS, 1.0),
        )
    state.set_progress(progress)
    return state


def _build_cap_chord(
    scene: Scene, progress: float, with_labels: bool
) -> ClassroomState:
    tracker = ValueTracker(SECTION_START)
    authoring = _plane_interaction_authoring(
        scene,
        prefix="classroom:cap-chord",
        model=ConeModel.CLOSED_SINGLE,
        horizontal=0.0,
        tracker=tracker,
        policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        view=SINGLE_VIEW,
        painter_band=(20.0, 30.0),
    )
    state = ClassroomState(
        scene,
        "cap_chord_topology",
        tracker,
        SECTION_START,
        SECTION_END,
        (authoring,),
        (("closed", authoring.controller),),
    )
    if with_labels:
        _title(
            scene,
            "Why a base chord changes the finite section",
            "lateral conic → first base contact → lateral arc plus real base chord",
        )
        group = _stage_group(
            scene,
            (
                ("LATERAL ONLY", -3.25),
                ("FIRST VISIBLE", 0.0),
                ("ARC + CHORD", 3.25),
            ),
        )
        _bind_stage_overlay(
            state,
            group,
            (0.0, CAP_FIRST_VISIBLE_PROGRESS, 1.0),
        )
    state.set_progress(progress)
    return state


_BUILDERS: dict[str, Callable[[Scene, float, bool], ClassroomState]] = {
    "conic_family_transition": _build_topology,
    "closed_vs_open": _build_closed_open,
    "hidden_curve_policies": _build_policies,
    "projection_degeneration": _build_projection,
    "cap_chord_topology": _build_cap_chord,
}


def build_classroom_state(
    scene: Scene,
    lesson_id: str,
    *,
    progress: float = 0.0,
    with_labels: bool = False,
) -> ClassroomState:
    """Build one lesson at a normalized progress on the production call path."""

    try:
        builder = _BUILDERS[lesson_id]
    except KeyError as exc:
        raise ValueError(f"unknown classroom lesson {lesson_id!r}") from exc
    scene.camera.background_color = BACKGROUND_COLOR
    return builder(scene, float(progress), bool(with_labels))


class _ClassroomLessonScene(Scene):
    lesson_id = ""
    run_time = 6.0

    def construct(self) -> None:
        spec = _SPEC_BY_ID[self.lesson_id]
        state = build_classroom_state(
            self,
            self.lesson_id,
            progress=spec.keyframes[0].progress,
            with_labels=True,
        )
        self.wait(0.55)
        previous = spec.keyframes[0].progress
        for index, keyframe in enumerate(spec.keyframes[1:], start=1):
            distance = abs(keyframe.progress - previous)
            self.play(
                state.tracker.animate.set_value(state.value_at(keyframe.progress)),
                run_time=max(0.7, self.run_time * distance),
                rate_func=smooth,
            )
            self.wait(0.65 if index < len(spec.keyframes) - 1 else 0.75)
            previous = keyframe.progress
        state.restore()


class ConicFamilyTransitionLesson(_ClassroomLessonScene):
    lesson_id = "conic_family_transition"
    run_time = 7.0


class ClosedVsOpenConeLesson(_ClassroomLessonScene):
    lesson_id = "closed_vs_open"


class HiddenCurvePoliciesLesson(_ClassroomLessonScene):
    lesson_id = "hidden_curve_policies"


class ProjectionDegenerationLesson(_ClassroomLessonScene):
    lesson_id = "projection_degeneration"


class CapChordTopologyLesson(_ClassroomLessonScene):
    lesson_id = "cap_chord_topology"


__all__: Sequence[str] = (
    "BACKGROUND_COLOR",
    "CAP_CONTACT_PROGRESS",
    "CAP_FIRST_VISIBLE_PROGRESS",
    "ClassroomLessonSpec",
    "ClassroomState",
    "CapChordTopologyLesson",
    "ClosedVsOpenConeLesson",
    "ConicFamilyTransitionLesson",
    "HiddenCurvePoliciesLesson",
    "LESSON_SPECS",
    "LessonKeyframe",
    "PARABOLA_PROGRESS",
    "PROJECTION_OBLIQUE_PROGRESS",
    "ProjectionDegenerationLesson",
    "build_classroom_state",
    "classroom_lesson_specs",
    "classroom_projection_view",
)

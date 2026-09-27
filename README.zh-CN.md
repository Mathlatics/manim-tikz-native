# manim-tikz-native 中文说明

[![CI](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml/badge.svg)](https://github.com/Mathlatics/manim-tikz-native/actions/workflows/ci.yml)
[![Python 3.11 / 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Manim 0.20.1](https://img.shields.io/badge/Manim-0.20.1-6c55a3.svg)](https://www.manim.community/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一套面向数学教学动画的语义几何工具链：把文档明确支持的 TikZ 子集编译成可编辑
的原生 Manim 对象，并为显式登记的多面体、二次曲面、圆锥曲线和教学构造提供
解析的平行投影遮挡。

项目会保留稳定对象身份；当源码或几何无法认证时明确失败，不会把内容压成 SVG、
位图，也不会猜一个看起来差不多的 `z_index`。

[English](README.md) · [项目文档](docs/README.md) ·
[中文操作指南](docs/user-guide.zh-CN.md) · [公共 API](docs/public-api.md) ·
[示例](examples/) · [贡献说明](CONTRIBUTING.md)

> **版本说明：**当前正式发布版是
> [`v0.1.1`](https://github.com/Mathlatics/manim-tikz-native/releases/tag/v0.1.1)。
> GitHub `main` 还包含 `CHANGELOG.md` 中列在 `Unreleased` 下的已审查、未发布能力；
> 不能把所有主干功能都说成已经包含在 v0.1.1 安装包中。

![椭圆、抛物线和双曲线过渡](https://raw.githubusercontent.com/Mathlatics/manim-tikz-native/main/examples/classroom_cone_sections/gallery/contact-sheets/conic_family_transition.png)

## 项目包含什么

### 1. 语义 TikZ 编译

- 把受控的二维/三维 TikZ 编译成真正的 `Line`、`Polygon`、`Circle`、
  `Ellipse`、`Dot`、`Tex`、`MathTex`、箭头和角标对象；
- 保存命名坐标、路径、语义 ID 和几何关系，供后续动画继续驱动；
- 三维图既可按源码视角固定投影，也可保留世界坐标交给 Manim 相机；
- 用 source project 从作者源码可重复生成 ShapeAsset、合成计划、相机镜头和
  Manim 源码等派生产物。

### 2. 解析几何与自动遮挡

- 闭合凸多面体、有限开放面、铰链、自由直线、移动截平面、复制体交接和抽离二面角；
- 有限球、封闭/开放圆柱、封闭圆锥、开放单锥壳和有限开放双锥壳；
- 圆、椭圆、抛物线、双曲线、相切点、轮廓线、截口圆、接触圆和真实截线的
  解析可见性；
- 在支持范围内，为半透明教学图形建立确定的从远到近绘制关系。

### 3. 平行相机与固定容量 Manim 运行时

- `ParallelCameraState` 同时保存观察方向、世界目标点、画面锚点和缩放，并支持
  平面正视、斜视和精确侧视；
- 命名镜头、安全切换、目标跟随、截面时间线、拓扑双 bank、视口事务和受限的
  多 Rig 全局协调；
- 预先分配 Cairo 槽位，保持虚线相位和对象身份；更新失败时保留上一正确帧并
  完整恢复，不在 updater 里临时替换 Mobject。

稳定的架构依赖方向是：

```text
Geometry -> Topology -> Visibility -> Compositor -> Manim bindings
几何         拓扑          可见性          绘制关系       Manim 绑定
```

前四层不依赖 Manim。详细说明见[项目架构](docs/architecture.md)和
[几何内核分层](docs/geometry-kernel-layers.md)。

## 应该从哪个入口开始

| 目标 | 推荐入口 | 起步文档 |
| --- | --- | --- |
| 把 TikZ 编译成原生 Manim 对象 | `compile_document()` + 原生 renderer | [首个 TikZ 场景](docs/user-guide.zh-CN.md#路线一把-tikz-编译成原生-manim-对象) |
| 从 TikZ 作者源码重建派生产物 | `tikz-native-project` | [源码权威项目](docs/source-authoritative-projects.md) |
| 处理闭合凸体的隐藏线 | `OcclusionScene3D` | [自动遮挡](docs/automatic-occlusion.md) |
| 处理开放面或铰链 | `OpenFaceScene3D` | [开放面示例](examples/open_face_visibility/README.md) |
| 给凸体加入一个移动截平面，并可附加自由直线 | `ConvexSectionScene3D` | [凸截面示例](examples/convex_sections/README.md) |
| 制作一个有限二次曲面截面动画 | `QuadricSectionRig` | [二次曲面 Quick Start](docs/quadric-authoring-workflow.md) |
| 处理没有截平面的多个分离二次曲面和曲线 | `QuadricOcclusion3D` | [二次曲面遮挡](docs/quadric-occlusion.md) |
| 制作回调式或拓扑变化截面 | `QuadricSection3D` / `QuadricSectionTransition3D` | [二次曲面创作](docs/quadric-authoring-workflow.md) |
| 协调 `OPEN_DOUBLE` 双锥截面 | `CompositeQuadricSection3D` | [二次曲面示例](examples/quadrics/README.md) |
| 计算丹德林球 | `compute_dandelin_construction()` | [丹德林球契约](docs/dandelin-spheres-v1.md) |
| 实时重算丹德林曲线实虚 | `DandelinOcclusion3D` | [丹德林使用指南](docs/user-guide.zh-CN.md#路线五选择正确的丹德林路径) |
| 在 Manim 中播放语义相机镜头 | `MultiProjectionCamera` + `play_parallel_camera_shot_sequence()` | [相机镜头示例](examples/parallel_camera_shots/README.md) |

完整 Python 和 JSON 接口见[公共 API](docs/public-api.md)。

## 五分钟起步

项目 CI 验证 Python 3.11 和 3.12；发布证据环境固定为 Python 3.12.13。
包依赖固定 Manim Community 0.20.1。

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

系统还需要 Cairo/Pango、FFmpeg、XeLaTeX、TikZ、`dvisvgm`、Fandol 和
Latin Modern 字体。完整安装方法见[中文操作指南](docs/user-guide.zh-CN.md#一选择经过验证的环境)。

检查四个版本化入口：

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

直接渲染仓库中的圆锥截面起步动画：

```bash
python -m manim --renderer cairo -r 480,270 --fps 15 \
  --media_dir media/quadric-quickstart \
  examples/quadrics/quadric_section_rig_quick_start.py \
  ConeSectionRigQuickStart
```

如果希望先从 TikZ 开始，请照着
[包含 `figure.tex` 和 `scene.py` 的完整两文件示例](docs/user-guide.zh-CN.md#路线一把-tikz-编译成原生-manim-对象)
操作。

## 圆锥截面 Quick Start

普通作者只描述“截平面平移/旋转”这样的数学动作。高层 Rig 会自动处理有限截线、
实线/虚线、平面与曲面的绘制关系、固定对象槽位和失败回滚：

```python
from math import pi

from manim import Scene
from polyhedron_visibility.quadrics import ConeSpec, QuadricSectionRig, SectionPlane


class ConeSectionLesson(Scene):
    def construct(self):
        cone = ConeSpec("cone", (0, 0, -1.5), (0, 0, 1), pi / 6, (0, 4))
        plane = SectionPlane(
            "cut", (0, 0, -0.4), (0.45, 0, 1), u_axis=(0, 1, 0)
        )
        with QuadricSectionRig(
            self,
            surface=cone,
            section_id="cone-section",
            plane=plane,
            paint_policy="depth_aware_diagrammatic",
        ).session() as section:
            self.play(section.animate_plane_shift(0.6), run_time=2)
            self.play(
                section.animate_plane_rotation(
                    axis=(0, 0, 1), angle=pi / 3, pivot=cone.apex
                ),
                run_time=2,
            )
```

调构图时使用 `render_profile="preview"`、480×270、15 fps；课堂成片改用
`render_profile="final"`、960×540、30 fps。完整 Preview / Final /
Release-Evidence 区别见[有限二次曲面创作工作流](docs/quadric-authoring-workflow.md)。

## 丹德林能力边界

丹德林功能有几种有意不同的显示契约：

| 路径 | 已认证内容 | 没有声称的内容 |
| --- | --- | --- |
| `DandelinSection3D` | 解析构造和普通圆锥截线 | 球—锥面的物理遮挡；辅助对象位于顶层教学带。 |
| `DandelinOcclusion3D` | 在固定 Cairo 槽位中随实时相机重算曲线实线/虚线片段 | 物理隐藏面或截平面填充。 |
| depth-aware 空间 TikZ | 冻结视角下的边界实线/虚线片段 | 运动、相机镜头或物理隐藏面。 |
| `depth_aware_teaching_transparent` | 静态空间视图、单锥瓣圆/椭圆/抛物线的曲线片段与课堂绘制顺序；必须显示接触圆 | 运动、其他视图、光学透明或不透明实体隐藏面。 |
| nested-tangent 场景协调器 | 一个圆锥/圆柱母面、一个相切平面、恰好两只相切球和登记过的边界 | 任意多对象遮挡。 |

[圆锥面—圆柱面切换示例](examples/dandelin_cone_cylinder_switch/README.md)使用狭义
nested-tangent 路径：平面边、母面边、球轮廓、接触圆、真实截线和教学轴线都进入
同一套解析片段图。半透明填充仍是教学图层；这个示例的最终 Manim 组装也仍属于
场景代码，不是通用 `DandelinConeCylinderSwitch3D` 公共 facade。

## 必须理解的边界

- 这不是任意 TikZ 转换器；不支持的语法会明确失败，不会改用 SVG 或位图。
- 自动遮挡要求稳定、显式的拓扑；不能从任意 `VGroup` 或 mesh 猜出可靠模型。
- 自动遮挡和二次曲面的正式绑定使用 Cairo 与平行投影；透视和 OpenGL 等价绑定
  不属于 v1 承诺。
- 普通全局多二次曲面路径要求严格分离。双锥和丹德林相切构型使用的是明确、狭义
  的协调器，不是一般相交曲面求解器。
- 教学透明的 painter order 不等于物理光照、折射或通用透明材质模拟。
- 仓库包含渲染器无关的 motion / section timeline 契约，但不包含浏览器/PPT 编辑器、
  应用层通用时间线数据库、ShapeAsset 数据库或预览缓存服务。

精确支持与拒绝条件见[支持的 TikZ 子集](docs/supported-tikz.md)、
[自动遮挡](docs/automatic-occlusion.md)和
[有限圆锥截面 v1 契约](docs/quadric-section-v1-contract.md)。

## 示例导航

| 示例 | 内容 |
| --- | --- |
| [解析椭圆动画](examples/analytic_geometry_ellipse_demo/README.md) | 由 geometry rig 驱动的语义 TikZ 对象。 |
| [凸体截面](examples/convex_sections/README.md) | 自由直线、移动截面和准确平面/实体透明排序。 |
| [抽离二面角](examples/derived_dihedral_extraction/README.md) | 复制交接、分离、统一线/面排序和往返。 |
| [二次曲面示例](examples/quadrics/README.md) | 球、圆柱、圆锥、圆锥曲线族和拓扑切换。 |
| [高中圆锥截面课堂](examples/classroom_cone_sections/README.md) | 五段带审查关键帧的教学场景。 |
| [丹德林课堂三幕](examples/classroom_dandelin_spheres/README.md) | 椭圆、抛物线、双曲线的静态教学叠加。 |
| [丹德林圆锥—圆柱切换](examples/dandelin_cone_cylinder_switch/README.md) | 两只相切球与连续变化的母面。 |
| [语义相机镜头](examples/parallel_camera_shots/README.md) | 命名平行视角、精确侧视、目标跟随和回滚。 |
| [源码项目相机镜头](examples/source_project_camera_shots/README.md) | 作者相机 JSON 与可重建派生产物。 |

## 测试与贡献

开发、构建需要安装 `.[dev]`。请按修改范围选择测试层级；下面三条层级命令是
备选，不是每次都要依次重复执行：

```bash
python scripts/run_ci_test_tier.py core
python scripts/run_ci_test_tier.py cairo-smoke
python scripts/run_ci_test_tier.py all
python -m build
python -m twine check dist/*
```

高清关键帧、完整运动扫描、可复现构建和 MP4 证据另行运行。修改运行时或发布证据
之前，请阅读[贡献说明](CONTRIBUTING.md)、
[维护与发布指南](docs/maintainer-guide.md)和
[扩展 Cairo 验收](docs/extended-quadric-ci.md)。

## 许可与安全

MIT，见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

TikZ 编译会调用外部 TeX 与 Manim 工具。不要在包含敏感文件的环境中，以不受限
shell escape 处理不可信源码；详见 [SECURITY.md](SECURITY.md)。

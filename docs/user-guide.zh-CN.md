# 中文操作指南

这份指南从全新 checkout 开始，带你完成安装、首个 TikZ 原生场景、源码项目、
自动遮挡、二次曲面、丹德林构造和平行相机动画。命令尽量直接使用仓库内已经验证
的示例，避免只给一段无法独立运行的代码。

[English](user-guide.md) · [文档导航](README.md) ·
[公共 API](public-api.md)

## 一、选择经过验证的环境

包元数据写的是 Python 3.11 及以上，但当前 GitHub CI 实际覆盖 Python 3.11 和
3.12，发布证据固定使用 Python 3.12.13。为了尽量复现正式环境，推荐使用
Python 3.12；系统里的 `python3` 即使版本更高，也不代表已经通过本项目验证。

项目固定使用 Manim Community `0.20.1`，另外需要：

- Manim 所需的 Cairo、Pango 和 `pkg-config`；
- 用于视频输出与检查的 FFmpeg、`ffprobe`；
- 用于公式和中文的 XeLaTeX、TikZ、`dvisvgm`、Fandol 与 Latin Modern 字体；
- 完整测试和证据流程使用的 Poppler 工具。

Manim 官方的[本地安装说明](https://docs.manim.community/en/stable/installation.html)
列出了不同系统上的 Cairo/Pango 依赖。macOS 常用的基础命令是：

```bash
brew install cairo pango pkg-config ffmpeg poppler
```

TeX 建议使用包含 XeLaTeX、TikZ、Fandol 和 Latin Modern 的完整
[MacTeX](https://tug.org/mactex/)。

Debian / Ubuntu 可以直接采用项目 CI 的依赖清单：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  dvisvgm ffmpeg libcairo2-dev libpango1.0-dev pkg-config poppler-utils \
  texlive-fonts-recommended texlive-lang-chinese texlive-latex-extra \
  texlive-xetex
```

## 二、从 GitHub checkout 安装

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
export TIKZ_NATIVE_REPO="$PWD"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

如果你的 Python 3.12 命令名称不同，先确认 `python --version` 确实是 3.11 或
3.12，再创建环境。普通使用者安装 `.[test]`；需要构建发行包和维护项目的贡献者
改装 `.[dev]`。

先检查外部工具：

```bash
python -m manim --version
printf 'n\n' | python -m manim checkhealth
command -v xelatex
command -v dvisvgm
command -v ffmpeg
command -v ffprobe
command -v pdftoppm
kpsewhich FandolSong-Regular.otf
kpsewhich latinmodern-math.otf
```

然后检查四个版本化 Bridge。每条命令都会输出一份 JSON 健康信息：

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

最后从仓库外面验证导入。这样可以发现“只有当前目录恰好是源码根目录，所以看似
能导入”的坏 editable 环境：

```bash
cd "$TIKZ_NATIVE_REPO"
(cd /tmp && "$TIKZ_NATIVE_REPO/.venv/bin/python" -c \
  'import tikz_native, polyhedron_visibility; print("imports ok")')
```

下面各条路线会继续使用 `TIKZ_NATIVE_REPO`。如果换了新终端，请先激活虚拟环境，
并把这个变量重新设为 checkout 的绝对路径。

## 路线一：把 TikZ 编译成原生 Manim 对象

先建立一个独立练习目录，并复制仓库自带的起步图形：

```bash
quickstart_directory="$(mktemp -d /tmp/manim-tikz-quickstart.XXXXXX)"
cp "$TIKZ_NATIVE_REPO/tikz_native/examples/native_friendly_figure.tex" \
  "$quickstart_directory/figure.tex"
cd "$quickstart_directory"
```

创建 `scene.py`：

```python
from pathlib import Path

from manim import Scene
from tikz_native import NativeManimRenderer, compile_document


class NativeTikzScene(Scene):
    def construct(self):
        document = compile_document(Path("figure.tex"))
        picture = document.pictures[0]
        if picture.unsupported:
            raise ValueError("Unsupported TikZ: " + "; ".join(picture.unsupported))

        figure = NativeManimRenderer(scene_unit_per_cm=1.0).render(picture)
        self.add(figure.group)
        self.wait()
```

使用 Cairo 渲染：

```bash
python -m manim --renderer cairo -ql \
  --media_dir media/quickstart scene.py NativeTikzScene
cd "$TIKZ_NATIVE_REPO"
```

命令没有加 `-p`，所以在 SSH 或无桌面环境中也能运行；本机希望渲染后自动打开
视频时再加 `-p`。得到的图形由真正的 Manim 对象组成，并保留语义 ID 到对象的
映射，不是一整张不可编辑的 SVG。

三维 TikZ 有两个不同出口：

- `NativeFixedViewRenderer`：按源码投影压成普通二维 Manim 对象；
- `NativeManim3DRenderer`：保留世界坐标，供 Manim 三维相机继续观察。

在加入裁剪、任意路径、装饰或自定义宏之前，请先看
[支持的 TikZ 子集](supported-tikz.md)。

## 路线二：使用“源码权威”项目

源码项目把 TikZ、可选的 motion / camera 配置和渲染意图当成长期保存的作者输入；
ShapeAsset、合成计划、生成源码和 build manifest 都是可以随时重建的派生产物。

把仓库示例复制到仓库外再试完整命令：

```bash
source_project_directory="$(mktemp -d /tmp/tikz-native-camera-demo.XXXXXX)"
cp -R "$TIKZ_NATIVE_REPO/examples/source_project_camera_shots/." \
  "$source_project_directory/"
(
  cd "$source_project_directory"
  tikz-native-project build project.json
  tikz-native-project status project.json
  tikz-native-project rebuild project.json
  tikz-native-project clean project.json
)
```

| 命令 | 含义 |
| --- | --- |
| `build` | 构建缺失或已过期的节点，复用仍然有效的节点。 |
| `status` | 只读检查 fresh / stale / missing，不写输出目录。 |
| `rebuild` | 忽略缓存，重新生成全部派生节点。 |
| `clean` | 只删除已经验证属于该项目的 `derivedOutput`。 |

`status` 在全部新鲜时退出 0，在缺失或过期时退出 1；两者都会返回合法 JSON。
退出 2 才表示输入、所有权、安全检查或构建失败。输出目录标记、锁、原子发布和
回滚规则见[源码权威项目](source-authoritative-projects.md)。

`tikz-native-source-v3` 是严格的机器接口：`run` 需要版本化 JSON、源码 SHA-256
和当前 Provider revision。它返回定义与辅助函数源码，并不自动生成一个可直接
交给 Manim 的 `Scene` 类。普通作者优先使用 source project 或仓库里已有宿主场景，
不要把 Source v3 输出误当成可直接渲染的视频脚本。

## 路线三：给普通 Manim 场景加入自动遮挡

先按几何类型选择入口：

| 几何对象 | 推荐入口 | 仓库示例 |
| --- | --- | --- |
| 闭合凸多面体 | `OcclusionScene3D` | `examples/polyhedron_visibility/cube_auto_occlusion.py` |
| 有限开放面或铰链 | `OpenFaceScene3D` | `examples/open_face_visibility/dihedral_auto_occlusion.py` |
| 闭合实体、一个截平面和可选自由直线 | `ConvexSectionScene3D` | `examples/convex_sections/convex_sections_demo.py` |
| 从原实体复制出的两面二面角 | `ExtractedDihedralScene3D` | `examples/derived_dihedral_extraction/` |

先渲染三个代表场景：

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -ql \
  examples/polyhedron_visibility/cube_auto_occlusion.py \
  CubeAutoOcclusionDemo

python -m manim --renderer cairo -ql \
  examples/open_face_visibility/dihedral_auto_occlusion.py \
  DihedralAutoOcclusionDemo

python -m manim --renderer cairo -ql \
  examples/convex_sections/convex_sections_demo.py \
  CombinedSectionAndLineDemo
```

这里的“自动”是指登记好几何和拓扑后，线段分段、虚实判断与绘制顺序自动计算；
它不会从任意 `VGroup` 猜出哪些多边形是面、哪些重合点是同一顶点。作者仍要登记
稳定顶点、最大凸面、语义线条及其相邻面。求解器使用平行投影，输入模糊或超出
契约时会明确失败，而不是退回手写 `z_index`。

`ConvexSectionScene3D` 在冻结或绑定之前必须声明恰好一个截平面，不能把它当成
“只处理自由直线”的入口。若已经有冻结后的 visibility model，只想直接绑定面明暗，
可看[公共 API](public-api.md#automatic-face-depth-cues)里的低层
`DepthCuedAutoOcclusion3D`。

## 路线四：制作二次曲面与圆锥曲线动画

固定拓扑的截平面平移或旋转优先使用 `QuadricSectionRig`。它会统一计算截线、
可见/隐藏片段、截平面深度区域、曲面顺序、固定 Manim 槽位和失败回滚。

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -r 480,270 --fps 15 \
  --media_dir media/quadric-quickstart \
  examples/quadrics/quadric_section_rig_quick_start.py \
  ConeSectionRigQuickStart
```

调构图时使用 `render_profile="preview"`、480×270、15 fps；课堂成片改成
`render_profile="final"`、960×540、30 fps。分辨率和帧率仍由 Manim 命令控制，
因为控制器建立时渲染器已经存在。

- 普通回调式截面使用 `QuadricSection3D`；
- 椭圆 → 抛物线 → 双曲线的族切换使用 `QuadricSectionTransition3D`；
- 没有截平面、只需协调彼此分离的二次曲面和曲线时使用 `QuadricOcclusion3D`；
- `OPEN_DOUBLE` 的受限双锥公共截面使用 `CompositeQuadricSection3D`。

第一阶段 `QuadricSectionRig` 会冻结一个静态平行投影；只要显示平面或截线，它就会
在播放前拒绝精确侧视。需要已支持的 AREA → LINE → AREA 相机切换时，应使用低层
`QuadricSection3D` 并传入实时相机投影。若要预编译并统一协调作者镜头、截面时间线、
拓扑 banks 和 preflight，则走另一条 `compile_parallel_section_rig_from_shots()` 路径。

公开解析曲面包括有限球、封闭或开放圆柱、封闭单锥、开放单锥壳和有限开放双锥
壳。生产绑定只支持 Cairo 与平行投影。一般相交的多个二次曲面、任意多个可见
截平面、透视和 OpenGL 正式绑定仍在 v1 范围外。

继续开发前先读[有限二次曲面创作工作流](quadric-authoring-workflow.md)，不要一开始
就绕过高层入口手接底层 compositor。

## 路线五：选择正确的丹德林路径

项目中有几条用途不同的丹德林路径，不能把它们都概括成“物理自动遮挡”：

| 路径 | 已认证内容 | 重要边界 |
| --- | --- | --- |
| `DandelinSection3D` 课堂入口 | 解析丹德林构造和普通圆锥截线 | 球与辅助线是静态顶层教学叠加。 |
| `DandelinOcclusion3D` | 在固定 Cairo 槽位中随实时相机重算曲线实线/虚线片段 | 填充只是教学层，而且绑定不拥有截平面填充。 |
| depth-aware 空间 TikZ | 冻结视角下的边界实线/虚线片段 | 运动和相机镜头会拒绝，填充也不是物理隐藏面。 |
| `depth_aware_teaching_transparent` | 静态空间视图中，单锥瓣圆/椭圆/抛物线的曲线片段与课堂绘制顺序 | 必须显示接触圆；运动和其他视图会拒绝，教学透明也不是光学透明。 |
| `SceneOcclusionCoordinator` 的 nested-tangent 路径 | 一个圆锥/圆柱母面、一个相切平面、恰好两只相切球和登记过的解析边界 | 是狭义构型，不是任意对象通解。 |

先看静态三幕课堂场景：

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir media/classroom-dandelin \
  examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py \
  DandelinThreeConicsLesson
```

再看完整的圆锥面—圆柱面切换：

```bash
python -m manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir media/dandelin-cone-cylinder-switch \
  examples/dandelin_cone_cylinder_switch/dandelin_cone_cylinder_switch.py \
  DandelinConeCylinderSwitch
```

切换动画把平面边、母面边、球轮廓、接触圆、真实截线和教学轴线都登记为解析
边界源，逐帧重算实线/虚线。半透明填充仍表达课堂图层，不声称逐像素模拟真实
透明材质。这个示例的最终 Manim 组装代码也仍属于场景示范，并不是已经封装好的
通用 `DandelinConeCylinderSwitch3D` 公共入口。

受控 TikZ 的固定三视图路线见
[TikZ-native 丹德林三视图](../examples/tikz_dandelin_views/README.md)。

## 路线六：使用语义平行相机镜头

`ParallelCameraState` 把可逆平行视角、世界目标点、画面锚点和缩放放在同一状态
中；`ParallelCameraShotSequence` 再加入镜头名称、时长、停留、提示词和安全切换。
这两类只是不可变的作者数据；真正播放时，Scene 要使用 `MultiProjectionCamera`，
再调用 `play_parallel_camera_shot()` 或 `play_parallel_camera_shot_sequence()`。

```bash
cd "$TIKZ_NATIVE_REPO"
python -m manim --renderer cairo -ql \
  examples/parallel_camera_views/scene.py \
  TargetOrbitCameraDemo PlaneViewReductionDemo AnchorZoomCameraDemo

python -m manim --renderer cairo -ql -r 480,270 --fps 6 \
  examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
  SingleConeSectionShotDemo
```

“沿平面观察”仍是合法的三维相机，只是该平面在屏幕上退化为一条有限线段。协调
截面运行时能处理已认证的 AREA → LINE → AREA 切换；透视相机不能交给二次曲面
和自动遮挡管线。

## 输出目录与运行位置

- 除非指南另有说明，仓库示例命令都从仓库根目录运行。
- 本地 Manim 输出放在 `media/` 或仓库外。`media/` 已被 Git 忽略；`artifacts/`
  不是通用临时目录。
- source project 只拥有清单中写明的 `derivedOutput`，不要把作者文件放进去。
- Manim 可能在命令工作目录中生成 `media/Tex` 缓存，这是正常派生数据。

## 常见问题

### 仓库根目录能导入，运行示例却报 `ModuleNotFoundError`

先确认虚拟环境已激活，再执行安装部分的“仓库外导入”检查。macOS 上 editable
`.pth` 可能继承 Finder 的 hidden 标志，Python 3.12 会跳过它。只修复虚拟环境并
重装：

```bash
chflags -R nohidden "$TIKZ_NATIVE_REPO/.venv"
"$TIKZ_NATIVE_REPO/.venv/bin/python" -m pip install \
  --force-reinstall --no-deps -e "$TIKZ_NATIVE_REPO"
```

`PYTHONPATH=$PWD` 可以暂时用于定位问题，不应该替代正常的隔离环境安装。

### 公式、中文或渲染失败

检查 XeLaTeX、`dvisvgm`、FFmpeg 和前面列出的两种字体；确认使用项目固定的
Manim 0.20.1。直接安装最新 Manim 并不等于与当前 checkout 兼容。

### 编译器拒绝普通 TikZ 能画出的内容

这是受控子集编译器，不是任意 TikZ 转换器。请把源码缩成最小示例，对照
[支持的 TikZ 子集](supported-tikz.md)。项目不会偷偷改用 SVG 或位图。

### `tikz-native-project status` 返回 1

先读它输出的 JSON。退出 1 只是派生产物缺失或已过期；确认计划后执行 `build`。
退出 2 才表示输入、所有权、安全检查或构建失败。

### 二次曲面或丹德林场景拒绝 OpenGL / perspective

切换到 Cairo 和受支持的平行视角。这些不是随意的视觉偏好，而是生产契约的一部分。

### Bridge `run` 报源码哈希或 revision 不匹配

重新计算源码 SHA-256，并读取当前 `health` 输出。Bridge 请求绑定的是准确源码快照
和 Provider revision，不能悄悄拿另一版代码执行旧请求。

## 继续阅读

- [项目文档导航](README.md)
- [公共 API](public-api.md)
- [贡献说明](../CONTRIBUTING.md)
- [维护与发布指南](maintainer-guide.md)

# TikZ → 原生 Manim：工具接入完整说明

> 迁入说明：Provider 当前位于 `tex-to-mathcapture-ppt/tools/tikz-native-provider/`。
> 本文中的 `tikz_native/`、`scripts/` 和 `tests/` 均相对于该目录；旧研究场景、
> `reports/` 与 `media/` 路径只作为迁入前验收证据，不属于运行时 Provider。

本文面向需要把 TeX 文档中的 TikZ 图接入“TeX → PPT → Manim 动画”工具链的开发者。
它描述当前工作区已经实现的真实能力、推荐的接入分层、数据契约、二维和三维渲染顺序、
动态关系、错误处理、缓存、并发、安全和验收要求。

本文中的“原生 Manim”不是指对象不继承 `VMobject`。`Line`、`Polygon`、`MathTex`
本来就是 Manim 对象体系的一部分。这里的准确含义是：

- 线、面、点、角标、标签分别创建为具有明确语义的 Manim 对象；
- 每个对象拥有稳定 ID，可以单独出现、消失、变形或跟随几何关系运动；
- 不把整张 TikZ 先转换为一块 SVG、位图或无语义的通用路径；
- 遇到当前子集之外的 TikZ 语法时报告并停止，不做静默降级。

当前标准化子集的权威定义是：

- [`tikz-native-subset-v0.1.md`](tikz-native-subset-v0.1.md)
- [`../tikz_native/subset_v0_1.json`](../tikz_native/subset_v0_1.json)

已有的源码编写规范和阶段基线分别见：

- [`tikz-native-manim-conversion-guide.md`](tikz-native-manim-conversion-guide.md)
- [`tikz-native-v0.1-baseline.md`](tikz-native-v0.1-baseline.md)
- [`tikz-native-3d-prototype.md`](tikz-native-3d-prototype.md)

本文不承诺支持完整 PGF/TikZ。它规定的是一个可验证、可扩展、适合动画的受控子集。

## 1. 总体结论

推荐把转换过程接成下面七层，而不是把“解析、预览、正式渲染、写入 PPT”做成一个不可拆分的按钮：

```text
输入 TeX
  ↓
选择零参数图形宏 / 提取 tikzpicture
  ↓
编译为语义中间表示 DocumentSpec
  ↓
A/B/C 兼容性门禁
  ↓
实例化为二维或三维原生 Manim 对象
  ↓
静态复刻验收 + 动态关系/时间线配置
  ↓
输出 PNG/MP4 + manifest/report，最后由 PPT 层引用
```

工具接入时应坚持以下四条总原则：

1. `DocumentSpec` 是 TikZ 和 Manim 之间的稳定中间层，不能让 UI 直接依赖解析器内部正则。
2. “静态转换成功”“对象可独立动画”“几何关系已经动态化”“教学时间线已编排”是四种不同状态。
3. 转换阶段保持真实比例，不对整图额外 `scale()`；构图只通过相机视野、画布和 PPT 布局处理。
4. 所有正式输出都必须保留源文件哈希、入口宏、图编号、对象 ID、兼容性报告和渲染参数。

## 2. 当前代码入口

核心模块位于 `tikz_native/`：

| 模块 | 作用 |
| --- | --- |
| `compiler.py` | 解析受控 TikZ 子集，生成 `DocumentSpec` |
| `macro_frontend.py` | 读取简单宏，并物化一个零参数图形入口宏 |
| `compatibility.py` | 按 A/B/C 子集生成机器可读兼容性报告 |
| `manim_renderer.py` | 把二维 `ObjectSpec` 创建为原生 Manim 对象 |
| `manim_renderer_3d.py` | 保留世界坐标，创建三维对象、动态标签和遮挡关系 |
| `projection_3d.py` | 把 TikZ `3d view` 或显式基向量转换为 3×3 投影矩阵 |
| `animation.py` | 生成确定性的对象揭示顺序 |
| `dynamic_geometry.py` | 把主动参数和几何依赖绑定到已经创建的对象 |
| `occlusion_3d.py` | 计算平行投影下有限面片对线段的遮挡区间 |

公开 Python 入口为：

```python
from tikz_native import (
    audit_document_compatibility,
    compile_document,
)
```

命令行入口为：

```text
scripts/convert_tikz_native.py
scripts/audit_tikz_native_compatibility.py
```

需要注意：当前转换器生成的是语义清单和运行时对象，不是自动生成一份完整教学场景
`.py`。具体主动对象、运动区间、讲解顺序和动画时长仍属于消费端工具或场景模板。

### 2.1 从现有工具导入模块

当前目录还不是一个可由 `pip install` 安装的独立 Python 包。Bridge 以 Provider 根目录
作为 `PYTHONPATH` 独立运行；迁入前的演示场景曾另外依赖外部
`MultiProjectionCamera`，但 Provider 包本身不依赖它。可以使用：

```bash
cd /path/to/tex-to-mathcapture-ppt/tools/tikz-native-provider
PYTHONPATH="$PWD" .venv-manim/bin/python -m tikz_native.bridge health
```

不要在多个业务模块中各自散落 `sys.path.insert(...)`。正式长期接入时，更合适的做法是把
这一层收敛到一个适配器进程，或再把转换器整理为可安装包；无论选择哪一种，都应保持公共
`compile_document` / renderer 接口不变。

## 3. 运行环境和依赖

当前工作区验证环境使用 Manim Community、XeLaTeX、FFmpeg 和 Poppler。工具在启动时至少应检查：

```bash
.venv-manim/bin/python -c 'import manim; print(manim.__version__)'
xelatex --version
ffmpeg -version
pdftoppm -v
```

标签使用 `MathTex` / `Tex` 和 XeLaTeX。白色描边标签还会调用高分辨率 TeX 探针与
`pdftoppm` 测量实际 TikZ node 外框和墨迹偏移，因此仅有 Manim 还不够。

默认 `TexTemplate` 目前引用本机字体目录 `/Users/leocyan/Library/Fonts/` 中的宋体、
黑体、仿宋和 Latin Modern 文件。接入其他机器时必须选择一种方案：

- 启动时检查这些字体文件并阻断缺失环境；或
- 由工具注入自己的 `TexTemplate`，同时把模板内容及字体文件版本写入缓存键和报告。

不能在字体缺失时悄悄换字体，因为字形尺寸变化会直接改变标签位置和白色描边 node 的测量结果。

## 4. 输入契约

### 4.1 普通 TeX 文档

若文档直接包含一个或多个 `tikzpicture`，可以直接编译：

```python
from pathlib import Path
from tikz_native import compile_document

document = compile_document(Path("/absolute/path/document.tex"))
```

`document.pictures` 按源码中物化后的出现顺序保存图形。`PictureSpec.index` 是从 1
开始的报告编号，Python 列表下标仍从 0 开始。工具界面应显示前者，不要把两者混用。

### 4.2 图形宏库

若文档把各幅图定义为宏，应选择一个零参数入口宏：

```python
document = compile_document(
    Path("/absolute/path/figures.tex"),
    entry_macro="SquarePyramidFig",
)
```

入口名前导反斜杠可以省略。当前宏前端的边界是：

- 可以物化选中的零参数入口宏；
- 可以处理受控的简单递归展开和部分纯数值辅助宏；
- 入口宏若要求参数会直接报错；
- 未解析的 `#1` 等参数占位符会直接报错；
- 任意条件分支、复杂 pgfkeys、带副作用的宏不能当作稳定接口。

因此，推荐为每一幅需要接入工具的图保留一个稳定的零参数包装宏：

```tex
\newcommand{\SquarePyramidFig}{%
  \begin{tikzpicture}[3d view={40.4}{23.8}]
    % ...
  \end{tikzpicture}%
}
```

不要让工具通过正则猜测应调用哪个多参数底层宏。

### 4.3 输入快照

接入工具时，转换作业至少记录：

```json
{
  "source_path": "/absolute/path/document.tex",
  "source_sha256": "...",
  "entry_macro": "SquarePyramidFig",
  "picture_index": 1,
  "subset_version": "v0.1"
}
```

正式渲染应从已经记录哈希的输入快照读取。若编辑器中的 TeX 在排队后又发生变化，
应创建新作业，不能让同一个作业 ID 的输入内容漂移。

## 5. 语义中间表示

### 5.1 `DocumentSpec`

`compile_document(...)` 返回 `DocumentSpec`。主要字段包括：

| 字段 | 含义 |
| --- | --- |
| `source_path` | 源文件绝对路径 |
| `source_sha256` | 源文件内容哈希 |
| `entry_macro` | 实际物化的图形宏；无则为 `None` |
| `colors` | 已解析的颜色定义 |
| `pictures` | 一个或多个 `PictureSpec` |
| `warnings` | 文档级非致命信息 |

可以直接输出机器可读清单：

```python
document.write_json(output_dir / "manifest.json")
```

### 5.2 `PictureSpec`

一幅图的主要字段为：

| 字段 | 含义 |
| --- | --- |
| `index` | 从 1 开始的图编号 |
| `scale` | TikZ 图形几何比例 |
| `dimension` | `2` 或 `3` |
| `projection_3d` | 三维图的视角来源和 3×3 矩阵 |
| `coordinates` | 命名点到二维/三维坐标的映射 |
| `coordinate_dependencies` | 插值、平移、垂足等构造关系 |
| `named_paths` | 带方向的命名直线、椭圆或圆 |
| `intersections` | 路径求交关系、交点名和排序参数 |
| `objects` | 可实例化的 `ObjectSpec` 列表 |
| `occlusion_relations` | 三维线段与有限面片的遮挡关系 |
| `warnings` | B 级或其他需要保留的信息 |
| `unsupported` | 严格模式必须阻断的内容 |

### 5.3 `ObjectSpec`

每个对象至少具有：

- `id`：稳定动画接口；
- `kind`：`line`、`polygon`、`dot`、`label` 等；
- `geometry`：端点、中心、半径、顶点等几何参数；
- `style`：线宽、颜色、透明度、虚线、字体和 node 信息；
- `z_index`：静态层级；
- `source_line`、`raw`：错误定位线索；
- `placement`：标签 anchor、偏移、路径位置和 `sloped` 信息。

对象 ID 是工具编排动画的主要接口，例如：

```python
figure.objects["line.P.Q"]
figure.objects["dot.P"]
figure.objects["label.P.P"]
figure.objects["right_angle.R.H.P"]
```

同一输入、同一入口宏和同一子集版本下，稳定 ID 不应因为预览、分辨率或相机取景而改变。

当前 `source_line` 对宏展开后的对象还不能始终精确回指到原始语句行。错误界面应同时显示：

```text
源文件 + 入口宏 + 图编号 + 对象 ID + source_line + raw
```

不要只显示行号。

## 6. A/B/C 兼容性门禁

### 6.1 等级含义

| 等级 | 含义 | 工具行为 |
| --- | --- | --- |
| A | 已有原生对象或关系接口，具备动态化基础 | 可以继续，但仍需显式选择主动参数和时间线 |
| B | 静态安全、近似或只属于 TeX 版面 | 允许继续，必须保留警告 |
| C | 当前没有可靠原生语义映射 | 严格阻断，不得退回 SVG/位图/通用路径 |

A 级并不等于“自动生成完整动画”。例如，`name intersections` 可以稳定保存交点
身份，但 TikZ 并没有说明究竟让直线转动、让点移动，还是只逐项揭示。

### 6.2 Python 门禁

工具接入优先使用同一个已编译文档做审计，避免再次解析不同输入：

```python
from tikz_native import audit_document_compatibility, compile_document

document = compile_document(source, entry_macro=entry_macro)
compatibility = audit_document_compatibility(document)

if compatibility["static_status"] != "pass":
    raise RuntimeError("TikZ-native compatibility blocked")
```

主要状态为：

```text
static_status  = pass | blocked
dynamic_status = native-relations-ready-explicit-driver-required | blocked
```

`dynamic_status` 中的 `explicit-driver-required` 是正常状态，不是错误。

### 6.3 命令行门禁

普通文档可以运行：

```bash
cd /Users/leocyan/Documents/Code/Manim/manim_scenes
.venv-manim/bin/python scripts/audit_tikz_native_compatibility.py \
  --input '/absolute/path/document.tex' \
  --output-dir reports/tikz_native/job-001
```

返回码：

- `0`：没有 C 级发现；
- `2`：存在 C 级发现，严格转换应停止；
- 其他非零：解析、依赖或程序错误。

当前审计 CLI 没有 `--entry-macro` 参数。宏库接入不能先用转换命令选择一个宏、
再让审计命令扫描另一份范围；应使用上面的 Python API 对同一个 `DocumentSpec` 审计。

## 7. 当前支持范围

### 7.1 动态安全的核心对象和关系

当前 A 级核心包括：

- 命名二维坐标和三维 `(x,y,z)` 坐标；
- 受控纯数值宏；
- 两点插值、中心对称、坐标平移和点到直线的正交投影；
- 命名直线、椭圆、圆；
- 带有向 `sort by` 的直线—直线、直线—椭圆/圆求交；
- `Line`、`Polygon`、`Ellipse`、`Circle`、`Dot` / `Dot3D`；
- `Stealth` 箭头；
- 普通标签、路径标签、`sloped` 标签和三维 billboard 标签；
- 二维角弧与直角记号；
- pt 线宽、显式虚线节距、HTML 颜色、xcolor 混色和透明度；
- `3d view={azimuth}{elevation}` 和显式 `x/y/z={(u,v)}` 投影基向量。

### 7.2 B 级内容

当前典型 B 级内容包括：

- `dashed`、`densely dashed`：使用近似节距；
- `baseline`、`trim right`：属于 TeX 外部行盒，不进入 Manim 几何；
- 已证明安全的冗余 `scope[draw=none]` 情形。

需要视觉精确的虚线应改写为：

```tex
\draw[dash pattern=on 1.6pt off 1.7pt] (A)--(B);
```

### 7.3 C 级内容

当前严格阻断的主要类型包括：

- 一般 Bézier、一般 `arc`、`plot`、`smooth`；
- `clip`、decoration、pattern、shade、gradient；
- 一般复杂 node、`text width`、自动换行和 matrix；
- 任意嵌套 scope 变换与完整样式继承；
- 任意参数宏、条件分支和复杂 pgfkeys；
- 复杂曲线多交点、相切合并、交点消失和重新出现；
- 任意空间曲线、任意平面内空间圆弧和一般三维角标；
- 非凸实体、曲面和透视投影下的一般自动遮挡。

“C 级”表示当前程序尚未建立可回归的原生映射，不表示 TikZ 本身写错了。

### 7.4 项目定制特例

当前渲染器已经针对工作区中的 `\mathWhiteOutline` 标签、部分矩形 node、
`transform shape` 和部分 `canvas is plane` 写法做了专门处理。这些能力不能外推为
“完整 node 引擎”或“完整 canvas 变换支持”。新工具应把已验证的确切写法当作
兼容性特例；改变 node 选项后仍需重新审计和对图回归。

三维线—有限凸面动态遮挡也已经在当前代码和专项测试中实现，但
`subset_v0_1.json` 的 feature registry 尚未给遮挡关系单列 feature。因此现阶段的
A/B/C 报告不会单独证明遮挡已经验收；工具必须同时检查 `occlusion_relations`、完成实际
实例化，并运行三维遮挡专项回归。以后将这项能力正式纳入公共子集时，应发布 registry
更新或新子集版本，不能仅修改说明文字。

## 8. 静态原生实例化

### 8.1 二维图

```python
from tikz_native.manim_renderer import NativeManimRenderer

picture = document.pictures[0]
if picture.dimension != 2:
    raise ValueError("expected 2D picture")
if picture.unsupported:
    raise RuntimeError("; ".join(picture.unsupported))

renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
figure = renderer.render(picture)

self.add(figure.group)
```

`NativeFigure` 提供：

- `picture`：原始 `PictureSpec`；
- `objects`：`object_id -> Mobject`；
- `group`：整幅图的 `VGroup`；
- `warnings`：非致命警告。

### 8.2 三维图

三维图必须使用保留世界坐标的渲染器，并先恢复 TikZ 视角：

```python
import numpy as np
from manim import ThreeDScene
from multi_projection_camera import MultiProjectionCamera
from tikz_native.manim_renderer_3d import NativeManim3DRenderer


class NativeTikz3DScene(ThreeDScene):
    def __init__(self, **kwargs):
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def construct(self):
        picture = document.pictures[0]
        if picture.dimension != 3 or picture.projection_3d is None:
            raise ValueError("expected native 3D picture")

        renderer = NativeManim3DRenderer(scene_unit_per_cm=1.0)
        figure = renderer.render(picture)

        matrix = np.asarray(picture.projection_3d.matrix, dtype=float)
        camera: MultiProjectionCamera = self.camera
        camera.set_projection_matrix(matrix, view_center=figure.view_center)

        # 顺序非常重要：先设相机，再绑定，再加入场景。
        renderer.bind_labels_to_camera(figure, camera)
        renderer.bind_occlusions_to_camera(figure, camera)

        self.add(figure.world_group)
        self.add_fixed_orientation_mobjects(*figure.fixed_orientation_labels)
```

`Native3DFigure` 把空间对象和屏幕朝向标签分开保存：

- `world_group`：面、线、点等三维对象；
- `fixed_orientation_labels`：始终正对屏幕的标签；
- `view_center`：相机应使用的图形中心；
- `occlusion_groups`：已经绑定的逻辑遮挡线容器。

## 9. 尺寸、字体和线宽

这是工具接入最容易再次引入回归的部分。

### 9.1 TikZ `scale` 的准确语义

默认情况下：

- `tikzpicture` 的 `scale` 缩放坐标和几何尺寸；
- 不缩放 node 文字；
- 不应被转换器再次作用到字体、pt 线宽、点半径或箭头尺寸；
- 只有源码显式使用 `transform shape` 时，node 形状和文字才随图形变换。

当前渲染器已经按此规则实现。消费端不能再对 `figure.group`、
`figure.world_group` 或其父组调用 `scale()` / `scale_to_fit_*()`，否则会重新把字体和
线宽一起缩放，破坏 TikZ 语义。

真实比例模式使用：

```python
renderer = NativeManimRenderer(scene_unit_per_cm=1.0)
```

画面放不下时，依次选择：

1. 扩大相机 frame；
2. 增大输出分辨率；
3. 拆分纵览；
4. 在 PPT 布局层统一放置最终资产。

不要逐图压缩内部对象。

### 9.2 字体规则

当前标签规则为：

- 默认 TeX 文档类为 `11pt`；
- 标准 `\normalsize` 实际约为 10.95pt；
- `\small` 等显式命令保留；
- 所有数学标签和混合文本中的数学片段默认注入 `\displaystyle`；
- TikZ `scale` 不改变标签尺寸。

工具若允许用户切换字体模板，必须把模板视为一次新的渲染配置，重新做标签尺寸和位置验收。

### 9.3 线宽规则

TikZ 的 pt 线宽与 Manim/Cairo 的屏幕描边并非数值一一相等。当前工作区经过视觉回归后，
几何描边使用 `3.8` 的 pt 换算系数，白色标签描边使用独立的 `2.15` 系数：

```python
NativeManimRenderer(
    scene_unit_per_cm=1.0,
    stroke_width_per_pt=3.8,
    label_outline_stroke_width_per_pt=2.15,
)
```

这些是渲染标定参数，不是数学几何参数。更换 Manim、Cairo、分辨率或抗锯齿环境后，应使用
TeX—Manim 对照图重新标定，不能只看源码数字。

## 10. 标签位置与白色描边

### 10.1 普通二维标签

转换器保留：

- `above`、`below`、`left`、`right` 及组合 anchor；
- pt 级 `xshift` / `yshift`；
- endpoint node 与 path node 的差别；
- `pos` 和 `sloped`；
- TikZ node 的内边距语义。

`(A)--node[pos=.5]{$d$}(B)` 与 `(A)--(B)node{$B$}` 不是同一种标签，
工具的源码编辑器不要为了格式化而交换 node 在路径中的位置。

### 10.2 白色描边标签

白色描边标签不能只给 Manim 字形加一圈普通 stroke 后仍沿用可见墨迹外框。TikZ 的
anchor 依赖完整 node 盒子，公式的视觉墨迹中心又不一定等于 TeX 盒子中心。

当前实现会：

1. 用 XeLaTeX 渲染一个高分辨率测量探针；
2. 测量 node 外框和黑色墨迹的相对偏移；
3. 缓存测量结果；
4. 按真实 node 盒子重新计算 anchor；
5. 分开标定白色 halo 和普通几何线宽。

工具接入时不要绕过这一过程，也不要对描边标签另做经验性 `shift()`。若白色描边结果有误，
应把标签源码、字体模板、node 选项和探针缓存键一起记录下来排查。

### 10.3 三维标签

三维标签不仅要正对屏幕，还要在每一帧按当前投影矩阵重新解释 `above`、`left` 等屏幕方向。
因此必须调用：

```python
renderer.bind_labels_to_camera(figure, camera)
```

这能维持 anchor、字号和方向，但不会自动解决所有标签—标签碰撞。较大的相机运动区间仍应配置：

- 分段 anchor；
- 必要时的可见性切换；
- 单独的碰撞检查；
- 教学上可接受的相机路径。

不能通过缩小字体掩盖碰撞。

## 11. 自动揭示和教学时间线

转换器可以生成确定性的基线揭示层：

```text
fills
  → coordinate_frame
  → solid_geometry
  → auxiliary_geometry
  → markers
  → points
  → labels
```

相关 API：

```python
from tikz_native.animation import (
    play_named_reveal,
    play_semantic_reveal,
    semantic_animation_layers,
)
```

基线揭示适合验证“每个对象是否真的能独立动画”，不代表唯一教学顺序。正式教学动画应保存一份
独立于 TikZ 的时间线，例如：

```json
{
  "schema_version": 1,
  "steps": [
    {"action": "create", "targets": ["line.A.B", "line.A.D"]},
    {"action": "grow", "targets": ["dot.P"]},
    {"action": "write", "targets": ["label.P.P"]}
  ]
}
```

推荐让时间线只引用稳定对象 ID，不引用 `submobjects[3]` 这类不稳定位置。

## 12. 动态几何的三种层级

### 12.1 刚体级动画

如果整幅图只需平移、旋转或整体淡入，可以直接动画 `figure.group`。这不建立新的几何依赖，
也不等于“点、线按约束运动”。

### 12.2 对象级揭示和变换

使用稳定 ID 分别对线、点、标签调用 `Create`、`FadeIn`、`Write`、`Transform`。
结构明显不同的公式标签不要直接做点级 `Transform`，应优先：

- `FadeOut` + `FadeIn`；或
- 结构确实适合时使用 `TransformMatchingTex`。

### 12.3 约束驱动动画

真正的动态几何应有一个或少数主动参数，然后逐帧重算从动对象。例如：

```text
角度 θ
  → 有向直线 l
  → l 与椭圆的交点 Q、P
  → 中心对称点 R
  → 线段、填充、点、标签、垂足和直角记号
```

当前二维统一绑定器为 `NativeMotionBinder`，支持绑定：

- 实线、箭头和显式虚线；
- 点；
- 多边形；
- 普通标签；
- 路径标签；
- 直角记号。

主动参数通常用 `ValueTracker` 提供：

```python
from manim import ValueTracker
from tikz_native.dynamic_geometry import (
    EllipseChordDriver,
    NativeMotionBinder,
)

theta = ValueTracker(initial_angle)
driver = EllipseChordDriver.from_named_intersection(
    theta.get_value,
    picture,
    relation_index=0,
    pivot_name="F",
)
binder = NativeMotionBinder(figure, renderer)

binder.bind_line(
    "line.Lstart.Lend",
    lambda: driver.state().line_start,
    lambda: driver.state().line_end,
)
binder.bind_dot("dot.P", lambda: driver.state().p)
binder.bind_dot("dot.Q", lambda: driver.state().q)
```

这只是接口示意。实际对象 ID 必须从当前 `manifest.json` 读取，不能照抄示例。

### 12.4 驱动配置属于消费端

当前编译器会保留 TikZ 已经表达出的构造关系，但不会从静态图猜测主动对象。推荐工具另存一份
动态配置：

```json
{
  "driver": {
    "id": "theta",
    "type": "value_tracker",
    "initial": 0.66,
    "range": [0.38, 0.94]
  },
  "relation": {
    "type": "ellipse_chord",
    "intersection_index": 0,
    "pivot": "F"
  },
  "timeline": [
    {"to": 0.94, "duration": 2.0},
    {"to": 0.38, "duration": 2.0}
  ]
}
```

这份 JSON 是推荐的工具层契约，不是当前编译器自动识别的 TikZ 语法。

## 13. 动态交点的身份

推荐的 TikZ 写法是：

```tex
\path[name path=C]
  (O) ellipse [x radius=2,y radius={sqrt(3)}];
\path[name path=l]
  (Lstart)--(Lend);
\path[name intersections={
  of=C and l,
  sort by=l,
  by={Q,P}
}];
```

`sort by=l` 沿 `Lstart → Lend` 的路径方向排序，所以 `Q`、`P` 的身份有明确来源。
Manim 逐帧求交时必须保留同一条有向路径，不能依赖数值求交函数碰巧返回的数组顺序。

反转 `Lstart` 和 `Lend` 会反转交点顺序，这是语义变化，不是随机错误。

动态区间还必须满足：

- 直线持续与曲线有两个有效交点；
- 不穿过相切、交点合并或消失的拓扑边界；
- 若必须穿越边界，应先设计对象出现/合并/消失策略，并把该能力作为新 feature 回归。

## 14. 三维相机

### 14.1 初始视角

`3d view={azimuth}{elevation}` 按 TikZ perspective 库定义生成正交相机矩阵，
不是通过截图估算。显式 `x/y/z={(u,v)}` 则保留一般工程投影的两个屏幕基向量，并补充一个
确定的深度方向。

初始帧必须先调用：

```python
camera.set_projection_matrix(matrix, view_center=figure.view_center)
```

之后才能绑定标签和遮挡。

### 14.2 相机过渡选择

`MultiProjectionCamera` 提供两类过渡：

```python
self.play(camera.animate_to_matrix(target_matrix, view_center=center))
self.play(camera.animate_orbit_to_matrix(target_matrix, view_center=center))
```

选择原则：

- `animate_orbit_to*` 只用于起点和终点都是正交相机姿态的情况；
- `3d view` 生成的正交姿态可以使用球面轨道；
- 一般显式 `x/y/z` 基向量可能包含剪切或不等比缩放，应使用一般矩阵插值
  `animate_to*`；
- 不得把工程投影强行解释为透视相机或刚体相机旋转。

### 14.3 相机取景

`camera.set_zoom(...)` 属于最终构图，不应改写几何对象。用于 TeX—Manim 同尺度对照时，
应固定统一的 zoom、frame、分辨率和裁切规则；用于 PPT 版面时，则由上层布局统一决定。

## 15. 三维动态遮挡

### 15.1 当前实现

编译器会把项目中已经语义化的线—面遮挡宏保存为 `OcclusionRelationSpec`。当前识别的
项目宏包括：

- `DrawSpaceLineBehindHorizontalFace`；
- `DrawSpaceLineBehindTriFace`；
- `DrawSpaceLineBehindParallelogramFace`；
- `DrawSpacePlaneInteraction`。

每一条逻辑遮挡线会在场景开始前创建：

- 遮挡前可见段槽位；
- 面后隐藏段槽位；
- 遮挡后可见段槽位；
- 显式虚线需要的固定原生 `Line` 池。

相机运动时只修改这些已有 `Line` 的端点和透明度，不新增、删除或替换子对象。因此 Cairo
在第一次 `play` 开始时就已经看到完整对象拓扑，不会在第一次相机动画结束后突然跳变。

静态图也应调用同一个绑定函数并停留在初始视角，从而与后续动态视频使用同一套遮挡结构。

### 15.2 调用时序

正确顺序：

```text
render figure
  → set camera matrix
  → bind labels
  → bind occlusions
  → add world_group / labels
  → play camera animation
```

错误做法是先把静态虚线加入场景，等第一段相机动画结束后才替换成动态遮挡线。

### 15.3 揭示时的对象选择

绑定遮挡后，源码中的若干静态线对象会从 `world_group` 中移除，并由一个逻辑关系容器替代。
制作三维揭示动画时应：

1. 收集所有 `relation.object_ids`；
2. 普通线集合中排除这些源码成员；
3. 加入 `figure.occlusion_groups.values()`；
4. 需要按 ID 寻址时使用 `relation.id`。

否则可能同时绘制旧静态线和新动态线。

### 15.4 动点和动面

默认遮挡计算读取 TikZ 的固定坐标。若点或面也在运动，必须提供实时坐标：

```python
renderer.bind_occlusions_to_camera(
    figure,
    camera,
    coordinate_provider=lambda name: current_coordinates[name],
)
```

`coordinate_provider` 应返回未乘 `picture.scale` 的原始二维/三维坐标，缩放由渲染器统一处理。

### 15.5 当前边界

动态遮挡目前只支持：

- 平行投影；
- 三角形、平行四边形等有限凸面片；
- 线段与面片之间的单一遮挡区间。

仍未自动解决：

- 透视投影下逐点变化的视线；
- 非凸面、多区间遮挡、曲面；
- 多个透明面片在相机运动中的填充前后重排；
- 一般闭合实体的全部可见面排序。

显式虚线槽位容量按绑定时的整条线长度预分配。只有相机运动时线长不变，不会触发容量问题。
若 `coordinate_provider` 允许线段显著变长，必须在绑定前按最大可能长度预留容量或扩展 API；
当前超出容量会抛出：

```text
animated occlusion line exceeded its preallocated dash capacity
```

工具不能捕获后继续输出残缺视频。

## 16. 推荐的 TikZ 源码规范

### 16.1 命名所有教学对象

```tex
\coordinate (P) at (1,1.5);
\coordinate (Q) at (-1.8,-0.6);
\draw (P)--(Q);
```

匿名坐标可以静态转换，但不适合作为长期动画接口。

### 16.2 独立动画对象分开写

```tex
\draw (A)--(B);
\draw (B)--(C);
\draw (C)--(A);
```

虽然转换器可以拆折线，分开写能让对象 ID、错误定位和教学顺序更稳定。

### 16.3 填充和轮廓分开

```tex
\fill[fill=lectureteal!14] (A)--(B)--(C)--cycle;
\draw (A)--(B);
\draw (B)--(C);
\draw (C)--(A);
```

这样面和每一条边都能单独动画。

### 16.4 几何关系优先使用 TikZ 原生语义

优先使用：

- 命名坐标；
- `name path` / `name intersections` / `sort by`；
- calc 插值、平移和投影；
- 明确的 `(x,y,z)` 世界坐标；
- 集中写在 `tikzpicture` 上的三维视角。

只有 TikZ 原生无法稳定表达主动参数时，才在工具层增加动态配置。

### 16.5 样式显式化

推荐明确写出：

```tex
line width=0.85pt,
line cap=round,
line join=round,
dash pattern=on 1.6pt off 1.7pt,
opacity=0.72
```

颜色混合 `lecturegold!30` 与透明度 `opacity=.3` 是两种不同语义。

### 16.6 避免隐藏关键关系

重复绘制可以封装，但关键点、路径方向、交点命名、面拓扑和主动对象候选不应只藏在多参数宏、
循环局部变量或条件分支里。

### 16.7 三维图保留真实坐标

不要把每个三维点提前手算为二维投影坐标。应保留：

```tex
\coordinate (A) at (0,0,0);
\coordinate (B) at (3,0,0);
```

并在图级写 `3d view` 或显式基向量。这样 Manim 才能从其他角度重新观察。

## 17. 工具层作业模型

建议每次转换建立独立作业目录：

```text
jobs/<job-id>/
  input/
    source.tex
    input.json
  compile/
    manifest.json
    animation_plan.json
    compatibility.json
    report.json
    report.md
  preview/
    static.png
    comparison.png
  render/
    scene.py
    output.mp4
    poster.png
  package/
    asset.json
```

推荐状态机：

```text
received
  → compiled
  → compatibility_pass | blocked
  → instantiated
  → static_qa_pass
  → dynamic_config_required | dynamic_ready
  → preview_rendered
  → formal_rendered
  → packaged
```

任何阶段失败都保留前面的报告，不覆盖为笼统的“转换失败”。

### 17.1 推荐作业 JSON

```json
{
  "schema_version": 1,
  "job_id": "tikz-native-20260803-001",
  "input": {
    "source_path": "/absolute/path/source.tex",
    "source_sha256": "...",
    "entry_macro": "SquarePyramidFig",
    "picture_index": 1
  },
  "conversion": {
    "subset_version": "v0.1",
    "scene_unit_per_cm": 1.0,
    "strict_native": true
  },
  "render": {
    "quality": "preview",
    "renderer": "cairo",
    "pixel_width": 1280,
    "pixel_height": 720,
    "frame_rate": 30,
    "background": "#FFFFFF"
  },
  "outputs": {
    "manifest": "compile/manifest.json",
    "compatibility": "compile/compatibility.json",
    "video": "render/output.mp4"
  }
}
```

### 17.2 推荐缓存键

缓存键至少包含：

```text
source_sha256
+ entry_macro
+ picture_index
+ subset_version
+ compiler/renderer code revision
+ Manim version
+ TexTemplate hash
+ font file fingerprints
+ scene_unit_per_cm
+ stroke calibration
+ camera matrix/zoom/frame
+ render resolution/fps/background
+ animation config hash
```

白色描边测量缓存还必须包含标签内容、字号、node 选项、字体模板和探针版本。

不能只用源文件路径做缓存键。

## 18. 预览、正式渲染和 PPT 写入边界

工具界面应把以下操作分开：

| 操作 | 是否生成媒体 | 是否修改 PPT |
| --- | --- | --- |
| 解析/兼容性检查 | 否 | 否 |
| 原生对象实例化 | 通常否 | 否 |
| 静态预览 | 是，临时 PNG | 否 |
| 动画预览 | 是，低清 MP4 | 否 |
| 正式渲染 | 是，正式 MP4/PNG | 否 |
| 写入/替换 PPT | 使用已验收资产 | 是 |

一次预览失败不应写回 PPT，也不应覆盖上一个正式成片。正式资产建议按内容哈希命名，
PPT 层只在用户确认后切换引用。

Manim 原生对象存在于 Python 场景运行时。把视频插入 PowerPoint 后，PowerPoint 不会保留
每条线、每个点为可编辑形状。因此应同时保留：

- 原始 TeX；
- `manifest.json`；
- 动态配置和时间线；
- 场景版本；
- 正式媒体文件。

以后需要修改某个点或动画时，从这些源资产重新渲染，不要从 MP4 反向恢复对象。

## 19. 并发和进程隔离

多个作业并行渲染时，应为每个作业分配独立的：

- `media_dir`；
- 临时目录；
- scene 文件名和类名；
- report/output 目录；
- Manim partial movie 目录；
- 需要时的 TeX 缓存目录。

不要让两个作业同时写同一个 `manifest.json`、`Tex/` 或 `partial_movie_files/`。

对包含大量 `MathTex` 或白色描边探针的批量任务，最稳妥的初始策略是：

- 解析可以并行；
- 同一缓存目录内的 TeX 编译串行；
- 正式 Manim 渲染使用独立进程和独立 `media_dir`；
- 作业完成后再原子地发布最终资产索引。

## 20. 安全要求

TikZ 几何前端主要读取文本，但 `MathTex` / `Tex` 会实际运行 XeLaTeX。若工具允许上传不受信任的
TeX，必须把标签编译视为代码执行边界：

- 在隔离用户或容器中运行；
- 禁用 shell escape；
- 限制可读写目录；
- 限制 CPU、内存、文件大小和运行时间；
- 不把用户文本拼接为 shell 命令；
- 只通过参数数组调用子进程；
- 对输入路径做规范化和允许目录校验；
- 不允许 TeX 读取 PPT、密钥、浏览器配置或其他作业目录。

本机受信任教学文档可以使用较轻量的限制，但工具架构仍应保留这一边界。

## 21. 错误分类和用户提示

推荐不要只返回一段 stderr，而是使用以下错误类别：

| 类别 | 示例 | 是否重试 |
| --- | --- | --- |
| `INPUT_ERROR` | 文件不存在、入口宏不存在、宏需要参数 | 修改输入后重试 |
| `UNSUPPORTED_TIKZ` | C 级语法、`picture.unsupported` 非空 | 先标准化源码或开发新 feature |
| `INSTANTIATION_ERROR` | 字体缺失、MathTex 编译失败、对象参数非法 | 修复环境或标签源码 |
| `DYNAMIC_DOMAIN_ERROR` | 相切、零长度路径、交点消失 | 修改运动区间或拓扑策略 |
| `CAMERA_ERROR` | 对非正交矩阵使用球面轨道 | 切换矩阵插值 |
| `OCCLUSION_ERROR` | 透视遮挡、虚线池容量不足 | 修改方案或扩展实现 |
| `RENDER_ERROR` | FFmpeg、Cairo、磁盘或进程失败 | 保留日志后重试 |
| `QA_FAILED` | 渲染完成但静态/动态验收不通过 | 不发布、不写 PPT |

对象实例化错误已经包含图编号、对象 ID 和 kind。工具应把异常链和 `raw` 源语句一起展示。

## 22. 静态验收

每个新增来源或 feature 至少检查：

1. `unsupported` 为空，B 级警告已确认；
2. 所有对象都能实际实例化；
3. 命名点、线段端点、圆锥曲线尺寸和投影关系正确；
4. 填充、边线、虚线、点、标签层级正确；
5. 颜色、混色、透明度、线宽、虚线节距和箭头尺寸正确；
6. 标签 anchor、路径位置、`sloped`、字号、中文和 `\displaystyle` 正确；
7. 白色描边标签使用真实 node 测量后的位置正确；
8. 三维初始相机与 TikZ 视角一致；
9. 没有对单图整体二次缩放；
10. 生成同尺度并排图和必要的透明叠图。

“能成功输出 PNG”不等于静态验收通过。

## 23. 动态验收

动态视频还要检查：

1. 对象 ID 在动画前后保持稳定；
2. 主动参数和有效区间有明确记录；
3. 在区间起点、中点、终点抽查全部几何不变量；
4. 有向交点身份不交换；
5. 动线、点、填充、标签、垂足和角标同步；
6. 零长度线、相切、交点消失等边界已被排除或显式处理；
7. 标签不缩放、不碎裂，必要时分段切换 anchor；
8. 三维标签始终正对屏幕且 anchor 随相机更新；
9. 动态遮挡从第一帧即生效，没有第一次动画后的跳变；
10. 相机返回 TikZ 视角后，末帧与初始静态复刻一致；
11. 低清预览和正式分辨率各做一次渲染检查；
12. 检查实际 MP4 的分辨率、帧率、时长和关键帧，而不是只看命令返回码。

## 24. 命令行工作流

### 24.1 生成转换清单

```bash
cd /Users/leocyan/Documents/Code/Manim/manim_scenes

.venv-manim/bin/python scripts/convert_tikz_native.py \
  --input '/absolute/path/document.tex' \
  --entry-macro SquarePyramidFig \
  --output-dir reports/tikz_native/job-001 \
  --instantiate
```

产物：

- `manifest.json`：完整语义中间表示；
- `animation_plan.json`：确定性基线揭示；
- `report.json`：对象统计、警告、严格门禁、实例化类和包围盒；
- `report.md`：人可读报告。

返回码 `0` 表示严格门禁通过，`2` 表示未通过。

### 24.2 运行兼容性审计

不使用入口宏的普通文档：

```bash
.venv-manim/bin/python scripts/audit_tikz_native_compatibility.py \
  --input '/absolute/path/document.tex' \
  --output-dir reports/tikz_native/job-001
```

宏库应使用 Python API，确保审计和实例化复用同一个 `DocumentSpec`。

### 24.3 渲染场景

```bash
.venv-manim/bin/python -m manim -ql \
  scenes/your_generated_scene.py \
  YourPreviewScene \
  --media_dir reports/tikz_native/job-001/media
```

正式成片再改用约定的正式质量、分辨率和帧率。不要让预览和正式渲染共用可覆盖的输出路径。

## 25. 推荐的工具适配器

建议在现有工具中增加一个薄适配层，而不是复制编译器代码：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tikz_native import audit_document_compatibility, compile_document


@dataclass
class TikzNativeCompileResult:
    document: Any
    compatibility: dict[str, Any]
    picture_index: int

    @property
    def picture(self):
        return self.document.pictures[self.picture_index - 1]


def compile_tikz_native_job(
    source: Path,
    *,
    entry_macro: str | None,
    picture_index: int,
) -> TikzNativeCompileResult:
    document = compile_document(source, entry_macro=entry_macro)
    if not 1 <= picture_index <= len(document.pictures):
        raise IndexError("picture_index out of range")

    compatibility = audit_document_compatibility(document)
    if compatibility["static_status"] != "pass":
        raise RuntimeError("TikZ-native strict compatibility blocked")

    return TikzNativeCompileResult(
        document=document,
        compatibility=compatibility,
        picture_index=picture_index,
    )
```

适配层只负责：

- 固定输入快照；
- 调用公共 API；
- 做兼容性门禁；
- 写出中间清单；
- 选择二维/三维场景模板；
- 传入动态配置；
- 收集渲染产物和日志。

它不应重新实现 TikZ 解析、标签 anchor 或遮挡算法。

## 26. 新 feature 的扩展流程

遇到未支持语法时，不应先问“怎样把它画出来”，而应依次完成：

1. 建立最小 TikZ 样例；
2. 定义它的几何和动画语义；
3. 在编译器中保存稳定对象或关系；
4. 使用明确的原生 Manim 类实现；
5. 加入对象 ID 和 manifest 测试；
6. 生成 TikZ—Manim 静态对比；
7. 检查多个参数状态的动态不变量；
8. 覆盖反向、退化、拓扑变化和失败行为；
9. 更新 `subset_v0_1.json` 的 feature 等级，或发布新子集版本；
10. 更新本文和阶段基线。

若只能保证静态近似，先列为 B；不能可靠复刻则保持 C。不能为了提高“成功率”增加 SVG 兜底。

## 27. 发布门禁清单

工具把资产写入 PPT 前，建议要求以下项目全部通过：

```text
[ ] 源文件哈希与作业记录一致
[ ] 入口宏和图编号明确
[ ] A/B/C 审计无 C 级发现
[ ] picture.unsupported 为空
[ ] --instantiate 通过
[ ] 未使用 SVG、位图、通用路径兜底
[ ] 未对整图做二次 scale/scale_to_fit
[ ] 字体模板和字体文件已记录
[ ] 11pt、\displaystyle、显式字号命令已检查
[ ] 线宽和白色描边标定未漂移
[ ] 静态 TeX—Manim 对比通过
[ ] 主动参数、有效区间和时间线已保存
[ ] 动态不变量和标签碰撞检查通过
[ ] 三维标签在相机运动中稳定
[ ] 动态遮挡从第一帧起稳定
[ ] 正式 MP4/PNG 的分辨率、帧率、时长已核实
[ ] 正式资产未被预览覆盖
[ ] PPT 写入使用已验收资产且可回滚
```

## 28. 当前最重要的已知限制

在工具 UI 中应明确展示，而不是藏在日志里：

1. 当前支持的是标准化 TikZ 子集，不是完整 TikZ。
2. 静态 TikZ 通常没有唯一主动对象和教学时间线。
3. A 级关系具备动态基础，但仍要显式配置运动区间。
4. 复杂 node、曲线路径、clip、pattern、matrix 等仍可能阻断。
5. 动态交点不能无策略穿越相切或消失事件。
6. 三维一般工程投影不一定能使用正交球面相机轨道。
7. 三维遮挡目前限于平行投影和有限凸面片。
8. 三维标签 anchor 会动态更新，但一般碰撞规避尚未自动化。
9. 白色描边标签依赖字体、XeLaTeX 和高分辨率测量环境。
10. PPT 中的 MP4 不保留 Manim 对象可编辑性，源清单和动态配置必须长期保存。

## 29. 建议的第一版接入范围

为了让工具尽快得到稳定结果，第一版建议只开放：

- 选择 TeX 文件；
- 选择零参数图形宏和图编号；
- 运行严格解析、A/B/C 审计和 `--instantiate`；
- 显示对象清单、稳定 ID、警告和阻断项；
- 生成无二次缩放的静态预览；
- 生成语义基线揭示动画；
- 对已有关系模板选择主动参数和运动区间；
- 对三维图恢复 TikZ 视角并启用动态标签/遮挡；
- 分开生成预览 MP4、正式 MP4 和 PPT 写入动作。

暂时不要在第一版里承诺：

- 任意 TikZ 全自动转换；
- 从最终静态图自动推断唯一教学动画；
- 任意三维实体的全自动可见面排序；
- 自动修复所有标签碰撞；
- 把 Manim 子对象直接变为 PowerPoint 可编辑形状。

这样接入后，工具得到的不是一个“尽量画出来”的黑盒，而是一条可以定位、回归、扩展并用于
正式教学视频生产的原生转换管线。

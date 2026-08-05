# TikZ 到原生 Manim 的受控转换规范

> 迁入说明：本文形成于独立 Manim 研究目录阶段。当前 Provider 根目录为
> `tex-to-mathcapture-ppt/tools/tikz-native-provider/`；文中的 `manim_scenes/tikz_native`、
> `manim_scenes/scripts` 和 `manim_scenes/tests` 分别对应当前的 `tikz_native/`、
> `scripts/` 和 `tests/`，场景、报告和媒体路径仍是外部历史证据。

当前阶段性能力已冻结为 `v0.1`。版本范围、语义指纹、渲染证据与复现命令见
[`tikz-native-v0.1-baseline.md`](tikz-native-v0.1-baseline.md)。后续扩展不得覆盖
该基线；有意改变对象结构或视觉证据时应建立新版本。

A/B/C 兼容性等级、推荐源码写法与新增 feature 的完成标准见
[`tikz-native-subset-v0.1.md`](tikz-native-subset-v0.1.md)。

三维正投影试验、相机矩阵推导、标签策略和当前限制见
[`tikz-native-3d-prototype.md`](tikz-native-3d-prototype.md)。该原型继续使用本规范的
“原生对象 + 严格报告”原则，不把三维图先压成二维路径。

若要把转换过程接入现有 TeX/PPT 工具，请使用包含数据契约、作业状态、缓存、
并发、错误处理与发布门禁的
[`tikz-native-tool-integration-guide.md`](tikz-native-tool-integration-guide.md)。

## 1. 目标与边界

这套程序面向“同一张数学图既要保留 TikZ 静态排版，又要能在 Manim 中逐项动画”的场景。

这里的“原生 Manim”指直接创建有明确语义的 `Line`、`Polygon`、`Ellipse`、`Circle`、`Dot`、`Arc`、`RightAngle`、`MathTex`、`Tex` 等对象。它们在 Manim 内部当然仍继承自 `Mobject/VMobject`，但转换器不会把整张 TikZ 先变成 SVG，也不会用一个无语义的通用 `VMobject` 重新描点。

程序采用“受控子集 + 失败即报告”的策略，而不是假装实现完整 TikZ：

```text
TeX 文档
  -> 提取 tikzpicture
  -> 展开受支持的宏、循环和数值表达式
  -> 建立坐标表和语义对象清单
  -> 创建原生 Manim 对象
  -> 输出 manifest.json、report.json、report.md
  -> 用稳定对象 ID 编写动画
```

遇到新语法时，程序把它写入 `unsupported` 并让严格门禁失败；不会静默退回 SVG、位图或通用路径。

## 2. 当前程序

- 编译器：`manim_scenes/tikz_native/compiler.py`
- 原生对象工厂：`manim_scenes/tikz_native/manim_renderer.py`
- 三维正投影换算：`manim_scenes/tikz_native/projection_3d.py`
- 三维原生对象工厂：`manim_scenes/tikz_native/manim_renderer_3d.py`
- 原生动画调度：`manim_scenes/tikz_native/animation.py`
- 动态几何绑定：`manim_scenes/tikz_native/dynamic_geometry.py`
- 统一命令：`manim_scenes/scripts/convert_tikz_native.py`
- 本题场景：`manim_scenes/scenes/national_2026_18_native.py`
- 三维样例场景：`manim_scenes/scenes/tikz_native_3d_demo.py`
- 回归测试：`manim_scenes/tests/test_tikz_native_compiler.py`
- 三维回归测试：`manim_scenes/tests/test_tikz_native_3d.py`

在 `manim_scenes` 目录运行：

```bash
.venv-manim/bin/python scripts/convert_tikz_native.py \
  --input '/Users/leocyan/Documents/讲评课/2026年全国一卷第18题.tex' \
  --output-dir reports/tikz_native/2026_national_1_18 \
  --instantiate
```

`--instantiate` 不只检查语法，还会实际构造每一个 Manim 对象，能抓到字体、TeX 编译和对象参数层面的错误。

渲染静态总览、全图自动揭示与代表图构造动画：

```bash
.venv-manim/bin/python -m manim -ql --format=png \
  scenes/national_2026_18_native.py \
  National2026TikzNativeAllGallery

.venv-manim/bin/python -m manim -ql \
  scenes/national_2026_18_native.py \
  National2026TikzNativeAllGalleryReveal

.venv-manim/bin/python -m manim -ql \
  scenes/national_2026_18_native.py \
  National2026TikzNativeFigure04Construction

.venv-manim/bin/python -m manim -ql \
  scenes/national_2026_18_native.py \
  National2026TikzNativeFigure01DrivenMotion

.venv-manim/bin/python -m manim -ql \
  scenes/national_2026_18_native.py \
  National2026TikzNativeFigure04DrivenMotion
```

## 3. 当前映射

| TikZ 语义 | Manim 对象 | 动画粒度 |
| --- | --- | --- |
| 两点间实线 | `Line` | 每条线独立 |
| 显式虚线 | 多个 `Line` 组成的 `VGroup` | 整条虚线独立，虚线段仍可访问 |
| `Stealth` 箭头 | `Line` + `StealthTip` | 每条箭头独立 |
| 闭合 `\fill` | `Polygon`，默认无描边 | 每块填充独立 |
| 椭圆、圆 | `Ellipse`、`Circle` | 每个轮廓独立 |
| `\fill (...) circle (...pt)` | `Dot` | 每个点独立 |
| 数学标签 | `MathTex` | 每个标签独立 |
| 中英文混合标签 | `Tex` | 每个标签独立 |
| `pic angle` | `Arc` + 独立角标签 | 弧与标签分别独立 |
| `pic right angle` | `RightAngle` | 每个直角记号独立 |

对象保存在 `NativeFigure.objects` 中，键是稳定语义 ID，例如：

```python
figure.objects["line.P.Q"]
figure.objects["dot.P"]
figure.objects["label.P.P"]
figure.objects["right_angle.R.H.P"]
```

只要坐标命名和对象结构稳定，就可以直接对这些对象调用 `Create`、`FadeIn`、`Transform`、`animate` 等动画。

## 4. 最有利于转换的 TikZ 写法

### 4.1 所有教学语义点都命名

推荐：

```tex
\coordinate (P) at (1,1.5);
\coordinate (Q) at (-1.8,-0.6);
\draw (P) -- (Q);
```

不推荐把关键点直接写成匿名数字坐标：

```tex
\draw (1,1.5) -- (-1.8,-0.6);
```

匿名坐标仍可转换，但只能得到 `line`、`line.2` 之类按出现顺序生成的 ID；源码一旦插入新线，后续 ID 可能漂移，不利于动画映射。

### 4.2 需要独立动画的线，尽量一条 `\draw` 写一个语义对象

```tex
\draw (P) -- (Q);
\draw (Q) -- (R);
\draw (R) -- (P);
```

当前程序也会把 `(P)--(Q)--(R)--cycle` 拆成独立线段，但分开写更容易给每条线安排叙事顺序，也更容易定位报错。

### 4.3 填色和轮廓分开写

```tex
\fill[fill=lectureteal!14] (P) -- (Q) -- (R) -- cycle;
\draw[draw=lectureline] (P) -- (Q) -- (R) -- cycle;
```

这样会得到一块无描边 `Polygon` 和三条独立 `Line`。不要依赖 `\filldraw`，也不要只靠外层 `scope` 同时控制多种语义；否则静态图虽然简短，动画时填充和边线很难分开。

### 4.4 明确区分“端点标签”和“线段标签”

端点标签放在线段之后：

```tex
\draw (A) -- (B) node[above] {$B$};
```

线段标签放在连接符之后、终点之前，并建议显式写 `pos`：

```tex
\draw (A) -- node[pos=0.5,below] {$\Delta x$} (B);
```

`(A) -- node {...} (B)` 的 TikZ 默认位置是线段中点；`(A) -- (B) node {...}` 的默认位置是当前端点。两种写法只差 node 的位置，却是完全不同的语义。

### 4.5 只有需要沿线旋转时才写 `sloped`

```tex
\draw (R) -- node[pos=0.58,above,sloped] {$h_Q$} (H);
```

不写 `sloped` 时，标签保持页面水平，即使所在的线段是竖直线。转换器已按这个规则区分；不要把所有线段标签统一加 `sloped`。

### 4.6 样式尽量显式

推荐在图或命令上明确写出：

```tex
\begin{tikzpicture}[
  scale=1.1,
  line width=0.85pt,
  line cap=round,
  line join=round
]
\draw[draw=lectureblue!82!black] (A) -- (B);
\draw[
  draw=lecturegold!85!black,
  dash pattern=on 1.6pt off 1.7pt
] (A) -- (C);
```

`dashed`、`densely dashed` 在不同 TikZ/Manim 环境中的节距并不完全相同。若虚线节奏需要视觉复刻，使用显式 `dash pattern`。

### 4.7 颜色混合与透明度分别写

```tex
\fill[fill=lecturegold!30,opacity=0.72] ...;
```

`lecturegold!30` 是与白色按比例混色，不等于 30% 透明度；`opacity=0.72` 才控制透明度。两者不要混为一项。

当前颜色定义建议使用：

```tex
\definecolor{lecturegold}{HTML}{B98A24}
```

转换器目前读取 HTML 六位颜色和 xcolor 的 `A!p!B` 混色。

### 4.8 数值宏保持纯数值、无副作用

当前可稳定处理：

```tex
\pgfmathsetmacro{\r}{0.72}
\coordinate (U) at (1.45,{1.45*\r});
```

以及简单的无参数 `\gdef` 和简单点表 `\foreach`。避免在一张图中混入条件判断、随机数、复杂 key handler、文件读写或依赖当前 TeX 盒子尺寸的计算。

### 4.9 calc 坐标只使用清楚的几何构造

当前稳定支持：

```tex
($(A)!0.4!(B)$)       % 线性插值
($(A)!(P)!(B)$)       % P 到 AB 的垂足
($(A)+(0.2,-0.1)$)    % 坐标平移
```

如果一个点在后续会被标记、连线或动画，最好先把 calc 结果命名：

```tex
\coordinate (H) at ($(A)!(P)!(B)$);
```

### 4.10 优先用命名路径和原生交点表达相交关系

当点来自直线与圆锥曲线求交时，优先保留 TikZ 的构造语义：

```tex
\path[name path=C]
  (O) ellipse [x radius=2,y radius={sqrt(3)}];
\path[name path=l]
  (Lstart) -- (Lend);
\path[name intersections={
  of=C and l,
  sort by=l,
  by={Q,P}
}];
```

`sort by=l` 按 `Lstart -> Lend` 的有向路径排列交点。转换清单同时保存两条命名路径、路径方向、`Q/P` 的排序序号和初始参数；静态渲染使用求得的坐标，动态场景可以按同一方向逐帧重新求交。反转直线端点会反转交点身份，因此动画更新时必须保持路径方向连续。

当前支持直线—椭圆（圆视为半轴相等的椭圆）和直线—直线求交，并要求 `sort by` 指向其中的直线路径。复杂曲线多交点、相切合并和按闭合曲线路径排序仍进入 `unsupported`。

### 4.11 角和直角必须使用命名点，并显式给半径

```tex
\pic[
  draw=lectureblue,
  line width=0.62pt,
  angle radius=14pt,
  angle eccentricity=1.45,
  "$\varphi$"
] {angle=Xdir--Q--U};

\pic[
  draw=lecturered,
  angle radius=8pt
] {right angle=U--Q--D};
```

三点顺序决定弧的起点、顶点和终点，也决定直角记号所在象限。修改顺序后必须做视觉回归，不能只检查角度数值。

### 4.12 图形几何与 TeX 版面修正分开

`baseline=...`、`trim right=...` 是 TeX 排版层选项，不是几何对象。转换器会记录但不把它们应用到 Manim 坐标。若图需要在 PPT 中对齐，应在幻灯片布局层设置统一锚点，不要把 `trim` 当作几何修正。

### 4.13 字体由共享模板统一管理

当前共享模板显式使用 `11pt` 文档类。TikZ 的 `scale` 只缩放坐标几何，不缩放标签字形；默认标签继承 11pt 正文字号。标准 11pt 类中的 `\normalsize` 为 10.95pt，`\small` 为 10pt，转换器把这些命令原样交给 XeLaTeX，不再用人工比例估算。

所有标签中的数学片段默认显式加入 `\displaystyle`，包括纯公式、`\small` 公式以及“公式＋中文”的混合标签。例如 `{$x$正方向}` 会渲染为 `{$\displaystyle x$正方向}`，`{\small $\frac ba$}` 会同时保留 `\small` 和 `\displaystyle`。若源码随后明确写出 `\textstyle` 等命令，LaTeX 的后置命令仍可覆盖默认显示样式。

纵览图也不得对包含标签的单图 `VGroup` 调用 `scale` 或 `scale_to_fit_*`。空间不足时只能统一扩大相机视野、提高输出分辨率或拆分纵览，不能逐图压缩。这样不同 `tikzpicture scale` 下的同一普通标签保持完全相同的尺寸；只有源码明确写了 `\small` 等命令时才改变字号。

标签源码只写内容和必要的字号命令，不要在单张 TikZ 中硬编码某台机器上的西文字体名。当前环境中，显式设置 `Latin Modern Roman` 曾被新版 `fontspec` 判为找不到字体；交给 XeLaTeX 默认数学字体后恢复稳定。中文字体统一由 Manim 的 `TexTemplate` 设置。

### 4.14 三维图必须显式声明视角

推荐直接使用 TikZ 原生视角：

```tex
\begin{tikzpicture}[3d view={40.4}{23.8},scale=1.15]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (3,0,0);
  \coordinate (D) at (0,2,0);
  \coordinate (A1) at (0,0,2.4);
  \draw (A) -- (B);
  \draw (A) -- (D);
  \draw (A) -- (A1);
\end{tikzpicture}
```

转换器按 TikZ `perspective` 库的正投影定义计算初始相机矩阵，同时保留 `(x,y,z)`
世界坐标，因此之后可以直接转动 Manim 相机。也支持显式 `x={(...)},y={(...)},z={(...)}`
基向量；两种写法都比在每个点上手算二维投影更有利于动态转换。

三维标签仍继承 11pt 正文字号和默认 `\displaystyle`，不随 `scale` 变化。标签保持
正对屏幕，并在相机移动时按当前投影重新计算 `above/below/left/right` 偏移，避免把
TikZ 初始视角下的二维偏移刚性带到其他视角。

## 5. 当前自动转换级别

### 可直接自动转换

- 命名坐标和数值坐标；
- 直线、折线、闭合多边形填充；
- 椭圆、圆、点、Stealth 箭头；
- 简单标签、路径标签、数学公式、中英文混合标签；
- `angle`、`right angle`；
- HTML 颜色、xcolor 混色、透明度、显式虚线节距；
- 当前文档使用的简单宏、循环、数值表达式和 calc 坐标；
- 原生 `name path` 直线/椭圆/圆，以及按有向直线 `sort by` 的交点；
- 交点、插值、垂足和平移坐标的构造依赖清单。
- 显式 `(x,y,z)` 坐标；
- `3d view={azimuth}{elevation}` 正投影视角；
- 显式 `x/y/z` 二维基向量；
- 原生空间线段、平面多边形、`Dot3D` 与 fixed-orientation 标签。

### 需要新增规则后才能自动转换

- Bézier 曲线、一般 `arc`、plot/smooth 路径；
- `clip`、pattern、shade、gradient、decoration；
- 复杂 node 形状、边框、`text width`、自动换行、matrix；
- 嵌套 scope 的旋转、缩放、平移和样式继承；
- 自定义箭头库、复杂 path shortening；
- 依赖 TeX 盒子测量的坐标；
- 任意参数宏、条件分支和复杂 pgfkeys。
- 三维空间曲线、任意平面内圆弧和三维 `pic` 角标；
- `tdplot_rotated_coords`、一般 `canvas is plane` 变换，以及尚未登记语义的
  自定义遮挡宏；
- 透明面片在相机运动中的自动前后重排，以及由静态源码自动推断运动依赖。

这些情况当前应进入 `unsupported`，由人决定补充哪一种“有语义的原生对象”映射，而不是退化成一整块 SVG。

## 6. 自动动画与教学动画

### 6.1 自动揭示

统一命令除 `manifest.json` 与转换报告外，还会输出 `animation_plan.json`。程序把每幅图的稳定对象 ID 自动归入以下播放层：

```text
fills
  -> coordinate_frame
  -> solid_geometry
  -> auxiliary_geometry
  -> markers
  -> points
  -> labels
```

这是一套确定性的“基线揭示”，适合批量检查所有转换对象是否能独立播放。它不会把整图淡入：实线、箭头、圆和角记号用 `Create`，点用 `GrowFromCenter`，标签用 `Write` 或逐标签 `FadeIn`；显式虚线仍是一个可寻址对象，但内部小线段会沿路径方向依次 `Create`。

`play_semantic_reveal(...)` 可以直接把这套规则应用到任意一幅或多幅 `NativeFigure`。如果要表达具体教学过程，则用 `play_named_reveal(...)` 按稳定 ID 编排。例如第 4 图依次构造坐标轴、椭圆与直线 $l$、点 $P,Q,R$、三角形、边长标签、垂线、距离标签和直角记号。

这里必须保留一条边界：静态 TikZ 只描述最终图形，通常没有记录“先给点还是先连线”“辅助线在哪一句讲解后出现”等叙事信息。因此程序能自动保证对象可动画、层次可用，却不能可靠猜出唯一教学顺序。需要精确讲解时，稳定命名点和稳定对象 ID 仍是最可靠的接口。

### 6.2 依赖驱动的几何运动

“对象逐项出现”和“图形真的运动”是两层不同的问题。后一种不能只对最终 `VGroup` 做平移或旋转，而必须每帧重算几何依赖。例如第 1 图采用一个角度参数驱动：

```text
直线 l 绕焦点 F 转动
  -> 重新计算 l 与椭圆的交点 P、Q
  -> 令 R=-P
  -> 同步更新点、标签、三角形边、填充区域和虚线 QO
```

`NativeMotionBinder` 把这些依赖函数绑定到转换后原有的 `Line`、`Dot`、`Polygon` 和标签上。实线直接更新端点；填充每帧重建为原生 `Polygon`；虚线按新的长度重新生成原生小线段，避免把旧虚线整体拉伸；标签只移动，不缩放、不重新排版。因此 TikZ `scale`、11pt 字号和默认 `\displaystyle` 规则在运动中都不改变。

第 1 图现已改为 TikZ 原生 `name path + name intersections + sort by`：manifest 同时保存椭圆、直线的有向路径、$Q/P$ 的交点顺序以及 $R=$(O)!-1!(P)$ 的插值依赖。动态场景从这些路径读取椭圆半轴、初始直线方向和显示长度，不再重复写死这组几何参数。静态 TikZ 仍然不能唯一说明“让直线主动还是让点主动”，因此主动对象和运动区间继续由动画场景显式选择。

动态标签还有一个静态复刻中不存在的新问题：固定 anchor 在某些运动区间可能与坐标轴或其他标签碰撞。本示例将直线角度限制在 $22^{\circ}$ 到 $54^{\circ}$；更大的运动范围应增加分段 anchor 或碰撞规避，而不应通过缩放字体解决。

第 4 图进一步复用了同一驱动器，并加入投影依赖：每帧把 $R$ 正交投影到 $PQ$ 得到 $H$，同步更新 $RH$、直角记号、路径标签 $d$ 和 $\lvert PQ\rvert$。路径标签保留初始的切向/法向偏移并随线段旋转；直角记号根据 $R,H,P$ 三点重新创建为原生 `RightAngle`。这证明动态绑定层已经覆盖“动点—动线—垂足—路径标签—角标”链路，而不仅是第一幅图的点线联动。

## 7. 本次实际发现并修复的问题

1. `-- node {...} (B)` 被误判为起点标签，造成第 13、14 图的 `A` 与 `\Delta x` 重叠。已按 TikZ 语义修为默认 `pos=0.5`，并加入测试。
2. `\fill` 曾错误继承默认描边。静态图被后续 `\draw` 覆盖后看不出问题，但独立播放填充时会提前出现轮廓。现已固定为默认无描边。
3. 路径标签曾统一按线段方向旋转。现已改为只有 `sloped` 才旋转，竖直线上的普通标签保持水平。
4. `lecturegold!30` 曾容易被误当作透明度。现按 xcolor 混色计算，透明度单独保留。
5. Manim 标签模板显式设置 `Latin Modern Roman` 时，当前 XeLaTeX/fontspec 环境报字体不存在。已撤去机器相关的西文字体声明。
6. 不同公式直接 `Transform` 会在中间帧产生字形碎裂。当前动画规则是：几何对象直接变换；结构变化较大的标签先淡出、再淡入；只有结构足够相似时才考虑 `TransformMatchingTex`。
7. `baseline`、`trim right` 在原文中存在，但只影响 TeX 页面布局。报告会保留警告，不将其伪装成 Manim 几何。
8. 旧纵览图曾为适应固定网格而逐图调用 `scale_to_fit_*`，导致标签二次缩放。现已删除所有逐图缩放，改为统一扩大相机视野；相同标签在 `scale=0.8` 与 `scale=1.5` 下的宽高回归为完全一致。
9. 小字号和中英混合标签过去通过 `Tex` 的行内数学路径渲染，分式可能退回 `\textstyle`。现对每个标签数学片段显式注入 `\displaystyle`，并设门禁防止遗漏或重复注入。
10. 普通 `FadeIn(VGroup(...))` 虽然能让画面出现，却不能证明对象真的可单独控制。现改为按稳定 ID 生成每个对象自己的动画；虚线也按内部原生线段依次绘制。
11. 自动揭示并不等于几何运动。现新增依赖驱动层，第 1 图的 $P,Q,R$、相关线段、两块填充、虚线及标签均由同一个直线角度参数实时更新。
12. 第 4 图进一步加入动态垂足、斜线路径标签和直角记号；这些对象需要按几何约束重新计算，不能随父组做近似旋转或拉伸。
13. 原先手算二次方程得到的第一幅图交点已改为 TikZ 原生命名路径求交；转换器新增有向直线排序，反转直线端点的回归测试确认交点顺序随路径方向一致反转。
14. 第一幅动态场景原先重复写死椭圆半轴、焦点、线段长度和初始角度；现由命名路径与交点关系构造驱动器，源码和动画几何只保留一份事实来源。

## 8. 每次扩展转换器时必须回归的项目

1. 解析门禁：所有 TikZ 语句要么转换，要么进入 `unsupported`。
2. 原生门禁：不得调用 `VMobject(...)`、`SVGMobject(...)`、`ImageMobject(...)` 或通用描点兜底。
3. 几何：命名点坐标、投影垂直性、插值比例、外接曲线尺寸。
4. 样式：颜色、混色、透明度、线宽、虚线节距、箭头尺寸。
5. 标签：端点/中点语义、anchor、offset、`sloped`、中英文和公式字体。
6. 标记：角弧方向、角标签半径、直角象限。
7. 层级：填充、结构线、虚线、点、标签的遮挡顺序。
8. 动画：对象 ID 稳定；几何 Transform 无跳变；标签过渡无字形碎裂。
9. 输出：静态 PNG、完整 MP4、分辨率、帧率和时长均可复核。
10. 纵览：不得逐图缩放；相同普通标签跨图尺寸一致，显式字号命令除外。
11. 数学样式：每个标签公式必须含显式 `\displaystyle`；纯公式、小字号和混合文本都要覆盖。
12. 动态约束：抽查多个参数值，确认动点仍满足原曲线、共线、对称、垂直等几何约束；返回初始参数后最终帧应恢复静态复刻状态。
13. 动态标签：检查整个运动区间的遮挡与碰撞；必要时切换 anchor，不得用缩小字体掩盖问题。
14. 交点身份：检查 `sort by` 路径方向、交点参数单调性和反向路径行为；不得依赖求交函数未经约定的返回顺序。
15. 三维视角：静态初始帧必须与 TikZ 的 `3d view` 或显式基向量一致；不得用透视相机近似正投影。
16. 三维标签：相机全程转动时检查 fixed-orientation、屏幕方向 anchor、字号和 `\displaystyle` 均保持稳定。
17. 三维遮挡：检查面、边、点的深度关系；若转换器尚未实现动态排序，必须在报告中明确标记，不能把偶然正确的初始层级当作已支持。

目前 manifest 中对象的 `source_line` 只能稳定指到所属 `tikzpicture` 的起始行；宏展开后的对象还不能精确反查到原始语句行号。定位新错误时应同时使用图编号、对象 ID 和 `raw` 源语句。

示例源码见 `manim_scenes/tikz_native/examples/native_friendly_figure.tex`。

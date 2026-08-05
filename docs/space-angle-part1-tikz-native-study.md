# 《空间角专题复习（一）》TikZ → 原生 Manim 研究记录

> 迁入说明：本文是迁入 Git 前的专题研究记录，原 TeX、报告、图片和视频仍位于外部
> 教学/研究目录；Provider 源码现由
> `tex-to-mathcapture-ppt/tools/tikz-native-provider/` 统一管理。

## 1. 当前结论

本轮没有修改原始 TeX，而是把它当作宏库输入，按 `\previewFig` 中的具体图形宏逐个展开、编译和实例化。

- 源文件共有 20 个 `tikzpicture` 源码块。
- 独立预览列出 21 个入口；宏展开后实际对应 25 个 picture 实例。
- 21 个入口现已全部通过严格编译和原生实例化：20 个无警告，1 个仅含非致命警告。
- 25 幅图共得到 527 个独立原生对象，实例化失败为 0，`unsupported` 为 0。
- 全部 25 幅图均已按同一物理尺度生成 TikZ/Manim 并排图和红青叠加图；没有对 Manim 图组再做 fit 或缩放。

这里的“通过”不是“生成一张 SVG 后整体导入”，而是点、线、面和标签都具有独立对象身份。遇到未支持语法时，程序写入 `unsupported` 并停止严格门禁，不静默跳过，也不退回 SVG、位图或通用 `VMobject` 路径。

完整逐入口结果见：

- [`support-matrix.md`](../reports/space_angle_part1_native/batch_audit/support-matrix.md)
- [`audit.json`](../reports/space_angle_part1_native/batch_audit/audit.json)

## 2. 已经打通的转换链路

### 2.1 以“具体图形宏”为入口

原文件不是 20 段可以直接截取的裸 TikZ，而是带有 Base 宏、参数、比较图和预览包装器的双模式宏库。转换器现在可以用：

```bash
.venv-manim/bin/python scripts/convert_tikz_native.py \
  --input '/Users/leocyan/Documents/Code/Codex/output/空间角专题复习（一）_tikz.tex' \
  --entry-macro SquarePyramidFig \
  --output-dir reports/space_angle_part1_native/square_pyramid \
  --instantiate
```

宏前端会递归展开 `\newcommand`、`\renewcommand`、`\providecommand` 的具体包装器，代入 `#1`、`#2` 等参数，再把得到的 `tikzpicture` 交给语义编译器。图前用 `\pgfmathsetmacro` 计算的视角参数也会先求值，因此比较图中的参数化视角不再直接卡在 `scale=#1` 或未定义视角宏上。

### 2.2 当前可保留的三维语义

当前已覆盖：

- 数值 `scale`，只作用于几何坐标，不二次缩放字体；
- `space view={(x_x,x_y),(y_x,y_y),(z_x,z_y)}`；
- 显式 `x/y/z={(u,v)}` 投影基向量；
- PGF 角度制的 `sin`、`cos`、`tan`；
- `\defPoint`、`\defPointShift`、`\pointOnSpaceLine`；
- 命名样式及 `xcolor` 混色；
- 点、折线、多个互不连续的子路径、闭合面、`filldraw`；
- `node[pt]` 和 `\fill ... circle ... node...`；
- `\node[options] at (P)` 与 `\node at (P) [options]` 两种原生顺序；
- 常规 label anchor、pt 偏移、字号、背景和 `\mathWhiteOutline`；
- 所有数学标签自动使用 `\displaystyle`，默认正文基准为 11pt；
- 普通标签作为朝向屏幕的独立对象，相机变化时重新计算屏幕偏移。
- 三类线—面遮挡宏会保留线段、遮挡面和实虚样式的关系，并在初始 TikZ 视角下解析为原生线段；
- `\DrawSpacePlaneInteraction` 会生成独立平面、边界和遮挡边段；
- 三维 `angle`、`right angle` 和 `anglemark` 会生成原生圆弧/折线/扇形与独立标签；
- `canvas is yz/xy plane` 的节点坐标会提升回三维坐标；`transform shape` 会同时接受 picture scale 和局部平面投影产生的旋转、缩放、斜切；
- `\csname pt3d@...` 点分量可从编译器坐标表读取，公垂线两图可以完整实例化；
- `pptstep` 静态编译时不再破坏相邻命令，但阶段元数据尚未进入动画计划。

转换器还增加了一个重要的严格性保护：若一条未支持的自定义命令后面紧跟 `\draw` 或 `\node`，不会再丢掉前缀后假装转换成功。这样自动遮挡宏缺失时，支持矩阵会诚实地显示为未通过。

## 3. 全部入口与 25 幅实际图

21 个 `previewFig` 入口均已通过。三个比较入口会展开为多幅实际图：`SkewLinesFig` 为 3 幅，`LinePlaneAngleCompareFig` 与 `DihedralAngleCompareFig` 各为 2 幅，因此最终回归总数为 25 幅。

唯一非致命警告来自 `TriangularPrismLinePlaneAngleProblemFig` 中的 `\fill(Q)`：它没有圆半径或 node 尺寸，TikZ 本身也不会形成可见点，因此转换器将其忽略。这个警告不涉及丢失可见几何。

全量纵览和逐图证据：

- [`all_25_pair_contact_sheet.png`](../reports/space_angle_part1_native/full_catalog/all_25_pair_contact_sheet.png)
- [`manifest.json`](../reports/space_angle_part1_native/full_catalog/manifest.json)

## 4. 同物理尺度静态对照

全部场景使用 `scene_unit_per_cm=1.0`、相机 `zoom=1.0`，没有对图形组或单幅图再做 fit/scale。低清 Manim 输出中 1 个场景单位为 60 像素；原 TikZ PDF 以 152.4 dpi 栅格化，也得到 60 像素/厘米。对照只裁掉外部白边，没有调整任一侧大小。

25 幅图的裁切框中位差为：Manim 相对 TikZ 宽 `-1 px`、高 `0 px`；最大绝对差为宽 `3 px`、高 `2 px`。这证明 picture scale、投影基、11pt 标签基准和总体锚点没有发生系统性缩放漂移。

线宽还需要单独按最终栅格墨迹校准。旧换算 `2.15 Manim stroke / TeX pt` 在 Cairo 的 60 px/cm 低清输出中受到抗锯齿影响，视觉上明显比 TikZ 细。以 `ObliqueLineProjectionFig` 中不与其他对象重叠的斜线为截面，TikZ 有效墨迹宽度为 `2.254 px`；将原生几何线宽倍率统一提高到 `3.8` 后，Manim 为 `2.228 px`，差约 `1.1%`。这条倍率同时用于二维和三维的实线、虚线、轮廓、圆弧、直角标及 node 边框，并保留源码中各个 `line width` 之间的比例；文字的白色 halo 仍使用原先独立校准的 `2.15`，不会因几何线条加粗而变厚。

标签之前显得偏细、偏小的主要原因已经定位：原讲义预览使用 `unicode-math`、`Latin Modern Math` 的 `FakeBold=2`，中文使用宋体；旧模板使用普通 MathTex 与苹方。当前 Manim TeX 模板已与源样式的字体配置对齐，并继续让普通标签不随 picture scale 改变；只有源码明确写了 `transform shape` 的 node 才随 picture scale 缩放。

本轮还对 25 幅图中的 162 个非空标签做了 TikZ 物理 node 中心与 Manim 投影中心的逐项比较。全部标签平均误差为 `0.266 pt`、中位误差为 `0.316 pt`、最大误差为 `0.786 pt`；其中 138 个普通标签平均误差为 `0.311 pt`、最大误差为 `0.786 pt`。24 个 `\mathWhiteOutline` 实例在启用专用 node 探针后，平均误差为 `0.010 pt`、中位误差为 `0.003 pt`、最大误差为 `0.030 pt`，已经不再是全量标签中的异常项。逐标签数据见 [`label_anchor_comparison.json`](../reports/space_angle_part1_native/label_anchor_comparison.json)。

### 4.1 `\mathWhiteOutline` 的专用位置与样式模型

目标文档源码有 20 处 `\mathWhiteOutline`；基础宏展开后，共形成 10 个最终图形入口中的 24 个标签实例，覆盖普通点标签、路径中点标签和角标。它不能完全按普通标签处理，也不宜把整段嵌套 TikZ 当作一个外部 SVG 导入。

本轮先用 `inner sep=0pt` 验证了 halo 本身近似对称，随后用同一 `\beta` 做了带实框的对照：不写方向键时黑字位于 node 中心；写成 `above left=-1pt` 后，外层 node 不仅变大，黑字还会在框内向左上移动。根因是 `above/right/below/left` 在外层 node 排版前安装的画布平移，会进入 `\mathWhiteOutline` 内部的嵌套 `tikzpicture`。普通文本没有这一层嵌套，所以只有白描边标签表现出这种特殊位置关系。

因此，普通 TeX `\hbox` 或“字形宽高 + 固定 padding”都不足以还原它。当前专用模型是：

1. 核心公式仍由独立的原生 `MathTex`/`Tex` 生成，并保持 11pt 与 `\displaystyle`；
2. 可见白边由 Manim background stroke 生成，描边参数继续来自宏的可选 pt 值；
3. 前景字色固定为黑色，因为源宏内部明确写了 `text=black`，不能继承蓝色投影线等父路径颜色；
4. 将源码中的方向键、anchor 与 inner/outer sep 按原顺序送入一个最小 XeLaTeX node 探针；
5. 探针以 600 dpi 测出真实 node 外框，以及黑色字形中心相对外框中心的偏移；
6. 测量结果按模板、公式和选项哈希缓存，Manim 只保留一个不可见占位框与一个可见公式对象。

探针只用于物理测量，不进入最终画面；最终标签仍可单独移动、淡入和变换，没有把整幅 TikZ 图或整个标签 SVG 作为不可拆分回退。专项对照见 [`white_outline_regression/all_25_pair_contact_sheet.png`](../reports/space_angle_part1_native/white_outline_regression/all_25_pair_contact_sheet.png)，原 TeX node 探针见 [`white_outline_probe`](../reports/space_angle_part1_native/white_outline_probe)。

## 5. 全量静态复刻后仍需回归的部分

### 5.1 动态遮挡不是普通虚线解析

文件大量使用：

- `\DrawSpaceLineBehindHorizontalFace`
- `\DrawSpaceLineBehindParallelogramFace`
- `\DrawSpacePlaneInteraction`

这些宏包含“线段或平面相对于观察方向的前后关系”。静态复刻可以在当前 TikZ 视角下算出可见段和隐藏段；但相机一动，交点、深度顺序、实虚线分段和虚线相位都可能变化，必须逐帧重算。仅把 TikZ 最初视角下的 dashed 样式复制到 Manim，运动后会出现本来应显露的边仍为虚线、应遮挡的边仍为实线。

当前编译器已经把这些宏识别成“线段 + 遮挡面 + 前后样式”的关系。渲染器在正式渲染前为每条关系预建固定数量的原生 `Line` 槽位，相机运动时逐帧重新求交，但只原地修改端点与透明度。静态图与动态视频因此使用同一对象结构，既保持 TikZ 初始视角的实虚线结果，也避免 Cairo 在动画分段边界出现旧线残留和跳变。

### 5.2 三维 `pic` 需要知道所在平面

二维 `angle`/`right angle` 只需三个投影点；三维角标还必须确定圆弧所在空间平面、半径的物理单位、相机退化时的方向以及标签朝向。当前这些入口因此被明确阻断，而不是错误地把三维点直接喂给二维 `Angle`。

当前已生成原生空间圆弧、直角折线、`anglemark` 扇形和独立文字，初始视角与源图一致。尚未完成的是相机运动时按新投影重新计算“屏幕定半径”的角标。

### 5.3 `canvas is ... plane` 与 `transform shape` 是另一类标签

普通 TikZ node 在投影位置上排版，但文字保持页面朝向；本轮已经能稳定映射为 Manim 的屏幕朝向标签。线面角比较图中的 node 则使用：

- `canvas is yz/xy plane at ...`
- `transform shape`
- `rectangle`、背景填色、`inner sep`
- `rotate=95`

当前已正确恢复它们的三维锚点、picture scale、矩形背景、边框、字号、显式旋转和局部平面的完整二维仿射，因此左右两幅静态图中的彩色说明框会像 TikZ 一样倾斜。若相机运动，仍需要把局部平面基绑定到相机 updater，逐帧重算这组仿射矩阵。

### 5.4 `\csname pt3d@...` 把几何关系藏进了 TeX 状态

两个公垂线入口直接从 `math-handout-common.sty` 的内部宏表读取点坐标。当前转换器已经能把本文件使用的 `\csname pt3d@点名@分量` 映射到自己的坐标登记表，所以两图已通过。这个实现仍不是完整 TeX 引擎；任意动态控制序列名、复杂嵌套条件和一般 `clip` 仍不属于保证子集。

更稳定的做法不是实现一整套 TeX，而是把常用关系提升为可识别的几何原语，例如“点平移”“线性插值”“平面内张成点”“公垂线足点”。这样 TikZ 和 Manim 共用同一关系，动画时也能明确谁是主动对象、谁是从动对象。

### 5.5 `pptstep` 应成为动画阶段，而不是被删除

`ProjectionOfFigureFig` 已经用 `pptstep[id=...,name=...]` 表达教学阶段。当前静态编译会安全跨过环境边界并保留全部几何，但尚未把 id/name 写入对象元数据，因此自动动画仍会丢掉这部分最有价值的信息。

正确方向是：

- 把 step id 写进稳定对象 ID 或对象的阶段元数据；
- 区分首次出现、持续存在、变色/淡化和退出；
- 允许 scene 按 step id 生成基础 reveal；
- 教学节奏仍由场景显式调整，不从一张静态图猜测主动对象。

### 5.6 TikZ `space view` 不一定是刚性三维相机

本文件常用的 `space view={(-0.35,-0.35),(1,0),(0,1)}` 是仿射/斜投影基，不是正交旋转矩阵，因此不能直接作为球面 orbit 的起点。样例视频使用一般矩阵插值从精确 TikZ 视角过渡到等轴测视角，再返回原视角；中间过程允许仿射形变。

若希望动画始终表现为刚性相机转动，TikZ 源应同时保存方位角、仰角和必要的投影类型，或使用可还原为正交相机的 `3d view`。否则“静态完全一致”和“全过程刚性 orbit”不能同时自动保证。

### 5.7 外部样式文件是可复现性依赖

当前 XeLaTeX 实际解析的是：

```text
/Users/leocyan/Library/texmf/tex/latex/math-handout/math-handout-common.sty
SHA-256 022057593c4dc97ca3ae267c6d49368ee07b7a09a1f96696619870d54da1ba90
```

它与 Manim 工作区根目录约 4.7KB 的同名旧文件不同。`space view`、点坐标登记、遮挡和 `\mathWhiteOutline` 等语义来自这份约 80KB 的用户 texmf 文件。换机器或改变 `TEXMF` 查找顺序时，单独复制目标 TeX 并不能保证复现；批量报告应继续记录实际样式路径与哈希。

## 6. 更有利于转换和动画的 TikZ 写法

### 6.1 保留具体、零参数的公开入口

Base 宏可以参数化，但每一幅最终使用的图最好提供一个零参数包装器：

```tex
\newcommand{\MyConcreteFig}{%
  \MyBaseFig{0.85}{-12}{...}%
}
```

转换器选择 `\MyConcreteFig`，得到确定的 scale、视角和图层；不要让最终入口还要求调用者补 `#1`。

### 6.2 关系优先于内部坐标宏

推荐继续使用或扩展这类语义写法：

```tex
\defPoint{A}{0}{0}{0}
\defPointShift{B}{A}{2}{0}{1}
\pointOnSpaceLine{M}{A}{B}{0.5}
```

不建议在图形正文里直接写：

```tex
\pgfmathsetmacro{\x}{\csname pt3d@A@x\endcsname + ...}
```

如果原生 TikZ 确实没有合适关系，再新增一个名字明确、参数显式的公共几何宏，同时给转换器添加同名 handler。

### 6.3 视角要区分“精确斜投影”和“可旋转相机”

- 只要求静态复刻时，数值化 `space view` 或 `x/y/z` 基向量最直接。
- 以后要刚性转动时，优先保存 `3d view` 的方位角/仰角，或额外提供相机语义参数。
- 不要只留下由若干三角函数计算后的三个屏幕向量，却丢掉它们代表的相机类型。

### 6.4 标签明确区分两类

普通屏幕标签：

```tex
\node[above=3pt,right=1pt] at (P) {$P$};
```

它应在 Manim 中保持朝向屏幕，位置跟随点，字号不随 TikZ scale 改变。只有确实要贴在空间平面上的标签才使用 `canvas is ... plane` 和 `transform shape`，并最好再包一层语义宏以显式给出平面。

### 6.5 遮挡继续写成关系，不要提前固化

对于将来要移动相机的图，`\DrawSpaceLineBehind...` 这类包含“线 + 面 + 前后样式”的写法比手工把线切成固定实线/虚线更有价值。需要做的是固定参数顺序、保证面的顶点顺序一致，并在转换器中实现该宏的动态语义。

### 6.6 保留阶段 id 和稳定点名

`pptstep` 的 id、点名和路径名应稳定，不要只靠绘制顺序表达教学阶段。点名也不要让转换器按 `A1`、`Ap` 或 `A'` 猜测拓扑；所有关系都以显式坐标和宏参数为准。

### 6.7 避免没有可见几何的命令

不要用 `\fill(Q)` 表示点。统一写成：

```tex
\node[pt] at (Q) {};
```

或：

```tex
\fill (Q) circle (0.8pt);
```

前者是 node 尺寸，默认不随 TikZ scale 变形；后者是实际路径，应按路径变换规则处理。

## 7. 动画样例说明

场景 `SpaceAngleSquarePyramidViewTransition` 先分别创建面、线、点和标签，再把相机从精确 TikZ 斜投影视角过渡到等轴测视角并返回。视频：

[`SpaceAngleSquarePyramidViewTransition.mp4`](../media/space_angle_part1_native/videos/space_angle_part1_native/720p30/SpaceAngleSquarePyramidViewTransition.mp4)

这个视频同时验证了两件事：

- 对象已经是独立的原生 Manim 对象，可以分别 `Create`、`FadeIn` 或绑定 updater；
- 语义遮挡线会随平行投影相机逐帧重新判定，静态场景和动态场景共用固定的原生 `Line` 槽位。

## 8. 验证命令

```bash
cd /Users/leocyan/Documents/Code/Manim/manim_scenes

# 21 个预览入口批量编译并实例化
/opt/homebrew/Cellar/manim/0.20.1/libexec/bin/python \
  scripts/audit_space_angle_tikz_native.py \
  --input '/Users/leocyan/Documents/Code/Codex/output/空间角专题复习（一）_tikz.tex' \
  --output-dir reports/space_angle_part1_native/batch_audit \
  --instantiate

# TikZ native 相关回归
.venv-manim/bin/python -m unittest discover -s tests -p 'test_tikz_native*.py' -v
.venv-manim/bin/python scripts/verify_tikz_native_baseline.py

# 25 幅实际图：同物理尺度 TikZ/Manim 并排图、叠加图与纵览图
/opt/homebrew/Cellar/manim/0.20.1/libexec/bin/python \
  scripts/render_space_angle_native_catalog.py \
  --output-dir reports/space_angle_part1_native/full_catalog

# 代表性静态图和动态视频
manim -ql -s --media_dir media/space_angle_part1_native \
  scenes/space_angle_part1_native.py \
  SpaceAngleSquarePyramidStatic \
  SpaceAngleCubeLinePlaneAngleStatic \
  SpaceAngleSquarePyramidFootPlaneStatic

manim -qm --media_dir media/space_angle_part1_native \
  scenes/space_angle_part1_native.py \
  SpaceAngleSquarePyramidViewTransition
```

本轮最终回归为 45 项 TikZ native 测试通过；目标文档 21/21 入口、25/25 幅图、527 个原生对象全部实例化，25 组并排图与 25 组叠加图完整。旧 v0.1 基线仍为 16 图、262 个语义对象，12 项渲染证据完整。目标 TeX 的 SHA-256 保持为 `bc557a03cb1c336d72d365a5323472335dc25f23283511a3951b5f8800d782bb`。

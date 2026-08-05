# TikZ → 原生 Manim v0.1 阶段性基线

> 迁入说明：本文保留 v0.1 建立时的报告路径和命令作为历史证据。当前 Provider 根目录
> 为 `tex-to-mathcapture-ppt/tools/tikz-native-provider/`，核心源码、脚本和测试分别位于
> `tikz_native/`、`scripts/` 和 `tests/`。

## 1. 这份基线解决什么问题

`v0.1` 是当前探索结果的可复现检查点，不是“已经支持完整 TikZ”的声明。

它固定了四类事实：

1. 输入 TeX 的确切版本；
2. 16 幅图转换后的对象、稳定 ID、动态依赖和严格门禁；
3. 已经实际检查过的静态图、动态视频和 TeX—Manim 对比图；
4. 当前仍然存在的能力边界。

以后修改解析器、渲染器或动态绑定层时，应先运行本基线。若结果变化，需要判断是回归错误还是有意升级；有意升级时新建 `v0.2`，不覆盖 `v0.1`。

## 2. 冻结输入

- 源文件：`/Users/leocyan/Documents/讲评课/2026年全国一卷第18题.tex`
- SHA-256：`92c66177ec43d1d73c0c2f9f1087d4f1b2ecbce62ba69217fc9de0764343a3e9`
- Python：3.12.13
- Manim Community：0.20.1
- XeTeX：TeX Live 2026
- FFmpeg：8.1.2

机器可读基线为：

- `manim_scenes/reports/tikz_native/2026_national_1_18/baseline-v0.1.json`

## 3. 当前已经冻结的能力

### 3.1 静态转换

- 16 个 `tikzpicture`；
- 262 个独立语义对象；
- 45 个点、58 条普通线、23 条箭头、8 个椭圆、1 个圆；
- 9 个填充多边形、9 个角弧、5 个原生直角记号；
- 85 个普通标签、10 个路径标签、9 个角标签；
- 所有对象都能实际实例化为有明确语义的 Manim 类；
- `unsupported` 为空；
- 未使用 SVG、位图、通用 `VMobject(...)` 或通用描点兜底。

### 3.2 字体与缩放

- 标签继承 11pt 正文字号；
- TikZ `scale` 只改变几何，不改变标签尺寸；
- 纵览场景不对单图内容调用 `scale` 或 `scale_to_fit_*`；
- 所有标签公式显式使用 `\displaystyle`；
- `\small` 与中英文混合标签保留各自的 TeX 语义。

### 3.3 对象级动画

- 262 个对象拥有稳定 ID；
- 填充、坐标框架、实线、辅助线、标记、点和标签可独立揭示；
- 显式虚线由原生小线段组成并沿路径依次创建；
- 教学顺序使用稳定对象 ID 显式编排，不从静态 TikZ 猜测唯一叙事。

### 3.4 动态几何

- 第一幅图保留 TikZ 原生 `name path + name intersections + sort by`；
- 有向直线排序在运动中稳定区分 `Q,P`；
- `R` 保留关于 `O`、`P` 的插值/中心对称依赖；
- 第一幅图的线、点、填充、虚线和标签可同步运动；
- 第四幅图已覆盖动点、动线、正交投影、路径标签和原生直角记号的依赖链；
- 结构不同的公式标签采用淡出/淡入，避免点级 `Transform` 碎字。

## 4. 语义指纹

基线不复制整份 329 KB manifest，而是保存以下紧凑门禁：

- 每幅图的对象数量；
- 每幅图按出现顺序排列的对象 ID 的 SHA-256；
- 全文对象类型统计；
- 自动动画分层统计；
- 坐标依赖类型统计；
- 命名路径求交关系；
- `unsupported` 和非致命警告数量。

完整调试信息仍在：

- `manim_scenes/reports/tikz_native/2026_national_1_18/manifest.json`
- `manim_scenes/reports/tikz_native/2026_national_1_18/animation_plan.json`
- `manim_scenes/reports/tikz_native/2026_national_1_18/report.json`

## 5. 冻结的渲染证据

| 证据 | 证明内容 |
| --- | --- |
| `artifacts/all_figures_true_scale_11pt.png` | 16 图真实尺度纵览，不逐图缩放 |
| `artifacts/representative_true_scale_11pt.png` | 五幅高风险图的字号、标签和复杂标记 |
| `artifacts/figure01_native_intersections_driven_1080p.mp4` | 有向交点身份和第一幅图的完整联动 |
| `artifacts/figure04_construction_1080p.mp4` | 稳定对象 ID 的教学构造顺序 |
| `artifacts/figure04_driven_motion_1080p.mp4` | 投影、路径标签、直角记号动态依赖 |
| `artifacts/figure13_to_14.mp4` | 对象级过渡与标签安全替换 |
| `artifacts/figure01_tex_vs_manim_same_scale.png` | TeX 与 Manim 同尺度并排比较 |
| `artifacts/figure01_tex_vs_manim_overlay_aligned.png` | 几何轮廓对齐和样式差异定位 |

每项证据的字节数和 SHA-256 都写入 `baseline-v0.1.json`。验证脚本会检查文件是否缺失或被意外覆盖。

## 6. 当前边界

以下内容没有被 `v0.1` 宣称为动态安全：

- 复杂曲线多交点、相切合并和交点消失；
- 任意 Bézier、一般 `arc`、plot/smooth；
- `clip`、decoration、pattern、shade 和 gradient；
- 复杂 node、matrix、自动换行和碰撞避让；
- 嵌套 scope 的旋转、缩放、平移与完整样式继承；
- 任意参数宏、复杂 PGF 表达式和完整表达式依赖图；
- 从静态 TikZ 自动推断唯一教学时间线。

普通 `dashed`、`densely dashed` 仍使用近似基准节距；需要精确节奏时应写显式 `dash pattern`。`baseline`、`trim right` 等 TeX 版面参数仍不进入 Manim 几何层。

## 7. 验证命令

在 `manim_scenes` 目录运行：

```bash
.venv-manim/bin/python scripts/convert_tikz_native.py \
  --input '/Users/leocyan/Documents/讲评课/2026年全国一卷第18题.tex' \
  --output-dir reports/tikz_native/2026_national_1_18 \
  --instantiate

.venv-manim/bin/python scripts/verify_tikz_native_baseline.py

.venv-manim/bin/python -m unittest \
  tests.test_tikz_native_compiler

.venv-manim/bin/python -m unittest discover -s tests
```

其中：

- 第一条重新生成 manifest、动画计划和转换报告；
- 第二条重新解析 TeX，检查语义指纹、原生对象政策和全部冻结证据；
- 第三条运行 TikZ 专项测试；
- 第四条检查整个 `manim_scenes` 工作区。

若只想检查语义而不核对 PNG/MP4 哈希，可运行：

```bash
.venv-manim/bin/python scripts/verify_tikz_native_baseline.py \
  --skip-evidence
```

## 8. 基线通过的含义

基线通过表示：当前 TeX 输入仍然生成同一组语义对象、对象顺序和已知依赖，原生对象门禁仍然成立，冻结的视觉与动画证据没有被意外替换。

它不表示所有 TikZ 都已支持，也不表示所有可能的运动区间都安全。新增关系仍然必须配套静态和动态回归样例。

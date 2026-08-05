# TikZ → 原生 Manim 三维转换原型

> 迁入说明：本文保留迁入前的三维原型与产物路径作为历史证据。当前受 Git 管理的
> Provider 根目录为 `tex-to-mathcapture-ppt/tools/tikz-native-provider/`；核心源码路径
> 应从旧 `manim_scenes/tikz_native/` 映射到当前 `tikz_native/`。

## 1. 本轮结果

首个三维原型已经跑通以下链路：

```text
TikZ 三维坐标
→ 读取 TikZ 视角
→ 保存世界坐标与 3×3 投影矩阵
→ 原生 Manim 点、线、面、箭头和标签
→ 原视角静态复刻
→ 相机沿三维轨道转动
→ 返回 TikZ 原视角
```

演示源码：

- `latex/tikz_native_3d_demo/tikz_native_3d_demo.tex`

转换与渲染代码：

- `manim_scenes/tikz_native/projection_3d.py`
- `manim_scenes/tikz_native/manim_renderer_3d.py`
- `manim_scenes/scenes/tikz_native_3d_demo.py`

当前样例共生成 37 个独立语义对象：3 个面、12 条普通线、3 个箭头、8 个
三维点和 11 个标签。严格门禁通过，没有 SVG、位图或通用 VMobject 回退。

## 2. 视角映射

### 2.1 `3d view`

TikZ perspective 库对

```tex
3d view={azimuth}{elevation}
```

使用以下二维基向量：

```text
x = ( cos az, -sin az sin el )
y = ( sin az,  cos az sin el )
z = (      0,          cos el )
```

转换器按 TeX Live 中 `tikzlibraryperspective.code.tex` 的定义直接生成 Manim
相机矩阵，没有通过截图估算视角。首个样例使用工作区已有参数
`3d view={40.4}{23.8}`。

这三个基向量对应正交相机，因此可以使用球面相机轨道转到其他正交视角，再
精确返回 TikZ 原视角。

### 2.2 显式 `x/y/z` 基向量

以下 TikZ 写法也已能解析：

```tex
x={(-0.35cm,-0.35cm)},
y={(1cm,0cm)},
z={(0cm,1cm)}
```

转换器把三个二维向量组成投影矩阵的前两行，并用两行的叉积方向作为第三行，
只用于深度排序。这样静态画面与 TikZ 工程投影一致，同时仍保留原始三维点。

这类矩阵可能包含剪切或不等比例缩放，因此不能一律调用正交相机的球面轨道；
否则会把斜二测错误解释为真实相机姿态。

## 3. 原生对象映射

| TikZ 语义 | Manim 对象 | 说明 |
| --- | --- | --- |
| 空间线段 | `Line` | 端点为三维世界坐标，线宽保持屏幕 pt 语义 |
| 空间箭头 | `Line + StealthTip` | 箭头尺寸不随 TikZ `scale` 二次缩放 |
| 平面多边形 | `Polygon` | 顶点保留三维坐标 |
| 点标记 | `Dot3D` | 相机转动时仍保持球状点标记 |
| 标签 | `MathTex` / `Tex` | fixed orientation，默认 11pt、数学使用 `\displaystyle` |

TikZ `scale` 仍然只进入世界坐标，不进入字体大小。三维图不再通过额外缩放整个
Manim 组合来适应画面，而是调整相机取景。

## 4. 三维标签的新问题与处理

只让标签正对屏幕还不够。如果把原视角下的标签偏移保存成一个固定世界向量，
相机转动后 `above` 可能变成斜向，空间对角顶点甚至会让标签重合。

当前处理方式是：

1. 标签内容与字号保持不变；
2. 标签始终正对屏幕；
3. 每帧读取当前相机投影矩阵；
4. 把 TikZ 的 `above / below / left / right` 和 pt 偏移反解为当前世界偏移；
5. 相机返回原视角后，标签恢复原始位置。

这解决了 anchor 随相机失效的问题，但没有声称解决所有标签—标签碰撞。较复杂
的运动区间仍可能需要分段 anchor 或碰撞规避。

## 5. 当前仍需人工或后续扩展的部分

### 5.1 遮挡

编译器会把 `DrawSpaceLineBehindHorizontalFace`、
`DrawSpaceLineBehindTriFace`、`DrawSpaceLineBehindParallelogramFace` 和
`DrawSpacePlaneInteraction` 保存为显式的“线段—有限面”关系。三维场景在把图形
加入场景前统一绑定：

```python
renderer.bind_occlusions_to_camera(figure, camera)
```

每条源码遮挡线在渲染开始前预建两个可见段槽位和一个固定虚线池，所有子对象均
为原生 `Line`。相机运动时只原地修改端点和透明度，不新增、删除或替换子对象，
从而兼容 Cairo 在一次 `play` 开始时固定绘制对象列表的行为。静态场景也使用同一
结构，只停留在 TikZ 初始视角。静态编译和动态 updater 共用
`tikz_native/occlusion_3d.py` 的区间裁剪算法，因此原 TikZ 视角下二者应产生完全
相同的首帧。

当前动态接入有以下边界：

- 支持三角形、平行四边形等凸有限面片，以及平行投影相机运动；
- 动点或动面需要向 `bind_occlusions_to_camera` 提供实时
  `coordinate_provider`；
- 两个透明面片的填充前后顺序尚未随相机自动重排；
- 透视投影下的线—面逐点视线不同，尚未沿用这套平行射线算法；
- 闭合凸多面体仍可使用工作区已有 `ProjectedPolyhedron`，逐帧判断可见面和
  隐藏棱。

非凸实体、任意曲面和透明面交叠仍然需要更一般的遮挡算法。

### 5.2 空间圆弧与角标

二维 `arc`、`RightAngle` 不能直接放进三维世界。程序必须先知道圆弧所在平面、
法向量、起止方向和相机变化时的显示策略。当前遇到三维 `pic` 角标会严格报告，
不会画成错误的屏幕二维角标。

### 5.3 宏和 scope

工作区一些立体 TikZ 使用多参数 `\newcommand`、嵌套 `\foreach`、
`tdplot_rotated_coords`、`canvas is plane` 和自定义遮挡宏。这些不是单纯的三维
坐标解析问题，还涉及 TeX 宏展开、局部坐标系和样式继承。

更利于自动转换的写法是：

- 教学对象尽量使用显式命名坐标；
- 关键点保留真实 `(x,y,z)`，不要只写投影后的二维坐标；
- 视角集中写在 `tikzpicture` 选项中；
- 需要独立动画的棱、面、点和标签分开写；
- 若使用宏，尽量让宏只负责重复绘制，不隐藏关键几何关系；
- 固定视角的隐藏棱可以显式虚线；准备转动相机时，应给出实体面拓扑。

### 5.4 动态几何关系

相机转动已经是原生三维动画，但静态 TikZ 仍不会自动说明“哪个点主动运动、哪个
面随它变化”。点、线和面虽然已经是独立对象，主动参数和依赖关系仍需来自 TikZ
原生构造关系或动画场景显式指定。

## 6. 验证命令

在项目根目录运行：

```bash
manim_scenes/.venv-manim/bin/python \
  manim_scenes/scripts/convert_tikz_native.py \
  --input latex/tikz_native_3d_demo/tikz_native_3d_demo.tex \
  --output-dir manim_scenes/reports/tikz_native_3d_demo \
  --instantiate

manim_scenes/.venv-manim/bin/python -m manim -qm \
  manim_scenes/scenes/tikz_native_3d_demo.py \
  TikzNative3DCameraOrbit \
  --media_dir manim_scenes/media/tikz_native_3d
```

专项测试：

```bash
cd manim_scenes
.venv-manim/bin/python -m unittest \
  tests.test_tikz_native_3d \
  tests.test_tikz_native_compiler \
  tests.test_tikz_native_compatibility
```

最终全量回归为 108 项测试全部通过；既有二维 `v0.1` 基线仍为 16 图、262 个
语义对象，12 项渲染证据完整。

## 7. 产物

- 静态 TeX—Manim 对照：
  `manim_scenes/reports/tikz_native_3d_demo/tex_vs_manim_static.png`
- 相机动画：
  `manim_scenes/media/tikz_native_3d/videos/tikz_native_3d_demo/720p30/TikzNative3DCameraOrbit.mp4`
- 相机关键帧总览：
  `manim_scenes/reports/tikz_native_3d_demo/camera_orbit_contact_sheet.png`
- 转换报告：
  `manim_scenes/reports/tikz_native_3d_demo/report.md`
- 兼容性报告：
  `manim_scenes/reports/tikz_native_3d_demo/compatibility-v0.1.md`

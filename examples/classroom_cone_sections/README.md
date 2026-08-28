# 高中课堂圆锥截口场景库

这组示例把有限圆锥截口能力整理成五段可以直接用于备课的动画。场景只负责教学排版和节奏：截线求解、遮挡区间、实虚转换、统一绘制顺序和固定 Manim 对象都继续由正式接口 `QuadricSection3D` / `QuadricOcclusion3D` 完成，没有另建一套示例专用算法。

## 快速预览

在仓库根目录执行：

```bash
manim --renderer cairo --disable_caching -ql --fps 8 \
  --media_dir artifacts/classroom-cone-sections/preview \
  examples/classroom_cone_sections/classroom_cone_sections.py \
  ConicFamilyTransitionLesson \
  ClosedVsOpenConeLesson \
  HiddenCurvePoliciesLesson \
  ProjectionDegenerationLesson \
  CapChordTopologyLesson
```

高清发布：

```bash
manim --renderer cairo --disable_caching -qh --fps 30 \
  --media_dir artifacts/classroom-cone-sections/release \
  examples/classroom_cone_sections/classroom_cone_sections.py \
  ConicFamilyTransitionLesson \
  ClosedVsOpenConeLesson \
  HiddenCurvePoliciesLesson \
  ProjectionDegenerationLesson \
  CapChordTopologyLesson
```

当前生产绑定只支持 Cairo；使用 OpenGL 会按支持契约明确失败。

## 场景一：为什么会出现椭圆、抛物线和双曲线

场景类：`ConicFamilyTransitionLesson`

参数：圆锥半顶角为 30°；截平面法向量绕屏幕水平方向连续转动 0–1.40 rad；当截平面恰好平行于一条母线时到达精确抛物线。末帧继续转到约 80°，让有限双曲线支和抛物线在课堂投影中更容易区分。

数学结论：同一个圆锥的截线随平面倾角连续地从椭圆经过抛物线，再变成双曲线；曲线拓扑变化时，真实可见段与被遮挡虚线仍连续更新。

教师提示：

- 先让学生预测闭合曲线会在什么时候打开。
- 暂停在精确抛物线，追问“此时平面与圆锥的哪条直线平行？”
- 比较临界点前后的虚线，说明“被遮挡”并不等于曲线断裂。

关键帧：[椭圆 → 精确抛物线 → 双曲线](gallery/contact-sheets/conic_family_transition.png)

```bash
manim -ql --fps 8 examples/classroom_cone_sections/classroom_cone_sections.py ConicFamilyTransitionLesson
manim -qh --fps 30 examples/classroom_cone_sections/classroom_cone_sections.py ConicFamilyTransitionLesson
```

## 场景二：封闭圆锥体与张口圆锥壳

场景类：`ClosedVsOpenConeLesson`

参数：左右两侧使用相同的圆锥轴、半顶角、有限高度、截平面和观察方向；左侧模型为 `CLOSED_SINGLE`，右侧为 `OPEN_SINGLE`。

数学结论：封闭圆锥体的完整截面可以包含侧面弧与真实底面弦；张口圆锥壳没有底面，所以只能保留侧面截线，不能补出一条虚构的弦。

教师提示：

- 让学生判断黄色直线段究竟来自侧面还是底面。
- 临时遮住模型标签，让学生先根据截面边界判断左右模型。
- 强调青色开口圆周是壳的边界，但它不代表存在一个底面圆盘。

关键帧：[接触底面前 → 中间位置 → 穿过底面](gallery/contact-sheets/closed_vs_open.png)

```bash
manim -ql --fps 8 examples/classroom_cone_sections/classroom_cone_sections.py ClosedVsOpenConeLesson
manim -qh --fps 30 examples/classroom_cone_sections/classroom_cone_sections.py ClosedVsOpenConeLesson
```

## 场景三：三种隐藏曲线绘图策略

场景类：`HiddenCurvePoliciesLesson`

参数：三个几何完全相同的封闭圆锥和截平面，依次使用 `physical`、`diagrammatic`、`depth_aware_diagrammatic`。

数学结论：三种策略只改变“隐藏部分如何画”，不会改变截线几何。`physical` 不绘制不可见部分；`diagrammatic` 把教学虚线放在顶层；`depth-aware` 把虚线放在真实遮挡面之后，因此会受到半透明表面的颜色和透明度衰减。

教师提示：

- 先问哪一栏更接近真实视觉，再问哪一栏更方便讲解。
- 比较中栏和右栏虚线亮度，解释透明表面的前后绘制层级。
- 强调策略切换不是重新求了一条不同的圆锥曲线。

实现上，三栏使用同一个 `QuadricGeometryPrototype`：截平面分区只计算
一次，三栏分别生成自己的绘制策略和固定 Manim 槽位；横向排列使用
不改变深度的屏幕偏移。

关键帧：[较低截面 → 并排比较 → 较高截面](gallery/contact-sheets/hidden_curve_policies.png)

```bash
manim -ql --fps 8 examples/classroom_cone_sections/classroom_cone_sections.py HiddenCurvePoliciesLesson
manim -qh --fps 30 examples/classroom_cone_sections/classroom_cone_sections.py HiddenCurvePoliciesLesson
```

## 场景四：正投影、一般平行投影与侧视退化

场景类：`ProjectionDegenerationLesson`

参数：观察方向从 `(1,1,1)` 连续变到 `(0,1,0)`；在进度 0.45 处加入经过认证的斜投影；张口圆锥的开口圆周位于 `xy` 平面。

数学结论：圆周在一般平行投影下成为椭圆；观察方向逐渐进入圆周所在平面时，椭圆越来越扁；精确侧视时二维投影的秩降为 1，圆周成为一条有限线段，而不是无限直线。

教师提示：

- 追问“投影变成线段，是否说明三维圆周本身也退化了？”
- 把椭圆变扁与二维线性变换的秩联系起来。
- 指出侧视线段只占有限范围，不能沿支撑直线无限延长。

关键帧：[正投影 → 一般平行投影 → 精确侧视](gallery/contact-sheets/projection_degeneration.png)

```bash
manim -ql --fps 8 examples/classroom_cone_sections/classroom_cone_sections.py ProjectionDegenerationLesson
manim -qh --fps 30 examples/classroom_cone_sections/classroom_cone_sections.py ProjectionDegenerationLesson
```

## 场景五：平面碰到底面时，截面边界为什么改变

场景类：`CapChordTopologyLesson`

参数：封闭圆锥高度为 4、半顶角为 30°；截平面沿法向量 `(0.82,0,1)` 平移；第一次接触由解析式确定。由于接触瞬间的弦长度为 0、画面上没有可辨认的线段，中间关键帧在解析接触后再前进 0.03 归一化进度，显示第一条清楚可见的短弦。

数学结论：截平面未碰到底面时，有限截面只有侧面圆锥曲线；第一次接触底面以后，完整截面边界变成侧面弧加真实底面弦。这里变化的是有限实体的边界组成，不是椭圆/抛物线/双曲线的分类。

教师提示：

- 在接触前暂停，让学生预测下一刻会新增什么边界。
- 强调新增 chord（弦）来自封闭圆锥的真实底面。
- 比较接触瞬间的零长度弦与穿过底面后的有限弦。

关键帧：[纯侧面截线 → 接触后第一条可见短弦 → 侧面弧加底面弦](gallery/contact-sheets/cap_chord_topology.png)

```bash
manim -ql --fps 8 examples/classroom_cone_sections/classroom_cone_sections.py CapChordTopologyLesson
manim -qh --fps 30 examples/classroom_cone_sections/classroom_cone_sections.py CapChordTopologyLesson
```

## 重新生成审查关键帧

仓库保存 15 张 960×540 Cairo 关键帧、5 张三联对照图和一份带 SHA-256、绘制顺序及角色计数的 `gallery/manifest.json`。重新生成：

```bash
python scripts/generate_classroom_cone_section_gallery.py
```

生成结果是可重新生成的验收资料；教学几何的权威来源仍是场景参数与 renderer-neutral 内核。

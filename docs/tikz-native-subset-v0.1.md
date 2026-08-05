# TikZ → 原生 Manim 标准化子集 v0.1

> 迁入说明：当前权威机器定义位于 Provider 根目录下的
> `tikz_native/subset_v0_1.json`；本文中旧 `manim_scenes/` 报告路径保留为历史验收证据。

## 1. 目的

这份规范用于回答三个不同问题：

1. 某段 TikZ 能否被当前程序识别；
2. 它能否稳定复刻为原生 Manim 静态对象；
3. 它保存的信息是否足以建立动态几何关系。

三者不能再统一写成一个笼统的“支持”。机器可读 feature registry 位于：

- `manim_scenes/tikz_native/subset_v0_1.json`

当前 16 图的实际审计结果位于：

- `manim_scenes/reports/tikz_native/2026_national_1_18/compatibility-v0.1.json`
- `manim_scenes/reports/tikz_native/2026_national_1_18/compatibility-v0.1.md`

## 2. A/B/C 的准确含义

### A：动态安全

满足以下条件：

- 可以创建有明确语义的原生 Manim 对象；
- 需要独立动画的对象拥有稳定 ID；
- 如果 TikZ 写出了几何构造关系，manifest 会保留该关系；
- 可以通过统一绑定器实时更新对象，而不需要退回 SVG、位图或无语义通用路径。

A 级不表示程序能从静态 TikZ 自动猜出谁是主动对象，也不表示已经存在唯一教学时间线。例如直线—椭圆求交虽然属于 A 级，但仍需另行指定直线角度、运动区间和相切策略。

### B：静态安全

满足以下任一情况：

- 静态复刻可以接受，但视觉参数只能近似；
- 该选项属于 TeX 版面而不是 Manim 几何；
- 当前已证明可以安全忽略，但必须保留警告。

B 级不能静默升级为 A，也不能被当成 C 级错误。例如 `baseline` 和 `trim right` 只影响 TeX 行盒，应该留在报告中，由 PPT 布局层处理。

### C：不支持

当前没有可靠的原生语义映射。严格模式必须：

1. 记录图编号和原始语句；
2. 停止该文档的严格转换；
3. 禁止退回 SVG、位图或通用 VMobject 描点；
4. 由人决定新增哪一种原生关系和回归样例。

## 3. 当前 A 级子集

### 3.1 坐标和关系

- 命名坐标；
- 受控纯数值宏；
- 两点插值和外分，包括中心对称；
- 坐标平移；
- 点到命名直线的正交投影；
- `name path` 直线、椭圆和圆；
- 带有向 `sort by` 的直线—直线、直线—椭圆/圆求交。
- 三维命名坐标 `(x,y,z)`；
- perspective 库的 `3d view={方位角}{仰角}`；
- 显式 `x/y/z={(u,v)}` 工程投影基向量。

### 3.2 原生对象

- `Line`；
- `Line + StealthTip`；
- `Polygon`；
- `Ellipse`；
- `Circle`；
- `Dot`；
- 三维点使用 `Dot3D`，空间线段和面分别保留为原生 `Line`、`Polygon`；
- `MathTex`、`Tex`；
- 三维标签以 fixed-orientation 方式始终正对屏幕，并按当前相机矩阵重算 anchor；
- 路径标签；
- `Arc` 角弧；
- `RightAngle`。

### 3.3 样式

- pt 线宽独立换算；
- 显式 `dash pattern=on ... off ...`；
- HTML 颜色与 xcolor 混色；
- 描边、填充和整体透明度；
- 11pt、`\small` 等受控 TeX 字号命令；
- 数学标签默认显式 `\displaystyle`；
- TikZ `scale` 缩放几何但不缩放字体。

## 4. 当前 B 级子集

| Feature | 原因 | 建议写法 |
| --- | --- | --- |
| `style.dash_keyword` | `dashed`、`densely dashed` 使用近似节距 | 要求像素复刻时改写为显式 `dash pattern` |
| `layout.baseline` | 只影响 TeX 行盒 | 在 PPT/画布布局层设置锚点 |
| `layout.trim_right` | 只影响 TeX 外部留白 | 在布局层处理，不改变几何坐标 |
| `scope.redundant_draw_none` | 当前仅证明对 fill 可安全忽略 | 新图尽量删除这一冗余 scope 选项 |

## 5. 当前 C 级范围

- 一般 Bézier、一般 `arc`、plot 和 smooth；
- `clip`；
- decoration、pattern、shade、gradient；
- 复杂 node、`text width`、自动换行和 matrix；
- 嵌套 scope 旋转、缩放、平移与完整样式继承；
- 任意参数宏、条件分支和复杂 pgfkeys；
- 复杂曲线多交点和闭合路径排序；
- 相切、交点合并、消失与重新出现。
- 任意空间曲线、任意平面内圆弧和三维 `pic` 角标；
- 任意非凸实体或穿插线段的全自动遮挡。

这里的 C 级是当前实现状态，不是永远禁止。某项只有在同时增加原生映射、最小 TikZ fixture、静态对比、动态不变量和边界测试后，才能升级为 A 或 B。

## 6. 对 TikZ 源码的基本要求

### 6.1 命名教学对象

```tex
\coordinate (P) at (1,1.5);
\coordinate (Q) at (-1.8,-0.6);
\draw (P) -- (Q);
```

匿名坐标可以静态绘制，但不适合作为长期稳定的动画接口。

### 6.2 需要独立动画的对象分开写

```tex
\draw (P) -- (Q);
\draw (Q) -- (R);
\draw (R) -- (P);
```

转换器虽然会拆分折线，但分开写更容易保持语义和教学顺序。

### 6.3 填充和边界分离

```tex
\fill[fill=lectureteal!14] (P)--(Q)--(R)--cycle;
\draw (P)--(Q);
\draw (Q)--(R);
\draw (R)--(P);
```

### 6.4 动态求交保留有向路径

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

直线端点顺序就是交点身份的一部分，运动中不得随意反转。

### 6.5 三维图必须明确写出视角

正交相机优先使用：

```tex
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (2,1,3);
  \draw (A)--(B);
\end{tikzpicture}
```

TikZ 风格斜二测也可显式写成：

```tex
\begin{tikzpicture}[
  x={(-0.35cm,-0.35cm)},
  y={(1cm,0cm)},
  z={(0cm,1cm)}
]
```

`3d view` 会生成正交相机标架，可以沿球面轨道平滑转动。一般 `x/y/z`
基向量可能包含剪切或不等比例缩放，静态投影可以精确保留，但不能假装成普通
刚性相机旋转；若要改变视角，应使用一般矩阵过渡或另行指定目标正交相机。

三维图中的 `scale` 仍然只缩放世界几何，不缩放 11pt 标签。相机运动时，标签
保持正对屏幕，并逐帧重算 `above`、`below`、`left`、`right` 等屏幕方向偏移。
这只保证 anchor 语义，不自动解决所有标签之间的碰撞。

## 7. 审计命令

在 `manim_scenes` 目录运行：

```bash
.venv-manim/bin/python scripts/audit_tikz_native_compatibility.py \
  --input '/Users/leocyan/Documents/讲评课/2026年全国一卷第18题.tex' \
  --output-dir reports/tikz_native/2026_national_1_18
```

输出内容包括：

- 当前文档实际使用了哪些 A/B/C feature；
- 每项出现次数；
- B/C 项所在图编号和原始警告；
- 静态是否可继续；
- 动态使用前仍需提供哪些信息。

返回码约定：

- `0`：没有遇到 C 级语法；
- `2`：遇到 C 级语法，严格转换应停止。

## 8. 每次新增 feature 的完成标准

一项能力只有同时满足以下要求，才能写入 A 级：

1. 有最小原生 TikZ 样例；
2. 编译器保留稳定对象或关系；
3. 使用明确的原生 Manim 类；
4. 有对象 ID 与 manifest 测试；
5. 有 TikZ—Manim 静态对比；
6. 有多个参数状态的动态不变量测试；
7. 有反向、退化或拓扑边界测试；
8. 有明确的失败政策。

如果只能满足静态复刻，应先进入 B 级；不能安全复刻则保持 C 级。

## 9. 与 v0.1 阶段基线的关系

阶段基线冻结“当前程序对当前 16 图的实际结果”；本子集规范则定义“以后什么样的 TikZ 可以被称为静态安全或动态安全”。

新增本规范和兼容性报告不会修改 `baseline-v0.1.json` 中已经冻结的对象、依赖或渲染证据。

# manim-tikz-native 中文说明

这个项目把一套**受控的 TikZ 写法**转换成真正的 Manim 对象，而不是把整张
TikZ 图压成一张 SVG 或图片。转换后的直线、面、点、标签仍然有各自的名字，
因此可以继续使用 `ValueTracker`、`scene.play()` 和普通 Manim 动画来驱动。

它还包含两套自动遮挡能力：

- 闭合凸多面体：所有登记的凸面共同参与计算，语义边会自动切成可见实线和
  被遮挡虚线；
- 开放凸面：适合二面角、折叠板等不构成封闭多面体的图形，并能处理显式声明的
  铰链共边以及半透明面的前后次序。

在闭合凸多面体上，还可以继续加入两类对象：

- 独立直线：自动计算直线进入、穿过和离开实体的位置，实体内部的部分自动画成
  虚线，并可显示稳定的交点标记；
- 独立截面：移动一个无限延展的数学平面时，自动得到点、线段、三角形、四边形
  或更多边的凸截面，并让截面边界、原多面体边和辅助线一起参加遮挡计算。画面中
  的有限矩形默认会自动放大到覆盖完整立体图形，不需要作者预估平面尺寸。

当半透明截平面真正穿过实体时，还可以显式启用“准确透明排序”。模块会先把
完整平面分成截面内部与外部区域，再沿交线切开被穿过的实体面，最后把这些局部
三角片按当前平行视角从远到近排列。因此不再需要把整张平面粗暴地放在实体前面
或后面。

自动适配时，`half_width` 和 `half_height` 只是显示矩形的最小尺寸。模块会按
当前立体图形在平面两个方向上的完整范围增加 15% 留白；同一次动画中只会继续
扩大，不会缩小抖动。只有确实要表现一块有限薄片时，才设置
`plane_patch_mode="strict"`。

闭合凸多面体还可以启用“立体感提示层”：近面会保留较强的原色和不透明度，
远面、背向面会自动褪色；不同朝向还会在原色附近产生连续的冷暖色阶，投影外
轮廓会略微加粗。原有虚实线遮挡保持不变，截平面仍使用单独的教学配色。这是
为了让课件中的前后关系更容易读懂，并不是物理光照模拟。示例采用等轴视角，
避免正对单个大面而看不到其他面。

## 安装

```bash
git clone https://github.com/Mathlatics/manim-tikz-native.git
cd manim-tikz-native
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

另外需要安装 XeLaTeX、TikZ、Fandol 字体、Latin Modern 字体和 FFmpeg。

## 两个最直观的示例

```bash
# 闭合凸多面体
manim -pql examples/polyhedron_visibility/cube_auto_occlusion.py \
  CubeAutoOcclusionDemo

# 开放二面角
manim -pql examples/open_face_visibility/dihedral_auto_occlusion.py \
  DihedralAutoOcclusionDemo

# 直线穿过正方体、移动截面，以及两者组合
manim -pql examples/convex_sections/convex_sections_demo.py \
  LineThroughCubeDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  MovingPlaneSectionDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  CombinedSectionAndLineDemo
manim -pql examples/convex_sections/convex_sections_demo.py \
  AccurateTransparentSectionDemo

# 四种凸多面体的面明暗、轮廓和动态截面
manim -pql examples/convex_sections/other_convex_solids_demo.py \
  TetrahedronSectionDemo TriangularPrismSectionDemo \
  SquarePyramidSectionDemo OctahedronSectionDemo
```

## 需要特别理解的边界

这不是“任意 TikZ 转 Manim”。它只接受文档中明确列出的写法。遇到暂不支持的
复杂路径、裁剪、渐变、任意宏或拓扑突变时，转换会给出错误并停止，不会偷偷换成
SVG 或位图。这样做的目的，是保证动画中的每个对象都可读、可编辑、可恢复。

自动遮挡也要求明确的几何语义：顶点、面和需要显示的直线必须有稳定身份。
模块不会从一个任意 `VGroup` 中猜测拓扑。完整规则见：

- [公共 API](docs/public-api.md)
- [自动遮挡](docs/automatic-occlusion.md)
- [支持的 TikZ 子集](docs/supported-tikz.md)
- [架构说明](docs/architecture.md)

本仓库只提供可复用的 Manim 模块，不包含网页编辑器、PPT、时间线、ShapeAsset 或
ShapeState 数据。

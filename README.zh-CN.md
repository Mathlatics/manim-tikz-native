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

还可以从一个闭合凸多面体中选取两个相邻面，复制成一个可独立移动的二面角。
复制体刚出现时，原面与原棱会把显示权交给高亮的二面角，因此不会因为重叠绘制
而突然变深或变粗；平移刚开始时，原面与原棱会按二者在最终画面坐标中的分离
距离平滑恢复，而不是在第一张运动帧突然全部出现。分离距离达到默认的 `0.12`
画面单位后，原多面体与复制出的二面角会以完整强度同时参加全局线条遮挡。
启用准确透明排序后，两组半透明面发生穿插时会先被切成局部三角片，再按
当前视角逐片排列前后，而不是把整个二面角一律放在原立体的前面或后面。只有
两个有限多边形真正相交时才会切分；来自同一原始面、处于同一有效前后位置的连续
三角片会合并成一次透明填充，因此内部计算用的三角边不会显示在最终画面中。

“重合时只显示一次、分离时平滑恢复”现在不是二面角专用逻辑。模块会在复制发生时
冻结原顶点、原面、原棱与复制体之间的一一对应关系。整块多面体、截面或任意已登记
子结构都可以使用同一套交接计算：完全重合时复制体拥有画面，开始分离后只恢复与它
对应的原面和原棱，复制体本身始终保持完整强度。二面角控制器已经改为复用这套通用
模块，原有 `identity_handoff_distance` 用法保持不变。

现在这条准确模式还会把“面片、可见实线段、被遮挡虚线段”放进同一份前后关系
图。直线会在穿过面深度的位置以及投影交叉点处继续细分，然后 Manim 按局部深度
统一绘制。这样不仅虚实判断正确，线与面、线与线交叉处的最终像素层级也正确，
不会再把所有线条无条件盖在所有半透明面之上。

二面角移出后，还可以选择原多面体的任意一个面作为底面。模块会以原多面体全部
已登记顶点的几何中心定义旋转；复制出的二面角继承这个中心，发生平移后旋转中心
也随它一起平移。因此原立体和二面角会在各自当前位置绕自己的中心同步改变朝向，
而不是绕同一个固定世界点公转。示例还让原立体向一侧移动、二面角向另一侧移动，
最终分列画面两边；实线、虚线和透明面碎片仍使用同一份逐帧结果更新。

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

## 主要示例

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

# 从长方体、四面体、四棱锥中复制并移出一个二面角
manim -pql \
  examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  RectangularBoxDihedralDemo TetrahedronDihedralDemo \
  SquarePyramidDihedralDemo RectangularBoxDihedralRoundTripDemo
```

其中往返示例会让高亮复制体完成分离、同步旋转，再重新回到与原形状完全重合的
位置。反向过程仍使用同一套语义交接计算，因此末帧重新只显示一份形状，不会出现
两份半透明对象叠加变深。四面体示例专门展示纯分离交接；长方体与四棱锥示例还会
继续展示同步底面旋转。

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

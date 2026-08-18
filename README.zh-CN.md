# manim-tikz-native 中文说明

这个项目把一套**受控的 TikZ 写法**转换成真正的 Manim 对象，而不是把整张
TikZ 图压成一张 SVG 或图片。转换后的直线、面、点、标签仍然有各自的名字，
因此可以继续使用 `ValueTracker`、`scene.play()` 和普通 Manim 动画来驱动。

它还包含两套自动遮挡能力：

- 闭合凸多面体：所有登记的凸面共同参与计算，语义边会自动切成可见实线和
  被遮挡虚线；
- 开放凸面：适合二面角、折叠板等不构成封闭多面体的图形，并能处理显式声明的
  铰链共边以及半透明面的前后次序。

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

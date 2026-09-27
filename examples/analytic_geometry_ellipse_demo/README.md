# 椭圆解析几何驱动 Demo

这个 Demo 验证一条可继续扩展的链路：

```text
受支持的 TikZ 源码
  → PictureSpec 中的命名点、有向路径、交点和坐标依赖
  → ellipse_problem.motion.json 显式选择主动直线和绑定对象
  → ValueTracker 只修改角度 theta
  → P、Q、R、线段、填充、标签和角记号逐帧重算
```

## 文件

- `ellipse_problem.tex`：本题初始图，初始斜率为 `3/4`。
- `ellipse_problem.motion.json`：声明式主从关系、对象绑定和时间线。
- `scene.py`：只负责排版、数值展示和播放，不再逐个手写几何对象接线。

## 渲染

在仓库根目录中执行：

```bash
MEDIA_DIR=/absolute/path/to/an-empty-output-directory

python -m manim --renderer cairo -ql \
  examples/analytic_geometry_ellipse_demo/scene.py \
  EllipseAnalyticGeometryDriverDemo \
  --media_dir "$MEDIA_DIR"
```

## 这一版证明了什么

- TikZ 编译后的原生 Manim 对象可以直接运动，没有另画第二份图。
- 主动对象由配置显式选择，不从静态图猜测。
- `Q` 与 `P` 沿有向直线排序，运动中不会交换身份。
- `R` 根据 TikZ 保留的 `interpolation(O,P,-1)` 依赖重算，不在 Scene 中写死。
- 动态角弧和角标已纳入同一绑定机制。

这个示例只演示可复用的 Manim 模块，不依赖网页编辑器、PPT 或时间线。

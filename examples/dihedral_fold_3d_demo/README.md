# TikZ Native 三维折叠 Demo

这个 Demo 把项目中原先分散的四部分收到同一条可重复渲染的链路中：

```text
命名 TikZ 三维几何
  → Provider 生成稳定的原生 Manim 对象
  → motion-3d.json 声明“谁主动、谁跟随”
  → 平面 beta 绕公共棱 AB 折叠
  → M、N、线段 MN 和自动遮挡每帧更新
  → 相机切换后恢复 TikZ 入口视角和几何
```

## 文件

- `dihedral_fold.tex`：从已有二面角素材缩减出来的命名三维图。
- `motion-3d.json`：`tikz-native-motion-3d/v1` 声明式运动。
- `scene.py`：只组装 Provider 对象并播放 timeline，不再画第二份图。

## 几何关系

- `A-B` 是公共棱，也是折叠轴。
- `A-B-Alpha1-Alpha0` 是固定面 `alpha`。
- `A-B-Beta1-Beta0` 是活动面 `beta`。
- `M` 是活动面边 `Beta0-Beta1` 上的从动点。
- `N` 是 `M` 在公共棱 `AB` 上的投影垂足。
- `S-E` 是固定的紫色检测线，用来直观检查它被活动面遮挡的区间。
- `DrawSpacePlaneInteraction` 在 Provider 中保存为明确的线—面遮挡关系。

运动时只修改 `fold_angle`。其他点、面、标签和遮挡均从同一份
`PictureSpec` 中的稳定对象原地更新，不会为动画另外重画一套几何。

## 可复制的三维作者写法

第一版 Geometry Rig 优先读取明确写在 TikZ 里的语义，不会从生成后的
`objectId` 猜点名。制作类似素材时，可以直接沿用下面四类写法。

### 1. 声明折叠轴、固定面和活动面

```tex
\DeclareSpaceHinge
  {fold-angle}
  {A/B}
  {A/B/Alpha1/Alpha0}
  {A/B/Beta1/Beta0}
```

四个参数依次是：稳定的关系 ID、有方向的铰链轴、固定面顶点、活动面顶点。
这个命令不绘图，只把关系写进版本化的
`tikz-native-hinge-relation/v1`。轴的两个端点必须同时属于两个面；所有点
必须先声明为三维命名坐标。

### 2. 声明活动边上的从动点

```tex
\coordinate (M) at ($(Beta0)!0.67!(Beta1)$);
```

Provider 会把它保存为 `point_on_segment`：`M` 始终位于
`Beta0-Beta1` 上，参数为 `0.67`。活动面折叠时，`M` 自动跟随，不需要在
Native Clip 中再手写一次插值公式。

### 3. 声明投影点

```tex
\coordinate (N) at ($(A)!(M)!(B)$);
```

Provider 会把它保存为 `project_point_to_line`：`N` 始终是 `M` 在直线
`AB` 上的正交投影。这里必须使用真正的 TikZ projection 语义，不能用一
个碰巧落在 `AB` 上的固定插值点代替。

### 4. 声明面与动态遮挡

```tex
\DrawSpacePlaneInteraction
  [betaFace][betaCovered][alphaEdge][alphaHidden][betaEdge][betaHidden]
  {A/B/Alpha1/Alpha0}{A/B/Beta1/Beta0}

\DrawSpaceLineBehindParallelogramFace[probe][probeHidden]
  {S}{E}{A}{B}{Beta1}{Beta0}
```

第一条同时建立两个面的原生 Polygon、边线碎片和线—面遮挡关系；第二条
表示检测线 `SE` 可能被活动面遮挡。Geometry Rig 会把这些 raw fragments
归并为稳定的 `semanticGroups`，前端无需拆解 `objectId`。

相机切换不写在 TikZ 中。TikZ 只保存入口投影视角；`front`、`side`、
`top`、`oblique`、`isometric` 等相机操作由 Geometry Rig / Native Clip
时间线选择。v1 固定使用 `end_policy: restore_entry`：片段结束时恢复入口
几何和入口相机，不把临时三维姿态写成下一张快照的 ShapeState。

## 渲染

在 Provider 根目录中，使用安装了 Manim 0.20.x 的 Python：

```bash
python -m manim -ql \
  examples/dihedral_fold_3d_demo/scene.py \
  DihedralFold3DAnimationDemo \
  --media_dir /path/to/an-empty-media-directory
```

Demo 本身不包含任何用户目录的绝对路径。`scene.py` 只通过自身文件位置
找到 TikZ 和 motion 配置，因此整个 Provider 目录复制到其他位置后仍可运行。

## 这一版的边界

- 遮挡仍是平行投影下的有限三角形/平行四边形线—面关系。
- 折叠驱动已支持任意命名空间轴，但这个夹具特意使用易于核对的 `AB` 轴。
- 它是 Provider 内部实验 Demo，不会修改 ShapeState、PPT 作者数据或正式媒体。

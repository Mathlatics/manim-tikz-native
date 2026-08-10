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

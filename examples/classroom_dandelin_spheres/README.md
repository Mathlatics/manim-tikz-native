# 丹德林球课堂示例：一套构造，三种圆锥曲线

这个 16:9 Manim 场景分三幕依次展示：

1. 椭圆：两个丹德林球位于同一个锥瓣，球与截平面的两个切点成为椭圆的两个焦点；
2. 抛物线：截平面平行于一条圆锥母线时到达临界状态，只留下一个有限丹德林球；
3. 双曲线：完整的两支双曲线需要有限双锥，每个锥瓣各有一个丹德林球。

场景类为 `DandelinThreeConicsLesson`，文件为
`examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py`。

## 先看清楚画面语义

场景使用正式高层接口 `DandelinSection3D`。真实圆锥截线仍由现有
renderer-neutral 几何内核与截平面 compositor 计算；橙色球、球—圆锥接触圆、
焦点，以及前两幕中的准线，属于 **diagrammatic teaching overlay（教学辅助叠加层）**。

这层辅助图形用来解释数学构造，不表示系统已经认证了“球位于圆锥内部时”的
真实球—圆锥前后遮挡关系。换句话说，橙色线条显示在上方是教学排版选择，不能
把它解释为物理深度结果。接口公开的 `visibility_authoritative=False` 也明确记录了
这一点。

每一幕在 `attach()` 时才预留一个 Scene 级 painter band，并自动拆成圆锥截线、
教学叠加层和焦点三段；若前一幕或另一个控制器已占用首选范围，整组会一起上移，
不会只移动其中一层。`restore()` 会同时撤回画面对象、fixed-frame 注册和该 band。

## 低清预览

在仓库根目录执行：

```bash
manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir artifacts/classroom-dandelin-spheres/preview \
  examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py \
  DandelinThreeConicsLesson
```

低清版本适合检查构图、文字和课堂节奏。项目当前生产绑定只支持 Cairo，
不要把命令中的 renderer 改成 OpenGL。

## 高清发布

```bash
manim --renderer cairo --disable_caching -qh --fps 30 \
  --media_dir artifacts/classroom-dandelin-spheres/release \
  examples/classroom_dandelin_spheres/classroom_dandelin_spheres.py \
  DandelinThreeConicsLesson
```

Manim 默认画幅是 16:9；`-ql` 和 `-qh` 只改变输出质量，不改变这套课堂构图的
宽高比。

## 数学参数

三幕使用同一个半顶角为 30° 的直圆锥，圆锥轴为 `z` 轴，截平面法向量位于
`xz` 平面。记法向量与圆锥轴夹角的余弦为 `n·axis`：

| 幕次 | 圆锥模型 | 截平面位置 | `n·axis` | 数学结果 | 有限丹德林球 |
| --- | --- | --- | ---: | --- | ---: |
| 椭圆 | `OPEN_SINGLE` | `A + 1.5 axis` | 0.80 | 平面只切一个锥瓣，得到闭合椭圆 | 2 |
| 抛物线 | `OPEN_SINGLE` | `A + 3.0 axis` | `sin(30°)=0.50` | 平面平行于一条母线 | 1 |
| 双曲线 | `OPEN_DOUBLE` | `(0, 0.5, 0)` | 0.16 | 平面切过两个锥瓣，得到两支双曲线 | 2 |

这里使用有限圆锥范围，是为了让每个球都能被严格认证为完整地落在相应锥瓣内；
达到截断边界或数值容量上限时，高层接口会明确失败，而不是缩小球或猜一个看起来
合理的结果。

## 教师讲解提示

- 椭圆幕：先指出两个球各自与截平面相切的位置，再把两个切点命名为焦点；
- 抛物线幕：暂停比较截平面方向与母线方向，强调它是椭圆与双曲线之间的临界角；
- 双曲线幕：让学生观察两个球分居两个锥瓣，从而理解为什么完整双曲线需要双锥；
- 三幕都可以追问：“橙色圆为什么是接触圆，而不是球在截平面上的截面？”

双曲线幕采用开放双锥 compositor 已认证的精确侧视。在这个视角中，一条准线与
一个侧视接触圆会投影到同一条支撑直线上，系统无法为两层重合墨迹认证唯一顺序，
因此该幕显式关闭可选准线；球、接触圆、焦点和真实双曲线仍正常显示。这里选择
“少画一条辅助线并说明原因”，而不是猜测一个绘制层级。

本示例是静态三幕：每幕先一次性求解并附着固定对象，停留讲解后通过接口的
`restore()` 完整撤回，再进入下一幕；没有在 updater 中创建或替换 Mobject。

# 丹德林双球：圆锥面与圆柱面切换

这个示例只做一件事：让同一截平面下的两只丹德林球，随母面从圆锥连续过渡为
圆柱，再反向回到圆锥。

场景类：`DandelinConeCylinderSwitch`

## 数学含义

动画使用有限显示窗口内的旋转面

```text
r(z, p) = R + k0 (1 - p) z,    0 <= p <= 1
```

- `p=0` 时，下端半径为 0，是圆锥；
- `0<p<1` 时，圆锥顶点逐渐向下远离画面，显示窗口里看到的是圆台；
- `p=1` 时，斜率严格等于 0，母线互相平行，得到圆柱。

每一帧都会解析重算两只球的圆心和半径，并同时满足：

1. 球与当前圆锥面或圆柱面相切；
2. 球与固定截平面相切；
3. 橙色接触圆始终同时落在球面和当前母面上；
4. 两个黄色切点始终位于截平面上。

画面采用固定正交投影。每一帧都会通过仓库的解析截面合成器，把截平面拆成
`behind_surface`、`outside_projection`、`between_surface_sheets` 和
`in_front_of_surface` 四种深度区域，再把远球、近球和母面的前后层级合成到同一
绘制顺序中。灰蓝色/虚线表示被母面遮住的平面片段，青色/实线表示当前可见片段。

半透明填充仍是有几何证据的教学展示层，不宣称逐像素模拟了真实透明材质；解析
切触关系、平面分区和教学绘制顺序是每帧重新计算的。

## 低清预览

```bash
PYTHONPATH="$PWD" manim --renderer cairo --disable_caching -ql --fps 12 \
  --media_dir artifacts/dandelin-cone-cylinder-switch/preview \
  examples/dandelin_cone_cylinder_switch/dandelin_cone_cylinder_switch.py \
  DandelinConeCylinderSwitch
```

项目生产渲染契约是 Cairo；这个示例不使用 OpenGL。

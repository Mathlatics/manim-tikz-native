# Open-face automatic occlusion demo

This ordinary Manim scene registers two finite convex panels, their articulated
hinge, seven boundary edges, and one probe line. No per-line/per-face occlusion
relations are authored.

```bash
manim -pql examples/open_face_visibility/dihedral_auto_occlusion.py \
  DihedralAutoOcclusionDemo
```

During the fold, every registered line is compared with all eligible faces.
Visible and hidden spans are updated in stable solid/dashed slots, while the two
translucent face fills are ordered automatically.

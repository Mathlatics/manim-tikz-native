# Derived dihedral extraction demos

These scenes register one closed convex solid, copy two adjacent source faces,
and move the highlighted dihedral through and away from the original solid.
After separation, the rectangular-box and square-pyramid scenes rotate one
declared source face into the horizontal base position. The tetrahedron scene
deliberately isolates the copy handoff without a second rotation. The source
solid and copied dihedral use the source
solid's authored geometric center as the same center-relative rotation
definition. Each local placement moves that center with its entity, so the
source and copy rotate in place at opposite sides rather than orbiting one
fixed world-space pivot.
The source solid and copied dihedral participate in one global hidden-line
solve. `accurate_transparency=True` splits only finite face crossings into
independently sorted triangles, then batches consecutive triangles from the
same source face into one compound fill so their internal seams stay invisible.
The accurate mode also shares one painter graph across those face batches and
all visible/dashed line fragments. Lines are refined at local line/face depth
changes and projected line crossings, so the rendered Cairo pixels preserve
the same foreground/background result that the visibility solver computed.
Every automatically hidden blue or gold edge uses the same
`dash pattern=on 2pt off 2pt` rhythm after conversion into the demo's final
display coordinates. The dash phase remains anchored to the full semantic edge,
so an animated occlusion boundary clips the pattern instead of restarting it.

```bash
manim -pql examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  RectangularBoxDihedralDemo

manim -pql examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  TetrahedronDihedralDemo

manim -pql examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  SquarePyramidDihedralDemo

manim -pql examples/derived_dihedral_extraction/derived_dihedral_extraction_demo.py \
  RectangularBoxDihedralRoundTripDemo
```

The round-trip scene continues after the normal extraction sequence: it restores
the shared orientation and then moves the copied dihedral back onto its source.
This exercises the same geometry-driven identity handoff in reverse, so the
coincident final frame returns to one visible representation without a binary
opacity jump.

The first frame deliberately draws the highlighted copied dihedral in place of
the coincident source faces and edges.  This prevents double alpha blending and
duplicate edge pixels. As the copy starts moving, the paired source faces and
edges regain opacity through a short geometry-driven smoothstep rather than a
binary first-frame switch. Once the copy reaches the handoff distance, both
entities are rendered at full strength and occlude one another.

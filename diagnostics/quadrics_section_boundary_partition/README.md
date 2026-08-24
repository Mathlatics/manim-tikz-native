# Quadric section-boundary repair baseline

This diagnostic freezes the five pre-fix states used by the follow-up repair:

- `mainly_behind`
- `intersects`
- `near_tangent`
- `exact_parabola`
- `mainly_front`

It does not patch or replace production code. The capture records renderer-neutral
fragment counts, ray-classification counts, per-role screen areas, exact canonical
JSON, fixed Manim slot identity stability, and fill-only Cairo interior seam pixels.

Run from the repository root with the same Python environment used by Manim:

```bash
python diagnostics/quadrics_section_boundary_partition/capture.py \
  --output-dir /tmp/quadric-section-boundary-baseline
```

The Cairo metric deliberately hides strokes and semantic curves. Each role-union
mask is eroded by three pixels to exclude legitimate outer and role boundaries.
An interior pixel is then counted as a seam/deviation pixel when its RGB distance
from every legitimate flat Porter–Duff composite is greater than `8.0`. A region
painted with another legitimate flat role color is a painter-role error rather
than a triangulation seam, so it is deliberately not included in this count.

Python object addresses are process-local, so the committed evidence does not
pretend that raw `id()` values are portable. It records the fixed family size and
topology and verifies that every identity remains unchanged while the controller
updates through all five states.

# Semantic parallel-camera shot demos

These three short Cairo scenes exercise the renderer-neutral
`parallel-shot-sequence/v1` authoring contract and the Manim playback adapter.
All views remain parallel projections; perspective is deliberately out of
scope.

## Scenes

- `SemanticPlaneShotDemo` — an ordinary finite plane moves through
  `normal -> relative -> along -> return`.  The gold world-space point is the
  current `target`; the gold fixed-frame crosshair is its `screen_anchor`.
  The status label also shows `zoom`.
- `SingleConeSectionShotDemo` — a closed finite cone section follows
  `AREA -> LINE -> AREA`.  The plane fill disappears only in the certified
  exact edge-on frame, while the finite section remains as a line.
- `OpenDoubleSectionShotDemo` — an offset hyperbola on `OPEN_DOUBLE` follows
  `AREA -> LINE -> AREA`.  Surface sheets, section fragments, silhouettes,
  solid visible strokes, and dashed hidden strokes are recomputed from the
  same live camera state.

The two cone scenes pass `scene.camera` directly to their automatic-occlusion
controllers.  They do not maintain a second projection matrix, so camera
motion and visibility cannot silently drift apart.

## Low-cost Cairo previews

Run one scene at a time:

```bash
python -m manim --renderer cairo -ql -r 480,270 --fps 6 \
  examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
  SemanticPlaneShotDemo

python -m manim --renderer cairo -ql -r 480,270 --fps 6 \
  examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
  SingleConeSectionShotDemo

python -m manim --renderer cairo -ql -r 480,270 --fps 6 \
  examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
  OpenDoubleSectionShotDemo
```

For a faster functional dry run, lower `--fps` to `2`.  Keep
`--renderer=cairo`: the fixed-slot quadric bindings and these examples target
the Cairo renderer.

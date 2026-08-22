# Unified open-face Manim binding

The renderer-neutral open-face compositor computes a validated face/path
painter graph. The unified Manim binding maps that graph into a fixed Cairo
object pool without changing object topology during animation.

## Opt-in and compatibility

The default remains `compositing_mode="legacy"`. Unified mode is explicit:

```python
controller = scene_builder.controller(
    scene,
    projection=projection,
    style=style,
    compositing_mode="unified",
    paint_policy="diagrammatic",  # or "physical"
    painter_z_band=(20.0, 30.0),
)
```

This protects existing scenes which depend on authored face and line z-indices.
TikZ-native persistence and generated-source selection remain separate follow-up
work; this binding does not reinterpret old assets.

## Reserved painter band

Unified mode hides authored face fills and source strokes, then draws proxy
faces, solid fragments, and dashed fragments in one explicit z interval. Source
objects may share an authored z-index. The reserved interval must be finite,
increasing, and free of unrelated visible Scene drawables. It is revalidated on
reattach after a restore/detach cycle.

## Fixed slots and dash phase

Fragment identities are frame-local, so the binding maps each frame to
preallocated per-path slots in source-parameter order instead of treating a
`fragment_id` as permanent Mobject lineage. Hidden dash lines are preallocated
as well. Capacity limits cover fragment counts, dash counts, and the total proxy
Mobject count.

Dash phase remains anchored at source-path parameter zero; a new painter event
therefore clips the existing dash rhythm instead of restarting it.

## Transaction and last-good frame

A frame is fully computed and mapped before mutation. The runtime snapshots
managed proxy points, style arrays, opacity, z-index, and active painter-band
state. If any Manim setter or finalization callback fails, all managed state is
restored and the previous good frame remains current.

Authored source styles are restored by `restore()`/`detach()`. Reattachment
reuses the same preallocated proxy identities.

## Opacity lifecycle

While attached, animate `controller.display_mobject`, not the hidden authored
sources. Use `FadeOut(..., remover=False)` when the controller should remain
attached. An invisible opacity sentinel is the animation-owned value; the
updater reads its interpolated alpha and applies it to every face/path proxy, so
geometry updates do not cancel FadeIn/FadeOut.

## Scope

The binding supports Cairo, parallel projection, finite convex open faces,
straight semantic paths, and fixed topology. Persisted TikZ-native mode and a
self-contained generated-source runtime remain follow-up product work.

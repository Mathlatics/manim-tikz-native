# Parallel camera and section render sequences

`compile_parallel_section_sequence_from_shots()` is the renderer-neutral seam
between semantic parallel-camera shots and a certified `SectionTimeline`.  It
compiles the camera, cutting plane, topology handoff, semantic display state,
fixed bank capacity, cap-chord activation, painter order, and renderer affine
terms into one immutable sequence before a Scene is mutated.

Camera state, semantic shots, preflight, and frame transactions have their own
`parallel_camera_core` component identity.  The core has no TikZ compiler or
quadric dependency; the renderer-neutral quadric sequence depends on it, while
`MultiProjectionCamera` remains in the separate Manim-facing 3D motion
component.  This keeps cache invalidation aligned with the actual dependency
direction.

## Why the sequence has a render grid

Analytic section timelines contain exact key times, but a topology crossfade
also needs frames strictly inside its left and right windows.  A sequence may
therefore be compiled in either of two ways:

- omit `frame_rate` to get the smallest proof grid (all analytic keyframes plus
  one interior frame per non-empty crossfade side); or
- provide `frame_rate` to obtain Manim's real output-frame grid.  An analytic
  key time replaces a nearby nominal output time; it never inserts a physical
  frame.  Manim renders a play segment on `[0, run_time)`, so the final
  analytic endpoint replaces the last nominal output frame instead of becoming
  an extra, unrendered endpoint.

The camera sampler and real Manim playback share the named
`manim-smooth-v1` easing contract.  Intermediate offline camera evidence thus
matches `Scene.play`, not only its endpoints.

The provenance stores both grids one-to-one: `nominalFrameTimes` is the
renderer clock, while `evaluationTimes` contains the exact analytic times used
for geometry.  A critical time that differs from a shot boundary only by
floating-point roundoff is still assigned to that shot's exact endpoint.
Playback rejects a different live frame rate or shot source before `Scene.play`.
Absolute authoring times whose floating-point spacing is too coarse for the
local shot/frame duration are rejected instead of collapsing multiple frames.

```python
sequence = compile_parallel_section_sequence_from_shots(
    timeline,
    camera_shots,
    initial_camera,
    display_keyframes,
    limits=preflight_limits,
    painter_orders=painter_keyframes,
    semantic_bank_ids=("section-bank-a", "section-bank-b"),
    frame_rate=30,
    plane_patch_margin=0.08,
)
```

## Fixed semantic banks

`semantic_bank_ids` names two real `SECTION_CURVE` banks in the immutable
`SectionDisplayCatalog`; they are not synthetic preflight labels.  Every bank
must reserve both branch slots required by the section animation contract.
During a crossfade, `SectionBankRenderFrame` activates both banks in one
coordinated frame and records, per bank:

- the certified reference frame and geometry time;
- the semantic bank identity;
- the active branch count;
- the active isolated-point count;
- a digest of the freshly solved finite section geometry at `geometry_time`;
- the effective handoff opacity.

Each topology bank must own its own fixed slots for finite-surface `CAP_CHORD`
identities.  This lets both sides of a crossfade carry different cap states at
the same output frame.  Their per-bank capacity and analytic activation events
are included in the same preflight report.  Pure trim tangencies remain instant
cuts and use certified one-sided live geometry at the exact critical time,
avoiding a zero-length degenerate render.

## Finite plane patch and exact side view

`plane_patch_margin=None` means that no finite plane patch is drawn.  A finite
value uses the same analytic `fit_plane_display_patch()` contract as the Rig,
and the margin must be passed explicitly (the Rig default is `0.08`).  The
derived `FittedPlaneDisplayPatch` is serialized, channel-digested, and its four
world-space corners enter every safe-frame check.

When the camera looks exactly along the plane, those four corners naturally
project to the two endpoints of one finite line segment.  The preflight does
not invent thickness and does not treat the display patch as visibility truth.
If any `PLANE_FILL` or `PLANE_OUTLINE` slot is visible, `None` is rejected: a
visible plane must have a source-bound finite patch and runtime participant.

## Runtime gate and transaction order

Every runtime channel has a canonical SHA-256 digest embedded in its accepted
`ParallelPreflightFrame`.  Use the sequence-owned gate; constructing a gate
from only a report omits the section-specific digest functions.

```python
coordinator = ParallelFrameCoordinator()
gate = parallel_section_preflight_gate(sequence)
coordinator.add(gate.participant())
coordinator.add(parallel_screen_transform_guard(read_live_screen_transform))
coordinator.add(parallel_camera_frame_participant(camera))
coordinator.add(section_bank_frame_participant(bank_binding))
coordinator.add(section_plane_patch_participant(bank_binding))
coordinator.add(section_painter_order_participant(painter_binding))
coordinator.add(section_display_frame_participant(display_binding))

for frame in sequence.frames:
    coordinator.update(frame)
```

The bank and display bindings must each expose complete snapshot, apply, and
restore methods.  Preparation is read-only; if any commit partially mutates a
bank and then raises, the failing participant, the camera, and all earlier
participants are restored in reverse order.  A live inherited zoom, frame
center, or display offset that differs from preflight is rejected before any
commit.

The player verifies both participant identity and its audited binding kind and
phase; same-name no-op participants are not accepted.  Manim cache identity is
bound to the complete section-sequence digest and semantic segment identity.
Original shot `duration`/`hold` values are passed to `Scene.play` directly so
non-integral segment boundaries do not gain duplicate frames through cumulative
floating-point subtraction.

## Fail-closed boundaries

Compilation rejects, rather than guesses, when:

- the render grid omits an analytic key time or a crossfade interior;
- a display or painter keyframe changes inside an inserted crossfade frame
  without explicit render-frame values;
- a dynamic section slot is outside the two declared semantic banks;
- either bank has fewer than two fixed branch slots;
- cap-chord slots do not exactly match the finite-surface reservation;
- a finite plane patch differs from the source surface, evaluation plane, or
  explicit margin used by preflight;
- bank geometry is solved under a context or coefficient tolerance other than
  the policy serialized by `SectionTimeline`;
- painter evidence is empty;
- a runtime plane, display, bank, transition, or screen-transform channel no
  longer matches its accepted digest.

This contract remains parallel-projection-only.  Perspective cameras require
per-point view rays and are intentionally outside this sequence.

## One-controller Cairo binding

`compile_parallel_section_rig_from_shots()` performs a two-pass compile without
mutating the Scene.  The first pass fixes every non-painter input.  A single
unattached `QuadricOcclusion3D` then prepares its real numeric painter order for
each frame; the second pass binds that evidence into the final preflight report.
Only the final sequence can be attached and played.

Here `scene` must use `MultiProjectionCamera`.  Keep its inherited zoom at
`1`, inherited frame center at `(0, 0, 0)`, and the controller display offset
at `(0, 0)`; author target, final screen anchor, and semantic zoom in
`ParallelCameraState` so preflight and live rendering consume one affine state.

```python
catalog = build_parallel_section_rig_display_catalog(
    timeline,
    ("section-bank-a", "section-bank-b"),
    include_plane=True,
)
display = compile_section_display(
    catalog,
    SectionDisplayInstruction.for_mode("painted"),
)

binding = compile_parallel_section_rig_from_shots(
    scene,
    timeline,
    camera_shots,
    initial_camera,
    tuple(display for _ in timeline.samples),
    limits=preflight_limits,
    semantic_bank_ids=("section-bank-a", "section-bank-b"),
    frame_rate=30,
    plane_patch_margin=0.08,
)

binding.attach()
coordinator = binding.build_coordinator(scene.camera)
try:
    play_parallel_section_sequence(
        scene,
        binding.sequence,
        camera_shots,
        coordinator,
    )
finally:
    binding.restore()
```

Both topology banks use the same surface slots and the same managed painter
band.  A branch slot reserves two physical interval identities, so a periodic
seam can split without allocating a new Mobject.  A cap chord receives a
separate semantic identity in each bank.  Its live opacity is the product of
the bank handoff opacity and the semantic display opacity.

This first adapter excludes intrinsic surface silhouettes and cap rims from
the unified boundary solver.  Its `SURFACE_OUTLINE` role therefore uses the
controller's explicit `legacy_surface_stroke_fallback`: one static,
unoccluded teaching outline.  The controller default remains off, so existing
`include_surface_boundaries=False` callers do not acquire a new stroke.  A
future adapter that needs certified silhouette/cap-rim visibility must reserve
those semantic roles instead of treating this fallback as occlusion evidence.

The display participant is the only participant that calls
`QuadricOcclusion3D.update()`.  It verifies the committed z-order against the
preflight painter order.  Its transaction snapshot covers the complete Mobject
family, painter band, fragment maps, committed frames, last-input signatures,
and Cairo static image.  Pure geometry memoization may retain an exact-keyed
failed-frame entry, but it is not committed evidence and cannot change pixels.
A later participant failure therefore restores the whole last accepted frame
rather than only the visible points.

One attached binding owns one cached coordinator.  Repeated
`build_coordinator()` calls return that same object, so a stale second baseline
cannot roll back a newer committed frame.  Its participant set seals on the
first update and stays fixed across restored/re-attached playback sessions;
add any custom finalizer before the first frame.  `binding.restore()` first
restores an active coordinator, then releases the controller's Scene objects.

The initial binding deliberately fails before Scene ownership when:

- any frame activates an isolated `SECTION_POINT`; a true fixed point Mobject
  slot is required and a short line is not an acceptable substitute;
- a renderer-level `ParallelScreenTransform` is non-identity;
- surface or plane fill/outline opacity changes between frames (one constant
  multiplier per role is supported and is multiplied into the caller's
  `QuadricManimStyle`);
- generator, contour, or cap-rim semantic slots are requested.

These boundaries are explicit adapter limits, not limits of the
renderer-neutral sequence.  Ordinary finite conic branches, topology-bank
crossfades, cap chords, moving semantic cameras, finite plane patches, and an
exact rank-one side view are all handled by the one-controller binding.

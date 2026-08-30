# Source-authoritative projects

A source project keeps the authored TikZ file as the durable source of truth.
Optional motion JSON, parallel-camera shots, a Bridge request template, and
render intent are authored inputs as well. Python hooks are also optional, but
require a Bridge template because they are appended to generated Manim source.
ShapeAssets, compositing
plans, generated Manim source, and the build manifest are disposable outputs:
they may be deleted and rebuilt with the current Provider. Preview and final
media are downstream application outputs; this CLI neither creates nor removes
them.

This boundary avoids migrating disposable renderer output. It also prevents an
old implementation choice from becoming a manifest field or an automatic
fallback. Trusted authored Python hooks remain ordinary code and can explicitly
call lower-level or legacy APIs; projects should review them as source code.

## Project manifest

The public contract is
`tikz_native/schemas/tikz-native-source-project-v1.schema.json`. A minimal
project is:

```json
{
  "schemaVersion": "tikz-native-source-project/v1",
  "tikzSource": "figure.tex",
  "derivedOutput": ".tikz-native/derived",
  "renderIntent": {
    "paintPolicy": "diagrammatic",
    "projection": {
      "kind": "orthographic",
      "direction": [1, -1, -1]
    }
  }
}
```

All authored paths are relative to the manifest directory and must remain
inside it. `derivedOutput` is a dedicated build-owned location; do not put
authored files there.

The Provider writes `.tikz-native-owned.json` into that directory. A build may
create an absent directory or claim an existing empty directory. Before it
replaces a non-empty directory, or `clean` removes one, the Provider verifies
that the directory is not a symlink, the marker's manifest name and
project-relative output path match, and every entry is on the fixed
derived-output allowlist. A missing or mismatched marker on non-empty output,
or any unknown entry, makes the operation fail instead of deleting possibly
authored data. Because the identity is relative, the project and its derived
directory may be moved or copied together. The marker is internal build state
and must not be edited by hand.

The manifest deliberately does not persist `compositingMode`, a generated
ShapeAsset, or generated Python. Those describe a particular implementation,
not the mathematical source.

### Authored parallel-camera shots

A project may point to one renderer-neutral parallel-camera sequence:

```json
{
  "schemaVersion": "tikz-native-source-project/v1",
  "tikzSource": "figure.tex",
  "cameraShots": "camera-shots.json"
}
```

`camera-shots.json` must satisfy `parallel-shot-sequence/v1`. It stores complete
parallel-camera states—matrix, target, screen anchor, zoom, timing, transition,
and cue—rather than generated Manim code. The source file is authoritative. The
canonical `camera-shots.json` written under `derivedOutput`, the copy embedded
in generated source, and rendered media are disposable.

The first source-project slice deliberately rejects a manifest containing both
`cameraShots` and `motionJson`. Existing 3D motion documents can own camera
steps themselves, and two independent camera writers would make updater order
part of the result. A future coordinated scene-timeline contract may relax this
restriction; this build does not silently choose whichever writer runs last.

`renderIntent.projection` remains the static compositing entry/fallback
projection. An authored shot sequence is temporal data and changes the live
scene camera only when a host explicitly plays it. Source-project builds never
start playback as a side effect.

## Build and inspect

The installed CLI and its module form use the same implementation:

```bash
tikz-native-project build project.json
tikz-native-project status project.json
tikz-native-project rebuild project.json
tikz-native-project clean project.json

python -m tikz_native.source_project status project.json
```

Command JSON uses `resultFormat` value
`tikz-native-project-command-result/v1`. It is an operational result envelope,
not a `tikz-native-build-manifest/v1` document. Every non-error result writes
exactly one such JSON document to stdout; compiler and Manim diagnostics go to
stderr. `build`, `rebuild`, and `clean` return exit status 0 on success.
`status` returns 0 when everything is fresh and 1 when output is missing or
stale; both status results contain JSON. Invalid input or a build/safety failure
returns 2, writes no JSON to stdout, and explains the error on stderr.

The result envelope is operational and should not be persisted as author data.
All non-error variants contain `resultFormat` and `mode`. Build/rebuild add
`built`, `reused`, and `nodes`; status adds `fresh`, `manifestAction`, and
`nodes`; clean adds `removed`. Build/status variants also identify the project,
derived directory, and build-manifest path.

`build` reuses only outputs whose source fingerprints and relevant Provider
component revisions still match. `rebuild` ignores cache hits. `status` reports
fresh, stale, missing, and obsolete nodes plus a separate `manifestAction`,
without creating or changing the derived directory. `clean` is limited to
build-owned output and must not remove authored or unrelated files.

A build reads every authored input into one immutable snapshot, then holds the
project-directory lock while it constructs a complete sibling staging
directory. Output-parent components are traversed from the locked project
directory without following symlinks, and staged/cache files are written through
held directory descriptors. Path-backed compiler and Bridge inputs use one-file
system-temporary snapshots whose directory/file identities are held and checked;
their cleanup never recursively follows a mutable path. Immediately before and
again inside the reversible publication window, the build byte-compares all
inputs and staged node digests. Only an unchanged snapshot is published. Before
publication commits, an exception, failed parent-directory sync, or concurrent
source edit normally restores the previous output intact. New output names and
recovery names use no-replace renames, so a concurrently created directory is
preserved rather than overwritten. On macOS and Linux an existing output is
exchanged atomically; the portable fallback uses a rollback rename and therefore
has a brief namespace transition. If the filesystem operation used for rollback
itself fails, the command reports the exact internal sibling that still contains
the complete previous output. No inode is silently discarded, but the canonical
output name may then contain the uncommitted build or be absent and must be
restored manually before retrying.

All writers that rename the project directory or an ancestor of
`derivedOutput` must cooperate through the same project-directory lock. The
builder re-resolves and rechecks the named parent path around publication and
fails safely on detected replacement, while held directory descriptors keep
the transaction from escaping to a symlink target. No userspace sequence of
path checks can make an unrelated process's unsynchronised ancestor rename
atomic with publication, so such filesystem moves are outside the supported
concurrency contract.

`clean` first isolates and validates the complete owned output, then atomically
exchanges it with an empty sibling. Removing that empty directory is the sole
clean commit point, so a pre-commit failure never deletes only part of the old
output. Cleanup after a committed build or clean is best-effort. An interrupted
or concurrently changed filesystem may therefore leave an internal `.stage-*`,
`.rollback-*`, `.recovery-*`, `.concurrent-*`, or `.discard-*` sibling without
deleting the concurrent entry. After an ordinary failure the canonical output
still represents the previous build. After the separately reported double
rollback failure described above, treat the named recovery sibling—not the
canonical output name—as the previous build truth until manual recovery.

Depending on the project inputs, the derived directory can contain:

- `shape-asset.json`;
- `motion-asset.json`;
- `camera-shots.json`;
- `unified-compositing.json`;
- `generated_scene.py`;
- `build-manifest.json`.

`build-manifest.json` is evidence about a particular build, not an authored
project file. Its schema is packaged as
`tikz_native/schemas/tikz-native-build-manifest-v1.schema.json`.

## Component revisions and narrow invalidation

`source_project_build` owns the project orchestrator and both project schemas.
The generated open-face adapter has its own
`generated_open_face_visibility_3d` component. Each build node's cache key
incorporates only the relevant Provider component revisions that affect its
bytes; the build manifest records the union of revisions used by all present
nodes. As a result, a generated-source adapter change does not force an
unrelated ShapeAsset rebuild, while an orchestrator contract change cannot
silently reuse an output built under old rules.

The optional `camera_shots` node also records the `embedded_motion_3d`
component revision, because that component owns the strict
`parallel-shot-sequence/v1` parser and immutable camera-state implementation.
A camera-only source edit rebuilds `camera_shots`, `compositing`, and (when a
Bridge template is present) `generated_source`; it reuses `shape`.

Use `tikz-native health` to inspect the current component render and contract
revisions.

When `renderIntent.painterZBand` is omitted, the Provider hashes the TikZ bytes,
projection, `pictureIndex`, and `entryMacro` into one of 4096 deterministic
slots. Motion, paint-policy, and selection-only edits retain that figure's slot.
Each preferred slot is 1024 units wide and begins 2048 z-index units after the
previous slot, so different slots cannot overlap even though managed-band bounds
are inclusive. Generated scenes reserve the preferred band when the controller
attaches. If another generated controller already occupies it, including after
a true hash collision, the later controller is moved to the next available band
instead of overlapping it. An explicit `painterZBand` sets the preferred band
when authored figures need a coordinated starting range.

## Unified output, not automatic legacy fallback

Source-authoritative builds generate open-face scenes against the current
unified binding. If the current binding cannot be established, generation fails
closed. It does not silently emit or select the legacy compositor.

A Bridge request template is a JSON-object request seed, not a stored Bridge
response. It cannot contain `generatedSource`, Python/code fields, or generated
output. The builder always replaces `schema`, `operation`, `job_id`, and the
complete `input` object with the current source snapshot, hash, `pictureIndex`,
`entryMacro`, and Provider revision. A manifest `selection` uses the strict v3
fields and requires a Bridge template; it always overrides template selection,
while an absent manifest selection removes template selection. The local
`wholeFigureTargets` array controls Fade rewriting and is removed before the v3
request is validated. Exact whole-string template values may use
`${TIKZ_SOURCE}`, `${TIKZ_PATH}`, `${TIKZ_SHA256}`, `${MOTION_JSON}`,
`${HOOKS_SOURCE}`, `${PAINT_POLICY}`, `${PAINTER_Z_BAND}`,
`${PICTURE_INDEX}`, `${ENTRY_MACRO}`, or
`${EXPECTED_ASSET_PROVIDER_REVISION}`. Every invalidated generated-source build
calls the current v3 Bridge again. The derived build manifest records the
effective `pictureIndex`, `entryMacro`, and manifest `selection` as authoring
intent.

The unified AST rewrite runs before trusted hooks are appended inside a marked
hook block. Hook bytes are otherwise preserved, so comments, strings, and
ordinary hand-written animation code are not rewritten merely because they
mention a lower-level or legacy name.

When both a Bridge template and `cameraShots` are present, generated source
exports the immutable `TIKZ_NATIVE_CAMERA_SHOTS` sequence before the authored
hook block. Importing that module does not play the sequence or mutate a Scene.
Hooks and downstream hosts may consume the binding explicitly. In particular,
the build does not patch the older generated `local_camera_matrix` path: those
objects are already projected locally, so automatically applying a second
scene camera would be geometrically incorrect.

The low-level Manim binding retains an explicit legacy mode only so existing
hand-authored scenes can request their historical behavior. That compatibility
switch is not written into a source-project manifest and is not an automatic
fallback for new generated output.

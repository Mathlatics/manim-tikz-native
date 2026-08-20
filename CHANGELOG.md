# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- route the production closed-polyhedron `AutoOcclusion3D` solver through the
  shared `GeometryContext`, topology, and visibility layers while preserving
  the frozen v1 trace schema and interval classification;
- add an explicit tolerance-expanded visibility boundary mode for legacy trace
  compatibility without changing the exact-domain default used by new geometry;
- reject invalid, non-finite, or excessively deep PGF arithmetic with a stable
  `TikzNativeError` instead of leaking raw Python numeric exceptions, and
  recheck every TikZ length after unit conversion so line widths, radii,
  spacing, dashes, and arrow dimensions cannot become infinite;
- harden explicit TikZ line-versus-face occlusion with scale-aware local
  coordinates, normalized view rays, strict convex-face validation, and
  visible coplanar or boundary-only contacts; malformed, non-planar,
  degenerate, or non-convex faces now fail closed instead of making the whole
  semantic line appear visible;
- share that occlusion kernel with generated Manim v1/v2/v3 source so
  exported animations and the provider runtime cannot silently diverge;
- add differential regression coverage for generated/runtime parity, extreme
  scales, coplanar seams, malformed faces, and long-stroke/small-face cases;
- add a reusable one-solid-plus-one-derived-dihedral workflow for copying two
  adjacent faces, handing off coincident source pixels, and moving the copy as
  one rigid teaching object;
- replace the first-motion binary handoff with a smooth, geometry-driven
  activation envelope for the reappearing source faces and edges;
- extract that envelope into a reusable source-to-copy lineage contract for a
  whole registered solid or any copied face/stroke subset, while keeping the
  existing derived-dihedral authoring API compatible;
- solve source-solid and copied-dihedral semantic line visibility in one global
  frame, with stable visible/dashed slots and transactional restore;
- split intersecting translucent solid/copy faces into stable local triangles
  and sort only overlapping fragments from far to near;
- gate those splits by finite polygon crossings and batch consecutive
  same-source fragments into seam-free compound transparent fills;
- unify transparent face batches with visible and dashed stroke fragments in
  one deterministic Cairo painter graph, including line/face depth exchanges
  and projected line/line crossings;
- add synchronized base-plane rotation so a selected source face can become
  the horizontal bottom after the dihedral copy is separated;
- add rendered rectangular-box, tetrahedron, and square-pyramid examples,
  including one rectangular-box round trip that verifies the reverse identity
  handoff back to exact coincidence.

## 0.1.0 - 2026-08-18

Initial public alpha release.

- restricted semantic TikZ compiler and native 2D/3D Manim renderers;
- stable geometry relations and tracker-driven 2D/3D rigs;
- versioned bridge schemas and readable native Manim source generation;
- automatic hidden-line removal for closed convex polyhedra;
- automatic free-line/closed-convex-solid intersections with stable boundary
  markers;
- dynamic infinite-plane sections of closed convex solids with an automatically
  fitted, non-shrinking display patch, including stable point/segment/polygon
  transitions and global line occlusion;
- exact local transparent-fragment splitting and far-to-near compositing for
  one fitted plane patch intersecting one closed convex solid;
- reusable didactic face depth cues with continuous orientation tinting,
  distant-face fog, depth-aware opacity, and visible silhouette emphasis;
- automatic line occlusion and translucent face ordering for articulated open
  convex faces;
- portable TeX Live font defaults, Python packaging, examples, and CI.

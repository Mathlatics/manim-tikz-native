# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- use the classroom oblique-dimetric (`斜二测`) preset as the default
  multi-projection camera for ordinary polyhedra and a true orthographic
  isometric preset as the default Manim projection for quadrics and conic
  sections; explicit authored TikZ and caller-supplied projection matrices
  continue to take precedence;
- add analytic finite sphere, capped-cylinder, and one-nappe cone/frustum
  contracts under general parallel projection; solve and finitely trim circle,
  ellipse, parabola, hyperbola, and degenerate plane sections; partition
  semantic segments/arcs/conic branches at exact visibility and projected
  curve-crossing events; provide physical/diagrammatic painter policies, a
  fixed-capacity transactional Cairo binding, rotating-plane topology and
  moving-point traces, automatic fixed-capacity Manim handoff through exact
  ellipse/parabola/hyperbola events, automatic plane-patch fitting, and
  certified global ordering for a bounded set of strictly separated convex
  quadrics; add one-quadric/one-cutting-plane local compositing with adaptive
  rear/between/front patch partitioning, near-tangent feature guarding,
  seam-free contour merging, one managed painter band, fixed Manim identities,
  and transactional last-good-frame rollback; geometrically partition every
  plane fragment at outside/behind/between/front boundaries, certify the
  tangent-envelope approximation inside a bounded screen-space error band
  without raising the original 8192-fragment capacity, and add masked
  five-state Cairo color/seam, surface-opacity, outline, hidden-curve,
  high-resolution role-boundary, and continuous-motion regressions;
- preserve disjoint positive-winding pieces of one quadric section depth role
  near tangent events, certify each piece independently, and require exact
  combined-area recovery before publishing the frame;
- add a source-authoritative project format and the
  `tikz-native-project build/status/rebuild/clean` CLI: authored TikZ, optional
  motion/Bridge inputs, and render intent now deterministically regenerate
  disposable ShapeAsset, compositing, generated-Manim, and build-manifest
  outputs; cache reuse is component-revision scoped, authoritative JSON is
  strict, unified OpenFace generation fails closed, hidden stroke cap/join
  styles survive every binding path, and descriptor-based staged publication,
  rollback, and clean transactions preserve concurrent or unknown files;
- route the production closed-polyhedron `AutoOcclusion3D` solver through the
  shared `GeometryContext`, topology, and visibility layers while preserving
  the frozen v1 trace schema and interval classification;
- route open-face and section interval classification through the shared
  topology/visibility kernel, preserve dual face/logical-surface provenance,
  and replace their private whole-face and transparent-fragment graph sorts
  with the deterministic shared compositor;
- route the derived-dihedral unified face/stroke painter graph through the
  shared deterministic compositor while preserving its existing relation
  generator, lexicographic tie break, trace schema, and Cairo z-slot output;
- add a renderer-neutral open-face unified-compositing component that splits
  straight semantic paths at visibility, finite-face, and path-crossing painter
  events, preserves reverse-collinear parameter correspondence, emits validated
  face/path relations for diagrammatic or physical paint policy, and keeps the
  existing open-face v1 visibility trace unchanged; diagrammatic compositing
  consistently keeps semantic solid or dashed ink above overlapping face fill,
  while physical compositing follows actual depth;
- reuse each projected source-path intersection for fragmentation and painter
  relations, localize point events to adjacent fragments, sweep collinear
  overlap intervals in parameter order, and fail closed on an explicit
  fragment-pair-candidate limit instead of scanning every fragment pair;
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

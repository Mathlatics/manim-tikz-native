# Fast and extended quadric acceptance

The repository separates quick pull-request feedback from complete Cairo
release evidence.  The split changes when a test runs, not what the finite-cone
section contract promises.

## Pull-request CI

The ordinary `CI` workflow keeps the existing required check names
`test (3.11)` and `test (3.12)`.  Those jobs run the complete renderer-neutral
suite, deterministic parameter sweeps, Manim binding contracts, and the
320x180/480x270 Cairo pixel regressions.  A separate Python 3.12 smoke job
renders four fixed small animations, including the high-level section facade
and ellipse/parabola/hyperbola handoff.

The workflow runs once for each pull-request revision and once after the
accepted revision reaches `main`.  Pushes to a feature branch do not start a
second duplicate run alongside the pull-request run.

```bash
python scripts/run_ci_test_tier.py core
python scripts/run_ci_test_tier.py cairo-smoke
```

The exact deferred test identities are reviewed in
`.github/quadric-test-tiers.json`.  A direct `Scene.render()` test may not be
added without assigning it to either the small smoke set or the extended set.
Plain `python -m unittest discover` still runs every test for a deliberate
local full-suite check.

## Extended Cairo workflow

`Extended Quadric Acceptance` runs nightly, for a published release, or through
manual workflow dispatch.  It first executes the deferred high-resolution and
full-video tests, then generates a single evidence directory:

```bash
python scripts/run_ci_test_tier.py extended-cairo
python scripts/generate_quadric_extended_acceptance.py \
  --output /tmp/quadric-section-acceptance \
  --render-videos
```

The tier manifest also records the opt-in environment gates for the real 2D
and 3D motion-bridge renders.  The runner applies them before unittest
discovery, so `extended-cairo` cannot report success after silently skipping
those expensive render paths.

The workflow uploads only that explicit output directory.  It never uploads
the repository root or hidden Git metadata.

The bundle contains:

- 960x540 reviewed keyframes and contact sheets for closed/open cones,
  ellipse/parabola/hyperbola topology, the three hidden-curve policies,
  exact side-view trim rims, and cap-chord activation;
- complete MP4 scenes for all five scenarios;
- `evidence/acceptance.json`, including theoretical plane-depth roles, complete
  painter draw order, critical pixel coordinates, expected/actual RGB, and the
  applied tolerance;
- `evidence/fragment-ray-counts.csv` for both keyframes and every sampled
  motion frame;
- motion-sweep identity digests proving fixed Manim slots across the complete
  output-frame progress grid;
- per-scenario logs, SHA-256 artifact inventory, elapsed times, and explicit
  performance-budget results.

Whole-image hashes are deliberately not acceptance criteria.  The evidence
uses semantic interior probes and boundary-neighborhood probes so irrelevant
font or edge-antialiasing differences do not replace geometry review.

## Performance budgets

The versioned budgets and scenario grid live in
`tests/baselines/quadric-extended-acceptance-v1.json`.  The limits are broad
wall-clock guards against accidental complexity explosions, not benchmark
claims.  Exact fragment counts, ray counts, painter order, RGB evidence, and
fixed identities remain the primary acceptance data.

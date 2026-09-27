# Fast and extended quadric acceptance

The repository separates quick pull-request feedback from complete Cairo
release evidence.  The split changes when a test runs, not what the finite-cone
section contract promises.

## Pull-request CI

The ordinary `CI` workflow keeps the existing required check names
`test (3.11)` and `test (3.12)`.  Those jobs run the complete renderer-neutral
suite, deterministic parameter sweeps, Manim binding contracts, and the
320x180/480x270 Cairo pixel regressions. A separate Python 3.12 smoke job
renders a fixed small animation sample, including the high-level section
facade, mathematical-action Rig, and ellipse/parabola/hyperbola handoff.

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
Plain `python -m unittest discover` discovers every test, but it does not
enable the two opt-in real motion-render gates. For a deliberate local
full-suite check that includes those paths, use:

```bash
python scripts/run_ci_test_tier.py all
```

## Extended Cairo workflow

`Extended Quadric Acceptance` runs nightly, for a published release, or through
manual workflow dispatch. It first rebuilds and verifies the rolling
current-main release sidecar, then executes the deferred high-resolution and
full-video tests, and finally generates one evidence directory. The local
equivalent of the workflow order is:

```bash
python release/verify_quadric_section_release.py \
  --manifest release/quadric-section-v1-current-main-manifest.json \
  --evidence-json \
  /tmp/quadric-section-acceptance/evidence/release-manifest-verification.json \
  --failure-artifacts-directory \
  /tmp/quadric-section-acceptance/evidence/release-build-failure-artifacts

python scripts/run_ci_test_tier.py extended-cairo \
  --timings-json \
  /tmp/quadric-section-acceptance/evidence/extended-test-timings.json
python scripts/generate_quadric_extended_acceptance.py \
  --output /tmp/quadric-section-acceptance \
  --motion-sweep-workers 2 \
  --render-videos
```

The release verifier uses two builds by default. Reproduce the workflow with
the exact Python/build tool versions documented in
[`release/README.md`](../release/README.md); a different environment is useful
for diagnosis but is not release evidence.

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

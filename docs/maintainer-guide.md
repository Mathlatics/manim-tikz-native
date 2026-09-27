# Maintainer guide

This document describes the repository's actual development, CI, component
revision, and current-main evidence workflow. It complements the shorter
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Development environment

Use Python 3.11 or 3.12 for ordinary development; Python 3.12 is recommended.
The release and extended evidence environment is pinned more narrowly to
Python 3.12.13. The example below deliberately selects Python 3.12 rather than
whatever version a platform's unqualified `python3` happens to provide.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The Cairo jobs also need the TeX, FFmpeg, Cairo/Pango, and Poppler dependencies
listed in the [user guide](user-guide.md#1-choose-a-supported-environment) and
in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

Before changing code, verify the CLI contracts:

```bash
tikz-native health
tikz-native-rig-2d health
tikz-native-rig-3d health
tikz-native-source-v3 health
```

## Test tiers

Run the narrowest relevant unittest while editing, then choose the repository
tier that matches the change. These are alternatives, not a requirement to run
every line in sequence:

```bash
python -m unittest tests.test_relevant_module
python scripts/run_ci_test_tier.py core
python scripts/run_ci_test_tier.py cairo-smoke
python scripts/run_ci_test_tier.py extended-cairo
python scripts/run_ci_test_tier.py all
```

Use `--list` to inspect a tier without running it:

```bash
python scripts/run_ci_test_tier.py core --list
```

| Tier | Purpose |
| --- | --- |
| Targeted unittest | Fast feedback for the directly changed contract. |
| `core` | Ordinary PR suite: renderer-neutral tests, low-resolution regressions, package contracts, and deterministic sweeps. |
| `cairo-smoke` | Reviewed small real-Cairo movie set on Python 3.12. |
| `extended-cairo` | Deferred high-resolution and full-render tests; enables both real motion-render gates. |
| `all` | Every discovered test with the extended environment gates enabled. |

`python -m unittest discover` discovers every test but does not set the two
extended motion-render environment gates. It is therefore not equivalent to
`run_ci_test_tier.py all`.

Every new direct `Scene.render()` or movie test must be assigned to the
reviewed `cairo-smoke` or `extended-cairo` list in
`.github/quadric-test-tiers.json`. Do not silently add an expensive render to
ordinary PR feedback.

## Design invariants

- Keep the dependency direction Geometry → Topology → Visibility → Compositor
  → Manim bindings.
- Preserve stable semantic IDs, fixed Manim slot identity, deterministic JSON,
  and source restoration on normal or exceptional exit.
- Fail closed on unsupported geometry, capacity overflow, contradictory depth,
  or a stale source snapshot. Do not add SVG/bitmap or authored-z-index
  fallback paths.
- Keep the pure geometry/visibility layers independent of TikZ and Manim.
- Add renderer-neutral contract coverage and a real Manim/Cairo test when a
  runtime behavior changes.
- Document accepted TikZ syntax in `tikz_native/subset_v0_1.json` and
  `docs/supported-tikz.md`.
- Distinguish physical visibility, analytic curve visibility, and
  teaching-transparent painter order. A painter order is not automatically an
  optical or opaque hidden-surface claim.

## Component revision workflow

Persisted integrations use component-scoped implementation and contract
identities. A source edit that changes one component must not be hidden under
an old render/cache revision.

1. Make the implementation change and run the directly related tests.
2. Run the component revision test:

   ```bash
   python -m unittest tests.test_tikz_native_component_revisions
   ```

3. Inspect the recursively derived implementation revisions:

   ```bash
   python - <<'PY'
   from tikz_native.version import provider_component_implementation_revisions

   for name, revision in provider_component_implementation_revisions().items():
       print(name, revision)
   PY
   ```

4. For every affected component and dependent component, deliberately update
   the matching entries in:

   - `_UNRELEASED_COMPONENT_REVISIONS` in `tikz_native/version.py`;
   - `_DECLARED_IMPLEMENTATION_DIGESTS` in the same module; and
   - the expected current revisions in
     `tests/test_tikz_native_component_revisions.py`.

   The inspection command prints `component-sha256:<hex>`. Copy the same
   64-character `<hex>` in three different forms:

   - `_UNRELEASED_COMPONENT_REVISIONS`: `source-sha256:<hex>`;
   - `_DECLARED_IMPLEMENTATION_DIGESTS`: bare `<hex>`; and
   - the matching test constant: `source-sha256:<hex>`.

5. Rerun the component revision test and the relevant broader tiers.

Never rewrite `_PUBLIC_0_1_COMPONENT_REVISIONS`: it is the historical identity
of published `v0.1.1` bytes. Change a component **contract** revision only when
previously saved author data can no longer be read safely. An implementation
or rendering change normally refreshes the render/cache identity without
renaming the persisted contract.

## Package verification

The ordinary Python 3.12 CI job builds both distributions. Reproduce it with:

```bash
python -m build
python -m twine check dist/*
```

`README.md`, `README.zh-CN.md`, `CONTRIBUTING.md`, `docs/`, `examples/`,
`scripts/`, and `tests/` are included in the sdist through `MANIFEST.in`.
Consequently, a documentation-only change can leave the wheel unchanged while
still changing the normalized sdist hash and making current-main release
evidence stale.

## Pull-request checklist

An implementation or documentation PR should state:

- the intended change boundary and explicit non-goals;
- targeted, `core`, and any Cairo/Extended test results;
- whether a component implementation or contract revision changed;
- whether the wheel or sdist inputs changed;
- the relevant Cairo keyframes/evidence for a rendering change;
- documentation and `CHANGELOG.md` updates;
- whether a separate current-main evidence refresh is required after merge.

The ordinary CI jobs expected to pass are `test (3.11)`, `test (3.12)`, and
`cairo-smoke (3.12)`. The first two retain the existing branch-protection check
names; repository administrators must keep protection settings synchronized
with the workflow when changing required checks.

Use an ordinary merge commit for an attested implementation PR. The release
verifier records the reviewed implementation head and requires that head to be
an ancestor of the merge commit with the exact same Git tree. Squash or rebase
merging discards that reviewed-head relationship. An ordinary merge alone is
not enough if `main` advanced after the branch was reviewed: first integrate
the latest `main` into the branch, review and test that final head, and use that
updated head in the provenance record. Immediately after the merge, verify
both the ancestry and exact tree equality:

```bash
implementation_head=FINAL_REVIEWED_HEAD_SHA
merged_main=ORDINARY_MERGE_ON_MAIN_SHA
git merge-base --is-ancestor "$implementation_head" "$merged_main"
test \
  "$(git rev-parse "$implementation_head^{tree}")" = \
  "$(git rev-parse "$merged_main^{tree}")"
```

## Current-main evidence refresh

The repository has two sidecars with different lifetimes:

- `release/quadric-section-v1-release-manifest.json` is the frozen historical
  record for published `v0.1.1`; never relabel it as current work.
- `release/quadric-section-v1-current-main-manifest.json` follows the latest
  reviewed main-branch implementation.

Any merged change under package/production inputs—including README, docs,
examples, scripts, tests, and packaging metadata—makes the current-main
manifest stale. Refresh it in a separate evidence-only PR after the
implementation/docs merge and after main CI passes.

The evidence branch should change only the rolling current-main manifest. It
records:

- implementation base, reviewed head, implementation tree, and main merge;
- `build_artifacts.source_date_epoch`, exported by the verifier as
  `SOURCE_DATE_EPOCH`, from the implementation head;
- current component revisions;
- wheel and normalized-sdist SHA-256 values from two pinned builds; and
- evidence/fixture references that justify the changed support claims.

Install the exact release tooling before verification:

```bash
python -m pip install \
  build==1.6.0 \
  setuptools==84.0.0 \
  twine==7.0.0 \
  wheel==0.48.0
```

The verifier validates hashes already stored in the manifest; it does not fill
in hashes for a new build. Refresh them with this explicit probe-and-verify
sequence:

1. Update the provenance fields, artifact filenames if the package version
   changed, `build_artifacts.source_date_epoch`, current component revisions,
   and evidence/fixture references. Temporarily set both recorded SHA-256
   values to 64 zeroes so the probe must fail closed; never commit those
   placeholders.
2. Run one build with a fresh, empty failure-artifacts directory. A checksum
   failure is expected, and only that checksum failure is acceptable here:

   ```bash
   hash_probe_directory="$(mktemp -d /tmp/quadric-release-hash-probe.XXXXXX)"
   hash_probe_evidence=/tmp/quadric-release-hash-probe.json
   if python release/verify_quadric_section_release.py \
     --manifest release/quadric-section-v1-current-main-manifest.json \
     --build-count 1 \
     --evidence-json "$hash_probe_evidence" \
     --failure-artifacts-directory "$hash_probe_directory"
   then
     echo "error: the zero-hash probe unexpectedly passed" >&2
     exit 1
   fi

   python - "$hash_probe_evidence" <<'PY'
   import json
   import sys

   payload = json.load(open(sys.argv[1], encoding="utf-8"))
   assert payload["status"] == "fail"
   assert "SHA-256 does not match" in payload["error"]
   PY
   ```

3. Confirm that the directory contains a wheel, raw sdist, and normalized
   sdist. Compute the two hashes that the manifest records:

   ```bash
   ls "$hash_probe_directory"/run-1-*
   shasum -a 256 \
     "$hash_probe_directory"/run-1-*.whl \
     "$hash_probe_directory"/run-1-normalized-*.tar.gz
   ```

   On Linux, `sha256sum` is equivalent. Record the wheel digest under
   `build_artifacts.wheel.sha256` and the normalized tarball digest under
   `build_artifacts.sdist.sha256`; do not use the raw sdist digest.
4. Run the relevant evidence tests named by every new or changed fixture. The
   release verifier checks build and provenance data but does not resolve those
   test names on its own.
5. Run the clean two-build verification below. It must pass, and both runs must
   produce identical wheel and normalized-sdist digests:

```bash
verification_failure_directory="$(mktemp -d /tmp/quadric-release-final.XXXXXX)"
python release/verify_quadric_section_release.py \
  --manifest release/quadric-section-v1-current-main-manifest.json \
  --build-count 2 \
  --evidence-json /tmp/quadric-section-release-verification.json \
  --failure-artifacts-directory "$verification_failure_directory"
```

Before the evidence commit, the current-main drift in the verifier output
should be empty. After committing the sidecar, it must contain exactly
`release/quadric-section-v1-current-main-manifest.json`. Do not change the
recorded implementation merge to the evidence-only merge; that would make the
sidecar self-referential.

## Reproduce Extended Quadric Acceptance

The GitHub workflow executes these three stages in order:

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

The output includes semantic keyframes, contact sheets, painter traces, RGB
probes, motion sweeps, frame-performance traces, five MP4 scenes, logs, and a
SHA-256 inventory. Review both machine evidence and the rendered images/video;
a successful process exit alone is not visual acceptance.

After the evidence PR reaches `main`, manually dispatch `Extended Quadric
Acceptance` against the exact final main commit, download its uniquely named
artifact, verify the manifest hashes, and decode every MP4 with FFmpeg.

See [Fast and extended quadric acceptance](extended-quadric-ci.md) and
[Release evidence sidecars](../release/README.md) for the contract details.

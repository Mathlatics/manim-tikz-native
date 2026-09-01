# Contributing

Focused bug fixes and additions to the documented TikZ, geometry, visibility,
camera, and Manim contracts are welcome. Please keep the change boundary
explicit and include a minimal reproducer.

Read the [documentation index](docs/README.md) and the detailed
[maintainer guide](docs/maintainer-guide.md) before changing component
revisions or release evidence.

## Development setup

Project CI covers Python 3.11 and 3.12; Python 3.12 is recommended for local
development, while release evidence uses exactly Python 3.12.13. The command
below selects 3.12 explicitly so a newer system `python3` is not used by
accident.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The Cairo suites also require XeLaTeX/TikZ, Fandol and Latin Modern fonts,
`dvisvgm`, FFmpeg/ffprobe, Cairo/Pango, and Poppler. See the
[user guide](docs/user-guide.md#1-choose-a-supported-environment) for the
platform commands.

## Test the change

Start with the directly related unittest, then choose the reviewed tier that
matches the change. The following commands are alternatives rather than a
mandatory sequence:

```bash
python -m unittest tests.test_relevant_module
python scripts/run_ci_test_tier.py core
python scripts/run_ci_test_tier.py cairo-smoke
```

Use `python scripts/run_ci_test_tier.py all` for a final local full-suite
check. It enables the two opt-in real motion-render gates; plain unittest
discovery does not. Use `--list` to inspect a tier.

For package changes:

```bash
python -m build
python -m twine check dist/*
```

High-resolution keyframes and full videos run in the separate Extended Cairo
workflow. See [Fast and extended quadric acceptance](docs/extended-quadric-ci.md).

## Design rules

- Preserve `Geometry -> Topology -> Visibility -> Compositor -> Manim`
  dependency direction.
- Do not add an SVG, bitmap, or guessed-z-order fallback for unsupported input.
- Preserve stable semantic IDs, deterministic JSON, fixed Manim object
  identity, and transactional source restoration.
- Keep pure geometry and visibility code independent of TikZ and Manim.
- Add a renderer-neutral contract test and a real Manim/Cairo test for new
  runtime behavior.
- Document accepted syntax in both `tikz_native/subset_v0_1.json` and
  `docs/supported-tikz.md`.
- Refresh affected unreleased component revisions deliberately; never relabel
  the frozen public v0.1.1 revisions.
- Assign every direct movie render to `cairo-smoke` or `extended-cairo`.
- State whether a result is physical visibility, certified curve visibility,
  or teaching-transparent painter order.

## Pull request

Please include:

- the intended scope and non-goals;
- the exact tests and renders run;
- component revision and persisted-contract impact;
- wheel/sdist impact;
- keyframes or evidence for a Cairo rendering change;
- documentation/CHANGELOG updates; and
- whether a separate current-main evidence refresh is required after merge.

The current-main release manifest attests an exact reviewed head and Git tree.
Attested implementation PRs use an ordinary merge commit, followed by a
separate evidence-only refresh after main CI passes. README and docs are sdist
inputs, so a documentation change also refreshes the normalized-sdist evidence.
The complete procedure is in the [maintainer guide](docs/maintainer-guide.md#current-main-evidence-refresh).

Compiler changes should include a minimal `.tex` fixture. Security reports
should follow [SECURITY.md](SECURITY.md), not a public issue.

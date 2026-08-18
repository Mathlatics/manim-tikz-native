# Contributing

Bug reports and focused additions to the documented TikZ subset are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
```

## Design rules

- Do not add an SVG or bitmap fallback for unsupported TikZ.
- Preserve stable semantic object IDs and deterministic JSON output.
- Keep the pure visibility solver independent of TikZ and Manim.
- Add a strict contract test and a real Manim test for new runtime behavior.
- Dynamic bindings must preserve object identity and restore source state on
  normal and exceptional exit.
- Document new accepted syntax in `tikz_native/subset_v0_1.json`.
- Component revision changes must be deliberate and covered by revision tests.

Please include a minimal `.tex` fixture when changing the compiler.

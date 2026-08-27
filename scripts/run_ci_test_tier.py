#!/usr/bin/env python3
"""Run one deterministic unittest tier from the repository CI manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import unittest
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / ".github" / "quadric-test-tiers.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _flatten(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


@dataclass(frozen=True, slots=True)
class TierSelection:
    tier: str
    tests: tuple[unittest.TestCase, ...]
    all_ids: tuple[str, ...]
    tier_ids: dict[str, tuple[str, ...]]


def load_tier_manifest(path: Path = MANIFEST_PATH) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError("CI tier manifest must contain one JSON object")
    if value.get("schema") != "manim-tikz-native-ci-test-tiers/v1":
        raise ValueError("unsupported CI tier manifest schema")
    if value.get("default_tier") != "core":
        raise ValueError("the default CI tier must remain 'core'")
    tiers = value.get("tiers")
    if not isinstance(tiers, dict) or set(tiers) != {
        "cairo-smoke",
        "extended-cairo",
    }:
        raise ValueError("CI tiers must define cairo-smoke and extended-cairo")
    environments = value.get("tier_environment")
    if not isinstance(environments, dict) or set(environments) != {
        "extended-cairo"
    }:
        raise ValueError("CI tier environment must define extended-cairo")
    extended_environment = environments["extended-cairo"]
    if extended_environment != {
        "RUN_TIKZ_NATIVE_MOTION_3D_RENDER_TEST": "1",
        "RUN_TIKZ_NATIVE_MOTION_RENDER_TEST": "1",
    }:
        raise ValueError(
            "extended-cairo must enable both real motion-render test gates"
        )
    return value


def _apply_tier_environment(
    manifest: dict[str, object],
    tier: str,
) -> None:
    if tier not in {"extended-cairo", "all"}:
        return
    for name, value in manifest["tier_environment"]["extended-cairo"].items():
        os.environ[str(name)] = str(value)


def select_tier(
    tier: str,
    *,
    manifest_path: Path = MANIFEST_PATH,
) -> TierSelection:
    manifest = load_tier_manifest(manifest_path)
    _apply_tier_environment(manifest, tier)
    discovered = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern="test_*.py"
    )
    tests = tuple(_flatten(discovered))
    all_ids = tuple(item.id() for item in tests)
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("unittest discovery produced duplicate identities")
    all_id_set = set(all_ids)
    tiers = {
        name: tuple(str(item) for item in values)
        for name, values in manifest["tiers"].items()
    }
    assigned: dict[str, str] = {}
    for name, identities in tiers.items():
        if len(identities) != len(set(identities)):
            raise ValueError(f"tier {name!r} contains duplicate test identities")
        missing = tuple(item for item in identities if item not in all_id_set)
        if missing:
            raise ValueError(
                f"tier {name!r} names missing tests: {', '.join(missing)}"
            )
        for identity in identities:
            previous = assigned.setdefault(identity, name)
            if previous != name:
                raise ValueError(
                    f"test {identity!r} belongs to both {previous!r} and {name!r}"
                )
    if tier == "all":
        selected = tests
    elif tier == "core":
        selected = tuple(item for item in tests if item.id() not in assigned)
    elif tier in tiers:
        selected_ids = set(tiers[tier])
        selected = tuple(item for item in tests if item.id() in selected_ids)
    else:
        raise ValueError(
            f"unknown tier {tier!r}; choose core, cairo-smoke, extended-cairo, or all"
        )
    if not selected:
        raise RuntimeError(f"test tier {tier!r} resolved to an empty suite")
    return TierSelection(tier, selected, all_ids, tiers)


class _TimingResult(unittest.TextTestResult):
    def startTest(self, test: unittest.TestCase) -> None:  # noqa: N802
        self._test_started = perf_counter()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:  # noqa: N802
        elapsed = perf_counter() - self._test_started
        self.timings.append({"test": test.id(), "seconds": round(elapsed, 6)})
        super().stopTest(test)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timings: list[dict[str, object]] = []


class _TimingRunner(unittest.TextTestRunner):
    resultclass = _TimingResult


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tier",
        choices=("core", "cairo-smoke", "extended-cairo", "all"),
    )
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--failfast", action="store_true")
    parser.add_argument("--timings-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selection = select_tier(args.tier)
    if args.list_only:
        for test in selection.tests:
            print(test.id())
        print(
            json.dumps(
                {
                    "tier": selection.tier,
                    "selected": len(selection.tests),
                    "discovered": len(selection.all_ids),
                },
                sort_keys=True,
            )
        )
        return 0
    suite = unittest.TestSuite(selection.tests)
    started = perf_counter()
    result = _TimingRunner(
        verbosity=args.verbosity,
        failfast=args.failfast,
    ).run(suite)
    elapsed = perf_counter() - started
    summary = {
        "schema": "manim-tikz-native-ci-test-timing/v1",
        "tier": selection.tier,
        "discovered": len(selection.all_ids),
        "selected": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_seconds": round(elapsed, 6),
        "tests": result.timings,
    }
    if args.timings_json is not None:
        args.timings_json.parent.mkdir(parents=True, exist_ok=True)
        args.timings_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in summary.items() if key != "tests"}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

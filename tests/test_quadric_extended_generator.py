from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import generate_quadric_extended_acceptance as generator


def _scenario(
    scenario_id: str,
    samples: int,
    *critical: float,
) -> dict[str, object]:
    return {
        "id": scenario_id,
        "motion_samples": samples,
        "critical_progresses": list(critical),
    }


class _FakeProcessPoolExecutor:
    last_instance: "_FakeProcessPoolExecutor | None" = None

    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.tasks: tuple[generator._MotionSweepTask, ...] = ()
        type(self).last_instance = self

    def __enter__(self) -> "_FakeProcessPoolExecutor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def map(self, function, tasks):
        self.tasks = tuple(tasks)
        return tuple(function(task) for task in self.tasks)


class ExtendedAcceptanceGeneratorTests(unittest.TestCase):
    def test_parallel_sweeps_use_bounded_spawn_workers_and_preserve_order(
        self,
    ) -> None:
        scenarios = (
            _scenario("first", 35),
            _scenario("second", 49, 0.5),
            _scenario("third", 35),
        )
        spawn_context = object()

        def execute(task: generator._MotionSweepTask):
            return ({"scenario_id": task.scenario_id}, ())

        with TemporaryDirectory() as directory, patch.object(
            generator,
            "ProcessPoolExecutor",
            _FakeProcessPoolExecutor,
        ), patch.object(
            generator,
            "get_context",
            return_value=spawn_context,
        ), patch.object(
            generator,
            "_execute_motion_sweep",
            side_effect=execute,
        ):
            results = generator._run_motion_sweeps(
                scenarios,
                output=Path(directory),
                workers=2,
            )

        pool = _FakeProcessPoolExecutor.last_instance
        assert pool is not None
        self.assertEqual(pool.max_workers, 2)
        self.assertIs(pool.mp_context, spawn_context)
        self.assertEqual(
            tuple(task.scenario_id for task in pool.tasks),
            ("first", "second", "third"),
        )
        self.assertEqual(
            tuple(item[0]["scenario_id"] for item in results),
            ("first", "second", "third"),
        )
        self.assertEqual(pool.tasks[1].critical_progresses, (0.5,))

    def test_worker_persists_success_evidence_and_counts(self) -> None:
        sweep = {
            "scenario_id": "side-view",
            "sample_count": 35,
            "elapsed_seconds": 1.25,
        }
        with TemporaryDirectory() as directory, patch.object(
            generator,
            "_motion_sweep",
            return_value=(sweep, ()),
        ):
            output = Path(directory)
            result = generator._execute_motion_sweep(
                generator._MotionSweepTask(
                    "side-view",
                    35,
                    str(output),
                    (),
                )
            )
            evidence = json.loads(
                (
                    output
                    / "evidence"
                    / "motion-sweeps"
                    / "side-view.json"
                ).read_text(encoding="utf-8")
            )
            counts = (
                output / "evidence" / "motion-sweeps" / "side-view.csv"
            ).read_text(encoding="utf-8")

        self.assertEqual(result, (sweep, ()))
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["sweep"], sweep)
        self.assertIn("ray_classification_count", counts.splitlines()[0])

    def test_worker_persists_failure_before_reraising(self) -> None:
        with TemporaryDirectory() as directory, patch.object(
            generator,
            "_motion_sweep",
            side_effect=RuntimeError("synthetic sweep failure"),
        ):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "synthetic sweep failure"):
                generator._execute_motion_sweep(
                    generator._MotionSweepTask(
                        "failing",
                        35,
                        str(output),
                        (),
                    )
                )
            evidence = json.loads(
                (
                    output
                    / "evidence"
                    / "motion-sweeps"
                    / "failing.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(evidence["status"], "failed")
        self.assertIn("synthetic sweep failure", evidence["error"])

    def test_worker_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker count must be positive"):
            generator._run_motion_sweeps((), output=Path("unused"), workers=0)
        with self.assertRaisesRegex(ValueError, "must not exceed 2"):
            generator._run_motion_sweeps((), output=Path("unused"), workers=3)

    @unittest.skipUnless(
        os.environ.get("RUN_QUADRIC_MOTION_SWEEP_PROCESS_SMOKE") == "1",
        "real spawned Manim workers are an explicit acceptance smoke",
    )
    def test_real_spawned_workers_complete_two_small_managed_sweeps(self) -> None:
        scenarios = (
            _scenario("hidden_curve_policies", 2),
            _scenario("cap_chord_activation", 2),
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            results = generator._run_motion_sweeps(
                scenarios,
                output=output,
                workers=2,
            )
            evidence_files = tuple(
                sorted((output / "evidence" / "motion-sweeps").glob("*.json"))
            )

        self.assertEqual(
            tuple(item[0]["scenario_id"] for item in results),
            ("hidden_curve_policies", "cap_chord_activation"),
        )
        self.assertEqual(len(evidence_files), 2)


if __name__ == "__main__":
    unittest.main()

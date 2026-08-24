"""Compact motion-only Cairo evidence for the PR #12 production updater."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from manim import Scene, ValueTracker, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as diagnostic  # noqa: E402


class CairoConeDiagnosticVideoFast(Scene):
    """Traverse all five exact states with a small number of real Cairo frames."""

    def construct(self) -> None:
        self.camera.background_color = diagnostic.BACKGROUND_COLOR
        mode = os.environ.get("PR12_DIAGNOSTIC_MODE", "translucent")
        progress = ValueTracker(0.0)
        diagnostic.build_controller(
            self,
            progress.get_value,
            mode,
        ).attach()
        self.wait(1.0 / 6.0)
        for index in range(1, len(diagnostic.STATES)):
            self.play(
                progress.animate.set_value(float(index)),
                run_time=0.50,
                rate_func=linear,
            )
            self.wait(1.0 / 6.0)


__all__ = ["CairoConeDiagnosticVideoFast"]

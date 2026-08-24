"""Nine-frame stepped Cairo motion evidence for the merged PR #12 binding."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
from manim import Scene, ValueTracker

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene as diagnostic  # noqa: E402


class CairoConeDiagnosticVideoStepped(Scene):
    """Render nine real production updates from mainly-behind to mainly-front."""

    def construct(self) -> None:
        self.camera.background_color = diagnostic.BACKGROUND_COLOR
        mode = os.environ.get("PR12_DIAGNOSTIC_MODE", "translucent")
        progress = ValueTracker(0.0)
        controller = diagnostic.build_controller(
            self,
            progress.get_value,
            mode,
        ).attach()
        for value in np.linspace(0.0, float(len(diagnostic.STATES) - 1), 9):
            progress.set_value(float(value))
            controller.update()
            self.wait(1.0 / 3.0)


__all__ = ["CairoConeDiagnosticVideoStepped"]

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tikz_native import compile_document
from tikz_native.occlusion_3d import (
    parallel_occlusion_interval,
    parallel_view_direction,
    standalone_occlusion_source,
)


FACE = np.asarray(
    [
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=float,
)


class TikzNativeOcclusion3DTests(unittest.TestCase):
    def test_face_in_front_hides_only_the_projected_interior(self) -> None:
        interval = parallel_occlusion_interval(
            (-2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            FACE,
            (0.0, 0.0, 1.0),
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        np.testing.assert_allclose(interval, (0.25, 0.75), atol=1.0e-7)

    def test_coplanar_and_near_coplanar_contacts_remain_visible(self) -> None:
        self.assertIsNone(
            parallel_occlusion_interval(
                (-2.0, 0.0, 1.0),
                (2.0, 0.0, 1.0),
                FACE,
                (0.0, 0.0, 1.0),
            )
        )
        self.assertIsNone(
            parallel_occlusion_interval(
                (-2.0, 0.0, 1.0 - 1.0e-12),
                (2.0, 0.0, 1.0 - 1.0e-12),
                FACE,
                (0.0, 0.0, 1.0),
            )
        )

    def test_view_direction_magnitude_does_not_change_the_interval(self) -> None:
        intervals = [
            parallel_occlusion_interval(
                (-2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                FACE,
                (0.0, 0.0, magnitude),
            )
            for magnitude in (1.0e-300, 1.0, 1.0e300)
        ]
        self.assertTrue(all(item is not None for item in intervals))
        first = np.asarray(intervals[0], dtype=float)
        for interval in intervals[1:]:
            np.testing.assert_allclose(interval, first, atol=1.0e-14)

    def test_uniform_model_scaling_is_stable(self) -> None:
        intervals = []
        for scale in (1.0e-6, 1.0, 1.0e6):
            intervals.append(
                parallel_occlusion_interval(
                    np.asarray((-2.0, 0.0, 0.0)) * scale,
                    np.asarray((2.0, 0.0, 0.0)) * scale,
                    FACE * scale,
                    (0.0, 0.0, 1.0),
                )
            )
        self.assertTrue(all(item is not None for item in intervals))
        baseline = np.asarray(intervals[1], dtype=float)
        for interval in intervals:
            np.testing.assert_allclose(
                interval,
                baseline,
                atol=5.0e-8,
                rtol=0.0,
            )

    def test_tiny_face_is_not_lost_on_a_very_long_stroke(self) -> None:
        interval = parallel_occlusion_interval(
            (-5.0e7, 0.0, 0.0),
            (5.0e7, 0.0, 0.0),
            [
                (-0.5, -1.0, 1.0),
                (0.5, -1.0, 1.0),
                (0.5, 1.0, 1.0),
                (-0.5, 1.0, 1.0),
            ],
            (0.0, 0.0, 1.0),
        )
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertGreater(interval[1] - interval[0], 5.0e-9)
        self.assertAlmostEqual((interval[0] + interval[1]) / 2.0, 0.5, places=12)

    def test_finite_convex_polygons_work_and_invalid_faces_fail_closed(self) -> None:
        pentagon = [
            (-1.0, -1.0, 1.0),
            (0.5, -1.2, 1.0),
            (1.4, 0.0, 1.0),
            (0.3, 1.3, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        self.assertIsNotNone(
            parallel_occlusion_interval(
                (-2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                pentagon,
                (0.0, 0.0, 1.0),
            )
        )

        nonconvex = [
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (0.0, -0.2, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        self.assertIsNone(
            parallel_occlusion_interval(
                (-2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                nonconvex,
                (0.0, 0.0, 1.0),
            )
        )
        self.assertIsNone(
            parallel_occlusion_interval(
                (-2.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                FACE,
                (0.0, 0.0, float("nan")),
            )
        )

    def test_projection_direction_is_scale_invariant_and_validated(self) -> None:
        np.testing.assert_allclose(
            parallel_view_direction(np.identity(3) * 1.0e-300),
            (0.0, 0.0, 1.0),
            atol=1.0e-14,
        )
        with self.assertRaisesRegex(ValueError, "linearly dependent"):
            parallel_view_direction(
                ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0))
            )
        with self.assertRaisesRegex(ValueError, "depth direction"):
            parallel_view_direction(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            parallel_view_direction(
                ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0))
            )

    def test_generated_kernel_is_self_contained_and_matches_runtime(self) -> None:
        namespace: dict[str, object] = {}
        source = (
            "from __future__ import annotations\n"
            "import numpy as np\n"
            + standalone_occlusion_source()
        )
        exec(compile(source, "<standalone-occlusion-3d>", "exec"), namespace)
        generated_interval = namespace["parallel_occlusion_interval"]
        generated_view = namespace["parallel_view_direction"]

        matrix = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=float,
        )
        np.testing.assert_allclose(
            generated_view(matrix),
            parallel_view_direction(matrix),
            atol=1.0e-14,
        )
        cases = (
            ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), FACE),
            ((-2.0, 0.0, 1.0), (2.0, 0.0, 1.0), FACE),
            (
                (-5.0e7, 0.0, 0.0),
                (5.0e7, 0.0, 0.0),
                np.asarray(
                    (
                        (-0.5, -1.0, 1.0),
                        (0.5, -1.0, 1.0),
                        (0.5, 1.0, 1.0),
                        (-0.5, 1.0, 1.0),
                    ),
                    dtype=float,
                ),
            ),
        )
        for start, end, face in cases:
            self.assertEqual(
                generated_interval(start, end, face, (0.0, 0.0, 1.0)),
                parallel_occlusion_interval(start, end, face, (0.0, 0.0, 1.0)),
            )

    def test_compiler_keeps_a_coplanar_relation_as_one_visible_stroke(self) -> None:
        source = r"""
\begin{tikzpicture}[
  x={(1cm,0cm)},
  y={(0cm,1cm)},
  z={(0cm,0cm)},
  edge/.style={black,thick},
  hidden/.style={black,densely dashed,thin}
]
  \coordinate (S) at (-2,0,0);
  \coordinate (E) at (2,0,0);
  \coordinate (A) at (-1,-1,0);
  \coordinate (B) at (1,-1,0);
  \coordinate (C) at (1,1,0);
  \coordinate (D) at (-1,1,0);
  \DrawSpaceLineBehindParallelogramFace[edge][hidden]
    {S}{E}{A}{B}{C}{D}
\end{tikzpicture}
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "coplanar-occlusion.tex"
            path.write_text(source, encoding="utf-8")
            picture = compile_document(path).pictures[0]

        self.assertFalse(picture.unsupported)
        self.assertEqual(len(picture.occlusion_relations), 1)
        relation = picture.occlusion_relations[0]
        self.assertEqual(len(relation.object_ids), 1)
        visibility = [
            next(
                item
                for item in picture.objects
                if item.id == object_id
            ).geometry["visibility"]
            for object_id in relation.object_ids
        ]
        self.assertEqual(visibility, ["visible"])


if __name__ == "__main__":
    unittest.main()

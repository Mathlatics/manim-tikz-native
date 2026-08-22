from __future__ import annotations

from dataclasses import replace
import unittest

from polyhedron_visibility.open_faces.contract import (
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceVisibilityModel,
)
from polyhedron_visibility.open_faces.unified_compositing import (
    OpenFaceUnifiedCompositingError,
    OpenFaceUnifiedCompositingLimits,
    compute_open_face_unified_compositing,
)
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind
from tests.test_open_face_unified_compositing import (
    IDENTITY_VIEW,
    _crossing_paths_model,
    _panel_probe_model,
    _parallel_faces_model,
)
from tikz_native.version import (
    COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING,
    provider_component_revisions,
)


def _view_collapsed_path_model() -> OpenFaceVisibilityModel:
    return OpenFaceVisibilityModel.from_dict(
        {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
            "visibilityGroupId": "view-collapsed-path",
            "vertices": [
                {"vertexId": "P0", "entryPosition": [0.0, 0.0, 0.0]},
                {"vertexId": "P1", "entryPosition": [0.0, 0.0, 1.0]},
                {"vertexId": "A", "entryPosition": [5.0, 5.0, 0.0]},
                {"vertexId": "B", "entryPosition": [6.0, 5.0, 0.0]},
                {"vertexId": "C", "entryPosition": [6.0, 6.0, 0.0]},
                {"vertexId": "D", "entryPosition": [5.0, 6.0, 0.0]},
            ],
            "faces": [
                {
                    "faceId": "remote-panel",
                    "logicalSurfaceId": "remote-surface",
                    "vertexIds": ["A", "B", "C", "D"],
                }
            ],
            "seams": [],
            "strokes": [
                {
                    "sourceEdgeId": "collapsed",
                    "vertexIds": ["P0", "P1"],
                }
            ],
        }
    )


class OpenFaceUnifiedContractHardeningTests(unittest.TestCase):
    def _valid_panel_frame(self):
        return compute_open_face_unified_compositing(
            _panel_probe_model(),
            projection_matrix=IDENTITY_VIEW,
        )

    def test_faces_must_match_visibility_identity_and_canonical_order(self) -> None:
        valid = self._valid_panel_frame()
        wrong = replace(
            valid.faces[0],
            item_id="face:wrong",
            face_id="wrong",
        )
        with self.assertRaisesRegex(ValueError, "exactly match visibility"):
            replace(valid, faces=(wrong,))

        multi = compute_open_face_unified_compositing(
            _parallel_faces_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        with self.assertRaisesRegex(ValueError, "canonical order"):
            replace(multi, faces=tuple(reversed(multi.faces)))

    def test_every_visibility_path_requires_painter_fragments(self) -> None:
        valid = self._valid_panel_frame()
        with self.assertRaisesRegex(ValueError, "missing visibility paths"):
            replace(valid, path_fragments=())

    def test_fragment_partition_rejects_gaps_overlaps_and_bad_order(self) -> None:
        valid = self._valid_panel_frame()
        self.assertGreaterEqual(len(valid.path_fragments), 3)
        first, second, *tail = valid.path_fragments

        gap_first = replace(
            first,
            parameter_interval=ParameterInterval(
                first.parameter_interval.start,
                first.parameter_interval.end - 0.05,
            ),
        )
        with self.assertRaisesRegex(ValueError, "contains a gap"):
            replace(
                valid,
                path_fragments=(gap_first, second, *tail),
            )

        overlap_first = replace(
            first,
            parameter_interval=ParameterInterval(
                first.parameter_interval.start,
                second.parameter_interval.start + 0.05,
            ),
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            replace(
                valid,
                path_fragments=(overlap_first, second, *tail),
            )

        with self.assertRaisesRegex(ValueError, "canonical source and parameter"):
            replace(valid, path_fragments=tuple(reversed(valid.path_fragments)))

    def test_fragment_visibility_and_occluders_must_match_trace(self) -> None:
        valid = self._valid_panel_frame()
        hidden_index = next(
            index
            for index, fragment in enumerate(valid.path_fragments)
            if fragment.visibility_kind is VisibilityKind.HIDDEN
        )
        hidden = valid.path_fragments[hidden_index]

        visible_copy = replace(
            hidden,
            visibility_kind=VisibilityKind.VISIBLE,
            occluder_face_ids=(),
            occluder_logical_surface_ids=(),
        )
        wrong_kind = list(valid.path_fragments)
        wrong_kind[hidden_index] = visible_copy
        with self.assertRaisesRegex(ValueError, "visibility kind disagrees"):
            replace(valid, path_fragments=tuple(wrong_kind))

        wrong_occluder = replace(
            hidden,
            occluder_face_ids=("wrong-face",),
            occluder_logical_surface_ids=("wrong-surface",),
        )
        wrong_provenance = list(valid.path_fragments)
        wrong_provenance[hidden_index] = wrong_occluder
        with self.assertRaisesRegex(ValueError, "occluders disagree"):
            replace(valid, path_fragments=tuple(wrong_provenance))

    def test_relations_must_use_canonical_identity_order(self) -> None:
        valid = compute_open_face_unified_compositing(
            _crossing_paths_model(),
            projection_matrix=IDENTITY_VIEW,
        )
        self.assertGreater(len(valid.order_relations), 1)
        with self.assertRaisesRegex(ValueError, "canonical identity order"):
            replace(
                valid,
                order_relations=tuple(reversed(valid.order_relations)),
            )

    def test_view_collapsed_path_fails_closed_instead_of_disappearing(self) -> None:
        with self.assertRaisesRegex(
            OpenFaceUnifiedCompositingError,
            "produced no painter fragments",
        ):
            compute_open_face_unified_compositing(
                _view_collapsed_path_model(),
                projection_matrix=IDENTITY_VIEW,
            )

    def test_fragment_face_candidate_budget_fails_before_relations(self) -> None:
        with self.assertRaisesRegex(
            OpenFaceUnifiedCompositingError,
            "fragment_face_candidates=3 exceeds limit 2",
        ):
            compute_open_face_unified_compositing(
                _panel_probe_model(),
                projection_matrix=IDENTITY_VIEW,
                limits=OpenFaceUnifiedCompositingLimits(
                    max_fragment_face_candidates=2
                ),
            )

        frame = compute_open_face_unified_compositing(
            _panel_probe_model(),
            projection_matrix=IDENTITY_VIEW,
            limits=OpenFaceUnifiedCompositingLimits(
                max_fragment_face_candidates=3
            ),
        )
        self.assertEqual(len(frame.path_fragments), 3)

    def test_print_current_unified_component_revision(self) -> None:
        revision = provider_component_revisions()[
            COMPONENT_OPEN_FACE_UNIFIED_COMPOSITING
        ]
        print(f"OPEN_FACE_UNIFIED_COMPONENT_REVISION={revision}", flush=True)
        self.assertTrue(revision.startswith(("source-sha256:", "component-sha256:")))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
import json

from ..trace import VisibilityFrame
from .contract import RigidTransform3D


DERIVED_DIHEDRAL_TRACE_SCHEMA = "manim-derived-dihedral-visibility-trace/v1"


@dataclass(frozen=True)
class DerivedDihedralVisibilityFrame:
    line_visibility: VisibilityFrame
    transform: RigidTransform3D
    coincident_source_face_ids: tuple[str, ...]
    suppressed_source_stroke_ids: tuple[str, ...]
    schema: str = DERIVED_DIHEDRAL_TRACE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "lineVisibility": self.line_visibility.to_dict(),
            "transform": self.transform.to_dict(),
            "coincidentSourceFaceIds": list(self.coincident_source_face_ids),
            "suppressedSourceStrokeIds": list(self.suppressed_source_stroke_ids),
        }


def canonical_derived_dihedral_trace_json(
    frame: DerivedDihedralVisibilityFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "DERIVED_DIHEDRAL_TRACE_SCHEMA",
    "DerivedDihedralVisibilityFrame",
    "canonical_derived_dihedral_trace_json",
]

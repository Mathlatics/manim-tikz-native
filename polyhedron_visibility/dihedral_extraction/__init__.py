"""Extract one movable dihedral copy from a closed convex solid."""

from .contract import (
    DERIVED_DIHEDRAL_MODEL_SCHEMA,
    DerivedBoundaryStrokeSpec,
    DerivedDihedralContractError,
    DerivedDihedralModel,
    DerivedDihedralSpec,
    RigidTransform3D,
)
from .base_plane import (
    BASE_PLANE_ROTATION_SCHEMA,
    BasePlaneRotation3D,
    BasePlaneRotationError,
)
from .solver import (
    DerivedDihedralSolverError,
    compute_derived_dihedral_visibility,
)
from .authoring import (
    DerivedDihedralAuthoringError,
    ExtractedDihedralEntity3D,
    ExtractedDihedralScene3D,
)
from .manim import ExtractedDihedralOcclusion3D
from .compositing import (
    DERIVED_DIHEDRAL_TRANSPARENT_COMPOSITING_SCHEMA,
    DerivedDihedralTransparentCompositingError,
    DerivedDihedralTransparentCompositingFrame,
    canonical_derived_dihedral_compositing_json,
    compute_derived_dihedral_transparent_compositing,
    transparent_dihedral_triangle_capacity,
)
from .compositing_manim import (
    DERIVED_DIHEDRAL_TRANSPARENT_BINDING_SCALE_LIMITS,
    DerivedDihedralTransparentBindingScaleLimits,
    DerivedDihedralTransparentLayer,
    DerivedDihedralTransparentManimError,
    guard_derived_dihedral_transparent_scale,
)
from .unified_compositing import (
    DERIVED_DIHEDRAL_UNIFIED_COMPOSITING_SCHEMA,
    DerivedDihedralUnifiedCompositingError,
    DerivedDihedralUnifiedCompositingFrame,
    UnifiedFaceBatch,
    UnifiedPaintRelation,
    UnifiedStrokeFragment,
    canonical_derived_dihedral_unified_compositing_json,
    compute_derived_dihedral_unified_compositing,
)
from .unified_compositing_manim import (
    DerivedDihedralUnifiedLayer,
    DerivedDihedralUnifiedManimError,
)
from .trace import (
    DERIVED_DIHEDRAL_TRACE_SCHEMA,
    DerivedDihedralVisibilityFrame,
    canonical_derived_dihedral_trace_json,
)

__all__ = [
    "BASE_PLANE_ROTATION_SCHEMA",
    "DERIVED_DIHEDRAL_MODEL_SCHEMA",
    "BasePlaneRotation3D",
    "BasePlaneRotationError",
    "DerivedBoundaryStrokeSpec",
    "DerivedDihedralContractError",
    "DerivedDihedralAuthoringError",
    "DerivedDihedralTransparentCompositingError",
    "DerivedDihedralTransparentCompositingFrame",
    "DerivedDihedralTransparentBindingScaleLimits",
    "DerivedDihedralTransparentLayer",
    "DerivedDihedralTransparentManimError",
    "DerivedDihedralUnifiedCompositingError",
    "DerivedDihedralUnifiedCompositingFrame",
    "DerivedDihedralUnifiedLayer",
    "DerivedDihedralUnifiedManimError",
    "DerivedDihedralModel",
    "DerivedDihedralSpec",
    "DerivedDihedralSolverError",
    "DerivedDihedralVisibilityFrame",
    "ExtractedDihedralEntity3D",
    "ExtractedDihedralOcclusion3D",
    "ExtractedDihedralScene3D",
    "RigidTransform3D",
    "DERIVED_DIHEDRAL_TRACE_SCHEMA",
    "DERIVED_DIHEDRAL_TRANSPARENT_COMPOSITING_SCHEMA",
    "DERIVED_DIHEDRAL_TRANSPARENT_BINDING_SCALE_LIMITS",
    "DERIVED_DIHEDRAL_UNIFIED_COMPOSITING_SCHEMA",
    "UnifiedFaceBatch",
    "UnifiedPaintRelation",
    "UnifiedStrokeFragment",
    "canonical_derived_dihedral_compositing_json",
    "canonical_derived_dihedral_unified_compositing_json",
    "canonical_derived_dihedral_trace_json",
    "compute_derived_dihedral_visibility",
    "compute_derived_dihedral_transparent_compositing",
    "compute_derived_dihedral_unified_compositing",
    "guard_derived_dihedral_transparent_scale",
    "transparent_dihedral_triangle_capacity",
]

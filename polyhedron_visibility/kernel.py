"""Public renderer-independent geometry-kernel surface.

Importing ``polyhedron_visibility.kernel`` does not import Manim.  The package
root uses lazy public exports so Python can reach this module without first
loading the renderer bindings.
"""

from .compositor import (
    CompositorCycleError,
    PainterConstraint,
    painter_ranks,
    stable_topological_sort,
)
from .geometry import (
    DEFAULT_GEOMETRY_CONTEXT,
    DEFAULT_RESOLVED_GEOMETRY_CONTEXT,
    GeometryContext,
    GeometryQuantity,
    ResolvedGeometryContext,
    coordinate_scale,
    resolve_geometry_context,
)
from .topology import (
    ParameterInterval,
    TaggedInterval,
    assert_exact_partition,
    coalesce_tagged_intervals,
    partition_parameter_domain,
)
from .visibility import (
    OcclusionInterval,
    VisibilityKind,
    VisibilitySpan,
    hidden_intervals,
    partition_visibility,
    visible_intervals,
)

__all__ = [
    "CompositorCycleError",
    "DEFAULT_GEOMETRY_CONTEXT",
    "DEFAULT_RESOLVED_GEOMETRY_CONTEXT",
    "GeometryContext",
    "GeometryQuantity",
    "OcclusionInterval",
    "PainterConstraint",
    "ParameterInterval",
    "ResolvedGeometryContext",
    "TaggedInterval",
    "VisibilityKind",
    "VisibilitySpan",
    "assert_exact_partition",
    "coalesce_tagged_intervals",
    "coordinate_scale",
    "hidden_intervals",
    "painter_ranks",
    "partition_parameter_domain",
    "partition_visibility",
    "resolve_geometry_context",
    "stable_topological_sort",
    "visible_intervals",
]

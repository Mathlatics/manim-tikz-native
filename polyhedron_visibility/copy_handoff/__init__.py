"""Continuous display ownership for geometry copied from another entity."""

from .contract import (
    COPY_IDENTITY_HANDOFF_SCHEMA,
    CopyHandoffContractError,
    CopyIdentityHandoffMap,
    CopyIdentityHandoffPolicy,
    CopyPrimitivePair,
    CopyVertexPair,
)
from .solver import (
    COPY_IDENTITY_HANDOFF_FRAME_SCHEMA,
    CopyIdentityHandoffFrame,
    CopyPrimitiveActivation,
    CopyVertexSeparation,
    compute_copy_identity_handoff,
)

__all__ = [
    "COPY_IDENTITY_HANDOFF_FRAME_SCHEMA",
    "COPY_IDENTITY_HANDOFF_SCHEMA",
    "CopyHandoffContractError",
    "CopyIdentityHandoffFrame",
    "CopyIdentityHandoffMap",
    "CopyIdentityHandoffPolicy",
    "CopyPrimitiveActivation",
    "CopyPrimitivePair",
    "CopyVertexPair",
    "CopyVertexSeparation",
    "compute_copy_identity_handoff",
]

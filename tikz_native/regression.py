from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from .animation import SEMANTIC_LAYER_ORDER, semantic_animation_layers


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regression evidence file."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_semantic_snapshot(document: Any) -> dict[str, Any]:
    """Build the compact, deterministic part of a TikZ-native baseline.

    The full manifest remains the detailed debugging artifact.  This snapshot
    deliberately keeps only compatibility gates: inventory, stable object-ID
    order, retained coordinate-dependency operations, and named intersections.
    """

    kind_counts = Counter(
        item.kind for picture in document.pictures for item in picture.objects
    )
    layer_counts = Counter()
    dependency_counts = Counter()
    picture_fingerprints: list[dict[str, Any]] = []
    intersections: list[dict[str, Any]] = []

    for picture in document.pictures:
        layers = semantic_animation_layers(picture, include_empty=True)
        layer_counts.update(
            {layer.name: len(layer.object_ids) for layer in layers}
        )

        ordered_ids = [item.id for item in picture.objects]
        ordered_id_digest = sha256(
            ("\n".join(ordered_ids) + "\n").encode("utf-8")
        ).hexdigest()
        picture_fingerprints.append(
            {
                "picture": picture.index,
                "object_count": len(ordered_ids),
                "ordered_object_ids_sha256": ordered_id_digest,
            }
        )

        dependency_counts.update(
            dependency.get("operation", "unknown")
            for dependency in picture.coordinate_dependencies.values()
        )
        intersections.extend(
            {
                "picture": picture.index,
                "path_a": relation.path_a,
                "path_b": relation.path_b,
                "sort_by": relation.sort_by,
                "coordinates": list(relation.coordinate_names),
            }
            for relation in picture.intersections
        )

    return {
        "picture_count": len(document.pictures),
        "object_count": sum(
            len(picture.objects) for picture in document.pictures
        ),
        "object_kind_counts": dict(sorted(kind_counts.items())),
        "animation_layer_counts": {
            name: layer_counts[name] for name in SEMANTIC_LAYER_ORDER
        },
        "picture_object_fingerprints": picture_fingerprints,
        "coordinate_dependency_operation_counts": dict(
            sorted(dependency_counts.items())
        ),
        "named_path_intersections": intersections,
        "unsupported_count": sum(
            len(picture.unsupported) for picture in document.pictures
        ),
        "warning_count": sum(
            len(picture.warnings) for picture in document.pictures
        ),
    }


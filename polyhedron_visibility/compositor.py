"""Compositor-layer primitives for deterministic painter ordering.

A constraint ``(farther, nearer)`` means that the farther item must be painted
first. The sorter is stable for otherwise unrelated items and reports cycles
explicitly instead of silently changing order from frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Generic, Hashable, Iterable, TypeVar


NodeT = TypeVar("NodeT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class PainterConstraint(Generic[NodeT]):
    farther: NodeT
    nearer: NodeT


class CompositorCycleError(ValueError):
    """Raised when depth constraints cannot form one painter order."""

    def __init__(self, unresolved: Iterable[Hashable]) -> None:
        self.unresolved = tuple(unresolved)
        names = ", ".join(repr(node) for node in self.unresolved)
        super().__init__(f"painter constraints contain a cycle: {names}")


def stable_topological_sort(
    nodes: Iterable[NodeT],
    constraints: Iterable[PainterConstraint[NodeT] | tuple[NodeT, NodeT]],
    *,
    key: Callable[[NodeT], object] | None = None,
) -> tuple[NodeT, ...]:
    """Return a deterministic far-to-near order.

    Input order is the final tie breaker. Duplicate nodes and duplicate edges
    are ignored. Constraint endpoints omitted from ``nodes`` are appended in
    first-seen order so callers cannot accidentally drop a visible fragment.
    """

    ordered_nodes: list[NodeT] = []
    seen: set[NodeT] = set()

    def add_node(node: NodeT) -> None:
        if node in seen:
            return
        seen.add(node)
        ordered_nodes.append(node)

    for node in nodes:
        add_node(node)

    normalized_constraints: list[tuple[NodeT, NodeT]] = []
    for constraint in constraints:
        if isinstance(constraint, PainterConstraint):
            farther, nearer = constraint.farther, constraint.nearer
        else:
            farther, nearer = constraint
        add_node(farther)
        add_node(nearer)
        normalized_constraints.append((farther, nearer))

    if key is None:
        key = lambda _node: 0

    index = {node: position for position, node in enumerate(ordered_nodes)}
    outgoing: dict[NodeT, set[NodeT]] = {
        node: set() for node in ordered_nodes
    }
    indegree: dict[NodeT, int] = {node: 0 for node in ordered_nodes}

    for farther, nearer in normalized_constraints:
        if farther == nearer or nearer in outgoing[farther]:
            continue
        outgoing[farther].add(nearer)
        indegree[nearer] += 1

    ready: list[tuple[object, int, NodeT]] = []
    for node in ordered_nodes:
        if indegree[node] == 0:
            heapq.heappush(ready, (key(node), index[node], node))

    result: list[NodeT] = []
    while ready:
        _, _, node = heapq.heappop(ready)
        result.append(node)
        for successor in sorted(
            outgoing[node],
            key=lambda item: (key(item), index[item]),
        ):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready,
                    (key(successor), index[successor], successor),
                )

    if len(result) != len(ordered_nodes):
        unresolved = [node for node in ordered_nodes if indegree[node] > 0]
        raise CompositorCycleError(unresolved)
    return tuple(result)


def painter_ranks(order: Iterable[NodeT]) -> dict[NodeT, int]:
    """Map one far-to-near order to deterministic integer draw ranks."""

    result: dict[NodeT, int] = {}
    for rank, node in enumerate(order):
        if node in result:
            raise ValueError(f"duplicate painter node: {node!r}")
        result[node] = rank
    return result

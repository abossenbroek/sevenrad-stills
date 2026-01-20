"""Graph model for TouchDesigner operator networks."""

from td_linter.graph.model import Connection, OperatorNode, TilePosition
from td_linter.graph.types import OperatorFamily

# NetworkGraphBuilder imported lazily to avoid circular imports
__all__ = [
    "Connection",
    "NetworkGraphBuilder",
    "OperatorFamily",
    "OperatorNode",
    "TilePosition",
]


def __getattr__(name: str) -> object:
    """Lazy import for NetworkGraphBuilder."""
    if name == "NetworkGraphBuilder":
        from td_linter.graph.builder import NetworkGraphBuilder

        return NetworkGraphBuilder
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

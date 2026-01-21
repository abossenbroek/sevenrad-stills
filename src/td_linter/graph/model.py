"""Data models for TouchDesigner operator networks."""

from dataclasses import dataclass, field
from pathlib import Path

from td_linter.graph.types import OperatorFamily


@dataclass(frozen=True)
class TilePosition:
    """Operator tile position and dimensions in the network editor."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class OperatorNode:
    """Represents a TouchDesigner operator in the network graph."""

    name: str  # e.g., "displace1"
    family: OperatorFamily  # e.g., TOP
    op_type: str  # e.g., "displace"
    path: str  # Full path: "project1/displace1"
    tile: TilePosition
    source_file: Path
    flags: dict[str, str] = field(default_factory=dict)
    inputs: list[tuple[int, str]] = field(default_factory=list)
    color: tuple[float, ...] | None = None

    @property
    def full_type(self) -> str:
        """Return the full type string like 'TOP:displace'."""
        return f"{self.family.value}:{self.op_type}"


@dataclass
class Connection:
    """Represents a connection between operators."""

    source_path: str  # Path to source operator
    target_path: str  # Path to target operator
    input_index: int  # Input index on the target
    is_missing: bool = False  # True if source doesn't exist


@dataclass
class ParseError:
    """Represents a parsing error."""

    file_path: Path
    line: int | None
    column: int | None
    message: str

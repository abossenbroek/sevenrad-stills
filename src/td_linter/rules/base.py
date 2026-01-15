"""Base classes for lint rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class Violation:
    """Represents a lint rule violation."""

    rule: str  # Rule ID: "no-invalid-cycles"
    message: str  # Human-readable message
    path: str  # Operator path
    severity: str = "error"  # error, warning, info
    source_file: Path | None = None
    line: int | None = None
    context: dict[str, object] = field(default_factory=dict)


class LintRule(ABC):
    """Base class for all lint rules."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the rule identifier, e.g., 'no-invalid-cycles'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable rule description."""
        ...

    @property
    def severity(self) -> str:
        """Return the default severity level."""
        return "error"

    @abstractmethod
    def check(self, graph: "nx.DiGraph") -> Iterator[Violation]:
        """Run the rule and yield violations."""
        ...

"""Base classes for lint rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, TypeVar, overload

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class Violation:
    """Represents a lint rule violation."""

    rule: str  # Rule ID: "no-invalid-cycles" or "C001"
    message: str  # Human-readable message
    path: str  # Operator path
    severity: str = "error"  # error, warning, info
    source_file: Path | None = None
    line: int | None = None
    context: dict[str, object] = field(default_factory=dict)


# Rule category codes
CATEGORIES = {
    "S": "syntax",
    "C": "connection",
    "T": "type",
    "R": "reference",
    "G": "glsl",
    "P": "python",
    "F": "performance",
}

# Type variable for option values
_T = TypeVar("_T")

# Type alias for option dictionaries (str, int, float, bool, list, dict)
OptionValue = str | int | float | bool | list[str] | dict[str, str]


class LintRule(ABC):
    """Base class for all lint rules."""

    # Rule options (can be set from config)
    _options: dict[str, OptionValue]

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule with optional configuration."""
        self._options = options or {}

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Return the rule code, e.g., 'C001'."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the rule name, e.g., 'no-invalid-cycles'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable rule description."""
        ...

    @property
    def id(self) -> str:
        """Return the rule identifier (alias for rule_id)."""
        return self.rule_id

    @property
    def category(self) -> str:
        """Return the category name based on rule_id prefix."""
        prefix = self.rule_id[0] if self.rule_id else ""
        return CATEGORIES.get(prefix, "unknown")

    @property
    def category_code(self) -> str:
        """Return the single-letter category code."""
        return self.rule_id[0] if self.rule_id else ""

    @property
    def severity(self) -> str:
        """Return the default severity level."""
        return "error"

    @property
    def options(self) -> dict[str, OptionValue]:
        """Return the rule options."""
        return self._options

    @overload
    def get_option(self, key: str) -> OptionValue | None: ...
    @overload
    def get_option(self, key: str, default: _T) -> OptionValue | _T: ...
    def get_option(
        self, key: str, default: _T | None = None
    ) -> OptionValue | _T | None:
        """Get a rule option with default fallback."""
        return self._options.get(key, default)

    @abstractmethod
    def check(self, graph: "nx.DiGraph") -> Iterator[Violation]:
        """Run the rule and yield violations."""
        ...

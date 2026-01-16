"""Base class for output formatters."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td_linter.rules.base import Violation


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the formatter name (e.g., 'text', 'json', 'sarif')."""
        ...

    @abstractmethod
    def format(
        self,
        violations: list["Violation"],
        project_path: str | None = None,
    ) -> str:
        """
        Format violations into output string.

        Args:
            violations: List of violations to format
            project_path: Optional project path for context

        Returns:
            Formatted string output
        """
        ...

    @property
    def supports_color(self) -> bool:
        """Whether this formatter supports color output."""
        return False

"""Diagnostic result types for linter output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DiagnosticSeverity(Enum):
    """LSP diagnostic severity levels."""

    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Position:
    """Position in a text document (0-indexed)."""

    line: int
    character: int

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Position:
        """Create from LSP position object."""
        return cls(line=data.get("line", 0), character=data.get("character", 0))


@dataclass
class Range:
    """Range in a text document."""

    start: Position
    end: Position

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> Range:
        """Create from LSP range object."""
        return cls(
            start=Position.from_lsp(data.get("start", {})),
            end=Position.from_lsp(data.get("end", {})),
        )


@dataclass
class Diagnostic:
    """A diagnostic message from the linter."""

    range: Range
    severity: DiagnosticSeverity
    message: str
    source: str
    code: str | int | None = None

    @classmethod
    def from_lsp(cls, data: dict[str, Any], source: str) -> Diagnostic:
        """Create from LSP diagnostic object."""
        severity_value = data.get("severity", 1)
        try:
            severity = DiagnosticSeverity(severity_value)
        except ValueError:
            severity = DiagnosticSeverity.ERROR

        return cls(
            range=Range.from_lsp(data.get("range", {})),
            severity=severity,
            message=data.get("message", "Unknown error"),
            source=source,
            code=data.get("code"),
        )

    def __str__(self) -> str:
        """Format diagnostic for display."""
        pos = f"{self.range.start.line + 1}:{self.range.start.character + 1}"
        severity_str = self.severity.name.lower()
        code_str = f" [{self.code}]" if self.code else ""
        return f"{pos}: {severity_str}{code_str}: {self.message}"


@dataclass
class LintResult:
    """Result of linting a single file."""

    filepath: str
    diagnostics: list[Diagnostic]
    success: bool

    @property
    def has_errors(self) -> bool:
        """Check if any diagnostics are errors."""
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        """Check if any diagnostics are warnings."""
        return any(d.severity == DiagnosticSeverity.WARNING for d in self.diagnostics)

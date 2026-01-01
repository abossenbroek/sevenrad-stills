"""LintError dataclass for validation results.

This module defines the LintError dataclass used to represent validation
errors, warnings, and informational messages from the linter.
"""

from __future__ import annotations

from dataclasses import dataclass

from max_linter.types import Severity


@dataclass
class LintError:
    """Represents a validation error with severity and location."""

    severity: Severity
    rule: str
    message: str
    object_id: str | None = None

    def __str__(self) -> str:
        location = f" ({self.object_id})" if self.object_id else ""
        return f"  [{self.severity.value}] {self.rule}{location}: {self.message}"

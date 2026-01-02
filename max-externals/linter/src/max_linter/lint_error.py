"""LintError dataclass for validation results.

This module defines the LintError dataclass used to represent validation
errors, warnings, and informational messages from the linter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "rule": self.rule,
            "message": self.message,
            "object_id": self.object_id,
        }

    def to_github_annotation(self, filepath: str) -> str:
        """Format as GitHub Actions workflow command.

        See: https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions
        """
        level = "error" if self.severity == Severity.ERROR else "warning"
        # Escape special characters for GitHub Actions
        msg = self.message.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
        location = f",title={self.rule}" if self.rule else ""
        obj_info = f" ({self.object_id})" if self.object_id else ""
        return f"::{level} file={filepath}{location}::{msg}{obj_info}"

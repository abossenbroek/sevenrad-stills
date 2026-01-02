"""Base validator mixin for Max linter.

This module provides the BaseValidatorMixin class that implements common
error/warning/info reporting methods used by all validator mixins.
"""

from __future__ import annotations

from max_linter.lint_error import LintError
from max_linter.types import Severity


class BaseValidatorMixin:
    """Base mixin providing error/warning/info reporting methods.

    This mixin provides common methods for creating LintError objects
    with different severity levels. All validator mixins should inherit
    from this class.

    Attributes:
        errors: List to collect ERROR severity issues.
        warnings: List to collect WARNING severity issues.
    """

    errors: list[LintError]
    warnings: list[LintError]

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error (validation failure).

        Args:
            rule: The rule code (e.g., 'ctx-001', 'signal-001').
            message: Human-readable description of the issue.
            object_id: Optional object ID where the issue was found.
        """
        self.errors.append(LintError(Severity.ERROR, rule, message, object_id))

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning (potential issue).

        Args:
            rule: The rule code (e.g., 'ctx-001', 'signal-001').
            message: Human-readable description of the issue.
            object_id: Optional object ID where the issue was found.
        """
        self.warnings.append(LintError(Severity.WARNING, rule, message, object_id))

    def info(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an informational message.

        Args:
            rule: The rule code (e.g., 'ctx-001', 'signal-001').
            message: Human-readable description of the issue.
            object_id: Optional object ID where the issue was found.
        """
        self.warnings.append(LintError(Severity.INFO, rule, message, object_id))

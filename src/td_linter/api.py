"""Programmatic API for td-linter.

This module provides a Python API for integrating td-linter into
other tools and scripts.

Example:
    from td_linter import lint, Severity

    violations = lint("myproject.toe.dir")
    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        sys.exit(1)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from td_linter.linter import run_lint
from td_linter.rules.base import Violation
from td_linter.rules.loader import LintConfig
from td_linter.rules.registry import get_registry

if TYPE_CHECKING:
    from td_linter.rules.base import LintRule


class Severity(str, Enum):
    """Severity levels for lint violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


def lint(
    path: str | Path,
    config_path: str | Path | None = None,
    select: list[str] | None = None,
    ignore: list[str] | None = None,
    validate_expressions: bool = False,
    validate_embedded: bool = True,
) -> list[Violation]:
    """Lint a .toe.dir project.

    Args:
        path: Path to .toe.dir directory
        config_path: Optional path to td-linter.yaml config
        select: Optional list of rule IDs or category codes to enable
        ignore: Optional list of rule IDs or category codes to disable
        validate_expressions: Enable expression validation in .parm files
        validate_embedded: Validate embedded GLSL/Python (default True)

    Returns:
        List of Violation objects found

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If path is not a directory

    Example:
        >>> violations = lint("myproject.toe.dir")
        >>> errors = [v for v in violations if v.severity == "error"]
        >>> print(f"Found {len(errors)} errors")
    """
    path = Path(path)

    if not path.exists():
        msg = f"Path not found: {path}"
        raise FileNotFoundError(msg)

    if not path.is_dir():
        msg = f"Not a directory: {path}"
        raise ValueError(msg)

    # Load configuration
    config_path_obj = Path(config_path) if config_path else None
    registry = get_registry(config_path_obj)

    # Apply select/ignore filters
    enabled_rules: list[LintRule]
    if select:
        enabled_rules = registry.select(select)
    else:
        enabled_rules = registry.enabled()

    if ignore:
        ignore_set = set(ignore)
        # Expand category codes
        expanded_ignore: set[str] = set()
        for pattern in ignore_set:
            if len(pattern) == 1:
                for rule in registry.by_category(pattern):
                    expanded_ignore.add(rule.rule_id)
            else:
                expanded_ignore.add(pattern)
        enabled_rules = [r for r in enabled_rules if r.rule_id not in expanded_ignore]

    return run_lint(
        path,
        validate_expressions=validate_expressions,
        validate_embedded=validate_embedded,
        rules=enabled_rules,
        config=registry.config,
    )


def lint_and_check(
    path: str | Path,
    fail_on_warning: bool = False,
    **kwargs: object,
) -> tuple[bool, list[Violation]]:
    """Lint a project and return pass/fail status.

    This is a convenience function for integration with build scripts.

    Args:
        path: Path to .toe.dir directory
        fail_on_warning: If True, warnings also cause failure
        **kwargs: Additional arguments passed to lint()

    Returns:
        Tuple of (passed: bool, violations: list[Violation])

    Example:
        >>> passed, violations = lint_and_check("myproject.toe.dir")
        >>> if not passed:
        ...     print("Linting failed")
        ...     sys.exit(1)
    """
    # Filter kwargs to only include valid lint() parameters
    lint_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("config_path", "select", "ignore", "validate_expressions", "validate_embedded")
    }
    violations = lint(path, **lint_kwargs)  # type: ignore[arg-type]

    if fail_on_warning:
        has_issues = any(v.severity in ("error", "warning") for v in violations)
    else:
        has_issues = any(v.severity == "error" for v in violations)

    return (not has_issues, violations)


# Re-export key types for convenience
__all__ = [
    "lint",
    "lint_and_check",
    "Severity",
    "Violation",
    "LintConfig",
]

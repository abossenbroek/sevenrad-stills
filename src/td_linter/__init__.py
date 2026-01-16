"""TouchDesigner Linter - Validate .toe.dir expanded projects.

This package provides a linter for TouchDesigner .toe.dir expanded projects.

Example:
    from td_linter import lint, Severity

    violations = lint("myproject.toe.dir")
    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        print("Linting failed!")
"""

__version__ = "0.1.0"

# Re-export public API
from td_linter.api import Severity, Violation, lint, lint_and_check

__all__ = [
    "__version__",
    "lint",
    "lint_and_check",
    "Severity",
    "Violation",
]

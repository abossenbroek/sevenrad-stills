"""Python rules (P): Python code validation."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, OptionValue, Violation


class PythonSyntaxRule(LintRule):
    """
    P001: Validate Python syntax in embedded code.

    Checks that Python code in .text files (scripts, callbacks) has
    valid syntax using Python's AST parser.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "P001"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "python-syntax"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate Python syntax in embedded code"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for Python syntax errors."""
        # Check for Python errors stored in graph metadata
        python_errors = graph.graph.get("python_errors", [])
        for error in python_errors:
            yield Violation(
                rule=self.rule_id,
                message=f"Python syntax error: {error.get('message', 'unknown error')}",
                path=error.get("path", "unknown"),
                severity=self.severity,
                source_file=error.get("source_file"),
                line=error.get("line"),
                context={"python_error": error},
            )


class PythonUndefinedNameRule(LintRule):
    """
    P002: Detect undefined names in Python code.

    Checks for references to undefined variables or functions in
    Python code, excluding TouchDesigner built-ins (op, me, parent, etc.).
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "P002"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "python-undefined-name"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect undefined names in Python code"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for undefined names."""
        # Check for undefined name warnings stored in graph metadata
        undefined_warnings = graph.graph.get("python_undefined_names", [])
        for warning in undefined_warnings:
            yield Violation(
                rule=self.rule_id,
                message=f"Undefined name: '{warning.get('name', 'unknown')}'",
                path=warning.get("path", "unknown"),
                severity=self.severity,
                source_file=warning.get("source_file"),
                line=warning.get("line"),
                context={
                    "undefined_name": warning.get("name"),
                    "scope": warning.get("scope"),
                },
            )


class TDExecuteDatCallbacksRule(LintRule):
    """
    P003: Check for proper Execute DAT callback signatures.

    Validates that Execute DAT callbacks (onStart, onCook, etc.)
    have the correct function signatures as expected by TouchDesigner.

    Note: Placeholder for future implementation.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "P003"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "td-execute-dat-callbacks"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Check Execute DAT callback signatures"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:  # noqa: ARG002
        """Check for incorrect callback signatures."""
        # Placeholder: Full implementation requires analyzing DAT scripts
        # for proper callback function signatures
        return iter(())

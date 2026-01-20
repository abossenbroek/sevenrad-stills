"""Syntax rules (S): File format validation."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, OptionValue, Violation


class ValidNFileSyntaxRule(LintRule):
    """
    S001: Validate .n file syntax.

    Ensures that .n (node definition) files have valid syntax and
    can be parsed without errors.

    Note: This rule is typically checked during graph building.
    Parse errors are reported as violations.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "S001"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "valid-n-file-syntax"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate .n file syntax is correct"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for .n file syntax errors."""
        # Check for parse errors stored in graph metadata
        parse_errors = graph.graph.get("n_file_errors", [])
        for error in parse_errors:
            err_msg = error.get("message", "unknown error")
            yield Violation(
                rule=self.rule_id,
                message=f"Invalid .n file syntax: {err_msg}",
                path=error.get("path", "unknown"),
                severity=self.severity,
                source_file=error.get("source_file"),
                line=error.get("line"),
                context={"parse_error": error},
            )


class ValidParmFileSyntaxRule(LintRule):
    """
    S002: Validate .parm file syntax.

    Ensures that .parm (parameter) files have valid syntax and
    can be parsed without errors.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "S002"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "valid-parm-file-syntax"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate .parm file syntax is correct"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for .parm file syntax errors."""
        # Check for parse errors stored in graph metadata
        parse_errors = graph.graph.get("parm_file_errors", [])
        for error in parse_errors:
            err_msg = error.get("message", "unknown error")
            yield Violation(
                rule=self.rule_id,
                message=f"Invalid .parm file syntax: {err_msg}",
                path=error.get("path", "unknown"),
                severity=self.severity,
                source_file=error.get("source_file"),
                line=error.get("line"),
                context={"parse_error": error},
            )


class TocCompletenessRule(LintRule):
    """
    S003: Validate TOC manifest completeness.

    Ensures that all entries in the .toc manifest file exist
    in the project directory.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "S003"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "toc-completeness"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate TOC entries exist in project"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for missing TOC entries."""
        # Check for TOC errors stored in graph metadata
        toc_errors = graph.graph.get("toc_errors", [])
        for error in toc_errors:
            yield Violation(
                rule=self.rule_id,
                message=f"TOC entry missing: {error.get('entry', 'unknown')}",
                path=error.get("entry", "unknown"),
                severity=self.severity,
                source_file=error.get("source_file"),
                context={"missing_entry": error.get("entry")},
            )

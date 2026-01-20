"""Reference rules (R): Operator reference validation."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, OptionValue, Violation


class ValidOperatorReferencesRule(LintRule):
    """
    R001: Validate operator references exist.

    Validates that operator references in inputs actually exist in
    the project.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "R001"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "valid-operator-references"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate operator path references exist"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for invalid operator references."""
        # Check all edges for missing references
        for source, target, data in graph.edges(data=True):
            if data.get("missing", False):
                actual_source = source.replace("MISSING:", "")
                yield Violation(
                    rule=self.rule_id,
                    message=(
                        f"Invalid reference: '{actual_source}' "
                        f"referenced by '{target}' does not exist"
                    ),
                    path=target,
                    severity=self.severity,
                    context={"invalid_ref": actual_source},
                )


class ValidPathReferencesRule(LintRule):
    """
    R002: Validate path references in parameters.

    Validates that path references in .parm files (expressions, strings)
    point to valid operators or resources.

    Note: This is a placeholder for future implementation that requires
    parsing .parm file expressions.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "R002"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "valid-path-references"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate path references in parameters"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:  # noqa: ARG002
        """Check for invalid path references."""
        # Placeholder: Full implementation requires parsing .parm files
        # for op() references, paths, and other operator references
        return iter(())

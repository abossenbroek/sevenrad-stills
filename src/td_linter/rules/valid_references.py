"""Rule: Validate operator references in .parm files."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, Violation


class ValidReferencesRule(LintRule):
    """
    Validates that operator references in parameters are valid.

    This rule checks that operators referenced in .parm files (via path
    references like ./operator or /path/to/operator) actually exist in
    the project.

    Note: This is a basic implementation. Full reference validation
    requires parsing .parm files and extracting path references.
    """

    @property
    def id(self) -> str:
        """Return rule identifier."""
        return "valid-operator-references"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Validate operator path references exist"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """
        Check for invalid operator references.

        Note: Currently validates based on graph edges. Full implementation
        would also parse .parm files for operator references in expressions.
        """
        # Check all edges for missing references
        for source, target, data in graph.edges(data=True):
            if data.get("missing", False):
                actual_source = source.replace("MISSING:", "")
                yield Violation(
                    rule=self.id,
                    message=(
                        f"Invalid reference: '{actual_source}' "
                        f"referenced by '{target}' does not exist"
                    ),
                    path=target,
                    severity="error",
                    context={"invalid_ref": actual_source},
                )

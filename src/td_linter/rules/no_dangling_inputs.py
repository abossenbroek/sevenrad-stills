"""Rule: Detect dangling (missing) input references."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, Violation


class NoDanglingInputsRule(LintRule):
    """
    Detects operators that reference non-existent inputs.

    When an operator references another operator that doesn't exist in the
    project, this indicates a broken connection that will cause errors at
    runtime.
    """

    @property
    def id(self) -> str:
        """Return rule identifier."""
        return "no-dangling-inputs"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect references to non-existent operators"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for dangling input references."""
        # Find all MISSING: placeholder nodes
        for node in graph.nodes:
            if node.startswith("MISSING:"):
                # Find which operators reference this missing node
                actual_path = node.replace("MISSING:", "")

                for _, target, data in graph.edges(node, data=True):
                    input_index = data.get("input_index", "?")
                    yield Violation(
                        rule=self.id,
                        message=(
                            f"Operator '{target}' references non-existent "
                            f"operator '{actual_path}' at input {input_index}"
                        ),
                        path=target,
                        severity=self.severity,
                        context={
                            "missing_ref": actual_path,
                            "input_index": input_index,
                        },
                    )

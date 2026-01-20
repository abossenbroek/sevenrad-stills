"""Rule: Detect invalid cycles in the operator network."""

from typing import Iterator

import networkx as nx

from td_linter.graph.types import FEEDBACK_OPERATORS, OperatorFamily
from td_linter.rules.base import LintRule, Violation


class NoInvalidCyclesRule(LintRule):
    """
    Detects cycles that are not legitimate feedback loops.

    Valid cycles include:
    - All-CHOP cycles (CHOPs support feedback natively)
    - Cycles containing feedback operators (feedback, timemachine, delay, etc.)

    Invalid cycles will cause infinite loops or undefined behavior in TD.
    """

    @property
    def id(self) -> str:
        """Return rule identifier."""
        return "no-invalid-cycles"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Disallow cycles without explicit feedback operators"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for invalid cycles in the graph."""
        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXError:
            return

        for cycle in cycles:
            if not self._is_valid_feedback(cycle, graph):
                cycle_str = " -> ".join(cycle)
                yield Violation(
                    rule=self.id,
                    message=f"Invalid cycle detected: {cycle_str} -> {cycle[0]}",
                    path=cycle[0],
                    severity="error",
                    context={"cycle": cycle},
                )

    def _is_valid_feedback(self, cycle: list[str], graph: nx.DiGraph) -> bool:
        """Check if a cycle is a legitimate feedback loop."""
        # Skip missing nodes
        if any(node.startswith("MISSING:") for node in cycle):
            return True  # Can't validate cycles with missing nodes

        # Get families and op_types for all nodes in cycle
        families: list[str] = []
        op_types: list[str] = []

        for node in cycle:
            node_data = graph.nodes.get(node, {})
            families.append(node_data.get("family", "UNKNOWN"))
            op_types.append(node_data.get("op_type", ""))

        # All CHOP: valid (CHOPs support native feedback)
        if all(f == "CHOP" for f in families):
            return True

        # Contains a feedback operator: valid
        for op_type in op_types:
            if op_type.lower() in FEEDBACK_OPERATORS:
                return True

        return False

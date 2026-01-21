"""Connection rules (C): Graph connectivity validation."""

from typing import Iterator

import networkx as nx

from td_linter.graph.types import FEEDBACK_OPERATORS
from td_linter.rules.base import LintRule, OptionValue, Violation


class NoInvalidCyclesRule(LintRule):
    """
    C001: Detect invalid cycles in the operator network.

    Detects cycles that are not legitimate feedback loops.

    Valid cycles include:
    - All-CHOP cycles (CHOPs support feedback natively)
    - Cycles containing feedback operators (feedback, timemachine, delay, etc.)

    Invalid cycles will cause infinite loops or undefined behavior in TD.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "C001"

    @property
    def name(self) -> str:
        """Return rule name."""
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
                    rule=self.rule_id,
                    message=f"Invalid cycle detected: {cycle_str} -> {cycle[0]}",
                    path=cycle[0],
                    severity=self.severity,
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
        return any(op_type.lower() in FEEDBACK_OPERATORS for op_type in op_types)


class NoDanglingInputsRule(LintRule):
    """
    C002: Detect dangling (missing) input references.

    Detects operators that reference non-existent inputs.

    When an operator references another operator that doesn't exist in the
    project, this indicates a broken connection that will cause errors at
    runtime.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "C002"

    @property
    def name(self) -> str:
        """Return rule name."""
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
                        rule=self.rule_id,
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

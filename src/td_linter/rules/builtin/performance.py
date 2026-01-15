"""Performance rules (F): Performance-related warnings."""

from typing import Iterator

import networkx as nx

from td_linter.rules.base import LintRule, OptionValue, Violation


class DeepNestingRule(LintRule):
    """
    F001: Detect deeply nested operator hierarchies.

    Deep nesting can indicate overly complex networks that are
    difficult to maintain and may have performance implications.

    Options:
        max_depth: Maximum allowed nesting depth (default: 10)
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "F001"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "deep-nesting"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect deeply nested operator hierarchies"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for deeply nested hierarchies."""
        max_depth_opt = self.get_option("max_depth", 10)
        max_depth = (
            int(max_depth_opt) if isinstance(max_depth_opt, (int, float, str)) else 10
        )

        # Find all nodes and check their depth
        for node in graph.nodes:
            if node.startswith("MISSING:"):
                continue

            depth = self._get_node_depth(node)
            if depth > max_depth:
                msg = f"Operator '{node}' has nesting depth {depth} (max: {max_depth})"
                yield Violation(
                    rule=self.rule_id,
                    message=msg,
                    path=node,
                    severity=self.severity,
                    context={
                        "depth": depth,
                        "max_depth": max_depth,
                    },
                )

    def _get_node_depth(self, node_path: str) -> int:
        """Calculate nesting depth from operator path."""
        # Count path separators (excluding leading /)
        path = node_path.lstrip("/")
        if not path:
            return 0
        return path.count("/") + 1


class ExcessiveInputsRule(LintRule):
    """
    F002: Detect operators with excessive input connections.

    Operators with many inputs can be difficult to maintain and
    may indicate a design that could be simplified.

    Options:
        max_inputs: Maximum allowed inputs per operator (default: 16)
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "F002"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "excessive-inputs"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect operators with too many inputs"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for operators with excessive inputs."""
        max_inputs_opt = self.get_option("max_inputs", 16)
        max_inputs = (
            int(max_inputs_opt) if isinstance(max_inputs_opt, (int, float, str)) else 16
        )

        for node in graph.nodes:
            if node.startswith("MISSING:"):
                continue

            # Count incoming edges (inputs)
            input_count = graph.in_degree(node)
            if input_count > max_inputs:
                msg = f"Operator '{node}' has {input_count} inputs (max: {max_inputs})"
                yield Violation(
                    rule=self.rule_id,
                    message=msg,
                    path=node,
                    severity=self.severity,
                    context={
                        "input_count": input_count,
                        "max_inputs": max_inputs,
                    },
                )

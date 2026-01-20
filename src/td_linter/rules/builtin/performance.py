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


class HeavyTextureChainsRule(LintRule):
    """
    F003: Detect long chains of TOP operators without caching.

    Long chains of texture operators can cause performance issues.
    Consider using Cache TOP or render-to-texture to break up chains.

    Options:
        max_chain_length: Maximum TOP chain length before warning (default: 8)
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "F003"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "heavy-texture-chains"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect long chains of TOP operators"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for long TOP chains."""
        max_length_opt = self.get_option("max_chain_length", 8)
        max_length = (
            int(max_length_opt) if isinstance(max_length_opt, (int, float, str)) else 8
        )

        # Cache operators that break chains
        cache_ops = frozenset({"cache", "rendertop", "feedback", "feedbacktop"})

        # Find TOP-only chains
        visited: set[str] = set()

        for node in graph.nodes:
            if node in visited or node.startswith("MISSING:"):
                continue

            node_data = graph.nodes.get(node, {})
            family = node_data.get("family", "")

            if family != "TOP":
                continue

            # Measure chain length from this node
            chain_length = self._measure_top_chain(graph, node, cache_ops, visited)

            if chain_length > max_length:
                yield Violation(
                    rule=self.rule_id,
                    message=f"TOP chain starting at '{node}' has {chain_length} operators (max: {max_length})",
                    path=node,
                    severity=self.severity,
                    context={
                        "chain_length": chain_length,
                        "max_chain_length": max_length,
                    },
                )

    def _measure_top_chain(
        self,
        graph: nx.DiGraph,
        start_node: str,
        cache_ops: frozenset[str],
        visited: set[str],
    ) -> int:
        """Measure the length of a TOP-only chain from a node."""
        chain_length = 0
        current = start_node

        while current and current not in visited:
            visited.add(current)

            node_data = graph.nodes.get(current, {})
            family = node_data.get("family", "")
            op_type = node_data.get("operator", "").lower()

            # Stop at non-TOP nodes or cache operators
            if family != "TOP" or op_type in cache_ops:
                break

            chain_length += 1

            # Move to next node in chain (follow outputs)
            successors = list(graph.successors(current))
            if len(successors) == 1:
                next_node = successors[0]
                next_data = graph.nodes.get(next_node, {})
                if next_data.get("family") == "TOP":
                    current = next_node
                else:
                    break
            else:
                break

        return chain_length


class UnoptimizedFeedbackRule(LintRule):
    """
    F004: Detect feedback loops without cache operators.

    Feedback loops should include cache operators for optimal
    performance. Without caching, feedback can cause excessive
    recomputation.

    Note: This rule checks for cycles that don't include known
    cache/delay operators.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "F004"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "unoptimized-feedback"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect feedback loops without cache operators"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "warning"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for feedback loops without cache operators."""
        # Operators that properly handle feedback
        feedback_ops = frozenset({
            "feedback", "feedbackchop", "feedbacktop",
            "cache", "delay", "lag", "timemachine",
        })

        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXNoCycle:
            return

        for cycle in cycles:
            # Check if any node in cycle is a proper feedback operator
            has_cache = False
            for node in cycle:
                if node.startswith("MISSING:"):
                    continue
                node_data = graph.nodes.get(node, {})
                op_type = node_data.get("operator", "").lower()
                if op_type in feedback_ops:
                    has_cache = True
                    break

            if not has_cache and len(cycle) > 0:
                first_node = cycle[0]
                yield Violation(
                    rule=self.rule_id,
                    message=f"Feedback loop at '{first_node}' has no cache/delay operators",
                    path=first_node,
                    severity=self.severity,
                    context={
                        "cycle_length": len(cycle),
                        "cycle_nodes": cycle[:5],  # First 5 nodes for context
                    },
                )


class CookEveryFrameRule(LintRule):
    """
    F005: Detect operators unnecessarily set to cook every frame.

    Some operators may be set to cook every frame when they don't
    need to, causing unnecessary computation.

    Note: This is a heuristic check based on operator metadata
    stored in the graph.
    """

    def __init__(self, options: dict[str, OptionValue] | None = None) -> None:
        """Initialize the rule."""
        super().__init__(options)

    @property
    def rule_id(self) -> str:
        """Return rule code."""
        return "F005"

    @property
    def name(self) -> str:
        """Return rule name."""
        return "cook-every-frame"

    @property
    def description(self) -> str:
        """Return rule description."""
        return "Detect operators unnecessarily cooking every frame"

    @property
    def severity(self) -> str:
        """Return default severity."""
        return "info"

    def check(self, graph: nx.DiGraph) -> Iterator[Violation]:
        """Check for operators cooking every frame unnecessarily."""
        # Operators that legitimately cook every frame
        time_based_ops = frozenset({
            "timer", "constant", "noise", "pattern", "ramp",
            "moviefilein", "videodevin", "audiodevin", "audiofilein",
            "lfo", "beat", "speed", "count", "audiospectrum",
        })

        for node in graph.nodes:
            if node.startswith("MISSING:"):
                continue

            node_data = graph.nodes.get(node, {})
            op_type = node_data.get("operator", "").lower()

            # Check if node has cook-every-frame flag set
            cook_flag = node_data.get("cook_every_frame", False)

            if cook_flag and op_type not in time_based_ops:
                yield Violation(
                    rule=self.rule_id,
                    message=f"Operator '{node}' is set to cook every frame",
                    path=node,
                    severity=self.severity,
                    context={
                        "operator_type": op_type,
                    },
                )

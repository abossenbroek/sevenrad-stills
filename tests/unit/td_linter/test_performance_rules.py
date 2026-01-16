"""Unit tests for performance rules (F001-F005).

These tests verify that performance rules correctly:
- Detect deeply nested hierarchies (F001)
- Detect excessive inputs (F002)
- Detect heavy texture chains (F003)
- Detect unoptimized feedback loops (F004)
- Detect unnecessary cook-every-frame operators (F005)
"""

import networkx as nx
import pytest

from td_linter.rules.builtin.performance import (
    CookEveryFrameRule,
    DeepNestingRule,
    ExcessiveInputsRule,
    HeavyTextureChainsRule,
    UnoptimizedFeedbackRule,
)


class TestDeepNestingRule:
    """Tests for F001: deep-nesting rule."""

    def test_rule_metadata(self) -> None:
        """Rule should have correct metadata."""
        rule = DeepNestingRule()
        assert rule.rule_id == "F001"
        assert rule.name == "deep-nesting"
        assert rule.severity == "warning"

    def test_shallow_nesting_no_violation(self) -> None:
        """Shallow nesting should not trigger violation."""
        rule = DeepNestingRule()
        graph = nx.DiGraph()

        # Create shallow hierarchy (depth 3)
        graph.add_node("/project/container/operator", family="TOP")
        graph.add_node("/project/container", family="COMP")
        graph.add_node("/project", family="COMP")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_deep_nesting_violation(self) -> None:
        """Deep nesting should trigger violation."""
        rule = DeepNestingRule(options={"max_depth": 5})
        graph = nx.DiGraph()

        # Create deep hierarchy (depth 8)
        deep_path = "/a/b/c/d/e/f/g/h"
        graph.add_node(deep_path, family="TOP")

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].rule == "F001"
        assert "nesting depth 8" in violations[0].message
        assert violations[0].context["depth"] == 8
        assert violations[0].context["max_depth"] == 5

    def test_custom_max_depth_option(self) -> None:
        """Custom max_depth option should be respected."""
        rule = DeepNestingRule(options={"max_depth": 3})
        graph = nx.DiGraph()

        graph.add_node("/a/b/c/d", family="TOP")  # depth 4

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].context["max_depth"] == 3

    def test_exactly_at_limit_no_violation(self) -> None:
        """Nesting exactly at limit should not trigger violation."""
        rule = DeepNestingRule(options={"max_depth": 5})
        graph = nx.DiGraph()

        graph.add_node("/a/b/c/d/e", family="TOP")  # depth 5

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_missing_nodes_skipped(self) -> None:
        """MISSING: nodes should be skipped."""
        rule = DeepNestingRule(options={"max_depth": 1})
        graph = nx.DiGraph()

        graph.add_node("MISSING:/very/deep/nested/path", family="MISSING")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_root_node_depth(self) -> None:
        """Root node should have depth 1."""
        rule = DeepNestingRule(options={"max_depth": 0})
        graph = nx.DiGraph()

        graph.add_node("/root", family="TOP")

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].context["depth"] == 1


class TestExcessiveInputsRule:
    """Tests for F002: excessive-inputs rule."""

    def test_rule_metadata(self) -> None:
        """Rule should have correct metadata."""
        rule = ExcessiveInputsRule()
        assert rule.rule_id == "F002"
        assert rule.name == "excessive-inputs"
        assert rule.severity == "warning"

    def test_few_inputs_no_violation(self) -> None:
        """Few inputs should not trigger violation."""
        rule = ExcessiveInputsRule()
        graph = nx.DiGraph()

        graph.add_node("/target", family="TOP")
        graph.add_node("/input1", family="TOP")
        graph.add_node("/input2", family="TOP")
        graph.add_edge("/input1", "/target", input_index=0)
        graph.add_edge("/input2", "/target", input_index=1)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_excessive_inputs_violation(self) -> None:
        """Excessive inputs should trigger violation."""
        rule = ExcessiveInputsRule(options={"max_inputs": 5})
        graph = nx.DiGraph()

        graph.add_node("/target", family="TOP")
        for i in range(10):
            graph.add_node(f"/input{i}", family="TOP")
            graph.add_edge(f"/input{i}", "/target", input_index=i)

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].rule == "F002"
        assert "10 inputs" in violations[0].message
        assert violations[0].context["input_count"] == 10
        assert violations[0].context["max_inputs"] == 5

    def test_custom_max_inputs_option(self) -> None:
        """Custom max_inputs option should be respected."""
        rule = ExcessiveInputsRule(options={"max_inputs": 3})
        graph = nx.DiGraph()

        graph.add_node("/target", family="TOP")
        for i in range(4):
            graph.add_node(f"/input{i}", family="TOP")
            graph.add_edge(f"/input{i}", "/target", input_index=i)

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].context["max_inputs"] == 3

    def test_exactly_at_limit_no_violation(self) -> None:
        """Inputs exactly at limit should not trigger violation."""
        rule = ExcessiveInputsRule(options={"max_inputs": 5})
        graph = nx.DiGraph()

        graph.add_node("/target", family="TOP")
        for i in range(5):
            graph.add_node(f"/input{i}", family="TOP")
            graph.add_edge(f"/input{i}", "/target", input_index=i)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_missing_nodes_skipped(self) -> None:
        """MISSING: nodes should be skipped even with many inputs."""
        rule = ExcessiveInputsRule(options={"max_inputs": 1})
        graph = nx.DiGraph()

        graph.add_node("MISSING:/target", family="MISSING")
        for i in range(10):
            graph.add_node(f"/input{i}", family="TOP")
            graph.add_edge(f"/input{i}", "MISSING:/target", input_index=i)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_no_inputs_no_violation(self) -> None:
        """Node with no inputs should not trigger violation."""
        rule = ExcessiveInputsRule()
        graph = nx.DiGraph()

        graph.add_node("/source", family="TOP")

        violations = list(rule.check(graph))
        assert len(violations) == 0


class TestHeavyTextureChainsRule:
    """Tests for F003: heavy-texture-chains rule."""

    def test_rule_metadata(self) -> None:
        """Rule should have correct metadata."""
        rule = HeavyTextureChainsRule()
        assert rule.rule_id == "F003"
        assert rule.name == "heavy-texture-chains"
        assert rule.severity == "warning"

    def test_short_chain_no_violation(self) -> None:
        """Short TOP chain should not trigger violation."""
        rule = HeavyTextureChainsRule()
        graph = nx.DiGraph()

        # Create short chain
        for i in range(5):
            graph.add_node(f"/top{i}", family="TOP", operator="blur")
            if i > 0:
                graph.add_edge(f"/top{i-1}", f"/top{i}")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_long_chain_violation(self) -> None:
        """Long TOP chain should trigger violation."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 5})
        graph = nx.DiGraph()

        # Create long chain of 10 TOPs
        for i in range(10):
            graph.add_node(f"/top{i}", family="TOP", operator="blur")
            if i > 0:
                graph.add_edge(f"/top{i-1}", f"/top{i}")

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].rule == "F003"
        assert "10 operators" in violations[0].message

    def test_cache_breaks_chain(self) -> None:
        """Cache operator should break the chain."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 3})
        graph = nx.DiGraph()

        # Create chain with cache in middle
        graph.add_node("/top0", family="TOP", operator="blur")
        graph.add_node("/top1", family="TOP", operator="blur")
        graph.add_node("/cache", family="TOP", operator="cache")
        graph.add_node("/top2", family="TOP", operator="blur")
        graph.add_node("/top3", family="TOP", operator="blur")

        graph.add_edge("/top0", "/top1")
        graph.add_edge("/top1", "/cache")
        graph.add_edge("/cache", "/top2")
        graph.add_edge("/top2", "/top3")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_feedback_breaks_chain(self) -> None:
        """Feedback operator should break the chain."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 3})
        graph = nx.DiGraph()

        # Create chain with feedback in middle
        graph.add_node("/top0", family="TOP", operator="blur")
        graph.add_node("/top1", family="TOP", operator="blur")
        graph.add_node("/fb", family="TOP", operator="feedback")
        graph.add_node("/top2", family="TOP", operator="blur")
        graph.add_node("/top3", family="TOP", operator="blur")

        graph.add_edge("/top0", "/top1")
        graph.add_edge("/top1", "/fb")
        graph.add_edge("/fb", "/top2")
        graph.add_edge("/top2", "/top3")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_non_top_breaks_chain(self) -> None:
        """Non-TOP operators should break the chain."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 3})
        graph = nx.DiGraph()

        # Create chain with CHOP in middle
        graph.add_node("/top0", family="TOP", operator="blur")
        graph.add_node("/top1", family="TOP", operator="blur")
        graph.add_node("/chop", family="CHOP", operator="constant")
        graph.add_node("/top2", family="TOP", operator="blur")
        graph.add_node("/top3", family="TOP", operator="blur")

        graph.add_edge("/top0", "/top1")
        graph.add_edge("/top1", "/chop")
        graph.add_edge("/chop", "/top2")
        graph.add_edge("/top2", "/top3")

        violations = list(rule.check(graph))
        # Chain breaks at /chop, so each segment is short
        assert len(violations) == 0

    def test_missing_nodes_skipped(self) -> None:
        """MISSING: nodes should be skipped."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 1})
        graph = nx.DiGraph()

        graph.add_node("MISSING:/top0", family="MISSING")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_branching_chain(self) -> None:
        """Branching chain should stop at the branch point."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 10})
        graph = nx.DiGraph()

        # Chain with branch
        graph.add_node("/top0", family="TOP", operator="blur")
        graph.add_node("/branch1", family="TOP", operator="blur")
        graph.add_node("/branch2", family="TOP", operator="blur")

        graph.add_edge("/top0", "/branch1")
        graph.add_edge("/top0", "/branch2")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_custom_max_chain_length(self) -> None:
        """Custom max_chain_length option should be respected."""
        rule = HeavyTextureChainsRule(options={"max_chain_length": 2})
        graph = nx.DiGraph()

        for i in range(4):
            graph.add_node(f"/top{i}", family="TOP", operator="blur")
            if i > 0:
                graph.add_edge(f"/top{i-1}", f"/top{i}")

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].context["max_chain_length"] == 2


class TestUnoptimizedFeedbackRule:
    """Tests for F004: unoptimized-feedback rule."""

    def test_rule_metadata(self) -> None:
        """Rule should have correct metadata."""
        rule = UnoptimizedFeedbackRule()
        assert rule.rule_id == "F004"
        assert rule.name == "unoptimized-feedback"
        assert rule.severity == "warning"

    def test_no_cycles_no_violation(self) -> None:
        """Graph without cycles should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/b", family="TOP", operator="blur")
        graph.add_edge("/a", "/b")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cycle_without_cache_violation(self) -> None:
        """Cycle without cache/delay should trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/b", family="TOP", operator="level")
        graph.add_node("/c", family="TOP", operator="composite")

        graph.add_edge("/a", "/b")
        graph.add_edge("/b", "/c")
        graph.add_edge("/c", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].rule == "F004"
        assert "no cache/delay" in violations[0].message

    def test_cycle_with_feedback_no_violation(self) -> None:
        """Cycle with feedback operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/fb", family="TOP", operator="feedback")

        graph.add_edge("/a", "/fb")
        graph.add_edge("/fb", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cycle_with_cache_no_violation(self) -> None:
        """Cycle with cache operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/cache", family="TOP", operator="cache")

        graph.add_edge("/a", "/cache")
        graph.add_edge("/cache", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cycle_with_delay_no_violation(self) -> None:
        """Cycle with delay operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="CHOP", operator="constant")
        graph.add_node("/delay", family="CHOP", operator="delay")

        graph.add_edge("/a", "/delay")
        graph.add_edge("/delay", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cycle_with_lag_no_violation(self) -> None:
        """Cycle with lag operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="CHOP", operator="math")
        graph.add_node("/lag", family="CHOP", operator="lag")

        graph.add_edge("/a", "/lag")
        graph.add_edge("/lag", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cycle_with_timemachine_no_violation(self) -> None:
        """Cycle with timemachine operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/tm", family="TOP", operator="timemachine")

        graph.add_edge("/a", "/tm")
        graph.add_edge("/tm", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_missing_nodes_skipped_in_cycle(self) -> None:
        """MISSING: nodes in cycles should be handled gracefully."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("MISSING:/b", family="MISSING", operator="")
        graph.add_node("/c", family="TOP", operator="level")

        graph.add_edge("/a", "MISSING:/b")
        graph.add_edge("MISSING:/b", "/c")
        graph.add_edge("/c", "/a")

        violations = list(rule.check(graph))
        # Cycle exists and has no proper cache, but missing node is skipped
        assert len(violations) == 1

    def test_multiple_cycles_multiple_violations(self) -> None:
        """Multiple unoptimized cycles should trigger multiple violations."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        # Cycle 1: a -> b -> a
        graph.add_node("/a", family="TOP", operator="blur")
        graph.add_node("/b", family="TOP", operator="level")
        graph.add_edge("/a", "/b")
        graph.add_edge("/b", "/a")

        # Cycle 2: c -> d -> c (separate)
        graph.add_node("/c", family="TOP", operator="composite")
        graph.add_node("/d", family="TOP", operator="blur")
        graph.add_edge("/c", "/d")
        graph.add_edge("/d", "/c")

        violations = list(rule.check(graph))
        assert len(violations) == 2

    def test_feedbackchop_operator_no_violation(self) -> None:
        """Cycle with feedbackchop operator should not trigger violation."""
        rule = UnoptimizedFeedbackRule()
        graph = nx.DiGraph()

        graph.add_node("/a", family="CHOP", operator="constant")
        graph.add_node("/fb", family="CHOP", operator="feedbackchop")

        graph.add_edge("/a", "/fb")
        graph.add_edge("/fb", "/a")

        violations = list(rule.check(graph))
        assert len(violations) == 0


class TestCookEveryFrameRule:
    """Tests for F005: cook-every-frame rule."""

    def test_rule_metadata(self) -> None:
        """Rule should have correct metadata."""
        rule = CookEveryFrameRule()
        assert rule.rule_id == "F005"
        assert rule.name == "cook-every-frame"
        assert rule.severity == "info"

    def test_no_cook_flag_no_violation(self) -> None:
        """Node without cook_every_frame flag should not trigger violation."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        graph.add_node("/op", family="TOP", operator="blur")

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cook_flag_false_no_violation(self) -> None:
        """Node with cook_every_frame=False should not trigger violation."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        graph.add_node("/op", family="TOP", operator="blur", cook_every_frame=False)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_cook_flag_true_violation(self) -> None:
        """Node with cook_every_frame=True should trigger violation."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        graph.add_node("/op", family="TOP", operator="blur", cook_every_frame=True)

        violations = list(rule.check(graph))
        assert len(violations) == 1
        assert violations[0].rule == "F005"
        assert "cook every frame" in violations[0].message
        assert violations[0].context["operator_type"] == "blur"

    def test_time_based_op_no_violation(self) -> None:
        """Time-based operators with cook_every_frame should not trigger violation."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        # These are legitimately time-based
        time_ops = [
            "timer", "constant", "noise", "pattern", "ramp",
            "moviefilein", "videodevin", "audiodevin", "audiofilein",
            "lfo", "beat", "speed", "count", "audiospectrum",
        ]

        for op in time_ops:
            graph.add_node(f"/{op}", family="CHOP", operator=op, cook_every_frame=True)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_mixed_operators(self) -> None:
        """Mixed operators should only flag non-time-based with cook_every_frame."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        # Time-based (should not trigger)
        graph.add_node("/timer", family="CHOP", operator="timer", cook_every_frame=True)
        graph.add_node("/noise", family="TOP", operator="noise", cook_every_frame=True)

        # Non-time-based (should trigger)
        graph.add_node("/blur", family="TOP", operator="blur", cook_every_frame=True)
        graph.add_node("/math", family="CHOP", operator="math", cook_every_frame=True)

        violations = list(rule.check(graph))
        assert len(violations) == 2
        violation_paths = {v.path for v in violations}
        assert "/blur" in violation_paths
        assert "/math" in violation_paths

    def test_missing_nodes_skipped(self) -> None:
        """MISSING: nodes should be skipped."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        graph.add_node("MISSING:/op", family="MISSING", operator="blur", cook_every_frame=True)

        violations = list(rule.check(graph))
        assert len(violations) == 0

    def test_case_insensitive_operator_match(self) -> None:
        """Operator type matching should be case-insensitive."""
        rule = CookEveryFrameRule()
        graph = nx.DiGraph()

        # Upper case operator type should still match time-based list
        graph.add_node("/timer", family="CHOP", operator="Timer", cook_every_frame=True)
        graph.add_node("/lfo", family="CHOP", operator="LFO", cook_every_frame=True)

        violations = list(rule.check(graph))
        assert len(violations) == 0


class TestPerformanceRulesDefaults:
    """Tests for default option values."""

    def test_deep_nesting_default_max_depth(self) -> None:
        """DeepNestingRule default max_depth should be 10."""
        rule = DeepNestingRule()
        graph = nx.DiGraph()

        # Depth 10 should not trigger
        graph.add_node("/1/2/3/4/5/6/7/8/9/10", family="TOP")

        violations = list(rule.check(graph))
        assert len(violations) == 0

        # Depth 11 should trigger
        graph.add_node("/1/2/3/4/5/6/7/8/9/10/11", family="TOP")

        violations = list(rule.check(graph))
        assert len(violations) == 1

    def test_excessive_inputs_default_max_inputs(self) -> None:
        """ExcessiveInputsRule default max_inputs should be 16."""
        rule = ExcessiveInputsRule()
        graph = nx.DiGraph()

        graph.add_node("/target", family="TOP")
        for i in range(16):
            graph.add_node(f"/input{i}", family="TOP")
            graph.add_edge(f"/input{i}", "/target", input_index=i)

        violations = list(rule.check(graph))
        assert len(violations) == 0

        # Add one more to exceed limit
        graph.add_node("/input16", family="TOP")
        graph.add_edge("/input16", "/target", input_index=16)

        violations = list(rule.check(graph))
        assert len(violations) == 1

    def test_heavy_texture_chains_default_max_length(self) -> None:
        """HeavyTextureChainsRule default max_chain_length should be 8."""
        rule = HeavyTextureChainsRule()
        graph = nx.DiGraph()

        # Chain of 8 should not trigger
        for i in range(8):
            graph.add_node(f"/top{i}", family="TOP", operator="blur")
            if i > 0:
                graph.add_edge(f"/top{i-1}", f"/top{i}")

        violations = list(rule.check(graph))
        assert len(violations) == 0

        # Add one more to exceed limit
        graph.add_node("/top8", family="TOP", operator="blur")
        graph.add_edge("/top7", "/top8")

        violations = list(rule.check(graph))
        assert len(violations) == 1

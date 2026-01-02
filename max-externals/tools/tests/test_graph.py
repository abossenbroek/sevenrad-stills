# ruff: noqa: S101
"""Tests for MaxhelpLinter graph-related functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from lint_maxhelp import LintGraph, MaxhelpLinter

from tests.conftest import create_test_patcher


class TestLintGraph:
    """Tests for LintGraph dataclass that builds shared graph state."""

    def test_build_from_valid_patcher(self, tmp_path: Path) -> None:
        """LintGraph.build() creates graph from patcher JSON."""
        # Create minimal patcher with jit.movie -> jit.gl.pix chain
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.filepath = test_file
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        assert isinstance(lint_graph.graph, nx.DiGraph)
        assert len(lint_graph.boxes) == 2
        assert "obj-1" in lint_graph.boxes
        assert "obj-2" in lint_graph.boxes

    def test_type_map_populated(self, tmp_path: Path) -> None:
        """Type map has entries for outlets."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Should have type info for outlet 0
        assert ("obj-1", 0) in lint_graph.type_map
        assert lint_graph.type_map[("obj-1", 0)] == "texture"

    def test_type_map_info_outlet(self, tmp_path: Path) -> None:
        """Type map correctly identifies info outlets."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Outlet 1 should be info type
        assert ("obj-1", 1) in lint_graph.type_map
        assert lint_graph.type_map[("obj-1", 1)] == "info"

    def test_empty_patcher_handled(self, tmp_path: Path) -> None:
        """Empty patcher doesn't crash."""
        data: dict[str, Any] = {"patcher": {"boxes": [], "lines": []}}
        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = data
        linter.boxes = {}

        lint_graph = LintGraph.build(data, linter)
        assert len(lint_graph.boxes) == 0
        assert len(lint_graph.type_map) == 0
        assert isinstance(lint_graph.graph, nx.DiGraph)

    def test_graph_has_edges_for_connections(self, tmp_path: Path) -> None:
        """Graph contains edges for patchlines."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Check that edge exists from outlet to inlet
        assert lint_graph.graph.has_edge(("obj-1", "out", 0), ("obj-2", "in", 0))
        # Check that box-level edge exists
        assert lint_graph.graph.has_edge(("obj-1", "box"), ("obj-2", "box"))

    def test_graph_nodes_have_box_attribute(self, tmp_path: Path) -> None:
        """Graph nodes store box reference in attributes."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Box node should have box attribute
        box_node = lint_graph.graph.nodes.get(("obj-1", "box"))
        assert box_node is not None
        assert "box" in box_node
        assert box_node["box"]["id"] == "obj-1"


class TestCycleDetection:
    """Tests for cycle detection in LintGraph for GPU feedback loops."""

    def test_no_cycles_in_linear_chain(self, tmp_path: Path) -> None:
        """Linear chain has no cycles."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-3",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
            {"source": ["obj-2", 0], "destination": ["obj-3", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) == 0

    def test_detects_direct_cycle(self, tmp_path: Path) -> None:
        """Direct pix->pix feedback detected as cycle."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect1",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect2",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
            {"source": ["obj-2", 0], "destination": ["obj-1", 1]},  # Feedback!
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1

    def test_orphans_detected(self, tmp_path: Path) -> None:
        """Orphaned objects (no connections) detected."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-orphan",
                "maxclass": "newobj",
                "text": "print",
                "numoutlets": 0,
                "numinlets": 1,
            },  # Not connected
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert "obj-orphan" in lint_graph.orphans

    def test_dead_branches_detected(self, tmp_path: Path) -> None:
        """Dead branches (not reaching display) detected."""
        boxes = [
            {
                "id": "obj-main",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-display",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-dead1",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dead2",
                "maxclass": "newobj",
                "text": "print",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {
                "source": ["obj-main", 0],
                "destination": ["obj-display", 0],
            },  # Main path to display
            {
                "source": ["obj-dead1", 0],
                "destination": ["obj-dead2", 0],
            },  # Dead branch
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        # obj-dead1 and obj-dead2 don't reach display
        assert "obj-dead1" in lint_graph.dead_branches
        assert "obj-dead2" in lint_graph.dead_branches

    def test_self_loop_detected(self, tmp_path: Path) -> None:
        """Self-referential loop detected as cycle."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.feedback",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-1", 0],
                "destination": ["obj-1", 1],
            },  # Self-loop
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1
        # Self-loop should appear as single-element cycle
        assert any(len(c) == 1 for c in lint_graph.cycles)

    def test_three_node_cycle(self, tmp_path: Path) -> None:
        """Three-node cycle (A->B->C->A) detected."""
        boxes = [
            {
                "id": "obj-a",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.a",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-b",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.b",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-c",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.c",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-a", 0], "destination": ["obj-b", 0]},
            {"source": ["obj-b", 0], "destination": ["obj-c", 0]},
            {"source": ["obj-c", 0], "destination": ["obj-a", 1]},  # Back to A
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1
        # Should have a 3-node cycle
        assert any(len(c) == 3 for c in lint_graph.cycles)

    def test_all_connected_to_display_no_dead_branches(self, tmp_path: Path) -> None:
        """When all boxes reach display, no dead branches."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-3",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
            {"source": ["obj-2", 0], "destination": ["obj-3", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.dead_branches) == 0

    def test_jit_window_also_display_sink(self, tmp_path: Path) -> None:
        """jit.window is also recognized as a display sink."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "jit.window",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        # All boxes reach the display sink
        assert len(lint_graph.dead_branches) == 0
        assert len(lint_graph.orphans) == 0

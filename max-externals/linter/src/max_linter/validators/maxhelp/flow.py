"""Flow validation mixin for Max help patchers.

This module provides the FlowValidatorMixin class that validates
signal flow and initialization order in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from max_linter.constants import JITTER_DISPLAY_SINKS

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class FlowValidatorMixin:
    """Mixin providing flow validation methods.

    Validates signal flow and initialization order for GPU objects.

    Rules:
        flow-001: Missing qmetro for video playback
        flow-002: Incomplete source->effect->display pipeline
        flow-003: jit.gl.pix inlet not connected
        init-001: jit.world not receiving loadbang
        init-002: jit.movie @drawto references non-existent context
        init-003: No delay between loadbang and jit.movie
    """

    # These attributes must be provided by the composing class
    lint_graph: LintGraph
    boxes: dict[str, dict[str, Any]]

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _find_boxes_by_type(self, type_prefix: str) -> list[str]:
        """Find boxes by type prefix."""
        raise NotImplementedError

    def _find_boxes_by_maxclass(self, maxclass: str) -> list[str]:
        """Find boxes by maxclass."""
        raise NotImplementedError

    def _get_box_text(self, box_id: str) -> str:
        """Get box text by ID."""
        raise NotImplementedError

    def _extract_context_name(self, text: str) -> str | None:
        """Extract context name from jit.world text."""
        raise NotImplementedError

    def _extract_drawto(self, text: str) -> str | None:
        """Extract @drawto value from text."""
        raise NotImplementedError

    def _validate_flow_rules(self) -> bool:
        """Validate signal flow rules using LintGraph.

        Checks for complete video pipelines and connected inlets.

        Rules:
            flow-001: Missing qmetro for video playback (ERROR)
            flow-002: Incomplete source->effect->display pipeline (ERROR)
            flow-003: jit.gl.pix inlet not connected (ERROR)

        Returns:
            True if all flow rules pass, False otherwise.
        """
        valid = True
        lint_graph = self.lint_graph  # Use pre-built LintGraph

        # Find key objects
        qmetros = self._find_boxes_by_type("qmetro")
        jit_movies = self._find_boxes_by_type("jit.movie")
        jit_gl_pix = self._find_boxes_by_type("jit.gl.pix")
        display_sinks: list[str] = []
        for box_id, box in self.boxes.items():
            maxclass = box.get("maxclass", "")
            text = box.get("text", "")
            if maxclass in JITTER_DISPLAY_SINKS or any(
                s in text for s in JITTER_DISPLAY_SINKS
            ):
                display_sinks.append(box_id)

        # flow-001: Missing qmetro for video playback
        # If there's a jit.movie, there should be a qmetro driving it
        if jit_movies and not qmetros:
            self.error(
                "flow-001",
                "No qmetro found for video playback. Add qmetro to drive jit.movie.",
            )
            valid = False

        # flow-002: Incomplete pipeline (source->effect->display)
        for pix_id in jit_gl_pix:
            # Check if pix reaches a display sink
            reaches_display = False
            for sink_id in display_sinks:
                try:
                    if nx.has_path(lint_graph.graph, (pix_id, "box"), (sink_id, "box")):
                        reaches_display = True
                        break
                except nx.NetworkXError:
                    continue

            if not reaches_display and display_sinks:
                self.error(
                    "flow-002",
                    "jit.gl.pix output not connected to display "
                    "(jit.pwindow/jit.window)",
                    pix_id,
                )
                valid = False

        # flow-003: jit.gl.pix inlet 0 not connected
        for pix_id in jit_gl_pix:
            inlet_0_node = (pix_id, "in", 0)
            has_input = any(
                edge[1] == inlet_0_node for edge in lint_graph.graph.edges()
            )

            if not has_input:
                self.error(
                    "flow-003",
                    "jit.gl.pix inlet 0 not connected - no texture input",
                    pix_id,
                )
                valid = False

        return valid

    def _validate_init_order(self) -> bool:
        """Validate initialization order for GPU objects.

        Rules:
            init-001: jit.world not receiving loadbang (ERROR)
            init-002: jit.movie @drawto references non-existent context (ERROR)
            init-003: No delay between loadbang and jit.movie (WARNING)

        Returns:
            True if all init rules pass, False otherwise.
        """
        valid = True
        lint_graph = self.lint_graph  # Use pre-built LintGraph

        loadbangs = self._find_boxes_by_maxclass("loadbang")
        loadbangs.extend(
            b
            for b in self._find_boxes_by_maxclass("newobj")
            if "loadbang" in self._get_box_text(b)
        )
        jit_worlds = self._find_boxes_by_type("jit.world")
        jit_movies = self._find_boxes_by_type("jit.movie")

        # init-001: Check jit.world receives loadbang
        for world_id in jit_worlds:
            has_loadbang_path = False
            for lb_id in loadbangs:
                try:
                    if nx.has_path(lint_graph.graph, (lb_id, "box"), (world_id, "box")):
                        has_loadbang_path = True
                        break
                except nx.NetworkXError:
                    continue
            if not has_loadbang_path:
                self.error(
                    "init-001",
                    "jit.world not receiving loadbang - "
                    "context may not initialize properly",
                    world_id,
                )
                valid = False

        # Build set of defined contexts
        defined_contexts: set[str] = set()
        for world_id in jit_worlds:
            ctx = self._extract_context_name(self._get_box_text(world_id))
            if ctx:
                defined_contexts.add(ctx)

        # init-002: Check jit.movie @drawto references valid context
        for movie_id in jit_movies:
            drawto = self._extract_drawto(self._get_box_text(movie_id))
            if drawto and drawto not in defined_contexts:
                self.error(
                    "init-002",
                    f"jit.movie @drawto '{drawto}' references non-existent context. "
                    f"Defined contexts: {defined_contexts or 'none'}",
                    movie_id,
                )
                valid = False

        # init-003: Check for delay between loadbang and jit.movie
        for movie_id in jit_movies:
            movie_text = self._get_box_text(movie_id)
            if "@output_texture 1" not in movie_text:
                continue
            for lb_id in loadbangs:
                try:
                    if nx.has_path(lint_graph.graph, (lb_id, "box"), (movie_id, "box")):
                        path = nx.shortest_path(
                            lint_graph.graph, (lb_id, "box"), (movie_id, "box")
                        )
                        path_texts = [self._get_box_text(node[0]) for node in path]
                        if not any("delay" in t or "pipe" in t for t in path_texts):
                            self.warning(
                                "init-003",
                                "No delay between loadbang and jit.movie - "
                                "may cause race condition. Add 'delay 100' "
                                "after loadbang.",
                                movie_id,
                            )
                except nx.NetworkXError:
                    continue

        return valid

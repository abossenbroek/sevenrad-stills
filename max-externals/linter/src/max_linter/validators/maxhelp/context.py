"""Context validation mixin for Max help patchers.

This module provides the ContextValidatorMixin class that validates
OpenGL context naming and initialization in Max/MSP help patchers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class ContextValidatorMixin:
    """Mixin providing context validation methods.

    Validates OpenGL context naming conventions and initialization order.

    Rules:
        ctx-001: Dots in context name (ERROR)
        ctx-002: @drawto references non-existent context (ERROR)
        ctx-003: Multiple jit.world with same context name (ERROR)
        context-naming: Context naming convention warnings
        context-init: Context initialization order warnings
    """

    # These attributes must be provided by the composing class
    filepath: Path | None
    graph: nx.DiGraph
    lint_graph: LintGraph

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _get_box_text(self, box_id: str) -> str:
        """Get box text by ID."""
        raise NotImplementedError

    def _find_boxes_by_type(self, type_prefix: str) -> list[str]:
        """Find boxes by type prefix."""
        raise NotImplementedError

    def _find_boxes_by_maxclass(self, maxclass: str) -> list[str]:
        """Find boxes by maxclass."""
        raise NotImplementedError

    def _extract_context_name(self, text: str) -> str | None:
        """Extract context name from object text like 'jit.world sr_ctx @visible 0'."""
        parts = text.split()
        if len(parts) >= 2:
            # Context name is usually the second word (after object name)
            candidate = parts[1]
            # Skip if it's an attribute (starts with @)
            if not candidate.startswith("@"):
                return candidate
        return None

    def _extract_drawto(self, text: str) -> str | None:
        """Extract @drawto value from object text."""
        match = re.search(r"@drawto\s+(\S+)", text)
        if match:
            return match.group(1)
        return None

    def _get_expected_context_name(self) -> str:
        """Derive expected context name from filename."""
        if self.filepath is None:
            return "sr_unknown_ctx"
        stem = self.filepath.stem  # e.g., "sr.bandswap"
        # Convert dots to underscores and add _ctx suffix
        name = stem.replace(".", "_") + "_ctx"
        return name

    def _validate_context_naming(self) -> bool:
        """Validate context names use underscores, not dots."""
        valid = True
        expected = self._get_expected_context_name()

        # Find all context-related objects
        jit_worlds = self._find_boxes_by_type("jit.world")
        jit_movies = self._find_boxes_by_type("jit.movie")
        jit_gl_pix = self._find_boxes_by_type("jit.gl.pix")
        jit_gl_texture = self._find_boxes_by_type("jit.gl.texture")

        contexts_found: set[str] = set()

        # Check jit.world context names
        for box_id in jit_worlds:
            text = self._get_box_text(box_id)
            ctx = self._extract_context_name(text)
            if ctx:
                contexts_found.add(ctx)
                if "." in ctx:
                    self.error(
                        "context-naming",
                        f"Context name '{ctx}' contains dots. "
                        f"Use underscores: '{ctx.replace('.', '_')}'",
                        box_id,
                    )
                    valid = False

        # Check @drawto attributes
        for box_id in jit_movies + jit_gl_pix + jit_gl_texture:
            text = self._get_box_text(box_id)
            ctx = self._extract_drawto(text)
            if ctx:
                contexts_found.add(ctx)
                if "." in ctx:
                    self.error(
                        "context-naming",
                        f"@drawto context '{ctx}' contains dots. Use underscores.",
                        box_id,
                    )
                    valid = False

        # Warn if context name doesn't match expected pattern
        for ctx in contexts_found:
            if ctx != expected and not ctx.startswith("sr_"):
                self.warning(
                    "context-naming",
                    f"Context name '{ctx}' may not follow convention. "
                    f"Expected: '{expected}'",
                )

        return valid

    def _validate_context_initialization(self) -> bool:
        """Validate that jit.world is properly initialized before jit.movie uses it.

        Uses networkx path finding to verify initialization order.
        """
        valid = True

        jit_worlds = self._find_boxes_by_type("jit.world")
        jit_movies = self._find_boxes_by_type("jit.movie")
        loadbangs = self._find_boxes_by_maxclass("newobj")
        loadbangs = [b for b in loadbangs if "loadbang" in self._get_box_text(b)]

        # Check each jit.movie with @output_texture
        for movie_id in jit_movies:
            text = self._get_box_text(movie_id)
            if "@output_texture" not in text:
                continue

            drawto = self._extract_drawto(text)
            if not drawto:
                self.warning(
                    "context-init",
                    "jit.movie has @output_texture but no @drawto context",
                    movie_id,
                )
                continue

            # Find corresponding jit.world
            matching_world = None
            for world_id in jit_worlds:
                world_text = self._get_box_text(world_id)
                world_ctx = self._extract_context_name(world_text)
                if world_ctx == drawto:
                    matching_world = world_id
                    break

            if not matching_world:
                self.error(
                    "context-init",
                    f"jit.movie uses @drawto {drawto} but no jit.world "
                    "with that context exists",
                    movie_id,
                )
                valid = False
                continue

            # Check if jit.world receives input from loadbang
            world_has_input = False
            for loadbang_id in loadbangs:
                try:
                    if nx.has_path(
                        self.graph, (loadbang_id, "box"), (matching_world, "box")
                    ):
                        world_has_input = True
                        break
                except nx.NetworkXError:
                    pass

            if not world_has_input:
                self.error(
                    "context-init",
                    f"jit.world {drawto} is not connected to loadbang - "
                    "context won't initialize",
                    matching_world,
                )
                valid = False

            # Check if there's a delay between loadbang and jit.movie
            for loadbang_id in loadbangs:
                try:
                    if nx.has_path(self.graph, (loadbang_id, "box"), (movie_id, "box")):
                        # Find the path and check for delay
                        paths = list(
                            nx.all_simple_paths(
                                self.graph, (loadbang_id, "box"), (movie_id, "box")
                            )
                        )
                        for path in paths:
                            has_delay = any(
                                "delay" in self._get_box_text(node[0])
                                for node in path
                                if isinstance(node, tuple) and len(node) >= 1
                            )
                            if not has_delay:
                                self.warning(
                                    "context-init",
                                    "Path from loadbang to jit.movie has no delay - "
                                    "may cause race condition",
                                    movie_id,
                                )
                except nx.NetworkXError:
                    pass

        return valid

    def _validate_context_rules(self) -> bool:
        """Validate OpenGL context rules using LintGraph.

        Uses self.lint_graph.boxes (built once at validation start) instead of
        rebuilding box lists via _find_boxes_by_type().

        Checks context naming, existence, and uniqueness.

        Rules:
            ctx-001: Dots in context name (ERROR)
            ctx-002: @drawto references non-existent context (ERROR)
            ctx-003: Multiple jit.world with same context name (ERROR)
        """
        valid = True

        # Use LintGraph.boxes instead of rebuilding lists
        # Filter boxes by type from the pre-built boxes dict
        jit_worlds: list[str] = []
        drawto_boxes: list[str] = []  # jit.movie, jit.gl.pix, jit.gl.texture

        for box_id, box in self.lint_graph.boxes.items():
            text = box.get("text", "")
            if text.startswith("jit.world"):
                jit_worlds.append(box_id)
            elif text.startswith(("jit.movie", "jit.gl.pix", "jit.gl.texture")):
                drawto_boxes.append(box_id)

        # Track defined contexts and their defining objects
        context_definitions: dict[str, list[str]] = {}  # ctx_name -> [box_ids]

        for world_id in jit_worlds:
            world_text = self._get_box_text(world_id)
            ctx = self._extract_context_name(world_text)
            if ctx:
                # ctx-001: Check for dots in context name
                if "." in ctx:
                    self.error(
                        "ctx-001",
                        f"Context name '{ctx}' contains dots - use underscores "
                        f"instead: '{ctx.replace('.', '_')}'",
                        world_id,
                    )
                    valid = False

                # Track for duplicate detection
                if ctx not in context_definitions:
                    context_definitions[ctx] = []
                context_definitions[ctx].append(world_id)

        # ctx-003: Check for duplicate context definitions
        for ctx_name, defining_boxes in context_definitions.items():
            if len(defining_boxes) > 1:
                self.error(
                    "ctx-003",
                    f"Multiple jit.world objects define context '{ctx_name}': "
                    f"{defining_boxes}",
                    defining_boxes[0],
                )
                valid = False

        # ctx-002: Check @drawto references exist
        defined_contexts = set(context_definitions.keys())

        for box_id in drawto_boxes:
            box_text = self._get_box_text(box_id)
            drawto = self._extract_drawto(box_text)

            if drawto:
                # ctx-001: Check for dots in @drawto value
                if "." in drawto:
                    self.error(
                        "ctx-001",
                        f"@drawto context '{drawto}' contains dots - use underscores",
                        box_id,
                    )
                    valid = False

                # ctx-002: Check context exists
                if drawto not in defined_contexts:
                    self.error(
                        "ctx-002",
                        f"@drawto references non-existent context '{drawto}'. "
                        f"Defined contexts: {defined_contexts or 'none'}",
                        box_id,
                    )
                    valid = False

        return valid

#!/usr/bin/env python3
"""
Graph-based linter for .maxhelp files.

Validates that help patchers have correct structure, initialization order,
complete signal flow, proper UI components for shader parameters, and proper
UI layout using networkx graph analysis.

Validation Features:
    - Context naming conventions (underscores, not dots)
    - OpenGL context initialization order
    - Complete GPU texture signal flow
    - Parameter UI controls and connections
    - UI component overlap detection

Usage:
    python lint_maxhelp.py help/*.maxhelp
    python lint_maxhelp.py --strict help/sr.bandswap.maxhelp
    python lint_maxhelp.py --verbose help/*.maxhelp

Exit codes:
    0: All validations passed
    1: One or more validations failed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx


class Severity(Enum):
    """Validation message severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class LintError:
    """Represents a validation error with severity and location."""

    severity: Severity
    rule: str
    message: str
    object_id: str | None = None

    def __str__(self) -> str:
        location = f" ({self.object_id})" if self.object_id else ""
        return f"  [{self.severity.value}] {self.rule}{location}: {self.message}"


class MaxhelpLinter:
    """Validates .maxhelp files using graph-based analysis."""

    def __init__(self, strict: bool = False, verbose: bool = False):
        self.strict = strict
        self.verbose = verbose
        self.errors: list[LintError] = []
        self.warnings: list[LintError] = []
        self.filepath: Path | None = None
        self.data: dict[str, Any] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.boxes: dict[str, dict[str, Any]] = {}

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add an error."""
        self.errors.append(LintError(Severity.ERROR, rule, message, object_id))

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add a warning."""
        self.warnings.append(LintError(Severity.WARNING, rule, message, object_id))

    def info(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Add an info message (only shown in verbose mode)."""
        if self.verbose:
            self.warnings.append(LintError(Severity.INFO, rule, message, object_id))

    def _build_graph(self) -> nx.DiGraph:
        """
        Build directed graph where nodes are (box_id, port_type, port_num) tuples.

        This allows tracking signal flow through the patcher.
        """
        graph = nx.DiGraph()

        # Add nodes for each box's inlets and outlets
        for box_wrapper in self.data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")

            # Store box reference in node attributes
            for i in range(box.get("numoutlets", 0)):
                graph.add_node((box_id, "out", i), box=box)
            for i in range(box.get("numinlets", 0)):
                graph.add_node((box_id, "in", i), box=box)

            # Also add a "box" node for easier lookup
            graph.add_node((box_id, "box"), box=box)

        # Add edges from patchlines
        for line in self.data.get("patcher", {}).get("lines", []):
            patchline = line.get("patchline", {})
            src = patchline.get("source", [])
            dst = patchline.get("destination", [])

            if len(src) >= 2 and len(dst) >= 2:
                src_node = (src[0], "out", src[1])
                dst_node = (dst[0], "in", dst[1])
                graph.add_edge(src_node, dst_node)

                # Also connect outlet to inlet at box level for path finding
                graph.add_edge((src[0], "box"), (dst[0], "box"))

        return graph

    def _get_box_text(self, box_id: str) -> str:
        """Get the text content of a box."""
        box = self.boxes.get(box_id, {})
        return str(box.get("text", ""))

    def _find_boxes_by_type(self, type_pattern: str) -> list[str]:
        """Find all boxes whose text starts with the given pattern."""
        result = []
        for box_id, box in self.boxes.items():
            text = box.get("text", "")
            if text.startswith(type_pattern):
                result.append(box_id)
        return result

    def _find_boxes_by_maxclass(self, maxclass: str) -> list[str]:
        """Find all boxes with the given maxclass."""
        result = []
        for box_id, box in self.boxes.items():
            if box.get("maxclass") == maxclass:
                result.append(box_id)
        return result

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

    def validate_file(self, filepath: Path) -> bool:
        """
        Validate a single .maxhelp file.

        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        self.filepath = filepath

        if not filepath.exists():
            self.error("file", f"File does not exist: {filepath}")
            return False

        if filepath.suffix != ".maxhelp":
            self.error(
                "file", f"File must have .maxhelp extension, got: {filepath.suffix}"
            )
            return False

        # Read and parse JSON
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            self.error("file", f"Failed to read file: {e}")
            return False

        try:
            self.data = json.loads(content)
        except json.JSONDecodeError as e:
            self.error("json", f"Invalid JSON syntax: {e.msg} at line {e.lineno}")
            return False

        # Build box lookup and graph
        self.boxes = {}
        for box_wrapper in self.data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")
            if box_id:
                self.boxes[box_id] = box

        self.graph = self._build_graph()

        # Run all validations
        valid = True
        valid &= self._validate_structure()
        valid &= self._validate_context_naming()
        valid &= self._validate_context_initialization()
        valid &= self._validate_signal_flow()
        valid &= self._validate_inlet_connections()
        valid &= self._validate_metadata()
        valid &= self._validate_parameter_ui()
        valid &= self._validate_no_overlaps()

        return valid

    def _validate_structure(self) -> bool:
        """Validate basic JSON structure."""
        valid = True

        if "patcher" not in self.data:
            self.error("structure", "Missing 'patcher' key")
            return False

        patcher = self.data["patcher"]

        required = ["boxes", "lines"]
        for field in required:
            if field not in patcher:
                self.error("structure", f"Missing required field: '{field}'")
                valid = False

        return valid

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
                        f"Context name '{ctx}' contains dots. Use underscores: '{ctx.replace('.', '_')}'",
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
                    f"Context name '{ctx}' may not follow convention. Expected: '{expected}'",
                )

        return valid

    def _validate_context_initialization(self) -> bool:
        """
        Validate that jit.world is properly initialized before jit.movie uses it.

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
                    f"jit.movie uses @drawto {drawto} but no jit.world with that context exists",
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
                    f"jit.world {drawto} is not connected to loadbang - context won't initialize",
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
                                    "Path from loadbang to jit.movie has no delay - may cause race condition",
                                    movie_id,
                                )
                except nx.NetworkXError:
                    pass

        return valid

    def _validate_signal_flow(self) -> bool:
        """
        Validate the GPU texture pipeline is complete.

        Expected flow: qmetro → jit.movie → jit.gl.pix → jit.pwindow
        """
        valid = True

        qmetros = [
            b
            for b in self._find_boxes_by_maxclass("newobj")
            if "qmetro" in self._get_box_text(b)
        ]
        jit_movies = self._find_boxes_by_type("jit.movie")
        jit_gl_pixs = self._find_boxes_by_type("jit.gl.pix")
        jit_pwindows = self._find_boxes_by_maxclass("jit.pwindow")

        # Check qmetro → jit.movie
        if qmetros and jit_movies:
            qmetro_to_movie = False
            for qmetro_id in qmetros:
                for movie_id in jit_movies:
                    try:
                        if nx.has_path(
                            self.graph, (qmetro_id, "box"), (movie_id, "box")
                        ):
                            qmetro_to_movie = True
                            break
                    except nx.NetworkXError:
                        pass
                if qmetro_to_movie:
                    break

            if not qmetro_to_movie:
                self.error(
                    "signal-flow",
                    "No path from qmetro to jit.movie - video won't update",
                )
                valid = False

        # Check jit.movie → jit.gl.pix
        if jit_movies and jit_gl_pixs:
            movie_to_pix = False
            for movie_id in jit_movies:
                for pix_id in jit_gl_pixs:
                    try:
                        if nx.has_path(self.graph, (movie_id, "box"), (pix_id, "box")):
                            movie_to_pix = True
                            break
                    except nx.NetworkXError:
                        pass
                if movie_to_pix:
                    break

            if not movie_to_pix:
                self.error(
                    "signal-flow",
                    "No path from jit.movie to jit.gl.pix - effect won't be applied",
                )
                valid = False

        # Check jit.gl.pix → jit.pwindow
        if jit_gl_pixs and jit_pwindows:
            pix_to_window = False
            for pix_id in jit_gl_pixs:
                for window_id in jit_pwindows:
                    try:
                        if nx.has_path(self.graph, (pix_id, "box"), (window_id, "box")):
                            pix_to_window = True
                            break
                    except nx.NetworkXError:
                        pass
                if pix_to_window:
                    break

            if not pix_to_window:
                self.error(
                    "signal-flow",
                    "No path from jit.gl.pix to jit.pwindow - output won't display",
                )
                valid = False

        return valid

    def _validate_inlet_connections(self) -> bool:
        """
        Validate that all required inlets are connected.

        For jit.gl.pix with multiple inlets, warn if not all are connected.
        """
        valid = True

        jit_gl_pixs = self._find_boxes_by_type("jit.gl.pix")

        for pix_id in jit_gl_pixs:
            box = self.boxes.get(pix_id, {})
            numinlets = box.get("numinlets", 1)

            if numinlets > 1:
                # Check which inlets are connected
                connected_inlets = set()
                for line in self.data.get("patcher", {}).get("lines", []):
                    patchline = line.get("patchline", {})
                    dst = patchline.get("destination", [])
                    if len(dst) >= 2 and dst[0] == pix_id:
                        connected_inlets.add(dst[1])

                missing = set(range(numinlets)) - connected_inlets
                if missing:
                    self.warning(
                        "inlet-connections",
                        f"jit.gl.pix has {numinlets} inlets but inlet(s) {sorted(missing)} not connected",
                        pix_id,
                    )

        return valid

    def _validate_metadata(self) -> bool:
        """Validate patcher metadata (description, tags)."""
        valid = True
        patcher = self.data.get("patcher", {})

        if not patcher.get("description"):
            self.warning("metadata", "Missing 'description' field")

        if not patcher.get("tags"):
            self.warning("metadata", "Missing 'tags' field")

        return valid

    def _are_boxes_connected(self, id1: str, id2: str) -> bool:
        """Check if two boxes are directly connected by a patchline."""
        try:
            # Check both directions
            if nx.has_path(self.graph, (id1, "box"), (id2, "box")):
                # Verify it's a direct connection (path length 1)
                path = nx.shortest_path(self.graph, (id1, "box"), (id2, "box"))
                if len(path) == 2:
                    return True
            if nx.has_path(self.graph, (id2, "box"), (id1, "box")):
                path = nx.shortest_path(self.graph, (id2, "box"), (id1, "box"))
                if len(path) == 2:
                    return True
        except nx.NetworkXError:
            pass
        return False

    def _validate_no_overlaps(self) -> bool:
        """
        Validate that UI components don't overlap each other.

        This method detects overlapping UI elements to help identify layout issues
        in Max help patchers. Overlaps are calculated using bounding box intersection
        based on each object's patching_rect [x, y, width, height].

        Overlap Detection Rules:
            - Significant overlap (>25% of smaller box): ERROR
            - Minor overlap (>100px² but <25%): WARNING
            - Tiny overlap (≤100px²): Ignored
            - Connected boxes: Allowed to overlap (common Max style)
            - Comments: Excluded (often used as labels)

        Interactive Object Types Checked:
            - dial, slider, button, toggle
            - number, flonum, message, newobj
            - umenu, jit.pwindow

        Bounding Box Calculation:
            For most objects, patching_rect directly provides [x, y, w, h].
            The width and height are used as-is from the Max patcher JSON.

        Connected Box Exception:
            Boxes that are directly connected by patchlines are allowed to overlap,
            as this is a common Max patching style (e.g., dial connected to number).

        Returns:
            True if no significant overlaps found, False otherwise
        """
        valid = True

        # UI element types that should not overlap (excluding comments)
        interactive_types = {
            "dial",
            "slider",
            "button",
            "toggle",
            "number",
            "flonum",
            "message",
            "newobj",
            "umenu",
            "jit.pwindow",
        }

        # Collect interactive boxes with their rectangles
        interactive_boxes: list[tuple[str, list[float]]] = []

        for box_id, box in self.boxes.items():
            maxclass = box.get("maxclass", "")
            if maxclass in interactive_types:
                rect = box.get("patching_rect", [])
                if len(rect) >= 4:
                    interactive_boxes.append((box_id, rect))

        # Check each pair for overlaps
        for i, (id1, rect1) in enumerate(interactive_boxes):
            x1, y1, w1, h1 = rect1[0], rect1[1], rect1[2], rect1[3]

            for id2, rect2 in interactive_boxes[i + 1 :]:
                x2, y2, w2, h2 = rect2[0], rect2[1], rect2[2], rect2[3]

                # Check for rectangle intersection
                h_overlap = x1 < x2 + w2 and x1 + w1 > x2
                v_overlap = y1 < y2 + h2 and y1 + h1 > y2

                if h_overlap and v_overlap:
                    # Skip if boxes are directly connected (intentional overlap)
                    if self._are_boxes_connected(id1, id2):
                        continue

                    # Calculate overlap area for severity assessment
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    min_area = min(w1 * h1, w2 * h2)

                    # Significant overlap (>25% of smaller box)
                    if overlap_area > 0.25 * min_area:
                        self.error(
                            "overlap",
                            f"Boxes '{id1}' and '{id2}' overlap "
                            f"({int(overlap_area)}px²)",
                        )
                        valid = False
                    elif overlap_area > 100:  # Only warn for overlaps > 100px²
                        self.warning(
                            "overlap",
                            f"Boxes '{id1}' and '{id2}' partially overlap "
                            f"({int(overlap_area)}px²)",
                        )

        return valid

    def _extract_gen_shader(self, text: str) -> str | None:
        """Extract @gen shader name from jit.gl.pix text."""
        match = re.search(r"@gen\s+(\S+)", text)
        if match:
            return match.group(1)
        return None

    def _strip_comments(self, code: str) -> str:
        """
        Strip C-style comments from code to avoid false positives in GLSL detection.

        Removes:
        - Single-line comments: // ...
        - Block comments: /* ... */

        Args:
            code: Source code string

        Returns:
            Code with comments replaced by whitespace (preserves line numbers)
        """
        result = []
        i = 0
        in_block_comment = False

        while i < len(code):
            if in_block_comment:
                # Look for end of block comment
                if code[i : i + 2] == "*/":
                    in_block_comment = False
                    result.append("  ")  # Replace */ with spaces
                    i += 2
                else:
                    # Preserve newlines for line number tracking
                    result.append("\n" if code[i] == "\n" else " ")
                    i += 1
            elif code[i : i + 2] == "/*":
                # Start of block comment
                in_block_comment = True
                result.append("  ")  # Replace /* with spaces
                i += 2
            elif code[i : i + 2] == "//":
                # Single-line comment - skip to end of line
                while i < len(code) and code[i] != "\n":
                    result.append(" ")
                    i += 1
            else:
                result.append(code[i])
                i += 1

        return "".join(result)

    def _find_genjit_file(self, shader_name: str) -> Path | None:
        """Find the .genjit file for a shader name."""
        if self.filepath is None:
            return None

        # Look in code/ directory relative to help/
        code_dir = self.filepath.parent.parent / "code"
        genjit_file = code_dir / f"{shader_name}.genjit"

        if genjit_file.exists():
            return genjit_file
        return None

    def _parse_genjit_params(self, genjit_path: Path) -> list[dict[str, Any]]:
        """
        Parse parameters from a .genjit file.

        Returns list of dicts with 'name' and 'default' keys.
        """
        params: list[dict[str, Any]] = []

        try:
            content = genjit_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return params

        # Find boxes with text starting with "param "
        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            text = box.get("text", "")
            if text.startswith("param "):
                parts = text.split()
                if len(parts) >= 3:
                    # Format: "param name default" or "param name default min max"
                    params.append(
                        {
                            "name": parts[1],
                            "default": parts[2],
                        }
                    )

        return params

    def _validate_genjit_format(
        self, genjit_path: Path, shader_name: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Validate that a .genjit file uses proper GenExpr format, not GLSL/XML.

        Detects GLSL/XML markers that indicate wrong shader format:
        - XML tags: <jit.gl.pix>, <param name=, <language name="glsl"
        - GLSL keywords: #version, uniform, void main(), gl_FragColor, texture2DRect

        Validates GenExpr requirements:
        - Has proper 'param name default' objects (not XML <param>)
        - Codebox uses GenExpr syntax (in1, out1, sample, norm, dim)
        - No user-defined function syntax (name() { ... })

        Args:
            genjit_path: Path to the .genjit file
            shader_name: Name of the shader for error messages

        Returns:
            Tuple of (is_valid, params_list)
        """
        valid = True
        params: list[dict[str, Any]] = []

        try:
            content = genjit_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self.error(
                "genjit-format",
                f"Invalid JSON in {shader_name}.genjit: {e.msg}",
            )
            return False, params
        except OSError as e:
            self.error(
                "genjit-format",
                f"Cannot read {shader_name}.genjit: {e}",
            )
            return False, params

        # Find codebox and param objects
        codebox_content: str | None = None
        has_param_objects = False

        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            maxclass = box.get("maxclass", "")
            text = box.get("text", "")

            # Check for proper param objects
            if maxclass == "newobj" and text.startswith("param "):
                has_param_objects = True
                parts = text.split()
                if len(parts) >= 3:
                    params.append({"name": parts[1], "default": parts[2]})

            # Get codebox content
            if maxclass == "codebox":
                codebox_content = box.get("code", "")

        if codebox_content is None:
            self.error(
                "genjit-format",
                f"No codebox found in {shader_name}.genjit",
            )
            return False, params

        # Strip comments before checking for GLSL markers to avoid false positives
        # (e.g., "uniform weighting" in a comment should not trigger GLSL detection)
        code_without_comments = self._strip_comments(codebox_content)

        # GLSL/XML markers that indicate wrong format
        glsl_xml_markers = [
            ("<jit.gl.pix>", "XML wrapper <jit.gl.pix> (use GenExpr codebox instead)"),
            ("</jit.gl.pix>", "XML closing tag </jit.gl.pix>"),
            ('<param name="', "XML parameter declaration <param name="),
            ('<language name="glsl"', "GLSL language declaration"),
            ("#version", "GLSL #version directive"),
            ("uniform ", "GLSL uniform declaration"),
            ("void main()", "GLSL main function"),
            ("gl_FragColor", "GLSL gl_FragColor output"),
            ("texture2DRect", "GLSL texture2DRect function"),
            ("varying ", "GLSL varying declaration"),
            ("<![CDATA[", "XML CDATA section"),
        ]

        for marker, description in glsl_xml_markers:
            if marker in code_without_comments:
                self.error(
                    "genjit-format",
                    f"GLSL/XML format detected in {shader_name}.genjit: {description}. "
                    f"Convert to GenExpr format with 'param name default' objects.",
                )
                valid = False

        # Check for invalid function definition syntax (GenExpr doesn't support this)
        # Pattern: identifier(params) { ... }
        func_def_pattern = re.compile(r"\b\w+\s*\([^)]*\)\s*\{")
        if func_def_pattern.search(code_without_comments):
            # Make sure it's not a valid GenExpr construct like if() { or for() {
            # by checking if it looks like a function definition
            lines = code_without_comments.split("\n")
            for line in lines:
                line = line.strip()
                # Skip control flow statements
                if any(
                    line.startswith(kw)
                    for kw in ["if", "else", "for", "while", "switch"]
                ):
                    continue
                # Check for function-like definition
                if func_def_pattern.match(line):
                    self.error(
                        "genjit-format",
                        f"Invalid function definition syntax in {shader_name}.genjit. "
                        f"GenExpr doesn't support user-defined functions. Inline the code instead.",
                    )
                    valid = False
                    break

        # Check for GenExpr requirements (only if not already detected as GLSL)
        if valid:
            genexpr_markers = ["out1", "in1", "sample(", "norm", "dim"]
            has_genexpr = any(marker in codebox_content for marker in genexpr_markers)

            if not has_genexpr:
                self.warning(
                    "genjit-format",
                    f"No GenExpr markers found in {shader_name}.genjit "
                    f"(expected: out1, in1, sample, norm, dim)",
                )

            # Check for param objects if parameters are used
            if not has_param_objects and params:
                self.warning(
                    "genjit-format",
                    f"No 'param' objects found in {shader_name}.genjit. "
                    f"Parameters should be declared as 'param name default' objects.",
                )

        return valid, params

    def _find_ui_controls(self) -> dict[str, list[str]]:
        """
        Find all UI control boxes in the help patcher.

        Returns dict mapping control type to list of box IDs.
        """
        ui_controls: dict[str, list[str]] = {
            "dial": [],
            "flonum": [],
            "number": [],
            "slider": [],
            "button": [],
            "toggle": [],
            "umenu": [],
        }

        for box_id, box in self.boxes.items():
            maxclass = box.get("maxclass", "")
            if maxclass in ui_controls:
                ui_controls[maxclass].append(box_id)

        return ui_controls

    def _get_interactive_control_ids(
        self, ui_controls: dict[str, list[str]]
    ) -> list[str]:
        """
        Get IDs of interactive UI controls (dial, slider, toggle, button).

        These are controls that provide visual/tactile interaction beyond
        simple text entry (number, flonum).
        """
        interactive_types = ["dial", "slider", "toggle", "button"]
        ids = []
        for ctrl_type in interactive_types:
            ids.extend(ui_controls.get(ctrl_type, []))
        return ids

    def _find_param_messages(self) -> dict[str, list[str]]:
        """
        Find message boxes that send parameters to jit.gl.pix.

        Returns dict mapping parameter name to list of box IDs.
        Looks for patterns like:
        - "param_name $1" (variable message)
        - "param_name VALUE" (fixed value, for cycling buttons)
        - "param1 V1, param2 V2, ..." (multi-param preset messages)
        - "prepend param_name" (newobj)
        """
        param_messages: dict[str, list[str]] = {}

        for box_id, box in self.boxes.items():
            text = box.get("text", "")
            maxclass = box.get("maxclass", "")

            # Check for message box with "$1" pattern: "param_name $1"
            if maxclass == "message" and "$1" in text:
                # Extract parameter name (first word before $1)
                parts = text.split()
                if len(parts) >= 2 and "$1" in text:
                    param_name = parts[0]
                    if param_name not in param_messages:
                        param_messages[param_name] = []
                    param_messages[param_name].append(box_id)

            # Check for message box with fixed or multi-param values
            elif maxclass == "message" and "$1" not in text:
                # Split by comma for multi-param messages like "perm_r 0, perm_g 1"
                segments = [s.strip() for s in text.split(",")]

                for segment in segments:
                    parts = segment.split()
                    if len(parts) == 2:
                        param_name = parts[0]
                        # Check if second part looks like a number (int or float)
                        try:
                            float(parts[1])
                            if param_name not in param_messages:
                                param_messages[param_name] = []
                            param_messages[param_name].append(box_id)
                        except ValueError:
                            pass

            # Check for newobj with "prepend param_name"
            elif maxclass == "newobj" and text.startswith("prepend "):
                parts = text.split()
                if len(parts) >= 2:
                    param_name = parts[1]
                    if param_name not in param_messages:
                        param_messages[param_name] = []
                    param_messages[param_name].append(box_id)

        return param_messages

    def _check_ui_to_pix_connection(
        self, ui_box_ids: list[str], param_box_ids: list[str], pix_ids: list[str]
    ) -> bool:
        """
        Check if there's a path from any UI control through param message to jit.gl.pix.

        Returns True if a valid connection chain exists.
        """
        for ui_id in ui_box_ids:
            for param_id in param_box_ids:
                # Check UI → param message connection
                try:
                    if nx.has_path(self.graph, (ui_id, "box"), (param_id, "box")):
                        # Check param message → jit.gl.pix connection
                        for pix_id in pix_ids:
                            if nx.has_path(
                                self.graph, (param_id, "box"), (pix_id, "box")
                            ):
                                return True
                except nx.NetworkXError:
                    pass

        return False

    def _validate_parameter_ui(self) -> bool:
        """
        Validate that shader parameters have corresponding UI controls.

        Checks:
        1. Each genjit parameter has a message/prepend to send it to jit.gl.pix
        2. Each parameter message is connected to jit.gl.pix
        3. Each parameter message has an upstream interactive UI control
           (dial, slider, toggle, button)
        """
        valid = True

        # Find all jit.gl.pix objects with @gen shaders
        jit_gl_pixs = self._find_boxes_by_type("jit.gl.pix")
        if not jit_gl_pixs:
            return valid  # No shaders to validate

        # Collect all shader parameters from referenced genjit files
        all_params: dict[str, list[dict[str, Any]]] = {}  # shader -> params

        for pix_id in jit_gl_pixs:
            text = self._get_box_text(pix_id)
            shader_name = self._extract_gen_shader(text)
            if not shader_name:
                continue

            genjit_path = self._find_genjit_file(shader_name)
            if not genjit_path:
                self.info(
                    "parameter-ui",
                    f"Could not find genjit file for shader '{shader_name}'",
                    pix_id,
                )
                continue

            # Validate genjit format and get parameters
            format_valid, params = self._validate_genjit_format(
                genjit_path, shader_name
            )
            if not format_valid:
                valid = False  # Propagate genjit format errors

            if params:
                all_params[shader_name] = params

        if not all_params:
            return valid  # No parameters to validate

        # Find parameter message/prepend boxes in help patcher
        param_messages = self._find_param_messages()

        # Find UI controls
        ui_controls = self._find_ui_controls()
        all_ui_ids = []
        for ids in ui_controls.values():
            all_ui_ids.extend(ids)

        # Get interactive controls (dial, slider, toggle, button)
        interactive_ui_ids = self._get_interactive_control_ids(ui_controls)

        # Check each shader's parameters
        for shader_name, params in all_params.items():
            for param in params:
                param_name = param["name"]

                # Check if there's a message/prepend for this parameter
                if param_name not in param_messages:
                    self.warning(
                        "parameter-ui",
                        f"No message or prepend found for parameter '{param_name}' "
                        f"from shader '{shader_name}'",
                    )
                    continue

                param_box_ids = param_messages[param_name]

                # Check if parameter message is connected to jit.gl.pix
                connected_to_pix = False
                for param_box_id in param_box_ids:
                    for pix_id in jit_gl_pixs:
                        try:
                            if nx.has_path(
                                self.graph, (param_box_id, "box"), (pix_id, "box")
                            ):
                                connected_to_pix = True
                                break
                        except nx.NetworkXError:
                            pass
                    if connected_to_pix:
                        break

                if not connected_to_pix:
                    self.warning(
                        "parameter-ui",
                        f"Parameter message '{param_name}' not connected to jit.gl.pix",
                    )

                # Check if there's ANY UI control connected to the parameter
                has_any_ui = self._check_ui_to_pix_connection(
                    all_ui_ids, param_box_ids, jit_gl_pixs
                )

                # Check if there's an INTERACTIVE control (dial, slider, toggle, button)
                has_interactive_ui = self._check_ui_to_pix_connection(
                    interactive_ui_ids, param_box_ids, jit_gl_pixs
                )

                if not has_any_ui:
                    # No UI at all - this is an error
                    self.error(
                        "parameter-ui",
                        f"No UI control found for parameter '{param_name}' "
                        f"from shader '{shader_name}'",
                    )
                    valid = False
                elif not has_interactive_ui:
                    # Has number/flonum but no interactive control - warning
                    self.warning(
                        "parameter-ui",
                        f"Parameter '{param_name}' has no interactive control "
                        f"(dial, slider, toggle, button) - only number/flonum",
                    )

        return valid

    def print_results(self, filepath: Path) -> None:
        """Print validation results."""
        if self.errors or self.warnings:
            print(f"\n{filepath}:")
            for error in self.errors:
                print(error)
            for warning in self.warnings:
                print(warning)
        elif self.verbose:
            print(f"\n{filepath}: OK")

    def has_errors(self) -> bool:
        """Check if there are any errors (or warnings in strict mode)."""
        return bool(self.errors or (self.strict and self.warnings))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate .maxhelp files with graph-based analysis"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Path(s) to .maxhelp files to validate",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show info messages and OK status",
    )
    args = parser.parse_args()

    linter = MaxhelpLinter(strict=args.strict, verbose=args.verbose)
    has_errors = False
    files_checked = 0
    files_failed = 0

    for filepath in args.files:
        # Expand directory to all .maxhelp files, or use single file
        files = list(filepath.glob("*.maxhelp")) if filepath.is_dir() else [filepath]

        for f in files:
            files_checked += 1
            valid = linter.validate_file(f)
            linter.print_results(f)

            if linter.has_errors():
                has_errors = True
                files_failed += 1

    # Summary
    print(f"\nValidated {files_checked} file(s), {files_failed} with issues")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

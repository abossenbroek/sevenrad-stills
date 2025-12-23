#!/usr/bin/env python3
"""
Graph-based linter for .maxhelp files.

Validates that help patchers have correct structure, initialization order,
and complete signal flow using networkx graph analysis.

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

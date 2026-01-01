"""Signal flow validation mixin for Max help patchers.

This module provides the SignalFlowValidatorMixin class that validates
GPU texture pipeline signal flow in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from max_linter.constants import JITTER_DISPLAY_SINKS

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class SignalFlowValidatorMixin:
    """Mixin providing signal flow validation methods.

    Validates GPU texture pipeline completeness and connection types.

    Rules:
        signal-flow: Missing pipeline components
        inlet-connections: Unconnected shader inlets
        connection-type: Type mismatch on connections
        type-001: Texture outlet -> matrix inlet
        type-002: Matrix outlet -> texture inlet
        type-003: Info outlet -> data inlet
        type-004: jit.movie in GPU pipeline without @output_texture 1
        type-005: jit.pwindow receiving texture without GPU context
        display-sink-type: Invalid source type for display sinks
    """

    # These attributes must be provided by the composing class
    graph: nx.DiGraph
    data: dict[str, Any]
    boxes: dict[str, dict[str, Any]]
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

    def _get_outlet_type(self, box: dict[str, Any], outlet_idx: int) -> str:
        """Get outlet type for a box."""
        raise NotImplementedError

    def _validate_signal_flow(self) -> bool:
        """Validate the GPU texture pipeline is complete.

        Expected flow: qmetro -> jit.movie -> jit.gl.pix -> jit.pwindow

        Note: jit.movie/jit.pwindow are only required for shader effect patchers
        (those with jit.gl.pix). Help patchers for C externals may not use this
        pipeline.
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

        # Determine if this is a shader effect patcher (has jit.gl.pix)
        is_shader_patcher = bool(jit_gl_pixs)

        # Required components only for shader effect patchers
        if is_shader_patcher:
            if not jit_movies:
                self.error(
                    "signal-flow",
                    "Missing jit.movie - shader effect needs a video input source",
                )
                valid = False

            if not jit_pwindows:
                self.error(
                    "signal-flow",
                    "Missing jit.pwindow - shader effect needs a video output display",
                )
                valid = False

        # Check qmetro -> jit.movie
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

        # Check jit.movie -> jit.gl.pix
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

        # Check jit.gl.pix -> jit.pwindow
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
        """Validate that all required inlets are connected.

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
                        f"jit.gl.pix has {numinlets} inlets but inlet(s) "
                        f"{sorted(missing)} not connected",
                        pix_id,
                    )

        return valid

    def _validate_connection_types(self) -> bool:
        """Validate that connected outlets and inlets have compatible types.

        Detects type mismatches like:
        - Texture outlet connected to matrix inlet
        - Info/dump outlet connected to texture or matrix inlet
        """
        valid = True

        for src, dst, data in self.graph.edges(data=True):
            # Only check port-to-port connections (not box-level edges)
            if src[1] != "out" or dst[1] != "in":
                continue

            outlet_type = data.get("outlet_type", "unknown")
            inlet_type = data.get("inlet_type", "unknown")

            src_box_id = src[0]
            dst_box_id = dst[0]
            src_box = self.boxes.get(src_box_id, {})
            dst_box = self.boxes.get(dst_box_id, {})
            src_text = src_box.get("text", src_box.get("maxclass", ""))
            dst_text = dst_box.get("text", dst_box.get("maxclass", ""))

            # Skip if types are unknown or compatible
            if outlet_type == "unknown" or inlet_type == "unknown":
                continue
            if inlet_type == "any":
                continue
            if outlet_type == "bang_or_message":
                continue

            # Texture -> Matrix is wrong
            if outlet_type == "texture" and inlet_type == "matrix":
                self.error(
                    "connection-type",
                    f"Texture outlet connected to matrix inlet: "
                    f"'{src_text}' outlet {src[2]} -> '{dst_text}' inlet {dst[2]}",
                    f"{src_box_id} -> {dst_box_id}",
                )
                valid = False

            # Matrix -> Texture is wrong
            if outlet_type == "matrix" and inlet_type == "texture":
                self.error(
                    "connection-type",
                    f"Matrix outlet connected to texture inlet: "
                    f"'{src_text}' outlet {src[2]} -> '{dst_text}' inlet {dst[2]}",
                    f"{src_box_id} -> {dst_box_id}",
                )
                valid = False

            # Info/dump outlet to texture or matrix inlet is likely wrong
            if outlet_type == "info" and inlet_type in ("texture", "matrix"):
                self.warning(
                    "connection-type",
                    f"Info outlet connected to {inlet_type} inlet: "
                    f"'{src_text}' outlet {src[2]} -> '{dst_text}' inlet {dst[2]}. "
                    "Info outlets typically output metadata, not image data.",
                    f"{src_box_id} -> {dst_box_id}",
                )

            # matrix_or_texture expects either matrix or texture
            if inlet_type == "matrix_or_texture":
                if outlet_type in ("texture", "matrix"):
                    continue  # Valid
                # Invalid - warn with actionable message
                self.warning(
                    "connection-type",
                    f"'{src_text}' outlet {src[2]} has type '{outlet_type}' "
                    f"but '{dst_text}' expects matrix or texture. "
                    "May cause runtime error.",
                    f"{src_box_id} -> {dst_box_id}",
                )

        return valid

    def _validate_strict_types(self) -> bool:
        """Validate strict type compatibility on connections.

        Uses LintGraph to check each connection for type mismatches.
        Implements type validation rules:
        - type-001: Texture outlet -> matrix inlet = ERROR
        - type-002: Matrix outlet -> texture inlet = ERROR
        - type-003: Info outlet -> data inlet = ERROR
        - type-004: jit.movie in GPU pipeline without @output_texture 1 = ERROR
        - type-005: jit.pwindow receiving texture without GPU context = ERROR

        Returns:
            True if validation passes (no errors), False otherwise.
        """
        valid = True
        lint_graph = self.lint_graph  # Use pre-built LintGraph

        # Check each edge for type compatibility
        for edge in lint_graph.graph.edges(data=True):
            src_node, dst_node, edge_data = edge

            # Only check outlet->inlet edges
            if len(src_node) < 3 or len(dst_node) < 3:
                continue
            if src_node[1] != "out" or dst_node[1] != "in":
                continue

            outlet_type = edge_data.get("outlet_type", "unknown")
            inlet_type = edge_data.get("inlet_type", "unknown")

            src_box_id = src_node[0]

            # type-001: Texture -> Matrix
            if outlet_type == "texture" and inlet_type == "matrix":
                self.error(
                    "type-001",
                    "Type mismatch: texture outlet cannot connect to matrix inlet",
                    src_box_id,
                )
                valid = False

            # type-002: Matrix -> Texture
            if outlet_type == "matrix" and inlet_type == "texture":
                self.error(
                    "type-002",
                    "Type mismatch: matrix outlet cannot connect to texture inlet. "
                    "Use jit.movie @output_texture 1 for GPU pipeline.",
                    src_box_id,
                )
                valid = False

            # type-003: Info -> Data inlet
            if outlet_type == "info" and inlet_type in ("texture", "matrix"):
                self.error(
                    "type-003",
                    "Type mismatch: info outlet (metadata) connected to data inlet",
                    src_box_id,
                )
                valid = False

        # type-004: jit.movie in GPU pipeline without @output_texture 1
        jit_movies = self._find_boxes_by_type("jit.movie")
        jit_gl_pix = self._find_boxes_by_type("jit.gl.pix")

        for movie_id in jit_movies:
            movie_text = self._get_box_text(movie_id)
            has_output_texture = "@output_texture 1" in movie_text

            # Check if connected to jit.gl.pix (GPU pipeline)
            for pix_id in jit_gl_pix:
                try:
                    if nx.has_path(
                        lint_graph.graph, (movie_id, "box"), (pix_id, "box")
                    ):
                        if not has_output_texture:
                            self.error(
                                "type-004",
                                "jit.movie connected to GPU pipeline (jit.gl.pix) "
                                "but missing @output_texture 1",
                                movie_id,
                            )
                            valid = False
                        break
                except nx.NetworkXError:
                    pass

        # type-005: jit.pwindow receiving texture without GPU context
        jit_pwindows = self._find_boxes_by_maxclass("jit.pwindow")
        jit_worlds = self._find_boxes_by_type("jit.world")

        for pwindow_id in jit_pwindows:
            # Find sources connected to inlet 0
            inlet_node = (pwindow_id, "in", 0)
            if inlet_node not in lint_graph.graph:
                continue

            for pred in lint_graph.graph.predecessors(inlet_node):
                if len(pred) < 3 or pred[1] != "out":
                    continue

                src_box_id = pred[0]
                src_outlet = pred[2]
                src_box = lint_graph.boxes.get(src_box_id, {})
                outlet_type = self._get_outlet_type(src_box, src_outlet)

                # Check if source outputs texture without GPU context
                if outlet_type == "texture" and not jit_worlds:
                    self.error(
                        "type-005",
                        "jit.pwindow receiving texture without GPU context. "
                        "Add jit.world to create OpenGL context or use "
                        "jit.matrix for CPU processing.",
                        pwindow_id,
                    )
                    valid = False

        return valid

    def _validate_display_sink_sources(self) -> bool:
        """Validate sources feeding Jitter display sinks have valid image types.

        Jitter display sinks (jit.pwindow, jit.window) expect matrix or texture
        data on inlet 0. This validation warns when sources have ambiguous
        outlet types that may cause 'object is not a valid matrix' runtime errors.
        """
        valid = True

        for box_id, box in self.boxes.items():
            maxclass = box.get("maxclass", "")
            text = box.get("text", "")

            # Check if this is a Jitter display sink
            is_display_sink = maxclass in JITTER_DISPLAY_SINKS or any(
                sink in text for sink in JITTER_DISPLAY_SINKS
            )

            if not is_display_sink:
                continue

            # Find all sources connected to inlet 0
            inlet_node = (box_id, "in", 0)
            if inlet_node not in self.graph:
                continue

            for pred in self.graph.predecessors(inlet_node):
                # Only check outlet connections (skip box-level edges)
                if len(pred) < 3 or pred[1] != "out":
                    continue

                src_box_id = pred[0]
                src_outlet = pred[2]
                src_box = self.boxes.get(src_box_id, {})

                outlet_type = self._get_outlet_type(src_box, src_outlet)
                src_text = src_box.get("text", src_box.get("maxclass", ""))

                # Valid: texture or matrix
                if outlet_type in ("texture", "matrix"):
                    continue

                # Invalid: bang_or_message, info, unknown
                if outlet_type == "bang_or_message":
                    self.warning(
                        "display-sink-type",
                        f"Object '{src_text}' has outlet type '{outlet_type}' "
                        f"but connects to {maxclass} which expects matrix/texture. "
                        "This may cause 'object is not a valid matrix' error. "
                        "If this object outputs a matrix, set outlettype to "
                        "['jit_matrix'].",
                        f"{src_box_id} -> {box_id}",
                    )
                elif outlet_type in ("info", "unknown"):
                    self.warning(
                        "display-sink-type",
                        f"Object '{src_text}' has ambiguous outlet type "
                        f"'{outlet_type}' connecting to {maxclass}. "
                        "Verify this produces valid image data.",
                        f"{src_box_id} -> {box_id}",
                    )

        return valid

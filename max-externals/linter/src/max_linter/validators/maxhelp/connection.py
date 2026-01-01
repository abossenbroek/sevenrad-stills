"""Connection type inference mixin for Max help patchers.

This module provides the ConnectionValidatorMixin class that handles
type inference for Jitter connections in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from max_linter.constants import JITTER_DISPLAY_SINKS


class ConnectionValidatorMixin:
    """Mixin providing connection type inference methods.

    Provides methods for determining outlet and inlet types for Jitter
    connections, used for type compatibility validation.
    """

    # These attributes must be provided by the composing class
    graph: nx.DiGraph

    def _get_outlet_type(self, box: dict[str, Any], outlet_idx: int) -> str:
        """Determine the type of data from an outlet.

        Returns one of: 'texture', 'matrix', 'info', 'bang_or_message', 'unknown'
        """
        text = box.get("text", "")
        maxclass = box.get("maxclass", "")

        # For Jitter objects, outlet 1+ is typically info/dump outlet
        is_jitter = (
            "jit." in text
            or maxclass.startswith("jit.")
            or maxclass in ("jit.pwindow",)
        )

        # First check explicit outlettype in box definition
        result = self._check_explicit_outlettype(box, outlet_idx, is_jitter)
        if result:
            return result

        # Infer type from object text for Jitter objects
        outlet_0_type = self._infer_jitter_outlet_type(text, maxclass)
        if outlet_0_type:
            return outlet_0_type if outlet_idx == 0 else "info"

        # Non-Jitter objects that output bang/message
        if "metro" in text or "loadbang" in text or maxclass == "loadbang":
            return "bang_or_message"

        return "unknown"

    def _check_explicit_outlettype(
        self, box: dict[str, Any], outlet_idx: int, is_jitter: bool
    ) -> str | None:
        """Check explicit outlettype array. Returns None if not found."""
        outlettype = box.get("outlettype", [])
        if outlet_idx >= len(outlettype):
            return None
        otype = outlettype[outlet_idx]
        if otype == "jit_gl_texture":
            return "texture"
        if otype == "jit_matrix":
            return "matrix"
        if otype == "":
            # Empty string: for Jitter objects outlet > 0 is info
            return "info" if is_jitter and outlet_idx > 0 else "bang_or_message"
        return None

    def _infer_jitter_outlet_type(self, text: str, maxclass: str) -> str | None:
        """Infer the outlet 0 type for Jitter objects.

        Returns None if not a Jitter object.
        """
        # jit.movie with @output_texture 1 outputs texture
        if "jit.movie" in text:
            return "texture" if "@output_texture 1" in text else "matrix"
        # jit.gl.pix and jit.gl.texture output texture
        if "jit.gl.pix" in text or "jit.gl.texture" in text:
            return "texture"
        # jit.matrix outputs matrix
        if "jit.matrix" in text or maxclass == "jit.matrix":
            return "matrix"
        return None

    def _get_expected_inlet_type(self, box: dict[str, Any], inlet_idx: int) -> str:
        """Determine the expected type of data for an inlet.

        Returns one of: 'texture', 'matrix', 'matrix_or_texture', 'any', 'unknown'
        """
        text = box.get("text", "")
        maxclass = box.get("maxclass", "")

        # jit.gl.pix expects texture input
        if "jit.gl.pix" in text:
            return "texture"

        # Jitter display sinks expect matrix or texture on inlet 0
        is_display_sink = maxclass in JITTER_DISPLAY_SINKS or any(
            sink in text for sink in JITTER_DISPLAY_SINKS
        )
        if is_display_sink:
            if inlet_idx == 0:
                return "matrix_or_texture"
            return "any"  # Other inlets can receive messages

        # jit.matrix objects expect matrix
        if "jit.matrix" in text and "@output_texture" not in text:
            return "matrix"

        # Most newobj can accept bang/message
        if maxclass == "newobj":
            return "any"

        return "unknown"

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

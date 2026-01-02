"""LintGraph dataclass for Max patcher graph analysis.

This module provides the LintGraph dataclass that encapsulates the networkx
graph structure built from a Max patcher, including type information for
connections, pre-computed cycles, orphan boxes, and dead branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import networkx as nx

from max_linter.constants import JITTER_DISPLAY_SINKS

if TYPE_CHECKING:
    pass


class TypeInferrer(Protocol):
    """Protocol for type inference callbacks.

    This protocol defines the interface that the linter must implement
    to provide type inference for outlet and inlet types.
    """

    def _get_outlet_type(self, box: dict[str, Any], outlet_idx: int) -> str:
        """Determine the type of data from an outlet."""
        ...

    def _get_expected_inlet_type(self, box: dict[str, Any], inlet_idx: int) -> str:
        """Determine the expected type of data for an inlet."""
        ...


@dataclass
class LintGraph:
    """Shared graph state for all validation rules.

    Built ONCE at start of validation, passed to ALL rules.
    This class encapsulates the graph structure built from a Max patcher,
    including outlet type information for type checking.

    Attributes:
        graph: Directed graph where nodes are (box_id, port_type, port_num) tuples.
               Port types are 'in', 'out', or 'box'.
        boxes: Dict mapping box_id to box properties dict.
        type_map: Dict mapping (box_id, outlet_idx) to outlet type string.
                  Types are: 'texture', 'matrix', 'info', 'bang_or_message', 'unknown'.
        cycles: Pre-computed cycles via nx.simple_cycles() on box-level graph.
                Each cycle is a list of box IDs forming the feedback loop.
        orphans: Box IDs with no connections (degree 0 in box-level graph).
        dead_branches: Box IDs not reaching any display sink (jit.pwindow, jit.window).
    """

    graph: nx.DiGraph
    boxes: dict[str, dict[str, Any]]
    type_map: dict[tuple[str, int], str]  # (box_id, outlet) -> type
    cycles: list[list[str]]  # Pre-computed cycles (list of box ID lists)
    orphans: set[str]  # Box IDs with no connections
    dead_branches: set[str]  # Box IDs not reaching display sinks

    @classmethod
    def build(cls, data: dict[str, Any], linter: TypeInferrer) -> LintGraph:
        """Build LintGraph from patcher JSON.

        Args:
            data: Parsed patcher JSON data with 'patcher' key containing
                  'boxes' and 'lines'.
            linter: Object implementing TypeInferrer protocol for type
                    inference methods.

        Returns:
            LintGraph with populated graph, boxes dict, type_map, cycles,
            orphans, and dead_branches.
        """
        graph = nx.DiGraph()
        boxes: dict[str, dict[str, Any]] = {}
        type_map: dict[tuple[str, int], str] = {}

        # Build a separate box-level graph for cycle/orphan/dead-branch detection
        box_graph = nx.DiGraph()

        # Extract boxes from patcher data
        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")
            if box_id:
                boxes[box_id] = box
                box_graph.add_node(box_id)

                # Add nodes for outlets and populate type_map
                for i in range(box.get("numoutlets", 0)):
                    graph.add_node((box_id, "out", i), box=box)
                    # Populate type_map for this outlet
                    outlet_type = linter._get_outlet_type(box, i)
                    type_map[(box_id, i)] = outlet_type

                # Add nodes for inlets
                for i in range(box.get("numinlets", 0)):
                    graph.add_node((box_id, "in", i), box=box)

                # Add a "box" node for easier lookup and path finding
                graph.add_node((box_id, "box"), box=box)

        # Add edges from patchlines with type information
        for line in data.get("patcher", {}).get("lines", []):
            patchline = line.get("patchline", {})
            src = patchline.get("source", [])
            dst = patchline.get("destination", [])

            if len(src) >= 2 and len(dst) >= 2:
                src_box_id, src_outlet = src[0], src[1]
                dst_box_id, dst_inlet = dst[0], dst[1]

                src_box = boxes.get(src_box_id, {})
                dst_box = boxes.get(dst_box_id, {})

                outlet_type = linter._get_outlet_type(src_box, src_outlet)
                inlet_type = linter._get_expected_inlet_type(dst_box, dst_inlet)

                # Add edge from outlet to inlet with type info
                graph.add_edge(
                    (src_box_id, "out", src_outlet),
                    (dst_box_id, "in", dst_inlet),
                    outlet_type=outlet_type,
                    inlet_type=inlet_type,
                )

                # Add box-level edge for path finding
                graph.add_edge((src_box_id, "box"), (dst_box_id, "box"))

                # Add edge to box-level graph for cycle detection
                box_graph.add_edge(src_box_id, dst_box_id)

        # Compute cycles using networkx simple_cycles on box-level graph
        cycles = list(nx.simple_cycles(box_graph))

        # Compute orphans - boxes with no edges at box level (degree 0)
        orphans = {box_id for box_id in boxes if box_graph.degree(box_id) == 0}

        # Compute dead branches - boxes not reaching display sinks
        # Display sinks: jit.pwindow, jit.window
        display_sinks: set[str] = set()
        for box_id, box in boxes.items():
            text = box.get("text", "")
            maxclass = box.get("maxclass", "")
            if maxclass in JITTER_DISPLAY_SINKS or any(
                sink in text for sink in JITTER_DISPLAY_SINKS
            ):
                display_sinks.add(box_id)

        # Find all nodes that can reach a sink (ancestors in directed graph)
        reaches_sink: set[str] = set()
        for sink in display_sinks:
            reaches_sink.add(sink)
            reaches_sink.update(nx.ancestors(box_graph, sink))

        # Dead branches are boxes that don't reach any sink and aren't orphans
        dead_branches = set(boxes.keys()) - reaches_sink - orphans

        return cls(
            graph=graph,
            boxes=boxes,
            type_map=type_map,
            cycles=cycles,
            orphans=orphans,
            dead_branches=dead_branches,
        )

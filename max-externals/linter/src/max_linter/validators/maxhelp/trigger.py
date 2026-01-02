"""Trigger order validation mixin for Max help patchers.

This module provides the TriggerValidatorMixin class that validates
trigger objects have proper outlet ordering in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class TriggerValidatorMixin:
    """Mixin providing trigger order validation methods.

    Validates that trigger objects fire outlets in correct order.

    Rules:
        trigger-001: Bang outlet fires before data outlet to same destination
    """

    # These attributes must be provided by the composing class
    lint_graph: LintGraph

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def _validate_trigger_order(self) -> bool:
        """Validate trigger objects don't send bang before message.

        Max fires outlets RIGHT to LEFT. For a trigger object like 't l b':
        - outlet 1 (b) fires FIRST
        - outlet 0 (l) fires SECOND

        If bang (b) is at a higher index than a data outlet (l/i/f/s) and both
        go to the same destination, the bang arrives before the data, which is
        typically wrong for dependent operations.

        Rules:
            trigger-001: Bang outlet fires before data outlet to same
                         destination (ERROR)

        Returns:
            True if all trigger rules pass, False otherwise.
        """
        valid = True
        lint_graph = self.lint_graph

        # Data types (non-bang) that should arrive before bang
        data_types = {"l", "list", "i", "int", "f", "float", "s", "symbol"}
        bang_types = {"b", "bang"}

        # Find all trigger objects
        trigger_boxes = [
            bid
            for bid, box in lint_graph.boxes.items()
            if box.get("text", "").startswith(("t ", "trigger "))
        ]

        for trig_id in trigger_boxes:
            text = lint_graph.boxes[trig_id].get("text", "")
            parts = text.split()
            if len(parts) < 2:
                continue

            outlets = parts[1:]  # e.g., ['l', 'b'] or ['list', 'bang']

            # Find destinations for each outlet from connections
            # Build a map: outlet_index -> list of (dest_box_id, dest_inlet)
            outlet_dests: dict[int, list[tuple[str, int]]] = {}
            for edge in lint_graph.graph.edges():
                src_node, dst_node = edge
                if (
                    isinstance(src_node, tuple)
                    and len(src_node) == 3
                    and src_node[0] == trig_id
                    and src_node[1] == "out"
                ):
                    outlet_idx = src_node[2]
                    if isinstance(dst_node, tuple) and len(dst_node) == 3:
                        dest_box_id = dst_node[0]
                        dest_inlet = dst_node[2]
                        if outlet_idx not in outlet_dests:
                            outlet_dests[outlet_idx] = []
                        outlet_dests[outlet_idx].append((dest_box_id, dest_inlet))

            # Find indices of bang and data outlets
            bang_outlets = [i for i, o in enumerate(outlets) if o.lower() in bang_types]
            msg_outlets = [i for i, o in enumerate(outlets) if o.lower() in data_types]

            # Check for problematic ordering:
            # If a bang outlet has a HIGHER index than a data outlet
            # AND both go to the same destination, the bang fires FIRST (wrong!)
            for bang_idx in bang_outlets:
                for msg_idx in msg_outlets:
                    # Higher index = fires first in Max (right to left)
                    if bang_idx > msg_idx:
                        # Check if they share any destination
                        bang_dests = set(outlet_dests.get(bang_idx, []))
                        msg_dests = set(outlet_dests.get(msg_idx, []))
                        shared_dests = bang_dests & msg_dests

                        if shared_dests:
                            # Get the types for the error message
                            bang_type = outlets[bang_idx]
                            msg_type = outlets[msg_idx]
                            dest_info = next(iter(shared_dests))

                            # Suggest corrected order (swap bang to be before
                            # data in text)
                            corrected = outlets.copy()
                            corrected[bang_idx], corrected[msg_idx] = (
                                corrected[msg_idx],
                                corrected[bang_idx],
                            )
                            corrected_text = f"{parts[0]} {' '.join(corrected)}"

                            self.error(
                                "trigger-001",
                                f"Trigger outlet order causes bang ({bang_type}) "
                                f"to fire before data ({msg_type}) to same "
                                f"destination '{dest_info[0]}' inlet {dest_info[1]}. "
                                f"Suggest: '{corrected_text}'",
                                trig_id,
                            )
                            valid = False

        return valid

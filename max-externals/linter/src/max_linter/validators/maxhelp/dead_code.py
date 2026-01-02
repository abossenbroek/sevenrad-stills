"""Dead code validation mixin for Max help patchers.

This module provides the DeadCodeValidatorMixin class that detects
orphaned objects, dead branches, and other unused code in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max_linter.constants import KNOWN_MAX_OBJECTS

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class DeadCodeValidatorMixin:
    """Mixin providing dead code validation methods.

    Detects orphaned objects, dead branches, and unknown objects.

    Rules:
        dead-001: Orphaned objects (no connections)
        dead-002: Dead branches not reaching output
        dead-003: Multiple sources to single inlet (race condition)
        unknown-object: Unknown object name
    """

    # These attributes must be provided by the composing class
    lint_graph: LintGraph
    data: dict[str, Any]

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _validate_dead_code(self) -> bool:
        """Validate for dead code patterns using LintGraph analysis.

        Detects orphaned objects, dead branches, and multiple sources to single inlet.

        Rules:
            dead-001: Orphaned objects (no connections) = ERROR
            dead-002: Dead branches not reaching output = ERROR
            dead-003: Multiple sources to single inlet (race) = WARNING

        Excluded from orphan detection (dead-001):
            - comment objects (labels)
            - panel objects (visual organization)
            - fpic objects (images)
            - live.comment objects

        Returns:
            True if all dead code rules pass, False otherwise.
        """
        valid = True
        lint_graph = self.lint_graph

        # Objects excluded from orphan detection (UI/comment elements)
        excluded_maxclasses = {"comment", "panel", "fpic", "live.comment"}

        # dead-001: Orphaned objects (no connections)
        for orphan_id in lint_graph.orphans:
            box = lint_graph.boxes.get(orphan_id, {})
            maxclass = box.get("maxclass", "")

            # Skip excluded object types
            if maxclass in excluded_maxclasses:
                continue

            self.error(
                "dead-001",
                f"Orphaned object with no connections: {box.get('text', maxclass)}",
                orphan_id,
            )
            valid = False

        # dead-002: Dead branches not reaching display sinks
        for dead_id in lint_graph.dead_branches:
            box = lint_graph.boxes.get(dead_id, {})
            maxclass = box.get("maxclass", "")

            # Skip excluded object types for dead branches too
            if maxclass in excluded_maxclasses:
                continue

            self.error(
                "dead-002",
                f"Dead branch not reaching display: {box.get('text', maxclass)}",
                dead_id,
            )
            valid = False

        # dead-003: Multiple sources to single inlet (race condition)
        # Build a map of (box_id, inlet) -> list of source connections
        inlet_sources: dict[tuple[str, int], list[str]] = {}
        for edge in lint_graph.graph.edges():
            src_node, dst_node = edge
            if (
                isinstance(dst_node, tuple)
                and len(dst_node) == 3
                and dst_node[1] == "in"
            ):
                dest_box_id = dst_node[0]
                dest_inlet = dst_node[2]
                key = (dest_box_id, dest_inlet)

                if key not in inlet_sources:
                    inlet_sources[key] = []

                # Extract source box ID
                if isinstance(src_node, tuple) and len(src_node) >= 1:
                    inlet_sources[key].append(src_node[0])

        # Report inlets with multiple sources
        for (box_id, inlet), sources in inlet_sources.items():
            if len(sources) > 1:
                self.warning(
                    "dead-003",
                    f"Multiple sources ({len(sources)}) to inlet {inlet}: "
                    f"race condition possible. Sources: {', '.join(sources)}",
                    box_id,
                )

        return valid

    def _validate_known_objects(self) -> bool:
        """Validate that all objects in the patcher are known/valid.

        This helps catch typos and non-existent objects like 'jit.gl.xfade'
        which would cause "No such object" errors in Max.

        Objects are considered valid if they:
        - Are in the KNOWN_MAX_OBJECTS set
        - Start with 'sr.' (project-specific objects)
        - Are special maxclass types (comment, button, dial, etc.)

        Returns:
            True if all objects are known, False otherwise.
        """
        valid = True

        # Max classes that don't use "text" for object name
        non_text_classes = {
            "comment",
            "button",
            "toggle",
            "dial",
            "slider",
            "number",
            "flonum",
            "message",
            "bpatcher",
            "inlet",
            "outlet",
            "live.dial",
            "live.slider",
            "live.button",
            "live.toggle",
            "live.numbox",
            "live.text",
            "live.menu",
            "panel",
            "fpic",
            "swatch",
            "multislider",
            "umenu",
            "ubumenu",
            "textbutton",
            "rslider",
            "kslider",
        }

        for box_wrapper in self.data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            box_id = box.get("id", "")
            maxclass = box.get("maxclass", "")

            # Skip non-newobj classes (they're built-in UI elements)
            if maxclass != "newobj":
                continue

            # Get object name from text field
            text = box.get("text", "")
            if not text:
                continue

            # Extract object name (first word, ignoring arguments)
            obj_name = text.split()[0] if text else ""
            if not obj_name:
                continue

            # Check if object is known
            is_known = (
                obj_name in KNOWN_MAX_OBJECTS
                or obj_name.startswith("sr.")  # Project-specific objects
                or obj_name.startswith("mc.")  # Max multichannel objects
                or obj_name.startswith("mcs.")  # Max multichannel signal
                or obj_name.startswith("gen~")  # Gen~ objects
                or obj_name.startswith("jit.gl.")  # All Jitter GL objects
                or obj_name.startswith("jit.anim.")  # Jitter animation
                or obj_name.startswith("jit.phys.")  # Jitter physics
                or obj_name in non_text_classes  # UI elements used as objects
            )

            if not is_known:
                self.warning(
                    "unknown-object",
                    f"Unknown object '{obj_name}' - may cause 'No such object' "
                    f"error in Max. Verify the object name is correct.",
                    box_id,
                )
                valid = False

        return valid

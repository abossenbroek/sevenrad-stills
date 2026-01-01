"""UI overlap validation mixin for Max help patchers.

This module provides the OverlapValidatorMixin class that detects
overlapping UI components in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class OverlapValidatorMixin:
    """Mixin providing UI overlap detection methods.

    Detects overlapping UI elements to help identify layout issues
    in Max help patchers.

    Rules:
        overlap-001: UI components overlap
    """

    # These attributes must be provided by the composing class
    lint_graph: LintGraph

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _are_boxes_connected(self, id1: str, id2: str) -> bool:
        """Check if two boxes are directly connected."""
        raise NotImplementedError

    def _validate_no_overlaps(self) -> bool:
        """Validate that UI components don't overlap each other.

        This method detects overlapping UI elements to help identify layout issues
        in Max help patchers. Overlaps are calculated using bounding box intersection
        based on each object's patching_rect [x, y, width, height].

        Overlap Detection Rules:
            - Significant overlap (>25% of smaller box): ERROR
            - Minor overlap (>100px^2 but <25%): WARNING
            - Tiny overlap (<=100px^2): Ignored
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
        # Use LintGraph.boxes for consistent state
        interactive_boxes: list[tuple[str, list[float]]] = []

        for box_id, box in self.lint_graph.boxes.items():
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
                            "overlap-001",
                            f"Boxes '{id1}' and '{id2}' overlap "
                            f"({int(overlap_area)}px^2)",
                        )
                        valid = False
                    elif overlap_area > 100:  # Only warn for overlaps > 100px^2
                        self.warning(
                            "overlap-001",
                            f"Boxes '{id1}' and '{id2}' partially overlap "
                            f"({int(overlap_area)}px^2)",
                        )

        return valid

"""Feedback loop validation mixin for Max help patchers.

This module provides the FeedbackValidatorMixin class that validates
GPU feedback loops have proper buffering in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max_linter.lint_graph import LintGraph


class FeedbackValidatorMixin:
    """Mixin providing feedback loop validation methods.

    Validates that GPU feedback loops have proper buffering.

    Rules:
        feedback-001: Unbuffered feedback (pix->pix without jit.gl.texture)
        feedback-002: Feedback buffer missing @name attribute
    """

    # These attributes must be provided by the composing class
    lint_graph: LintGraph

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _validate_feedback_loops(self) -> bool:
        """Validate feedback loops have proper buffering.

        GPU feedback loops in Max/Jitter require proper buffering with
        jit.gl.texture to avoid undefined behavior. This method checks
        all detected cycles for proper buffer usage.

        Rules:
            feedback-001: Unbuffered feedback (pix->pix without jit.gl.texture) = ERROR
            feedback-002: Feedback buffer missing @name attribute = WARNING

        Returns:
            True if all feedback rules pass, False otherwise.
        """
        valid = True

        for cycle in self.lint_graph.cycles:
            # cycle is a list of box IDs forming the feedback loop
            cycle_box_ids = cycle  # Already box IDs from LintGraph.build()

            # Check if jit.gl.texture exists in cycle (buffer)
            texture_boxes_in_cycle: list[str] = []
            for bid in cycle_box_ids:
                box = self.lint_graph.boxes.get(bid, {})
                text = box.get("text", "")
                if "jit.gl.texture" in text:
                    texture_boxes_in_cycle.append(bid)

            has_buffer = len(texture_boxes_in_cycle) > 0

            if not has_buffer:
                # Find jit.gl.pix objects in unbuffered loop
                pix_boxes = []
                for bid in cycle_box_ids:
                    box = self.lint_graph.boxes.get(bid, {})
                    text = box.get("text", "")
                    if "jit.gl.pix" in text:
                        pix_boxes.append(bid)

                if pix_boxes:
                    # Report error for unbuffered feedback
                    self.error(
                        "feedback-001",
                        f"Unbuffered GPU feedback loop detected. "
                        f"Add jit.gl.texture buffer between pix objects. "
                        f"Cycle: {' -> '.join(cycle_box_ids)}",
                        pix_boxes[0],
                    )
                    valid = False
            else:
                # Check if buffer has @name attribute (feedback-002)
                for tex_bid in texture_boxes_in_cycle:
                    box = self.lint_graph.boxes.get(tex_bid, {})
                    text = box.get("text", "")
                    if "@name" not in text:
                        self.warning(
                            "feedback-002",
                            "Feedback buffer jit.gl.texture missing @name attribute. "
                            "Add @name for reliable texture reference.",
                            tex_bid,
                        )

        return valid

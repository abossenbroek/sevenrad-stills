# ruff: noqa: S101
"""Tests for MaxhelpLinter feedback loop validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestFeedbackValidation:
    """Test feedback loop validation (feedback-001, feedback-002)."""

    def test_unbuffered_feedback_error(self, tmp_path: Path) -> None:
        """Direct pix->pix feedback without buffer = ERROR (feedback-001)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect1",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect2",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
            {"source": ["obj-2", 0], "destination": ["obj-1", 1]},  # Direct feedback!
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "feedback-001"]
        assert len(errors) >= 1
        assert (
            "unbuffered" in errors[0].message.lower()
            or "buffer" in errors[0].message.lower()
        )

    def test_buffered_feedback_no_error(self, tmp_path: Path) -> None:
        """pix->texture->pix feedback (properly buffered) = no error."""
        boxes = [
            {
                "id": "obj-pix1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect1",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-tex",
                "maxclass": "newobj",
                "text": "jit.gl.texture sr_ctx @name feedback_buf",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pix2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect2",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-pix1", 0], "destination": ["obj-tex", 0]},
            {"source": ["obj-tex", 0], "destination": ["obj-pix2", 0]},
            {
                "source": ["obj-pix2", 0],
                "destination": ["obj-pix1", 1],
            },  # Buffered feedback
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        feedback_errors = [e for e in linter.errors if e.rule == "feedback-001"]
        assert len(feedback_errors) == 0

    def test_buffer_missing_name_warning(self, tmp_path: Path) -> None:
        """jit.gl.texture in feedback loop without @name = WARNING (feedback-002)."""
        boxes = [
            {
                "id": "obj-pix1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect1",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-tex",
                "maxclass": "newobj",
                "text": "jit.gl.texture sr_ctx",  # No @name attribute!
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pix2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect2",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-pix1", 0], "destination": ["obj-tex", 0]},
            {"source": ["obj-tex", 0], "destination": ["obj-pix2", 0]},
            {"source": ["obj-pix2", 0], "destination": ["obj-pix1", 1]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        warnings = [w for w in linter.warnings if w.rule == "feedback-002"]
        assert len(warnings) >= 1
        assert "@name" in warnings[0].message

    def test_self_loop_unbuffered_error(self, tmp_path: Path) -> None:
        """Single pix feeding back to itself without buffer = ERROR (feedback-001)."""
        boxes = [
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.feedback",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-pix", 0], "destination": ["obj-pix", 1]},  # Self-loop
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "feedback-001"]
        assert len(errors) >= 1

    def test_buffer_with_name_no_warning(self, tmp_path: Path) -> None:
        """jit.gl.texture with @name in feedback loop = no warning."""
        boxes = [
            {
                "id": "obj-pix1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect1",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-tex",
                "maxclass": "newobj",
                "text": "jit.gl.texture sr_ctx @name my_buffer",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pix2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect2",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-pix1", 0], "destination": ["obj-tex", 0]},
            {"source": ["obj-tex", 0], "destination": ["obj-pix2", 0]},
            {"source": ["obj-pix2", 0], "destination": ["obj-pix1", 1]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        feedback_warnings = [w for w in linter.warnings if w.rule == "feedback-002"]
        assert len(feedback_warnings) == 0

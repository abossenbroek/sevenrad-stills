# ruff: noqa: S101
"""Tests for MaxhelpLinter trigger ordering validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestTriggerValidation:
    """Test trigger ordering validation (trigger-001)."""

    def test_t_l_b_wrong_order_error(self, tmp_path: Path) -> None:
        """t l b sends bang before message to same destination = ERROR (trigger-001).

        Max fires outlets RIGHT to LEFT, so 't l b' fires:
        - outlet 1 (b) first
        - outlet 0 (l) second

        If both go to the same destination, the bang arrives before the list,
        which is typically wrong (bang should trigger after data is ready).
        """
        boxes = [
            {
                "id": "obj-trigger",
                "maxclass": "newobj",
                "text": "t l b",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dest",
                "maxclass": "newobj",
                "text": "message",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-trigger", 0],
                "destination": ["obj-dest", 0],
            },  # l -> dest inlet 0
            {
                "source": ["obj-trigger", 1],
                "destination": ["obj-dest", 0],
            },  # b -> dest inlet 0
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "trigger-001"]
        assert len(errors) >= 1
        assert "bang" in errors[0].message.lower()
        assert (
            "before" in errors[0].message.lower()
            or "order" in errors[0].message.lower()
        )

    def test_t_b_l_correct_order_no_error(self, tmp_path: Path) -> None:
        """t b l sends message before bang = no error.

        Max fires RIGHT to LEFT, so 't b l' fires:
        - outlet 1 (l) first (sends list/message)
        - outlet 0 (b) second (sends bang after data)

        This is the CORRECT order for dependent operations.
        """
        boxes = [
            {
                "id": "obj-trigger",
                "maxclass": "newobj",
                "text": "t b l",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dest",
                "maxclass": "newobj",
                "text": "message",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-trigger", 0],
                "destination": ["obj-dest", 0],
            },  # b -> dest inlet 0
            {
                "source": ["obj-trigger", 1],
                "destination": ["obj-dest", 0],
            },  # l -> dest inlet 0
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "trigger-001"]
        assert len(errors) == 0

    def test_independent_outlets_no_error(self, tmp_path: Path) -> None:
        """Trigger outlets going to different destinations = no error.

        Even with 't l b' order, if outlets go to different destinations,
        the ordering doesn't matter.
        """
        boxes = [
            {
                "id": "obj-trigger",
                "maxclass": "newobj",
                "text": "t l b",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dest1",
                "maxclass": "newobj",
                "text": "message",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-dest2",
                "maxclass": "newobj",
                "text": "print",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {
                "source": ["obj-trigger", 0],
                "destination": ["obj-dest1", 0],
            },  # l -> dest1
            {
                "source": ["obj-trigger", 1],
                "destination": ["obj-dest2", 0],
            },  # b -> dest2
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "trigger-001"]
        assert len(errors) == 0

    def test_trigger_longform_syntax(self, tmp_path: Path) -> None:
        """'trigger' instead of 't' should also be validated."""
        boxes = [
            {
                "id": "obj-trigger",
                "maxclass": "newobj",
                "text": "trigger list bang",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dest",
                "maxclass": "newobj",
                "text": "message",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-trigger", 0],
                "destination": ["obj-dest", 0],
            },  # list -> dest inlet 0
            {
                "source": ["obj-trigger", 1],
                "destination": ["obj-dest", 0],
            },  # bang -> dest inlet 0
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "trigger-001"]
        assert len(errors) >= 1

    def test_multiple_data_types_with_bang(self, tmp_path: Path) -> None:
        """t i b with both to same destination = error."""
        boxes = [
            {
                "id": "obj-trigger",
                "maxclass": "newobj",
                "text": "t i b",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dest",
                "maxclass": "newobj",
                "text": "message",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-trigger", 0],
                "destination": ["obj-dest", 0],
            },  # i -> dest inlet 0
            {
                "source": ["obj-trigger", 1],
                "destination": ["obj-dest", 0],
            },  # b -> dest inlet 0
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "trigger-001"]
        assert len(errors) >= 1

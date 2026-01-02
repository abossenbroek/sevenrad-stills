# ruff: noqa: S101
"""Tests for MaxhelpLinter context naming and validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestContextNaming:
    """Test context naming validation."""

    def test_dots_in_context_error(self, tmp_path: Path) -> None:
        """Test that dots in context names trigger errors."""
        boxes = [
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr.effect.ctx @visible 0",  # Dots in context name
                "patching_rect": [100.0, 100.0, 200.0, 22.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        valid = linter.validate_file(test_file)

        assert not valid
        context_errors = [e for e in linter.errors if e.rule == "context-naming"]
        assert len(context_errors) > 0
        assert "sr_effect_ctx" in context_errors[0].message


class TestContextValidation:
    """Test context validation rules (ctx-001 to ctx-003)."""

    def test_dots_in_context_error(self, tmp_path: Path) -> None:
        """Context name with dots = ERROR (ctx-001)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.world sr.bad.ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "ctx-001"]
        assert len(errors) >= 1
        assert "underscore" in errors[0].message.lower()

    def test_dots_in_drawto_error(self, tmp_path: Path) -> None:
        """@drawto context with dots = ERROR (ctx-001)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.world sr_good_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.movie @drawto sr.bad.ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "ctx-001"]
        assert len(errors) >= 1
        assert "@drawto" in errors[0].message

    def test_nonexistent_drawto_error(self, tmp_path: Path) -> None:
        """@drawto references non-existent context = ERROR (ctx-002)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.movie @drawto wrong_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "ctx-002"]
        assert len(errors) >= 1

    def test_duplicate_context_error(self, tmp_path: Path) -> None:
        """Multiple jit.world with same name = ERROR (ctx-003)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "ctx-003"]
        assert len(errors) >= 1

    def test_valid_context_no_error(self, tmp_path: Path) -> None:
        """Valid context naming = no ctx errors."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx @visible 0",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.movie @drawto sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        ctx_errors = [e for e in linter.errors if e.rule.startswith("ctx-")]
        assert len(ctx_errors) == 0

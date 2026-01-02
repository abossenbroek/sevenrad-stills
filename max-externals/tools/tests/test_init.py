# ruff: noqa: S101
"""Tests for MaxhelpLinter initialization order validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestInitValidation:
    """Test initialization order validation (init-001, init-002, init-003)."""

    def test_jit_world_no_loadbang_error(self, tmp_path: Path) -> None:
        """jit.world without loadbang path = ERROR (init-001)."""
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
        errors = [e for e in linter.errors if e.rule == "init-001"]
        assert len(errors) >= 1
        assert "loadbang" in errors[0].message

    def test_movie_nonexistent_drawto_error(self, tmp_path: Path) -> None:
        """jit.movie @drawto to nonexistent context = ERROR (init-002)."""
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
        errors = [e for e in linter.errors if e.rule == "init-002"]
        assert len(errors) >= 1
        assert "wrong_ctx" in errors[0].message

    def test_no_delay_between_loadbang_and_movie_warning(self, tmp_path: Path) -> None:
        """Loadbang directly to jit.movie @output_texture = WARNING (init-003)."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx @visible 0",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1 @drawto sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-world", 0]},
            {"source": ["obj-lb", 0], "destination": ["obj-movie", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        warnings = [w for w in linter.warnings if w.rule == "init-003"]
        assert len(warnings) >= 1
        assert "delay" in warnings[0].message

    def test_valid_init_order_no_error(self, tmp_path: Path) -> None:
        """Proper loadbang->delay->jit.world->jit.movie = no init errors."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-delay",
                "maxclass": "newobj",
                "text": "delay 100",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx @visible 0",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1 @drawto sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-world", 0]},
            {"source": ["obj-lb", 0], "destination": ["obj-delay", 0]},
            {"source": ["obj-delay", 0], "destination": ["obj-movie", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        init_errors = [e for e in linter.errors if e.rule.startswith("init-")]
        assert len(init_errors) == 0

    def test_pipe_also_counts_as_delay(self, tmp_path: Path) -> None:
        """Using 'pipe' instead of 'delay' should also satisfy init-003."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-pipe",
                "maxclass": "newobj",
                "text": "pipe 100",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx @visible 0",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1 @drawto sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-world", 0]},
            {"source": ["obj-lb", 0], "destination": ["obj-pipe", 0]},
            {"source": ["obj-pipe", 0], "destination": ["obj-movie", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        init_003_warnings = [w for w in linter.warnings if w.rule == "init-003"]
        assert len(init_003_warnings) == 0

    def test_movie_without_output_texture_no_init003(self, tmp_path: Path) -> None:
        """jit.movie without @output_texture 1 should not trigger init-003."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_ctx @visible 0",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @drawto sr_ctx",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-world", 0]},
            {"source": ["obj-lb", 0], "destination": ["obj-movie", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        init_003_warnings = [w for w in linter.warnings if w.rule == "init-003"]
        assert len(init_003_warnings) == 0

    def test_flonum_not_initialized_error(self, tmp_path: Path) -> None:
        """flonum feeding param message without loadbang init = ERROR (init-004)."""
        boxes = [
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "text": "amount $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "init-004"]
        assert len(errors) >= 1
        assert "flonum" in errors[0].message.lower() or "amount" in errors[0].message

    def test_number_not_initialized_error(self, tmp_path: Path) -> None:
        """number feeding param message without loadbang init = ERROR (init-004)."""
        boxes = [
            {
                "id": "obj-number",
                "maxclass": "number",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "text": "seed $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-number", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "init-004"]
        assert len(errors) >= 1
        assert "number" in errors[0].message.lower() or "seed" in errors[0].message

    def test_flonum_initialized_no_error(self, tmp_path: Path) -> None:
        """flonum with loadbang->message init path = no init-004 error."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-init-msg",
                "maxclass": "message",
                "text": "0.5",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-param-msg",
                "maxclass": "message",
                "text": "amount $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-init-msg", 0]},
            {"source": ["obj-init-msg", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-param-msg", 0]},
            {"source": ["obj-param-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "init-004"]
        assert len(errors) == 0

    def test_dial_no_set_init_error(self, tmp_path: Path) -> None:
        """Dial feeding param chain without 'set N' from loadbang = ERROR (init-005)."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numoutlets": 1,
                "numinlets": 1,
                "size": 100.0,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "text": "amount $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
        ]
        lines = [
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "init-005"]
        assert len(errors) >= 1
        assert "dial" in errors[0].message.lower() or "set" in errors[0].message.lower()

    def test_dial_with_set_init_no_error(self, tmp_path: Path) -> None:
        """Dial with loadbang->set N->dial path = no init-005 error."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-set-msg",
                "maxclass": "message",
                "text": "set 50",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numoutlets": 1,
                "numinlets": 1,
                "size": 100.0,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "text": "amount $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-set-msg", 0]},
            {"source": ["obj-set-msg", 0], "destination": ["obj-dial", 0]},
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "init-005"]
        assert len(errors) == 0

    def test_dial_set_out_of_range_error(self, tmp_path: Path) -> None:
        """Dial set value out of position range triggers warning."""
        boxes = [
            {"id": "obj-lb", "maxclass": "loadbang", "numoutlets": 1, "numinlets": 0},
            {
                "id": "obj-set-msg",
                "maxclass": "message",
                "text": "set 150",  # Out of range for size 100
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numoutlets": 1,
                "numinlets": 1,
                "size": 100.0,
                "min": 0.0,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "text": "amount $1",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.effect",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-lb", 0], "destination": ["obj-set-msg", 0]},
            {"source": ["obj-set-msg", 0], "destination": ["obj-dial", 0]},
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-pix", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        # The out-of-range warning uses rule ID init-005-range
        range_warnings = [w for w in linter.warnings if w.rule == "init-005-range"]
        assert len(range_warnings) >= 1

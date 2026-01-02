# ruff: noqa: S101
"""Tests for MaxhelpLinter dead code detection functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestDeadCodeValidation:
    """Test dead code detection rules (dead-001, dead-002, dead-003)."""

    def test_orphan_detected_error(self, tmp_path: Path) -> None:
        """Object with no connections = ERROR (dead-001)."""
        boxes = [
            {
                "id": "obj-connected1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-connected2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-orphan",
                "maxclass": "newobj",
                "text": "delay 100",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-connected1", 0], "destination": ["obj-connected2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-001"]
        assert len(errors) >= 1
        assert errors[0].object_id is not None
        assert "obj-orphan" in errors[0].object_id

    def test_dead_branch_error(self, tmp_path: Path) -> None:
        """Subgraph not reaching display = ERROR (dead-002)."""
        boxes = [
            # Main pipeline reaching display
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            # Dead branch - connected but never reaches display
            {
                "id": "obj-dead1",
                "maxclass": "newobj",
                "text": "delay 100",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-dead2",
                "maxclass": "newobj",
                "text": "delay 200",
                "numoutlets": 1,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-movie", 0], "destination": ["obj-pwindow", 0]},
            {"source": ["obj-dead1", 0], "destination": ["obj-dead2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-002"]
        # Should detect both dead1 and dead2 as dead branches
        assert len(errors) >= 1
        dead_ids = [e.object_id for e in errors]
        assert "obj-dead1" in dead_ids or "obj-dead2" in dead_ids

    def test_multi_source_inlet_warning(self, tmp_path: Path) -> None:
        """Multiple sources to single inlet = WARNING (dead-003)."""
        boxes = [
            {
                "id": "obj-source1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-source2",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            # Both sources connected to same inlet 0 of pwindow
            {"source": ["obj-source1", 0], "destination": ["obj-pwindow", 0]},
            {"source": ["obj-source2", 0], "destination": ["obj-pwindow", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        warnings = [w for w in linter.warnings if w.rule == "dead-003"]
        assert len(warnings) >= 1
        assert warnings[0].object_id is not None
        assert "obj-pwindow" in warnings[0].object_id

    def test_comment_orphan_allowed(self, tmp_path: Path) -> None:
        """Comment objects should not be flagged as orphans."""
        boxes = [
            {
                "id": "obj-connected1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-connected2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-comment",
                "maxclass": "comment",
                "text": "This is a label",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-connected1", 0], "destination": ["obj-connected2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-001"]
        # Should not flag comment as orphan
        for error in errors:
            assert error.object_id != "obj-comment"

    def test_panel_orphan_allowed(self, tmp_path: Path) -> None:
        """Panel objects should not be flagged as orphans."""
        boxes = [
            {
                "id": "obj-connected1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-connected2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-panel",
                "maxclass": "panel",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-connected1", 0], "destination": ["obj-connected2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-001"]
        for error in errors:
            assert error.object_id != "obj-panel"

    def test_fpic_orphan_allowed(self, tmp_path: Path) -> None:
        """fpic (image) objects should not be flagged as orphans."""
        boxes = [
            {
                "id": "obj-connected1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-connected2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-fpic",
                "maxclass": "fpic",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-connected1", 0], "destination": ["obj-connected2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-001"]
        for error in errors:
            assert error.object_id != "obj-fpic"

    def test_live_comment_orphan_allowed(self, tmp_path: Path) -> None:
        """live.comment objects should not be flagged as orphans."""
        boxes = [
            {
                "id": "obj-connected1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-connected2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-live-comment",
                "maxclass": "live.comment",
                "text": "This is a live comment",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-connected1", 0], "destination": ["obj-connected2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        errors = [e for e in linter.errors if e.rule == "dead-001"]
        for error in errors:
            assert error.object_id != "obj-live-comment"

    def test_no_dead_code_valid_pipeline(self, tmp_path: Path) -> None:
        """Valid complete pipeline should not trigger dead code errors."""
        boxes = [
            {
                "id": "obj-toggle",
                "maxclass": "toggle",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-qmetro",
                "maxclass": "newobj",
                "text": "qmetro 30",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-toggle", 0], "destination": ["obj-qmetro", 0]},
            {"source": ["obj-qmetro", 0], "destination": ["obj-movie", 0]},
            {"source": ["obj-movie", 0], "destination": ["obj-pix", 0]},
            {"source": ["obj-pix", 0], "destination": ["obj-pwindow", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))
        linter = MaxhelpLinter()
        linter.validate_file(test_file)
        dead_errors = [e for e in linter.errors if e.rule.startswith("dead-")]
        assert len(dead_errors) == 0

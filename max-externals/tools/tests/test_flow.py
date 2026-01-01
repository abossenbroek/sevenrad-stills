# ruff: noqa: S101
"""Tests for MaxhelpLinter signal flow validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestFlowValidation:
    """Test signal flow validation rules (flow-001, flow-002, flow-003)."""

    def test_missing_qmetro_error(self, tmp_path: Path) -> None:
        """Missing qmetro for video = ERROR."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-3",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
            {"source": ["obj-2", 0], "destination": ["obj-3", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "flow-001"]
        assert len(errors) >= 1

    def test_pix_disconnected_inlet_error(self, tmp_path: Path) -> None:
        """jit.gl.pix with disconnected inlet = ERROR."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "flow-003"]
        assert len(errors) >= 1

    def test_pix_not_connected_to_display_error(self, tmp_path: Path) -> None:
        """jit.gl.pix output not reaching display = ERROR."""
        boxes = [
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
            # jit.pwindow exists but not connected to jit.gl.pix
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-qmetro", 0], "destination": ["obj-movie", 0]},
            {"source": ["obj-movie", 0], "destination": ["obj-pix", 0]},
            # NOTE: No connection from obj-pix to obj-pwindow
        ]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "flow-002"]
        assert len(errors) >= 1

    def test_complete_pipeline_no_error(self, tmp_path: Path) -> None:
        """Complete qmetro->movie->pix->pwindow pipeline = no flow errors."""
        boxes = [
            {
                "id": "obj-qmetro",
                "maxclass": "newobj",
                "text": "qmetro 30",
                "numoutlets": 1,
                "numinlets": 2,
            },
            {
                "id": "obj-toggle",
                "maxclass": "toggle",
                "numoutlets": 1,
                "numinlets": 1,
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

        flow_errors = [e for e in linter.errors if e.rule.startswith("flow-")]
        assert len(flow_errors) == 0

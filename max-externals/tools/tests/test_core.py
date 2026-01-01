# ruff: noqa: S101
"""Tests for MaxhelpLinter core validation (strict mode, required components)."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestStrictMode:
    """Test strict mode treats warnings as errors."""

    def test_strict_mode_warnings_fail(self, tmp_path: Path) -> None:
        """Test that warnings cause failure in strict mode."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "slider",
                "patching_rect": [100.0, 100.0, 200.0, 20.0],
            },
            {
                "id": "obj-2",
                "maxclass": "slider",
                # Minor overlap
                "patching_rect": [190.0, 100.0, 200.0, 20.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter(strict=True)
        linter.validate_file(test_file)

        # In strict mode, warnings should cause has_errors() to return True
        assert linter.has_errors()


class TestRequiredComponents:
    """Test that required components (video input/output) are validated."""

    def test_missing_jit_movie_error(self, tmp_path: Path) -> None:
        """Test that missing jit.movie triggers error."""
        boxes = [
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "jit.gl.pix @gen sr.effect",
                "patching_rect": [100.0, 100.0, 150.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        valid = linter.validate_file(test_file)

        assert not valid
        signal_errors = [e for e in linter.errors if e.rule == "signal-flow"]
        assert any(
            "jit.movie" in e.message or "video input" in e.message
            for e in signal_errors
        )

    def test_missing_jit_pwindow_error(self, tmp_path: Path) -> None:
        """Test that missing jit.pwindow triggers error."""
        boxes = [
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 2,
                "text": "jit.movie @output_texture 1",
                "patching_rect": [100.0, 100.0, 150.0, 22.0],
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "jit.gl.pix @gen sr.effect",
                "patching_rect": [100.0, 150.0, 150.0, 22.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        valid = linter.validate_file(test_file)

        assert not valid
        signal_errors = [e for e in linter.errors if e.rule == "signal-flow"]
        assert any("jit.pwindow" in e.message for e in signal_errors)

    def test_complete_pipeline_valid(self, tmp_path: Path) -> None:
        """Test that complete pipeline with all components is valid."""
        boxes = [
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 2,
                "text": "jit.movie @output_texture 1",
                "outlettype": ["jit_gl_texture", ""],
                "patching_rect": [100.0, 100.0, 150.0, 22.0],
            },
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "jit.gl.pix @gen sr.effect",
                "patching_rect": [100.0, 150.0, 150.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "patching_rect": [100.0, 200.0, 320.0, 180.0],
            },
        ]

        lines = [
            {"source": ["obj-movie", 0], "destination": ["obj-pix", 0]},
            {"source": ["obj-pix", 0], "destination": ["obj-pwindow", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have errors about missing components
        missing_errors = [
            e
            for e in linter.errors
            if e.rule == "signal-flow" and ("Missing" in e.message)
        ]
        assert len(missing_errors) == 0

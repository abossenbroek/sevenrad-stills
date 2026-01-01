# ruff: noqa: S101
"""Tests for MaxhelpLinter connection type validation."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


class TestConnectionTypes:
    """Test connection type validation (red-team fix)."""

    def test_texture_to_matrix_error(self, tmp_path: Path) -> None:
        """Test that texture->matrix connections trigger errors."""
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
                "id": "obj-matrix",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 2,
                "text": "jit.matrix 4 char 320 240",
                "patching_rect": [100.0, 150.0, 150.0, 22.0],
            },
        ]

        lines = [
            {
                "source": ["obj-movie", 0],
                "destination": ["obj-matrix", 0],
            }
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have an error about connection type mismatch
        connection_errors = [e for e in linter.errors if e.rule == "connection-type"]
        assert len(connection_errors) >= 1
        assert "texture" in connection_errors[0].message.lower()

    def test_matrix_to_texture_error(self, tmp_path: Path) -> None:
        """Test that matrix->texture connections trigger errors."""
        boxes = [
            {
                "id": "obj-movie",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 2,
                "text": "jit.movie",  # No @output_texture, outputs matrix
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
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

        lines = [
            {
                "source": ["obj-movie", 0],
                "destination": ["obj-pix", 0],
            }
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have an error about connection type mismatch
        connection_errors = [e for e in linter.errors if e.rule == "connection-type"]
        assert len(connection_errors) >= 1
        assert "matrix" in connection_errors[0].message.lower()

    def test_texture_to_texture_valid(self, tmp_path: Path) -> None:
        """Test that texture->texture connections are valid."""
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
        ]

        lines = [
            {
                "source": ["obj-movie", 0],
                "destination": ["obj-pix", 0],
            }
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have connection-type errors
        connection_errors = [e for e in linter.errors if e.rule == "connection-type"]
        assert len(connection_errors) == 0

    def test_info_outlet_to_texture_warning(self, tmp_path: Path) -> None:
        """Test that info outlet->texture inlet triggers warning."""
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
        ]

        # Connect outlet 1 (info) to jit.gl.pix inlet 0 (expects texture)
        lines = [
            {
                "source": ["obj-movie", 1],  # Info outlet
                "destination": ["obj-pix", 0],
            }
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have a warning about info outlet to texture inlet
        connection_warnings = [
            w for w in linter.warnings if w.rule == "connection-type"
        ]
        assert len(connection_warnings) >= 1
        assert "info" in connection_warnings[0].message.lower()


class TestDisplaySinkValidation:
    """Test generalized Jitter display sink validation."""

    def test_empty_outlettype_to_pwindow_warns(self, tmp_path: Path) -> None:
        """Any object with outlettype [''] to pwindow triggers warning."""
        boxes = [
            {
                "id": "obj-external",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "sr.maskgen",
                "outlettype": [""],  # Empty - should warn
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["jit_matrix", ""],
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]
        lines = [{"source": ["obj-external", 0], "destination": ["obj-pwindow", 0]}]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have a warning about display sink type
        display_warnings = [w for w in linter.warnings if w.rule == "display-sink-type"]
        assert len(display_warnings) >= 1
        assert "bang_or_message" in display_warnings[0].message

    def test_matrix_outlettype_valid(self, tmp_path: Path) -> None:
        """outlettype ['jit_matrix'] passes validation."""
        boxes = [
            {
                "id": "obj-external",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "sr.maskgen",
                "outlettype": ["jit_matrix"],  # Correct
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["jit_matrix", ""],
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]
        lines = [{"source": ["obj-external", 0], "destination": ["obj-pwindow", 0]}]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have display-sink-type warnings
        display_warnings = [w for w in linter.warnings if w.rule == "display-sink-type"]
        assert len(display_warnings) == 0

    def test_texture_outlettype_valid(self, tmp_path: Path) -> None:
        """outlettype ['jit_gl_texture'] passes validation."""
        boxes = [
            {
                "id": "obj-pix",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 2,
                "text": "jit.gl.pix @gen sr.effect",
                "outlettype": ["jit_gl_texture", ""],
                "patching_rect": [100.0, 100.0, 150.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["jit_matrix", ""],
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]
        lines = [{"source": ["obj-pix", 0], "destination": ["obj-pwindow", 0]}]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have display-sink-type warnings
        display_warnings = [w for w in linter.warnings if w.rule == "display-sink-type"]
        assert len(display_warnings) == 0

    def test_custom_external_with_empty_outlettype_warns(self, tmp_path: Path) -> None:
        """foo.myexternal with [''] to pwindow warns (name-agnostic)."""
        boxes = [
            {
                "id": "obj-custom",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "foo.myexternal @param 1",  # Different prefix than sr.*
                "outlettype": [""],  # Empty - should warn
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["jit_matrix", ""],
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]
        lines = [{"source": ["obj-custom", 0], "destination": ["obj-pwindow", 0]}]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should warn - validation is name-agnostic
        display_warnings = [w for w in linter.warnings if w.rule == "display-sink-type"]
        assert len(display_warnings) >= 1
        assert "foo.myexternal" in display_warnings[0].message

    def test_jit_op_chain_valid(self, tmp_path: Path) -> None:
        """bad_object -> jit.op -> pwindow: jit.op has valid outlettype."""
        boxes = [
            {
                "id": "obj-bad",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "text": "bad.object",
                "outlettype": [""],  # Would warn if direct
                "patching_rect": [100.0, 50.0, 100.0, 22.0],
            },
            {
                "id": "obj-op",
                "maxclass": "newobj",
                "numinlets": 2,
                "numoutlets": 2,
                "text": "jit.op @op +",
                "outlettype": ["jit_matrix", ""],  # jit.op has proper type
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
            },
            {
                "id": "obj-pwindow",
                "maxclass": "jit.pwindow",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["jit_matrix", ""],
                "patching_rect": [100.0, 150.0, 320.0, 180.0],
            },
        ]
        lines = [
            {"source": ["obj-bad", 0], "destination": ["obj-op", 0]},
            {"source": ["obj-op", 0], "destination": ["obj-pwindow", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # jit.op -> pwindow should NOT warn (jit.op has valid outlettype)
        display_warnings = [
            w
            for w in linter.warnings
            if w.rule == "display-sink-type" and "obj-op" in str(w.object_id)
        ]
        assert len(display_warnings) == 0

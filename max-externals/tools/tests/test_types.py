# ruff: noqa: S101
"""Tests for MaxhelpLinter type validation functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import (
    TYPE_COMPATIBLE,
    JitterType,
    MaxhelpLinter,
    types_compatible,
)

from tests.conftest import create_test_patcher


class TestJitterType:
    """Test JitterType enum and TYPE_COMPATIBLE matrix."""

    def test_texture_to_texture_compatible(self) -> None:
        """Texture->Texture is valid."""
        assert TYPE_COMPATIBLE[(JitterType.TEXTURE, JitterType.TEXTURE)] is True

    def test_texture_to_matrix_incompatible(self) -> None:
        """Texture->Matrix is INVALID (strict)."""
        assert TYPE_COMPATIBLE[(JitterType.TEXTURE, JitterType.MATRIX)] is False

    def test_matrix_to_texture_incompatible(self) -> None:
        """Matrix->Texture is INVALID (strict)."""
        assert TYPE_COMPATIBLE[(JitterType.MATRIX, JitterType.TEXTURE)] is False

    def test_info_to_data_incompatible(self) -> None:
        """Info outlet->data inlet is INVALID."""
        assert TYPE_COMPATIBLE[(JitterType.INFO, JitterType.TEXTURE)] is False
        assert TYPE_COMPATIBLE[(JitterType.INFO, JitterType.MATRIX)] is False

    def test_bang_to_message_compatible(self) -> None:
        """Bang can go to message inlet."""
        assert TYPE_COMPATIBLE[(JitterType.BANG, JitterType.MESSAGE)] is True

    def test_types_compatible_helper(self) -> None:
        """types_compatible() helper works."""
        assert types_compatible(JitterType.TEXTURE, JitterType.TEXTURE) is True
        assert types_compatible(JitterType.TEXTURE, JitterType.MATRIX) is False

    def test_matrix_to_matrix_compatible(self) -> None:
        """Matrix->Matrix is valid."""
        assert TYPE_COMPATIBLE[(JitterType.MATRIX, JitterType.MATRIX)] is True

    def test_unknown_to_unknown_compatible(self) -> None:
        """Unknown->Unknown is permissive."""
        assert TYPE_COMPATIBLE[(JitterType.UNKNOWN, JitterType.UNKNOWN)] is True

    def test_bang_to_texture_incompatible(self) -> None:
        """Bang cannot be image data."""
        assert TYPE_COMPATIBLE[(JitterType.BANG, JitterType.TEXTURE)] is False
        assert TYPE_COMPATIBLE[(JitterType.BANG, JitterType.MATRIX)] is False

    def test_message_to_texture_incompatible(self) -> None:
        """Message cannot be image data."""
        assert TYPE_COMPATIBLE[(JitterType.MESSAGE, JitterType.TEXTURE)] is False
        assert TYPE_COMPATIBLE[(JitterType.MESSAGE, JitterType.MATRIX)] is False

    def test_message_to_message_compatible(self) -> None:
        """Message can go to message inlet."""
        assert TYPE_COMPATIBLE[(JitterType.MESSAGE, JitterType.MESSAGE)] is True

    def test_jitter_type_values(self) -> None:
        """JitterType enum has correct string values."""
        assert JitterType.TEXTURE.value == "jit_gl_texture"
        assert JitterType.MATRIX.value == "jit_matrix"
        assert JitterType.TEXTURE_NAME.value == "texture_name"
        assert JitterType.BANG.value == "bang"
        assert JitterType.MESSAGE.value == "message"
        assert JitterType.INFO.value == "info"
        assert JitterType.UNKNOWN.value == "unknown"


class TestTypeValidation:
    """Tests for strict type validation rules (type-001 to type-005)."""

    def test_texture_to_matrix_error(self, tmp_path: Path) -> None:
        """Texture output to matrix inlet = ERROR (type-001)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.matrix 4 char 320 240",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "type-001"]
        assert len(errors) >= 1
        assert "type mismatch" in errors[0].message.lower()

    def test_matrix_to_texture_error(self, tmp_path: Path) -> None:
        """Matrix output to texture inlet = ERROR (type-002)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",  # No @output_texture, outputs matrix
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
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "type-002"]
        assert len(errors) >= 1
        assert "type mismatch" in errors[0].message.lower()

    def test_info_outlet_to_data_inlet_error(self, tmp_path: Path) -> None:
        """Info outlet to texture/matrix inlet = ERROR (type-003)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        # Connect outlet 1 (info) to jit.gl.pix inlet 0 (expects texture)
        lines = [{"source": ["obj-1", 1], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "type-003"]
        assert len(errors) >= 1
        assert "info outlet" in errors[0].message.lower()

    def test_movie_without_output_texture_in_gpu_pipeline_error(
        self, tmp_path: Path
    ) -> None:
        """jit.movie without @output_texture 1 connected to jit.gl.pix = ERROR."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @autostart 1",
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
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "type-004"]
        assert len(errors) >= 1

    def test_pwindow_texture_without_context_error(self, tmp_path: Path) -> None:
        """jit.pwindow receiving texture without GPU context = ERROR (type-005)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "jit.pwindow",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        errors = [e for e in linter.errors if e.rule == "type-005"]
        assert len(errors) >= 1
        assert "gpu context" in errors[0].message.lower()

    def test_valid_texture_chain_no_error(self, tmp_path: Path) -> None:
        """Valid texture->texture chain with GPU context = no type errors."""
        boxes = [
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_test_ctx @visible 0",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1 @drawto sr_test_ctx",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-3",
                "maxclass": "jit.pwindow",
                "numoutlets": 2,
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

        type_errors = [e for e in linter.errors if e.rule.startswith("type-")]
        assert len(type_errors) == 0

    def test_movie_with_output_texture_to_pix_valid(self, tmp_path: Path) -> None:
        """jit.movie with @output_texture 1 to jit.gl.pix = valid (no type-004)."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.test",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        type_004_errors = [e for e in linter.errors if e.rule == "type-004"]
        assert len(type_004_errors) == 0

    def test_pwindow_with_gpu_context_valid(self, tmp_path: Path) -> None:
        """jit.pwindow receiving texture with jit.world context = valid."""
        boxes = [
            {
                "id": "obj-world",
                "maxclass": "newobj",
                "text": "jit.world sr_test_ctx @visible 0",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
            {
                "id": "obj-2",
                "maxclass": "jit.pwindow",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        type_005_errors = [e for e in linter.errors if e.rule == "type-005"]
        assert len(type_005_errors) == 0

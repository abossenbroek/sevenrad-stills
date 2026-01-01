#!/usr/bin/env python3
# ruff: noqa: S101
"""
Tests for the MaxhelpLinter overlap detection and other validation features.

These tests verify that the linter correctly detects:
- Overlapping UI components
- Connected boxes (which should be allowed to overlap)
- Different object types with appropriate default sizes
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from lint_maxhelp import (
    TYPE_COMPATIBLE,
    JitterType,
    LintGraph,
    MaxhelpLinter,
    types_compatible,
)


def create_test_patcher(
    boxes: list[dict[str, Any]], lines: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create a minimal test patcher JSON structure."""
    if lines is None:
        lines = []

    return {
        "patcher": {
            "fileversion": 1,
            "boxes": [{"box": box} for box in boxes],
            "lines": [{"patchline": line} for line in lines],
        }
    }


class TestOverlapDetection:
    """Test overlap detection functionality."""

    def test_no_overlap_well_separated(self, tmp_path: Path) -> None:
        """Test that well-separated boxes don't trigger overlap warnings."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "dial",
                "patching_rect": [200.0, 100.0, 40.0, 40.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Check specifically for overlap errors (not other validation errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        overlap_warnings = [w for w in linter.warnings if w.rule == "overlap"]
        assert len(overlap_errors) == 0
        assert len(overlap_warnings) == 0

    def test_significant_overlap_error(self, tmp_path: Path) -> None:
        """Test that significant overlaps (>25%) trigger errors."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "dial",
                # Overlaps by 30x30 = 900px² (56% of 40x40=1600px²)
                "patching_rect": [110.0, 110.0, 40.0, 40.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        valid = linter.validate_file(test_file)

        assert not valid
        assert any(e.rule == "overlap" for e in linter.errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert "900" in overlap_errors[0].message

    def test_minor_overlap_warning(self, tmp_path: Path) -> None:
        """Test that minor overlaps (<25% but >100px²) trigger warnings."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "slider",
                "patching_rect": [100.0, 100.0, 100.0, 50.0],  # 5000px²
            },
            {
                "id": "obj-2",
                "maxclass": "slider",
                # Overlaps by 11x11 = 121px² (2.4% of 5000px², so <25% but >100px²)
                "patching_rect": [189.0, 139.0, 100.0, 50.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have a warning for partial overlap
        overlap_warnings = [w for w in linter.warnings if w.rule == "overlap"]
        assert len(overlap_warnings) > 0
        assert "partially overlap" in overlap_warnings[0].message

    def test_connected_boxes_allowed_to_overlap(self, tmp_path: Path) -> None:
        """Test that connected boxes are allowed to overlap."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "number",
                "numinlets": 1,
                "numoutlets": 2,
                # Intentionally overlaps with dial (common Max style)
                "patching_rect": [120.0, 110.0, 50.0, 22.0],
            },
        ]

        lines = [
            {
                "source": ["obj-1", 0],
                "destination": ["obj-2", 0],
            }
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should not report overlap for connected boxes
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert len(overlap_errors) == 0

    def test_comments_excluded_from_overlap(self, tmp_path: Path) -> None:
        """Test that comments are excluded from overlap detection."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "comment",
                # Overlaps with dial, but should be ignored
                "patching_rect": [110.0, 110.0, 100.0, 20.0],
                "text": "Label for dial",
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Check specifically for overlap errors (not other validation errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert len(overlap_errors) == 0

    def test_multiple_overlaps_all_reported(self, tmp_path: Path) -> None:
        """Test that multiple overlaps are all detected and reported."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "dial",
                "patching_rect": [110.0, 110.0, 40.0, 40.0],  # Overlaps obj-1
            },
            {
                "id": "obj-3",
                "maxclass": "dial",
                "patching_rect": [120.0, 120.0, 40.0, 40.0],  # Overlaps obj-2
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        # Should detect obj-1/obj-2 and obj-2/obj-3 overlaps
        assert len(overlap_errors) >= 2

    def test_different_object_types(self, tmp_path: Path) -> None:
        """Test overlap detection works for different Max object types."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "button",
                "patching_rect": [100.0, 100.0, 24.0, 24.0],
            },
            {
                "id": "obj-2",
                "maxclass": "toggle",
                "patching_rect": [110.0, 110.0, 24.0, 24.0],  # Overlaps button
            },
            {
                "id": "obj-3",
                "maxclass": "flonum",
                "patching_rect": [200.0, 100.0, 50.0, 22.0],
            },
            {
                "id": "obj-4",
                "maxclass": "message",
                "patching_rect": [210.0, 100.0, 50.0, 22.0],  # Overlaps flonum
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        # Should detect both button/toggle and flonum/message overlaps
        assert len(overlap_errors) >= 2

    def test_jit_pwindow_overlap(self, tmp_path: Path) -> None:
        """Test that jit.pwindow overlap detection works."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "jit.pwindow",
                "patching_rect": [30.0, 280.0, 320.0, 180.0],
            },
            {
                "id": "obj-2",
                "maxclass": "dial",
                # Overlaps with pwindow
                "patching_rect": [100.0, 300.0, 40.0, 40.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert len(overlap_errors) >= 1

    def test_edge_touching_no_overlap(self, tmp_path: Path) -> None:
        """Test that boxes touching at edges don't trigger overlap."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "dial",
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
            },
            {
                "id": "obj-2",
                "maxclass": "dial",
                # Touches right edge of obj-1 (140 = 100 + 40)
                "patching_rect": [140.0, 100.0, 40.0, 40.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Check specifically for overlap errors (not other validation errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert len(overlap_errors) == 0

    def test_vertical_separation(self, tmp_path: Path) -> None:
        """Test that vertically separated boxes don't trigger overlap."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "slider",
                "patching_rect": [100.0, 100.0, 200.0, 20.0],
            },
            {
                "id": "obj-2",
                "maxclass": "slider",
                # Below obj-1 (120 = 100 + 20)
                "patching_rect": [100.0, 125.0, 200.0, 20.0],
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Check specifically for overlap errors (not other validation errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap"]
        assert len(overlap_errors) == 0


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


class TestConnectionTypes:
    """Test connection type validation (red-team fix)."""

    def test_texture_to_matrix_error(self, tmp_path: Path) -> None:
        """Test that texture→matrix connections trigger errors."""
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
        """Test that matrix→texture connections trigger errors."""
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
        """Test that texture→texture connections are valid."""
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
        """Test that info outlet→texture inlet triggers warning."""
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
        """bad_object → jit.op → pwindow: jit.op has valid outlettype."""
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

        # jit.op → pwindow should NOT warn (jit.op has valid outlettype)
        display_warnings = [
            w
            for w in linter.warnings
            if w.rule == "display-sink-type" and "obj-op" in str(w.object_id)
        ]
        assert len(display_warnings) == 0


class TestDialDecimals:
    """Test dial-decimals validation for fractional output dials."""

    def test_dial_without_decimals_warns(self, tmp_path: Path) -> None:
        """Test that dial with fractional mult but no decimals triggers warning."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 100.0,
                "min": 0.0,
                "mult": 0.01,  # Fractional output, needs decimals=2
                # Missing "decimals" attribute
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should have warning about missing decimals
        decimal_warnings = [w for w in linter.warnings if w.rule == "dial-decimals"]
        assert len(decimal_warnings) >= 1
        assert "decimals" in decimal_warnings[0].message
        assert ">= 2" in decimal_warnings[0].message

    def test_dial_with_sufficient_decimals_valid(self, tmp_path: Path) -> None:
        """Test that dial with proper decimals passes validation."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 100.0,
                "min": 0.0,
                "mult": 0.01,
                "decimals": 2,  # Correct decimals
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have warning about decimals
        decimal_warnings = [w for w in linter.warnings if w.rule == "dial-decimals"]
        assert len(decimal_warnings) == 0

    def test_dial_with_integer_mult_valid(self, tmp_path: Path) -> None:
        """Test that dial with mult >= 1.0 doesn't need decimals."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 100.0,
                "min": 0.0,
                "mult": 1.0,  # Integer output
                # No "decimals" needed
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have warning about decimals for integer mult
        decimal_warnings = [w for w in linter.warnings if w.rule == "dial-decimals"]
        assert len(decimal_warnings) == 0

    def test_dial_mult_0001_requires_3_decimals(self, tmp_path: Path) -> None:
        """Test that mult=0.001 requires decimals >= 3."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 1000.0,
                "min": 0.0,
                "mult": 0.001,
                "decimals": 2,  # Insufficient - needs 3
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should warn about insufficient decimals
        decimal_warnings = [w for w in linter.warnings if w.rule == "dial-decimals"]
        assert len(decimal_warnings) >= 1
        assert ">= 3" in decimal_warnings[0].message


class TestDialInitRange:
    """Test dial set value range validation.

    The dial `set` message takes INTERNAL POSITION (0 to size), not output value.
    For a dial with size=50, mult=0.01: position 22 → output 0.22
    """

    def test_dial_set_value_out_of_position_range_warns(self, tmp_path: Path) -> None:
        """Test that set value outside dial position range triggers warning."""
        # dial with size=50 → position range 0-50
        # "set 100" is out of range (100 > 50)
        boxes = [
            {
                "id": "obj-loadbang",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["bang"],
                "patching_rect": [100.0, 50.0, 60.0, 22.0],
                "text": "loadbang",
            },
            {
                "id": "obj-set-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 80.0, 50.0, 22.0],
                "text": "set 100",  # OUT OF RANGE: position > size (50)
            },
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 110.0, 40.0, 40.0],
                "size": 50.0,
                "min": 0.0,
                "mult": 0.01,
                "decimals": 2,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 160.0, 60.0, 22.0],
            },
            {
                "id": "obj-param-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 190.0, 80.0, 22.0],
                "text": "gap_width $1",
            },
        ]
        lines = [
            {"source": ["obj-loadbang", 0], "destination": ["obj-set-msg", 0]},
            {"source": ["obj-set-msg", 0], "destination": ["obj-dial", 0]},
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-param-msg", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should warn about out-of-range set value
        range_warnings = [w for w in linter.warnings if w.rule == "dial-init-range"]
        assert (
            len(range_warnings) >= 1
        ), f"Expected dial-init-range warning, got: {linter.warnings}"
        assert "out of position range" in range_warnings[0].message
        assert "100" in range_warnings[0].message

    def test_dial_set_position_in_range_valid(self, tmp_path: Path) -> None:
        """Test that set value within dial position range passes validation."""
        # dial with size=50 → position range 0-50
        # "set 22" is valid (22 within 0-50) → output 22*0.01=0.22
        boxes = [
            {
                "id": "obj-loadbang",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["bang"],
                "patching_rect": [100.0, 50.0, 60.0, 22.0],
                "text": "loadbang",
            },
            {
                "id": "obj-set-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 80.0, 50.0, 22.0],
                "text": "set 22",  # VALID: position 22 within 0-50
            },
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 110.0, 40.0, 40.0],
                "size": 50.0,
                "min": 0.0,
                "mult": 0.01,
                "decimals": 2,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 160.0, 60.0, 22.0],
            },
            {
                "id": "obj-param-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 190.0, 80.0, 22.0],
                "text": "gap_width $1",
            },
        ]
        lines = [
            {"source": ["obj-loadbang", 0], "destination": ["obj-set-msg", 0]},
            {"source": ["obj-set-msg", 0], "destination": ["obj-dial", 0]},
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-param-msg", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have warning about out-of-range set value
        range_warnings = [w for w in linter.warnings if w.rule == "dial-init-range"]
        assert (
            len(range_warnings) == 0
        ), f"Unexpected dial-init-range warning: {range_warnings}"

    def test_dial_mult_1_set_integer_valid(self, tmp_path: Path) -> None:
        """Test that for mult=1.0, set with integer value is valid."""
        # dial with size=98 → position range 0-98
        # "set 14" is valid (14 within 0-98)
        boxes = [
            {
                "id": "obj-loadbang",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["bang"],
                "patching_rect": [100.0, 50.0, 60.0, 22.0],
                "text": "loadbang",
            },
            {
                "id": "obj-set-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 80.0, 50.0, 22.0],
                "text": "set 14",  # VALID: 14 is within 0-98
            },
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 110.0, 40.0, 40.0],
                "size": 98.0,
                "min": 0.0,
                "mult": 1.0,  # Output = position
            },
            {
                "id": "obj-number",
                "maxclass": "number",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 160.0, 60.0, 22.0],
            },
            {
                "id": "obj-param-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 190.0, 90.0, 22.0],
                "text": "scan_period $1",
            },
        ]
        lines = [
            {"source": ["obj-loadbang", 0], "destination": ["obj-set-msg", 0]},
            {"source": ["obj-set-msg", 0], "destination": ["obj-dial", 0]},
            {"source": ["obj-dial", 0], "destination": ["obj-number", 0]},
            {"source": ["obj-number", 0], "destination": ["obj-param-msg", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have warning about out-of-range set value
        range_warnings = [w for w in linter.warnings if w.rule == "dial-init-range"]
        assert (
            len(range_warnings) == 0
        ), f"Unexpected dial-init-range warning: {range_warnings}"


class TestCExternalDialRanges:
    """Test C external dial range validation using metadata."""

    def test_dial_below_param_min_error(self, tmp_path: Path) -> None:
        """Test that dial outputting below param min triggers error."""
        # Dial with size=50, mult=0.01 -> output range [0.0, 0.5]
        # sr.maskgen gap_width has min=0.001, so this should ERROR
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 50.0,
                # No min attribute - dial outputs [0, 0.5], starts at 0!
                "mult": 0.01,
                "decimals": 2,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 150.0, 60.0, 22.0],
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 180.0, 80.0, 22.0],
                "text": "gap_width $1",
            },
            {
                "id": "obj-maskgen",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["jit_matrix"],
                "patching_rect": [100.0, 220.0, 100.0, 22.0],
                "text": "sr.maskgen",
            },
        ]
        lines = [
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-maskgen", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should ERROR: dial min 0.0 < param min 0.001
        range_errors = [e for e in linter.errors if e.rule == "c-external-dial-range"]
        assert (
            len(range_errors) >= 1
        ), f"Expected c-external-dial-range error, got: {linter.errors}"
        assert "gap_width" in range_errors[0].message
        assert "EXCEEDS" in range_errors[0].message

    def test_dial_matching_param_bounds_valid(self, tmp_path: Path) -> None:
        """Test that dial with floatoutput=1 and min offset passes validation."""
        # Dial with floatoutput=1, min=0.001, size=499, mult=0.001
        # Output range: [0.001, 0.001 + 499*0.001] = [0.001, 0.5]
        # This exactly matches sr.maskgen gap_width bounds [0.001, 0.5]
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 499.0,
                "min": 0.001,  # Output offset with floatoutput=1
                "mult": 0.001,
                "decimals": 3,
                "floatoutput": 1,  # Required for min to work!
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 140.0, 60.0, 22.0],
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 170.0, 80.0, 22.0],
                "text": "gap_width $1",
            },
            {
                "id": "obj-maskgen",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["jit_matrix"],
                "patching_rect": [100.0, 210.0, 100.0, 22.0],
                "text": "sr.maskgen",
            },
        ]
        lines = [
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-maskgen", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have c-external-dial-range errors
        range_errors = [e for e in linter.errors if e.rule == "c-external-dial-range"]
        assert (
            len(range_errors) == 0
        ), f"Unexpected c-external-dial-range error: {range_errors}"

    def test_dial_exceeds_param_max_error(self, tmp_path: Path) -> None:
        """Test that dial exceeding param max triggers error."""
        # Dial with output range [0, 0.999] for gap_width (max is 0.5)
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 999.0,
                # No min - dial outputs [0, 0.999], exceeds max 0.5!
                "mult": 0.001,
                "decimals": 3,
            },
            {
                "id": "obj-flonum",
                "maxclass": "flonum",
                "numinlets": 1,
                "numoutlets": 2,
                "outlettype": ["", "bang"],
                "patching_rect": [100.0, 150.0, 60.0, 22.0],
            },
            {
                "id": "obj-msg",
                "maxclass": "message",
                "numinlets": 2,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [100.0, 180.0, 80.0, 22.0],
                "text": "gap_width $1",
            },
            {
                "id": "obj-maskgen",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["jit_matrix"],
                "patching_rect": [100.0, 220.0, 100.0, 22.0],
                "text": "sr.maskgen",
            },
        ]
        lines = [
            {"source": ["obj-dial", 0], "destination": ["obj-flonum", 0]},
            {"source": ["obj-flonum", 0], "destination": ["obj-msg", 0]},
            {"source": ["obj-msg", 0], "destination": ["obj-maskgen", 0]},
        ]

        patcher = create_test_patcher(boxes, lines)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should ERROR: dial max 1.0 > param max 0.5
        range_errors = [e for e in linter.errors if e.rule == "c-external-dial-range"]
        assert (
            len(range_errors) >= 1
        ), f"Expected c-external-dial-range error, got: {linter.errors}"
        assert "EXCEEDS" in range_errors[0].message

    def test_no_metadata_graceful(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Test that missing metadata file doesn't crash linter."""
        # Monkeypatch to return empty metadata
        monkeypatch.setattr(MaxhelpLinter, "_load_c_external_params", lambda _: {})

        # Create minimal patcher with sr.maskgen
        boxes = [
            {
                "id": "obj-maskgen",
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["jit_matrix"],
                "patching_rect": [100.0, 100.0, 100.0, 22.0],
                "text": "sr.maskgen",
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        # Should not raise, validation is skipped gracefully
        linter.validate_file(test_file)

        # No c-external-dial-range errors (validation skipped)
        range_errors = [e for e in linter.errors if e.rule == "c-external-dial-range"]
        assert len(range_errors) == 0


class TestDialFloatOutput:
    """Test that dials with float values require floatoutput=1."""

    def test_dial_float_mult_without_floatoutput_error(self, tmp_path: Path) -> None:
        """Dial with mult=0.001 but no floatoutput should ERROR."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 499.0,
                "mult": 0.001,  # Float mult - needs floatoutput!
                # Missing floatoutput: 1
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should ERROR: dial needs floatoutput=1
        float_errors = [e for e in linter.errors if e.rule == "dial-float-output"]
        assert (
            len(float_errors) >= 1
        ), f"Expected dial-float-output error, got: {linter.errors}"
        assert "floatoutput" in float_errors[0].message

    def test_dial_float_min_without_floatoutput_error(self, tmp_path: Path) -> None:
        """Dial with min=0.001 but no floatoutput should ERROR."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 100.0,
                "min": 0.001,  # Float min - needs floatoutput!
                "mult": 1.0,
                # Missing floatoutput: 1
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should ERROR: dial needs floatoutput=1
        float_errors = [e for e in linter.errors if e.rule == "dial-float-output"]
        assert (
            len(float_errors) >= 1
        ), f"Expected dial-float-output error, got: {linter.errors}"
        assert "floatoutput" in float_errors[0].message

    def test_dial_float_with_floatoutput_valid(self, tmp_path: Path) -> None:
        """Dial with mult=0.001 and floatoutput=1 should pass."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 499.0,
                "min": 0.001,
                "mult": 0.001,
                "floatoutput": 1,  # Correct!
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have dial-float-output errors
        float_errors = [e for e in linter.errors if e.rule == "dial-float-output"]
        assert (
            len(float_errors) == 0
        ), f"Unexpected dial-float-output error: {float_errors}"

    def test_dial_integer_mult_no_floatoutput_valid(self, tmp_path: Path) -> None:
        """Dial with mult=1.0 and min=0 doesn't need floatoutput."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 100.0,
                "min": 0.0,
                "mult": 1.0,  # Integer mult, min=0 - doesn't need floatoutput
                # No floatoutput needed
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should NOT have dial-float-output errors
        float_errors = [e for e in linter.errors if e.rule == "dial-float-output"]
        assert (
            len(float_errors) == 0
        ), f"Unexpected dial-float-output error: {float_errors}"

    def test_dial_integer_min_offset_without_floatoutput_error(
        self, tmp_path: Path
    ) -> None:
        """Dial with integer min offset (min=1) but no floatoutput should ERROR."""
        boxes = [
            {
                "id": "obj-dial",
                "maxclass": "dial",
                "numinlets": 1,
                "numoutlets": 1,
                "outlettype": ["float"],
                "patching_rect": [100.0, 100.0, 40.0, 40.0],
                "size": 19.0,
                "min": 1.0,  # Non-zero min - needs floatoutput even with integer values!
                "mult": 1.0,
                # Missing floatoutput: 1
            },
        ]

        patcher = create_test_patcher(boxes)
        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.validate_file(test_file)

        # Should ERROR: dial needs floatoutput=1 for min offset to work
        float_errors = [e for e in linter.errors if e.rule == "dial-float-output"]
        assert (
            len(float_errors) >= 1
        ), f"Expected dial-float-output error, got: {linter.errors}"
        assert "min=" in float_errors[0].message
        assert "floatoutput" in float_errors[0].message


class TestLintGraph:
    """Tests for LintGraph dataclass that builds shared graph state."""

    def test_build_from_valid_patcher(self, tmp_path: Path) -> None:
        """LintGraph.build() creates graph from patcher JSON."""
        # Create minimal patcher with jit.movie -> jit.gl.pix chain
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
        ]
        lines = [{"source": ["obj-1", 0], "destination": ["obj-2", 0]}]
        patcher = create_test_patcher(boxes, lines)

        test_file = tmp_path / "test.maxhelp"
        test_file.write_text(json.dumps(patcher))

        linter = MaxhelpLinter()
        linter.filepath = test_file
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        assert isinstance(lint_graph.graph, nx.DiGraph)
        assert len(lint_graph.boxes) == 2
        assert "obj-1" in lint_graph.boxes
        assert "obj-2" in lint_graph.boxes

    def test_type_map_populated(self, tmp_path: Path) -> None:
        """Type map has entries for outlets."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Should have type info for outlet 0
        assert ("obj-1", 0) in lint_graph.type_map
        assert lint_graph.type_map[("obj-1", 0)] == "texture"

    def test_type_map_info_outlet(self, tmp_path: Path) -> None:
        """Type map correctly identifies info outlets."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
                "outlettype": ["jit_gl_texture", ""],
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Outlet 1 should be info type
        assert ("obj-1", 1) in lint_graph.type_map
        assert lint_graph.type_map[("obj-1", 1)] == "info"

    def test_empty_patcher_handled(self, tmp_path: Path) -> None:
        """Empty patcher doesn't crash."""
        data: dict[str, Any] = {"patcher": {"boxes": [], "lines": []}}
        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = data
        linter.boxes = {}

        lint_graph = LintGraph.build(data, linter)
        assert len(lint_graph.boxes) == 0
        assert len(lint_graph.type_map) == 0
        assert isinstance(lint_graph.graph, nx.DiGraph)

    def test_graph_has_edges_for_connections(self, tmp_path: Path) -> None:
        """Graph contains edges for patchlines."""
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

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Check that edge exists from outlet to inlet
        assert lint_graph.graph.has_edge(("obj-1", "out", 0), ("obj-2", "in", 0))
        # Check that box-level edge exists
        assert lint_graph.graph.has_edge(("obj-1", "box"), ("obj-2", "box"))

    def test_graph_nodes_have_box_attribute(self, tmp_path: Path) -> None:
        """Graph nodes store box reference in attributes."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie @output_texture 1",
                "numoutlets": 2,
                "numinlets": 1,
            },
        ]
        patcher = create_test_patcher(boxes, [])

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)

        # Box node should have box attribute
        box_node = lint_graph.graph.nodes.get(("obj-1", "box"))
        assert box_node is not None
        assert "box" in box_node
        assert box_node["box"]["id"] == "obj-1"


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


class TestCycleDetection:
    """Tests for cycle detection in LintGraph for GPU feedback loops."""

    def test_no_cycles_in_linear_chain(self, tmp_path: Path) -> None:
        """Linear chain has no cycles."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
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

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) == 0

    def test_detects_direct_cycle(self, tmp_path: Path) -> None:
        """Direct pix->pix feedback detected as cycle."""
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
            {"source": ["obj-2", 0], "destination": ["obj-1", 1]},  # Feedback!
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1

    def test_orphans_detected(self, tmp_path: Path) -> None:
        """Orphaned objects (no connections) detected."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-orphan",
                "maxclass": "newobj",
                "text": "print",
                "numoutlets": 0,
                "numinlets": 1,
            },  # Not connected
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert "obj-orphan" in lint_graph.orphans

    def test_dead_branches_detected(self, tmp_path: Path) -> None:
        """Dead branches (not reaching display) detected."""
        boxes = [
            {
                "id": "obj-main",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-display",
                "maxclass": "jit.pwindow",
                "numoutlets": 1,
                "numinlets": 1,
            },
            {
                "id": "obj-dead1",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-dead2",
                "maxclass": "newobj",
                "text": "print",
                "numoutlets": 0,
                "numinlets": 1,
            },
        ]
        lines = [
            {
                "source": ["obj-main", 0],
                "destination": ["obj-display", 0],
            },  # Main path to display
            {
                "source": ["obj-dead1", 0],
                "destination": ["obj-dead2", 0],
            },  # Dead branch
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        # obj-dead1 and obj-dead2 don't reach display
        assert "obj-dead1" in lint_graph.dead_branches
        assert "obj-dead2" in lint_graph.dead_branches

    def test_self_loop_detected(self, tmp_path: Path) -> None:
        """Self-referential loop detected as cycle."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.feedback",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {
                "source": ["obj-1", 0],
                "destination": ["obj-1", 1],
            },  # Self-loop
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1
        # Self-loop should appear as single-element cycle
        assert any(len(c) == 1 for c in lint_graph.cycles)

    def test_three_node_cycle(self, tmp_path: Path) -> None:
        """Three-node cycle (A->B->C->A) detected."""
        boxes = [
            {
                "id": "obj-a",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.a",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-b",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.b",
                "numoutlets": 2,
                "numinlets": 2,
            },
            {
                "id": "obj-c",
                "maxclass": "newobj",
                "text": "jit.gl.pix @gen sr.c",
                "numoutlets": 2,
                "numinlets": 2,
            },
        ]
        lines = [
            {"source": ["obj-a", 0], "destination": ["obj-b", 0]},
            {"source": ["obj-b", 0], "destination": ["obj-c", 0]},
            {"source": ["obj-c", 0], "destination": ["obj-a", 1]},  # Back to A
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.cycles) >= 1
        # Should have a 3-node cycle
        assert any(len(c) == 3 for c in lint_graph.cycles)

    def test_all_connected_to_display_no_dead_branches(self, tmp_path: Path) -> None:
        """When all boxes reach display, no dead branches."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "newobj",
                "text": "jit.gl.pix",
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

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        assert len(lint_graph.dead_branches) == 0

    def test_jit_window_also_display_sink(self, tmp_path: Path) -> None:
        """jit.window is also recognized as a display sink."""
        boxes = [
            {
                "id": "obj-1",
                "maxclass": "newobj",
                "text": "jit.movie",
                "numoutlets": 2,
                "numinlets": 1,
            },
            {
                "id": "obj-2",
                "maxclass": "jit.window",
                "numoutlets": 1,
                "numinlets": 1,
            },
        ]
        lines = [
            {"source": ["obj-1", 0], "destination": ["obj-2", 0]},
        ]
        patcher = create_test_patcher(boxes, lines)

        linter = MaxhelpLinter()
        linter.filepath = tmp_path / "test.maxhelp"
        linter.data = patcher
        linter.boxes = {str(b["id"]): b for b in boxes}

        lint_graph = LintGraph.build(patcher, linter)
        # All boxes reach the display sink
        assert len(lint_graph.dead_branches) == 0
        assert len(lint_graph.orphans) == 0


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

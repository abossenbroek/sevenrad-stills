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

import pytest
from lint_maxhelp import MaxhelpLinter


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
        valid = linter.validate_file(test_file)

        assert valid
        assert len(linter.errors) == 0
        assert len([w for w in linter.warnings if w.rule == "overlap"]) == 0

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
        valid = linter.validate_file(test_file)

        # Should not report overlap for connected boxes
        assert valid
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
        valid = linter.validate_file(test_file)

        assert valid
        assert len([e for e in linter.errors if e.rule == "overlap"]) == 0

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
        valid = linter.validate_file(test_file)

        assert valid
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
        valid = linter.validate_file(test_file)

        assert valid
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

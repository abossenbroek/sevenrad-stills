# ruff: noqa: S101
"""Tests for MaxhelpLinter overlap detection functionality."""

from __future__ import annotations

import json
from pathlib import Path

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


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
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
        overlap_warnings = [w for w in linter.warnings if w.rule == "overlap-001"]
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
        assert any(e.rule == "overlap-001" for e in linter.errors)
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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
        overlap_warnings = [w for w in linter.warnings if w.rule == "overlap-001"]
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
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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

        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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

        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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

        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
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
        overlap_errors = [e for e in linter.errors if e.rule == "overlap-001"]
        assert len(overlap_errors) == 0

# ruff: noqa: S101
"""Tests for MaxhelpLinter dial validation functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lint_maxhelp import MaxhelpLinter

from tests.conftest import create_test_patcher


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

        # Should have error about missing decimals (dial-003)
        decimal_errors = [e for e in linter.errors if e.rule == "dial-003"]
        assert len(decimal_errors) >= 1
        assert "decimals" in decimal_errors[0].message
        assert ">= 2" in decimal_errors[0].message

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

        # Should NOT have error about decimals (dial-003)
        decimal_errors = [e for e in linter.errors if e.rule == "dial-003"]
        assert len(decimal_errors) == 0

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

        # Should NOT have error about decimals for integer mult (dial-003)
        decimal_errors = [e for e in linter.errors if e.rule == "dial-003"]
        assert len(decimal_errors) == 0

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

        # Should error about insufficient decimals (dial-003)
        decimal_errors = [e for e in linter.errors if e.rule == "dial-003"]
        assert len(decimal_errors) >= 1
        assert ">= 3" in decimal_errors[0].message


class TestDialInitRange:
    """Test dial set value range validation.

    The dial `set` message takes INTERNAL POSITION (0 to size), not output value.
    For a dial with size=50, mult=0.01: position 22 -> output 0.22
    """

    def test_dial_set_value_out_of_position_range_warns(self, tmp_path: Path) -> None:
        """Test that set value outside dial position range triggers warning."""
        # dial with size=50 -> position range 0-50
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
        range_warnings = [w for w in linter.warnings if w.rule == "init-005-range"]
        assert (
            len(range_warnings) >= 1
        ), f"Expected init-005-range warning, got: {linter.warnings}"
        assert "out of position range" in range_warnings[0].message
        assert "100" in range_warnings[0].message

    def test_dial_set_position_in_range_valid(self, tmp_path: Path) -> None:
        """Test that set value within dial position range passes validation."""
        # dial with size=50 -> position range 0-50
        # "set 22" is valid (22 within 0-50) -> output 22*0.01=0.22
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
        range_warnings = [w for w in linter.warnings if w.rule == "init-005-range"]
        assert (
            len(range_warnings) == 0
        ), f"Unexpected init-005-range warning: {range_warnings}"

    def test_dial_mult_1_set_integer_valid(self, tmp_path: Path) -> None:
        """Test that for mult=1.0, set with integer value is valid."""
        # dial with size=98 -> position range 0-98
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
        range_warnings = [w for w in linter.warnings if w.rule == "init-005-range"]
        assert (
            len(range_warnings) == 0
        ), f"Unexpected init-005-range warning: {range_warnings}"


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
        range_errors = [e for e in linter.errors if e.rule == "dial-001"]
        assert len(range_errors) >= 1, f"Expected dial-001 error, got: {linter.errors}"
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

        # Should NOT have dial-001 errors
        range_errors = [e for e in linter.errors if e.rule == "dial-001"]
        assert len(range_errors) == 0, f"Unexpected dial-001 error: {range_errors}"

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
        range_errors = [e for e in linter.errors if e.rule == "dial-001"]
        assert len(range_errors) >= 1, f"Expected dial-001 error, got: {linter.errors}"
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

        # No dial-001 errors (validation skipped)
        range_errors = [e for e in linter.errors if e.rule == "dial-001"]
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
        float_errors = [e for e in linter.errors if e.rule == "dial-002"]
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
        float_errors = [e for e in linter.errors if e.rule == "dial-002"]
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
        float_errors = [e for e in linter.errors if e.rule == "dial-002"]
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
        float_errors = [e for e in linter.errors if e.rule == "dial-002"]
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
        float_errors = [e for e in linter.errors if e.rule == "dial-002"]
        assert (
            len(float_errors) >= 1
        ), f"Expected dial-float-output error, got: {linter.errors}"
        assert "min=" in float_errors[0].message
        assert "floatoutput" in float_errors[0].message

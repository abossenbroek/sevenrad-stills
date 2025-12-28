#!/usr/bin/env python3
# ruff: noqa: S101
"""Test dial range validation functionality."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Add linter to path
sys.path.insert(0, str(Path(__file__).parent))
from lint_maxhelp import MaxhelpLinter


def test_calculate_dial_range() -> None:
    """Test _calculate_dial_range method with various configurations."""
    linter = MaxhelpLinter()

    # Standard 0-1 range: size=100, min=0, mult=0.01
    dial_box = {"size": 100.0, "min": 0.0, "mult": 0.01}
    min_out, max_out = linter._calculate_dial_range(dial_box)
    assert abs(min_out - 0.0) < 0.001, f"Expected min 0.0, got {min_out}"
    assert abs(max_out - 1.0) < 0.001, f"Expected max 1.0, got {max_out}"
    print("PASS: test_calculate_dial_range - standard 0-1")

    # Seed 0-1000 range: size=1000, min=0, mult=1.0
    dial_box = {"size": 1000.0, "min": 0.0, "mult": 1.0}
    min_out, max_out = linter._calculate_dial_range(dial_box)
    assert abs(min_out - 0.0) < 0.001, f"Expected min 0.0, got {min_out}"
    assert abs(max_out - 1000.0) < 0.001, f"Expected max 1000.0, got {max_out}"
    print("PASS: test_calculate_dial_range - seed 0-1000")

    # Scale 0.01-1 range: size=99, min=1, mult=0.01
    dial_box = {"size": 99.0, "min": 1.0, "mult": 0.01}
    min_out, max_out = linter._calculate_dial_range(dial_box)
    assert abs(min_out - 0.01) < 0.001, f"Expected min 0.01, got {min_out}"
    assert abs(max_out - 1.0) < 0.001, f"Expected max 1.0, got {max_out}"
    print("PASS: test_calculate_dial_range - scale 0.01-1")

    # Default values (should use size=100, min=0, mult=1.0)
    dial_box = {}
    min_out, max_out = linter._calculate_dial_range(dial_box)
    assert abs(min_out - 0.0) < 0.001, f"Expected min 0.0, got {min_out}"
    assert abs(max_out - 100.0) < 0.001, f"Expected max 100.0, got {max_out}"
    print("PASS: test_calculate_dial_range - defaults")


def test_parse_genjit_params() -> None:
    """Test _parse_genjit_params with new format."""
    linter = MaxhelpLinter()

    # Test with actual sr.corruption.genjit
    genjit_path = Path("code/sr.corruption.genjit")
    if not genjit_path.exists():
        print("SKIP: test_parse_genjit_params - sr.corruption.genjit not found")
        return

    params = linter._parse_genjit_params(genjit_path)

    # Should have 3 params: mode, intensity, seed
    assert len(params) == 3, f"Expected 3 params, got {len(params)}"

    # Check mode param
    mode_param = next((p for p in params if p["name"] == "mode"), None)
    assert mode_param is not None, "mode param not found"
    assert mode_param["min"] == 0.0, f"Expected mode min 0.0, got {mode_param['min']}"
    assert mode_param["max"] == 2.0, f"Expected mode max 2.0, got {mode_param['max']}"

    # Check intensity param
    intensity_param = next((p for p in params if p["name"] == "intensity"), None)
    assert intensity_param is not None, "intensity param not found"
    assert (
        intensity_param["min"] == 0.0
    ), f"Expected intensity min 0.0, got {intensity_param['min']}"
    assert (
        intensity_param["max"] == 1.0
    ), f"Expected intensity max 1.0, got {intensity_param['max']}"

    # Check seed param
    seed_param = next((p for p in params if p["name"] == "seed"), None)
    assert seed_param is not None, "seed param not found"
    assert seed_param["min"] == 0.0, f"Expected seed min 0.0, got {seed_param['min']}"
    assert (
        seed_param["max"] == 1000.0
    ), f"Expected seed max 1000.0, got {seed_param['max']}"

    print("PASS: test_parse_genjit_params")


def test_real_file_validation() -> None:
    """Test with actual help patchers to ensure no regressions."""
    # Test sr.corruption.maxhelp (should pass)
    corruption_path = Path("help/sr.corruption.maxhelp")
    if corruption_path.exists():
        linter = MaxhelpLinter()
        linter.validate_file(corruption_path)

        # Check no dial-range errors
        dial_range_errors = [e for e in linter.errors if e.rule == "dial-range"]
        assert (
            len(dial_range_errors) == 0
        ), f"sr.corruption should have no dial-range errors, got: {dial_range_errors}"
        print("PASS: test_real_file_validation - sr.corruption.maxhelp")
    else:
        print("SKIP: sr.corruption.maxhelp not found")

    # Test sr.bandswap.maxhelp (should fail - known mismatch)
    bandswap_path = Path("help/sr.bandswap.maxhelp")
    if bandswap_path.exists():
        linter = MaxhelpLinter()
        linter.validate_file(bandswap_path)

        # Should have dial-range errors for perm_r, perm_g, perm_b
        dial_range_errors = [e for e in linter.errors if e.rule == "dial-range"]
        assert (
            len(dial_range_errors) == 3
        ), f"sr.bandswap should have 3 dial-range errors, got: {len(dial_range_errors)}"

        # Check error messages contain expected info
        for error in dial_range_errors:
            assert "[0.00, 3.00]" in error.message, "Expected dial range [0.00, 3.00]"
            assert "[0.0, 2.0]" in error.message, "Expected param bounds [0.0, 2.0]"
        print("PASS: test_real_file_validation - sr.bandswap.maxhelp")
    else:
        print("SKIP: sr.bandswap.maxhelp not found")


def run_all_tests() -> int:
    """Run all test functions."""
    tests: list[Callable[[], None]] = [
        test_calculate_dial_range,
        test_parse_genjit_params,
        test_real_file_validation,
    ]

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

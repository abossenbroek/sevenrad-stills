#!/usr/bin/env python3
"""Test acceptance criteria for builtins.py ticket."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from max_linter.genexpr.builtins import (
    BUILTIN_FUNCTIONS,
    BUILTIN_VARIABLES,
    COMMON_VARIABLES,
)


def test_acceptance_criteria() -> None:
    """Test all acceptance criteria from the ticket."""
    print("Testing acceptance criteria...")

    # Assert 'sample' in BUILTIN_FUNCTIONS
    assert "sample" in BUILTIN_FUNCTIONS, "sample not in BUILTIN_FUNCTIONS"
    print("✓ sample in BUILTIN_FUNCTIONS")

    # Assert 'sin' in BUILTIN_FUNCTIONS
    assert "sin" in BUILTIN_FUNCTIONS, "sin not in BUILTIN_FUNCTIONS"
    print("✓ sin in BUILTIN_FUNCTIONS")

    # Assert len(BUILTIN_FUNCTIONS) >= 80
    count = len(BUILTIN_FUNCTIONS)
    assert count >= 80, f"BUILTIN_FUNCTIONS has {count} entries, expected >= 80"
    print(f"✓ BUILTIN_FUNCTIONS has {count} entries (>= 80)")

    # Assert 'in1' in BUILTIN_VARIABLES
    assert "in1" in BUILTIN_VARIABLES, "in1 not in BUILTIN_VARIABLES"
    print("✓ in1 in BUILTIN_VARIABLES")

    # Assert 'norm' in BUILTIN_VARIABLES
    assert "norm" in BUILTIN_VARIABLES, "norm not in BUILTIN_VARIABLES"
    print("✓ norm in BUILTIN_VARIABLES")

    # Assert 'i' in COMMON_VARIABLES
    assert "i" in COMMON_VARIABLES, "i not in COMMON_VARIABLES"
    print("✓ i in COMMON_VARIABLES")

    print("\n" + "=" * 60)
    print("All acceptance criteria passed!")
    print("=" * 60)

    print("\nStatistics:")
    print(f"  BUILTIN_FUNCTIONS: {len(BUILTIN_FUNCTIONS)} functions")
    print(f"  BUILTIN_VARIABLES: {len(BUILTIN_VARIABLES)} variables")
    print(f"  COMMON_VARIABLES: {len(COMMON_VARIABLES)} identifiers")

    # Show sample function signatures
    print("\nSample function signatures:")
    for fname in ["sample", "sin", "clamp", "mix", "vec"]:
        if fname in BUILTIN_FUNCTIONS:
            min_args, max_args, ret_type = BUILTIN_FUNCTIONS[fname]
            print(f"  {fname}: {min_args}-{max_args} args, returns {ret_type}")

    # Show sample variables
    print("\nSample built-in variables:")
    for vname in ["in1", "norm", "out1", "dim"]:
        if vname in BUILTIN_VARIABLES:
            print(f"  {vname}: {BUILTIN_VARIABLES[vname]}")


if __name__ == "__main__":
    test_acceptance_criteria()

#!/usr/bin/env python3
"""Standalone test script for GenExprParser - verifies acceptance criteria.

This script can be run directly without installing the package.
Usage: python test_parser_standalone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from max_linter.genexpr.parser import GenExprParser, ParseError


def test_basic_parsing() -> None:
    """Test that valid GenExpr code parses successfully."""
    print("Test 1: Basic parsing")
    parser = GenExprParser()
    tree = parser.parse("out1 = in1;")
    assert tree is not None
    print("  ✓ Basic parsing works")


def test_complex_valid_code() -> None:
    """Test parsing of more complex valid GenExpr code."""
    print("\nTest 2: Complex valid code")
    parser = GenExprParser()

    # Test assignment with function call
    tree = parser.parse("out1 = sample(in1, norm);")
    assert tree is not None
    print("  ✓ Function call parsing works")

    # Test multiple statements
    code = """
    color = sample(in1, norm);
    out1 = color * 0.5;
    """
    tree = parser.parse(code)
    assert tree is not None
    print("  ✓ Multiple statements parse correctly")

    # Test if statement
    code = """
    if (norm.x > 0.5) {
        out1 = sample(in1, norm);
    } else {
        out1 = vec4(0.0, 0.0, 0.0, 1.0);
    }
    """
    tree = parser.parse(code)
    assert tree is not None
    print("  ✓ If/else statement parsing works")

    # Test for loop
    code = """
    for (i = 0; i < 10; i++) {
        sum += i;
    }
    """
    tree = parser.parse(code)
    assert tree is not None
    print("  ✓ For loop parsing works")


def test_invalid_syntax() -> None:
    """Test that invalid GenExpr code raises ParseError with proper info."""
    print("\nTest 3: Invalid syntax error handling")
    parser = GenExprParser()

    # Missing expression after =
    try:
        parser.parse("out1 = ;")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1, f"Line number should be >= 1, got {e.line}"
        assert e.column >= 1, f"Column number should be >= 1, got {e.column}"
        print(f"  ✓ Error at line {e.line}, column {e.column}: {e.message[:50]}...")

    # Missing semicolon
    try:
        parser.parse("out1 = in1")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1
        assert e.column >= 1
        print(f"  ✓ Missing semicolon detected at line {e.line}, column {e.column}")

    # Invalid operator
    try:
        parser.parse("out1 @ in1;")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1
        assert e.column >= 1
        print(f"  ✓ Invalid operator detected at line {e.line}, column {e.column}")


def test_parser_caching() -> None:
    """Test that parser instance can be reused."""
    print("\nTest 4: Parser instance reuse")
    parser = GenExprParser()

    # Parse multiple times with same parser
    tree1 = parser.parse("out1 = in1;")
    tree2 = parser.parse("out1 = in2;")
    tree3 = parser.parse("color = sample(in1, norm);")

    assert tree1 is not None
    assert tree2 is not None
    assert tree3 is not None
    print("  ✓ Parser instance can be reused")


def test_parse_error_representation() -> None:
    """Test ParseError string representation."""
    print("\nTest 5: ParseError representation")
    error = ParseError("unexpected token", 5, 10)

    assert error.message == "unexpected token"
    assert error.line == 5
    assert error.column == 10
    assert "Line 5, column 10" in str(error)
    print(f"  ✓ ParseError str: {error}")
    print(f"  ✓ ParseError repr: {repr(error)}")


def main() -> None:
    """Run all acceptance criteria tests."""
    print("=" * 60)
    print("Testing GenExprParser acceptance criteria")
    print("=" * 60)

    try:
        test_basic_parsing()
        test_complex_valid_code()
        test_invalid_syntax()
        test_parser_caching()
        test_parse_error_representation()

        print("\n" + "=" * 60)
        print("All acceptance criteria tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

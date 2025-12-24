"""Test script for GenExprParser - verifies acceptance criteria."""

from __future__ import annotations

from max_linter.genexpr.parser import GenExprParser, ParseError


def test_basic_parsing() -> None:
    """Test that valid GenExpr code parses successfully."""
    parser = GenExprParser()
    tree = parser.parse("out1 = in1;")
    assert tree is not None
    print("✓ Basic parsing works")


def test_complex_valid_code() -> None:
    """Test parsing of more complex valid GenExpr code."""
    parser = GenExprParser()

    # Test assignment with function call
    tree = parser.parse("out1 = sample(in1, norm);")
    assert tree is not None
    print("✓ Function call parsing works")

    # Test multiple statements
    code = """
    color = sample(in1, norm);
    out1 = color * 0.5;
    """
    tree = parser.parse(code)
    assert tree is not None
    print("✓ Multiple statements parse correctly")

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
    print("✓ If/else statement parsing works")

    # Test for loop
    code = """
    for (i = 0; i < 10; i++) {
        sum += i;
    }
    """
    tree = parser.parse(code)
    assert tree is not None
    print("✓ For loop parsing works")


def test_invalid_syntax() -> None:
    """Test that invalid GenExpr code raises ParseError with proper info."""
    parser = GenExprParser()

    # Missing expression after =
    try:
        parser.parse("out1 = ;")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1, f"Line number should be >= 1, got {e.line}"
        assert e.column >= 1, f"Column number should be >= 1, got {e.column}"
        print(f"✓ Error at line {e.line}, column {e.column}: {e.message}")

    # Missing semicolon
    try:
        parser.parse("out1 = in1")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1
        assert e.column >= 1
        print(f"✓ Missing semicolon detected at line {e.line}, column {e.column}")

    # Invalid operator
    try:
        parser.parse("out1 @ in1;")
        raise AssertionError("Should have raised ParseError")
    except ParseError as e:
        assert e.line >= 1
        assert e.column >= 1
        print(f"✓ Invalid operator detected at line {e.line}, column {e.column}")


def test_parser_caching() -> None:
    """Test that parser instance can be reused."""
    parser = GenExprParser()

    # Parse multiple times with same parser
    tree1 = parser.parse("out1 = in1;")
    tree2 = parser.parse("out1 = in2;")
    tree3 = parser.parse("color = sample(in1, norm);")

    assert tree1 is not None
    assert tree2 is not None
    assert tree3 is not None
    print("✓ Parser instance can be reused")


def test_parse_error_representation() -> None:
    """Test ParseError string representation."""
    error = ParseError("unexpected token", 5, 10)

    assert error.message == "unexpected token"
    assert error.line == 5
    assert error.column == 10
    assert "Line 5, column 10" in str(error)
    print(f"✓ ParseError representation: {error}")
    print(f"✓ ParseError repr: {repr(error)}")


def test_parse_with_recovery_success() -> None:
    """Test parse_with_recovery with valid code."""
    parser = GenExprParser()

    # Valid code should return tree and empty error list
    tree, errors = parser.parse_with_recovery("out1 = in1;")
    assert tree is not None
    assert len(errors) == 0
    print("✓ parse_with_recovery returns tree and no errors for valid code")

    # Multiple valid statements
    code = """
    color = sample(in1, norm);
    out1 = color * 0.5;
    """
    tree, errors = parser.parse_with_recovery(code)
    assert tree is not None
    assert len(errors) == 0
    print("✓ parse_with_recovery handles multiple valid statements")


def test_parse_with_recovery_single_error() -> None:
    """Test parse_with_recovery with single syntax error."""
    parser = GenExprParser()

    # Single error should be detected
    tree, errors = parser.parse_with_recovery("out1 = ;")
    assert tree is None
    assert len(errors) >= 1
    assert errors[0].line >= 1
    assert errors[0].column >= 1
    print(f"✓ parse_with_recovery detected single error at line {errors[0].line}")


def test_parse_with_recovery_multiple_errors() -> None:
    """Test parse_with_recovery with multiple syntax errors."""
    parser = GenExprParser()

    # Code with multiple errors on different lines
    code = """out1 = ;
out2 @ in2;
out3 = in3"""

    tree, errors = parser.parse_with_recovery(code)
    assert tree is None
    assert len(errors) >= 2, f"Expected at least 2 errors, got {len(errors)}"

    # Verify errors are sorted by line number
    for i in range(len(errors) - 1):
        assert errors[i].line <= errors[i + 1].line

    print(f"✓ parse_with_recovery detected {len(errors)} errors:")
    for error in errors:
        print(f"  - Line {error.line}, column {error.column}")


def test_parse_with_recovery_mixed_valid_invalid() -> None:
    """Test parse_with_recovery with mix of valid and invalid lines."""
    parser = GenExprParser()

    # Mix of valid and invalid statements
    code = """color = sample(in1, norm);
out1 = ;
result = color * 0.5;
out2 @ in2;"""

    tree, errors = parser.parse_with_recovery(code)
    assert tree is None
    assert len(errors) >= 2

    print(
        f"✓ parse_with_recovery handles mixed valid/invalid code "
        f"({len(errors)} errors)"
    )


def main() -> None:
    """Run all acceptance criteria tests."""
    print("Testing GenExprParser acceptance criteria...\n")

    print("Test 1: Basic parsing")
    test_basic_parsing()
    print()

    print("Test 2: Complex valid code")
    test_complex_valid_code()
    print()

    print("Test 3: Invalid syntax error handling")
    test_invalid_syntax()
    print()

    print("Test 4: Parser instance reuse")
    test_parser_caching()
    print()

    print("Test 5: ParseError representation")
    test_parse_error_representation()
    print()

    print("Test 6: Error recovery - valid code")
    test_parse_with_recovery_success()
    print()

    print("Test 7: Error recovery - single error")
    test_parse_with_recovery_single_error()
    print()

    print("Test 8: Error recovery - multiple errors")
    test_parse_with_recovery_multiple_errors()
    print()

    print("Test 9: Error recovery - mixed valid/invalid")
    test_parse_with_recovery_mixed_valid_invalid()
    print()

    print("=" * 60)
    print("All acceptance criteria tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

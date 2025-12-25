"""Test script for GenExprValidator - verifies acceptance criteria."""

from __future__ import annotations

from max_linter.genexpr import GenExprValidator
from max_linter.results import DiagnosticSeverity


def test_validator_exists() -> None:
    """Test that GenExprValidator class exists and can be instantiated."""
    validator = GenExprValidator()
    assert validator is not None
    assert hasattr(validator, "validate")
    print("✓ GenExprValidator class exists and has validate() method")


def test_validate_returns_diagnostics() -> None:
    """Test that validate() returns a list of Diagnostic objects."""
    validator = GenExprValidator()

    # Test with valid code
    diagnostics = validator.validate("out1 = sample(in1, norm);")
    assert isinstance(diagnostics, list)
    print("✓ validate() returns a list")

    # Test that list items are Diagnostic objects
    invalid_code = "out1 = ;"
    diagnostics = validator.validate(invalid_code)
    assert len(diagnostics) > 0
    assert hasattr(diagnostics[0], "severity")
    assert hasattr(diagnostics[0], "message")
    assert hasattr(diagnostics[0], "range")
    print("✓ validate() returns Diagnostic objects")


def test_valid_code_no_diagnostics() -> None:
    """Test that valid code returns empty diagnostics list."""
    validator = GenExprValidator()

    valid_codes = [
        "out1 = sample(in1, norm);",
        "color = sample(in1, norm); out1 = color;",
        "out1 = vec4(norm.x, norm.y, 0.0, 1.0);",
        """
        color = sample(in1, norm);
        r = color.r * 2.0;
        g = color.g * 0.5;
        b = color.b * 1.5;
        out1 = vec4(r, g, b, color.a);
        """,
    ]

    for code in valid_codes:
        diagnostics = validator.validate(code)
        assert len(diagnostics) == 0, f"Valid code should have no diagnostics: {code}"

    print("✓ Valid code produces no diagnostics")


def test_syntax_error_detection() -> None:
    """Test that syntax errors are detected and converted to Diagnostics."""
    validator = GenExprValidator()

    # Missing expression after =
    diagnostics = validator.validate("out1 = ;")
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == DiagnosticSeverity.ERROR
    assert diagnostics[0].source == "genexpr-parser"
    assert diagnostics[0].code == "syntax-error"
    print(f"✓ Syntax error detected: {diagnostics[0].message}")

    # Missing semicolon
    diagnostics = validator.validate("out1 = in1")
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == DiagnosticSeverity.ERROR
    print("✓ Missing semicolon detected")

    # Invalid operator
    diagnostics = validator.validate("out1 @ in1;")
    assert len(diagnostics) == 1
    assert diagnostics[0].severity == DiagnosticSeverity.ERROR
    print("✓ Invalid operator detected")


def test_semantic_error_detection() -> None:
    """Test that semantic errors are detected."""
    validator = GenExprValidator()

    # Undefined variable warning
    diagnostics = validator.validate("out1 = undefined_var;")
    undefined_warnings = [d for d in diagnostics if "undefined_var" in d.message]
    assert len(undefined_warnings) > 0
    assert undefined_warnings[0].severity == DiagnosticSeverity.WARNING
    assert undefined_warnings[0].source == "genexpr-analyzer"
    print(f"✓ Undefined variable detected: {undefined_warnings[0].message}")

    # Unknown function error
    diagnostics = validator.validate("out1 = unknown_func(1, 2);")
    func_errors = [d for d in diagnostics if "unknown_func" in d.message.lower()]
    assert len(func_errors) > 0
    assert func_errors[0].severity == DiagnosticSeverity.ERROR
    assert func_errors[0].source == "genexpr-analyzer"
    print(f"✓ Unknown function detected: {func_errors[0].message}")

    # Invalid argument count
    diagnostics = validator.validate("out1 = sample(in1);")  # sample requires 2-3 args
    arg_errors = [d for d in diagnostics if "args" in d.message.lower()]
    assert len(arg_errors) > 0
    assert arg_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Invalid argument count detected: {arg_errors[0].message}")

    # Input assignment error
    diagnostics = validator.validate("in1 = vec4(0, 0, 0, 1);")
    input_errors = [d for d in diagnostics if "input" in d.message.lower()]
    assert len(input_errors) > 0
    assert input_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Input assignment error detected: {input_errors[0].message}")

    # Invalid swizzle
    diagnostics = validator.validate("out1 = in1.xyzwrgba;")  # Too many components
    swizzle_errors = [d for d in diagnostics if "swizzle" in d.message.lower()]
    assert len(swizzle_errors) > 0
    assert swizzle_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Invalid swizzle detected: {swizzle_errors[0].message}")


def test_missing_output_warning() -> None:
    """Test that missing out1 assignment generates a warning."""
    validator = GenExprValidator()

    diagnostics = validator.validate("x = 1; y = 2;")
    out1_warnings = [d for d in diagnostics if "out1" in d.message.lower()]
    assert len(out1_warnings) > 0
    assert out1_warnings[0].severity == DiagnosticSeverity.WARNING
    print(f"✓ Missing out1 assignment detected: {out1_warnings[0].message}")

    # With out1 assignment, no warning
    diagnostics = validator.validate("out1 = sample(in1, norm);")
    out1_warnings = [
        d
        for d in diagnostics
        if "out1" in d.message.lower() and "no assignment" in d.message.lower()
    ]
    assert len(out1_warnings) == 0
    print("✓ No warning when out1 is assigned")


def test_validator_caching() -> None:
    """Test that validator instance caches parser for performance."""
    validator = GenExprValidator()

    # Validate multiple times with same validator instance
    diagnostics1 = validator.validate("out1 = in1;")
    diagnostics2 = validator.validate("out1 = in2;")
    diagnostics3 = validator.validate("out1 = sample(in1, norm);")

    assert len(diagnostics1) == 0
    assert len(diagnostics2) == 0
    assert len(diagnostics3) == 0

    # Check that parser instance is cached
    assert hasattr(validator, "_parser")
    assert validator._parser is not None

    print("✓ Validator instance caches parser and can be reused")


def test_diagnostic_format() -> None:
    """Test that diagnostics have proper format."""
    validator = GenExprValidator()

    # Get a diagnostic from syntax error
    diagnostics = validator.validate("out1 = ;")
    assert len(diagnostics) > 0

    diag = diagnostics[0]

    # Check required fields
    assert hasattr(diag, "range")
    assert hasattr(diag, "severity")
    assert hasattr(diag, "message")
    assert hasattr(diag, "source")
    assert hasattr(diag, "code")

    # Check range structure
    assert hasattr(diag.range, "start")
    assert hasattr(diag.range, "end")
    assert hasattr(diag.range.start, "line")
    assert hasattr(diag.range.start, "character")

    # Check that line/column are 0-indexed
    assert diag.range.start.line >= 0
    assert diag.range.start.character >= 0

    print(f"✓ Diagnostic format is correct: {diag}")


def test_complex_shader() -> None:
    """Test validator on a complex realistic shader."""
    validator = GenExprValidator()

    shader_code = """
        // Gaussian blur horizontal pass
        blur_radius = 5.0;

        // Sample multiple pixels for blur
        color = vec4(0.0, 0.0, 0.0, 0.0);
        total_weight = 0.0;

        for (i = -5; i <= 5; i++) {
            offset = vec2(float(i) / dim.x, 0.0);
            weight = exp(-float(i * i) / (2.0 * blur_radius * blur_radius));
            color += sample(in1, norm + offset) * weight;
            total_weight += weight;
        }

        out1 = color / total_weight;
    """

    diagnostics = validator.validate(shader_code)

    # Should have no errors
    errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
    assert len(errors) == 0, f"Complex shader should have no errors, got: {errors}"

    print("✓ Complex realistic shader validates successfully")


def test_multiple_errors() -> None:
    """Test that validator can return multiple diagnostics."""
    validator = GenExprValidator()

    # Code with multiple semantic issues
    code = """
        out1 = undefined_var;
        result = unknown_func(1, 2);
        in1 = vec4(0, 0, 0, 1);
    """

    diagnostics = validator.validate(code)

    # Should have multiple diagnostics
    assert len(diagnostics) >= 2, "Should detect multiple issues"

    # Check for different types of issues
    undefined_errors = [d for d in diagnostics if "undefined_var" in d.message]
    function_errors = [d for d in diagnostics if "unknown_func" in d.message.lower()]
    input_errors = [d for d in diagnostics if "input" in d.message.lower()]

    assert len(undefined_errors) > 0, "Should detect undefined variable"
    assert len(function_errors) > 0, "Should detect unknown function"
    assert len(input_errors) > 0, "Should detect input assignment"

    print(f"✓ Multiple diagnostics returned: {len(diagnostics)} issues found")


def main() -> None:
    """Run all acceptance criteria tests."""
    print("Testing GenExprValidator acceptance criteria...\n")

    print("Test 1: Validator exists and is exported")
    test_validator_exists()
    print()

    print("Test 2: validate() returns list of Diagnostics")
    test_validate_returns_diagnostics()
    print()

    print("Test 3: Valid code produces no diagnostics")
    test_valid_code_no_diagnostics()
    print()

    print("Test 4: Syntax error detection")
    test_syntax_error_detection()
    print()

    print("Test 5: Semantic error detection")
    test_semantic_error_detection()
    print()

    print("Test 6: Missing output warning")
    test_missing_output_warning()
    print()

    print("Test 7: Validator instance caching")
    test_validator_caching()
    print()

    print("Test 8: Diagnostic format")
    test_diagnostic_format()
    print()

    print("Test 9: Complex shader validation")
    test_complex_shader()
    print()

    print("Test 10: Multiple errors handling")
    test_multiple_errors()
    print()

    print("=" * 60)
    print("All acceptance criteria tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Test script for SemanticAnalyzer - verifies acceptance criteria."""

from __future__ import annotations

from max_linter.genexpr.analyzer import SemanticAnalyzer
from max_linter.genexpr.parser import GenExprParser
from max_linter.results import DiagnosticSeverity


def test_undefined_variable_detection() -> None:
    """Test detection of undefined variables."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test undefined variable
    tree = parser.parse("out1 = undefined_var;")
    diagnostics = analyzer.analyze(tree)

    undefined_errors = [d for d in diagnostics if "undefined_var" in d.message]
    assert len(undefined_errors) > 0, "Should detect undefined variable"
    assert undefined_errors[0].severity == DiagnosticSeverity.WARNING
    print(f"✓ Detected undefined variable: {undefined_errors[0].message}")

    # Test that builtins don't trigger warnings
    tree = parser.parse("out1 = sample(in1, norm);")
    diagnostics = analyzer.analyze(tree)
    undefined_errors = [
        d
        for d in diagnostics
        if "undefined" in d.message.lower() or "before assignment" in d.message.lower()
    ]
    assert len(undefined_errors) == 0, "Builtins should not trigger undefined warnings"
    print("✓ Builtins (in1, norm) don't trigger undefined warnings")

    # Test that common variables don't trigger warnings
    tree = parser.parse("out1 = vec4(i, j, x, y);")
    diagnostics = analyzer.analyze(tree)
    undefined_errors = [
        d
        for d in diagnostics
        if "undefined" in d.message.lower() or "before assignment" in d.message.lower()
    ]
    assert len(undefined_errors) == 0, "Common variables should not trigger warnings"
    print("✓ Common variables (i, j, x, y) don't trigger warnings")


def test_function_validation() -> None:
    """Test function call validation."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test unknown function
    tree = parser.parse("out1 = unknown_func(1, 2);")
    diagnostics = analyzer.analyze(tree)
    unknown_errors = [d for d in diagnostics if "unknown_func" in d.message.lower()]
    assert len(unknown_errors) > 0, "Should detect unknown function"
    assert unknown_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Detected unknown function: {unknown_errors[0].message}")

    # Test too few arguments
    tree = parser.parse("out1 = sample(in1);")  # sample requires 2-3 args
    diagnostics = analyzer.analyze(tree)
    arg_errors = [d for d in diagnostics if "argument" in d.message.lower()]
    assert len(arg_errors) > 0, "Should detect insufficient arguments"
    assert arg_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Detected insufficient arguments: {arg_errors[0].message}")

    # Test too many arguments
    tree = parser.parse(
        "out1 = sample(in1, norm, 0, 0, 0);"
    )  # sample accepts max 3 args
    diagnostics = analyzer.analyze(tree)
    arg_errors = [d for d in diagnostics if "argument" in d.message.lower()]
    assert len(arg_errors) > 0, "Should detect excessive arguments"
    assert arg_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Detected excessive arguments: {arg_errors[0].message}")

    # Test valid function call
    tree = parser.parse("out1 = sample(in1, norm);")
    diagnostics = analyzer.analyze(tree)
    arg_errors = [d for d in diagnostics if "argument" in d.message.lower()]
    assert len(arg_errors) == 0, "Valid function call should not produce errors"
    print("✓ Valid function call produces no errors")


def test_swizzle_validation() -> None:
    """Test swizzle validation."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test swizzle with too many components
    tree = parser.parse("out1 = in1.xyzwrgba;")  # 8 components > 4
    diagnostics = analyzer.analyze(tree)
    swizzle_errors = [d for d in diagnostics if "swizzle" in d.message.lower()]
    assert len(swizzle_errors) > 0, "Should detect swizzle with too many components"
    assert swizzle_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Detected invalid swizzle length: {swizzle_errors[0].message}")

    # Test valid swizzles
    valid_swizzles = ["xyz", "rgba", "xy", "r", "xyzw"]
    for swizzle in valid_swizzles:
        tree = parser.parse(f"out1 = in1.{swizzle};")
        diagnostics = analyzer.analyze(tree)
        swizzle_errors = [d for d in diagnostics if "swizzle" in d.message.lower()]
        assert len(swizzle_errors) == 0, f"Valid swizzle '{swizzle}' should not error"
    print(f"✓ Valid swizzles ({', '.join(valid_swizzles)}) produce no errors")


def test_output_requirement() -> None:
    """Test that analyzer warns when out1 is not assigned."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test missing out1 assignment
    tree = parser.parse("x = 1; y = 2;")
    diagnostics = analyzer.analyze(tree)
    out1_warnings = [d for d in diagnostics if "out1" in d.message.lower()]
    assert len(out1_warnings) > 0, "Should warn when out1 is not assigned"
    assert out1_warnings[0].severity == DiagnosticSeverity.WARNING
    print(f"✓ Detected missing out1 assignment: {out1_warnings[0].message}")

    # Test with out1 assignment
    tree = parser.parse("out1 = sample(in1, norm);")
    diagnostics = analyzer.analyze(tree)
    out1_warnings = [
        d
        for d in diagnostics
        if "out1" in d.message.lower() and "no assignment" in d.message.lower()
    ]
    assert len(out1_warnings) == 0, "Should not warn when out1 is assigned"
    print("✓ No warning when out1 is assigned")


def test_input_protection() -> None:
    """Test that analyzer prevents assignment to input variables."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test assignment to in1
    tree = parser.parse("in1 = vec4(0, 0, 0, 1);")
    diagnostics = analyzer.analyze(tree)
    input_errors = [
        d for d in diagnostics if "input" in d.message.lower() and "in1" in d.message
    ]
    assert len(input_errors) > 0, "Should prevent assignment to in1"
    assert input_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Prevented assignment to in1: {input_errors[0].message}")

    # Test assignment to in2
    tree = parser.parse("in2 = vec4(1, 1, 1, 1);")
    diagnostics = analyzer.analyze(tree)
    input_errors = [
        d for d in diagnostics if "input" in d.message.lower() and "in2" in d.message
    ]
    assert len(input_errors) > 0, "Should prevent assignment to in2"
    assert input_errors[0].severity == DiagnosticSeverity.ERROR
    print(f"✓ Prevented assignment to in2: {input_errors[0].message}")

    # Test compound assignment to in1
    tree = parser.parse("in1 += vec4(0.1, 0.1, 0.1, 0);")
    diagnostics = analyzer.analyze(tree)
    input_errors = [
        d for d in diagnostics if "input" in d.message.lower() and "in1" in d.message
    ]
    assert len(input_errors) > 0, "Should prevent compound assignment to in1"
    print(f"✓ Prevented compound assignment to in1: {input_errors[0].message}")


def test_variable_tracking() -> None:
    """Test that analyzer correctly tracks variable definitions."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    # Test variable used after assignment
    tree = parser.parse("""
        color = sample(in1, norm);
        out1 = color * 0.5;
    """)
    diagnostics = analyzer.analyze(tree)
    undefined_errors = [
        d
        for d in diagnostics
        if "color" in d.message and "undefined" in d.message.lower()
    ]
    assert len(undefined_errors) == 0, "Variable defined before use should not error"
    print("✓ Variable defined before use produces no errors")

    # Test variable used in for loop
    tree = parser.parse("""
        for (i = 0; i < 10; i++) {
            sum += i;
        }
        out1 = vec4(sum, sum, sum, 1);
    """)
    diagnostics = analyzer.analyze(tree)
    # 'i' is defined in for loop init, 'sum' is common variable
    undefined_i_errors = [
        d for d in diagnostics if "i" in d.message and "undefined" in d.message.lower()
    ]
    assert (
        len(undefined_i_errors) == 0
    ), "Loop variable and common variable should not trigger errors"
    print("✓ Loop variable tracking works correctly")


def test_complex_shader() -> None:
    """Test analyzer on a realistic shader example."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    shader_code = """
        // Sample the input texture
        color = sample(in1, norm);

        // Apply some processing
        r = color.r * 2.0;
        g = color.g * 0.5;
        b = color.b * 1.5;

        // Clamp values
        r = clamp(r, 0.0, 1.0);
        g = clamp(g, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        // Output
        out1 = vec4(r, g, b, color.a);
    """

    tree = parser.parse(shader_code)
    diagnostics = analyzer.analyze(tree)

    # Should have no errors or warnings (all variables are defined)
    errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
    warnings = [d for d in diagnostics if d.severity == DiagnosticSeverity.WARNING]

    assert len(errors) == 0, f"Complex shader should have no errors, got: {errors}"
    assert (
        len(warnings) == 0
    ), f"Complex shader should have no warnings, got: {warnings}"
    print("✓ Complex realistic shader produces no diagnostics")


def test_diagnostic_format() -> None:
    """Test diagnostic object structure."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    tree = parser.parse("out1 = undefined_var;")
    diagnostics = analyzer.analyze(tree)

    assert len(diagnostics) > 0, "Should produce diagnostics"

    for diagnostic in diagnostics:
        # Check that diagnostic has required fields
        assert hasattr(diagnostic, "range"), "Diagnostic should have range"
        assert hasattr(diagnostic, "severity"), "Diagnostic should have severity"
        assert hasattr(diagnostic, "message"), "Diagnostic should have message"
        assert hasattr(diagnostic, "source"), "Diagnostic should have source"
        assert (
            diagnostic.source == "genexpr-analyzer"
        ), "Source should be genexpr-analyzer"

        # Check range structure
        assert hasattr(diagnostic.range, "start"), "Range should have start"
        assert hasattr(diagnostic.range, "end"), "Range should have end"
        assert hasattr(diagnostic.range.start, "line"), "Position should have line"
        assert hasattr(
            diagnostic.range.start, "character"
        ), "Position should have character"

        print(f"✓ Diagnostic format: {diagnostic}")


def main() -> None:
    """Run all acceptance criteria tests."""
    print("Testing SemanticAnalyzer acceptance criteria...\n")

    print("Test 1: Undefined variable detection")
    test_undefined_variable_detection()
    print()

    print("Test 2: Function call validation")
    test_function_validation()
    print()

    print("Test 3: Swizzle validation")
    test_swizzle_validation()
    print()

    print("Test 4: Output requirement")
    test_output_requirement()
    print()

    print("Test 5: Input protection")
    test_input_protection()
    print()

    print("Test 6: Variable tracking")
    test_variable_tracking()
    print()

    print("Test 7: Complex shader analysis")
    test_complex_shader()
    print()

    print("Test 8: Diagnostic format")
    test_diagnostic_format()
    print()

    print("=" * 60)
    print("All acceptance criteria tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

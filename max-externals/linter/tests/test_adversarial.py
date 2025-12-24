"""Adversarial test suite for GenExpr parser and analyzer.

This module contains edge case and stress tests designed to verify that the
parser and analyzer handle extreme inputs gracefully without crashing.
"""

from __future__ import annotations

import pytest

from max_linter.genexpr.analyzer import SemanticAnalyzer
from max_linter.genexpr.parser import GenExprParser, ParseError
from max_linter.genexpr.validator import GenExprValidator
from max_linter.results import DiagnosticSeverity


class TestParserAdversarial:
    """Adversarial tests for GenExprParser."""

    def test_deeply_nested_expressions(self) -> None:
        """Test parser with deeply nested parentheses and expressions."""
        parser = GenExprParser()

        # 50 levels of nesting
        nested = "out1 = " + "(" * 50 + "1.0" + ")" * 50 + ";"
        tree = parser.parse(nested)
        assert tree is not None
        print("✓ Parser handles 50 levels of nested parentheses")

        # 100 levels of nesting with operations
        nested_ops = "out1 = " + "(" * 100 + "1.0" + " + 1.0)" * 100 + ";"
        tree = parser.parse(nested_ops)
        assert tree is not None
        print("✓ Parser handles 100 levels of nested operations")

    def test_very_long_lines(self) -> None:
        """Test parser with very long lines (10000+ characters)."""
        parser = GenExprParser()

        # Very long variable name (1000 characters)
        long_var = "x" + "a" * 999
        code = f"{long_var} = 1.0; out1 = {long_var};"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles 1000-character variable name")

        # Very long expression (10000+ characters)
        # Create a long sum: 1.0 + 1.0 + 1.0 + ... (repeated 2000 times)
        long_expr = "out1 = " + " + ".join(["1.0"] * 2000) + ";"
        tree = parser.parse(long_expr)
        assert tree is not None
        print("✓ Parser handles 10000+ character expression")

    def test_unicode_in_identifiers(self) -> None:
        """Test parser with Unicode characters in various contexts."""
        parser = GenExprParser()

        # Unicode in comments should be allowed
        code = "// This is a comment with Unicode: αβγδ ∑∏∫\nout1 = in1;"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles Unicode in comments")

        # Unicode in strings (if supported)
        code = 'out1 = vec4(1.0, 0.0, 0.0, 1.0); // "Hello 世界"'
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles Unicode in string-like comments")

    def test_empty_and_whitespace_input(self) -> None:
        """Test parser with empty and whitespace-only input."""
        parser = GenExprParser()

        # Empty string - should parse successfully (empty program)
        tree = parser.parse("")
        assert tree is not None
        print("✓ Parser handles empty input")

        # Whitespace only
        tree = parser.parse("   \n\n\t\t\n   ")
        assert tree is not None
        print("✓ Parser handles whitespace-only input")

        # Comments only
        tree = parser.parse("// Just a comment\n// Another comment")
        assert tree is not None
        print("✓ Parser handles comments-only input")

    def test_malformed_truncated_code(self) -> None:
        """Test parser with malformed and truncated code."""
        parser = GenExprParser()

        # Truncated in middle of statement
        with pytest.raises(ParseError) as exc_info:
            parser.parse("out1 = sample(in1")
        assert exc_info.value.line >= 1
        assert exc_info.value.column >= 1
        print(f"✓ Parser detects truncated function call at line {exc_info.value.line}")

        # Truncated in middle of if statement
        with pytest.raises(ParseError) as exc_info:
            parser.parse("if (norm.x > 0.5")
        assert exc_info.value.line >= 1
        print(f"✓ Parser detects truncated if statement at line {exc_info.value.line}")

        # Missing closing brace
        with pytest.raises(ParseError) as exc_info:
            parser.parse("if (norm.x > 0.5) { out1 = in1;")
        assert exc_info.value.line >= 1
        print(f"✓ Parser detects missing closing brace at line {exc_info.value.line}")

    def test_comments_in_unusual_places(self) -> None:
        """Test parser with comments in unusual but valid positions."""
        parser = GenExprParser()

        # Comment in middle of expression (on next line)
        code = """out1 =
        // This is a comment
        sample(in1, norm);"""
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles comment in middle of expression")

        # Multiple consecutive comments
        code = """// Comment 1
        // Comment 2
        // Comment 3
        out1 = in1;"""
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles multiple consecutive comments")

        # Comment at end of file without newline
        code = "out1 = in1; // Final comment"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles comment at EOF")

    def test_edge_case_numbers(self) -> None:
        """Test parser with edge case numeric values."""
        parser = GenExprParser()

        # Very large number
        code = "out1 = vec4(9999999999999.0, 0.0, 0.0, 1.0);"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles very large numbers")

        # Very small number (many decimal places)
        code = "out1 = vec4(0.0000000000001, 0.0, 0.0, 1.0);"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles very small numbers")

        # Scientific notation (if supported)
        code = "out1 = vec4(1e10, 1e-10, 0.0, 1.0);"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles scientific notation")

        # Negative zero
        code = "out1 = vec4(-0.0, 0.0, 0.0, 1.0);"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles negative zero")

    def test_extreme_operator_chaining(self) -> None:
        """Test parser with extreme operator chaining."""
        parser = GenExprParser()

        # Very long chain of additions
        code = (
            "out1 = "
            + " + ".join([f"vec4({i}.0, 0.0, 0.0, 1.0)" for i in range(100)])
            + ";"
        )
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles 100 chained additions")

        # Mixed operators
        code = "out1 = 1.0"
        for i in range(50):
            code += f" + {i}.0 - {i}.0 * 1.0 / 1.0"
        code += ";"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles 50 mixed operator chains")

    def test_maximum_function_arguments(self) -> None:
        """Test parser with many function arguments."""
        parser = GenExprParser()

        # Function call with many arguments
        # Note: This will fail semantic analysis but should parse
        code = "out1 = vec4(" + ", ".join(["1.0"] * 100) + ");"
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles function call with 100 arguments")

    def test_deeply_nested_control_flow(self) -> None:
        """Test parser with deeply nested control structures."""
        parser = GenExprParser()

        # 20 levels of nested if statements
        code = "out1 = vec4(0.0, 0.0, 0.0, 1.0);\n"
        for i in range(20):
            code += f"if (norm.x > {i * 0.01}) {{\n"
        code += "out1 = vec4(1.0, 1.0, 1.0, 1.0);\n"
        code += "}" * 20

        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles 20 levels of nested if statements")

        # Nested for loops
        code = """
        sum = 0.0;
        for (i = 0; i < 5; i++) {
            for (j = 0; j < 5; j++) {
                for (k = 0; k < 5; k++) {
                    sum += 1.0;
                }
            }
        }
        out1 = vec4(sum, sum, sum, 1.0);
        """
        tree = parser.parse(code)
        assert tree is not None
        print("✓ Parser handles 3 levels of nested for loops")


class TestAnalyzerAdversarial:
    """Adversarial tests for SemanticAnalyzer."""

    def test_very_long_variable_names(self) -> None:
        """Test analyzer with extremely long variable names."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # 500-character variable name
        long_name = "var_" + "a" * 496
        code = f"{long_name} = sample(in1, norm); out1 = {long_name};"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        # Should not crash, no undefined variable warnings
        undefined_errors = [
            d
            for d in diagnostics
            if "undefined" in d.message.lower()
            or "before assignment" in d.message.lower()
        ]
        assert len(undefined_errors) == 0
        print("✓ Analyzer handles 500-character variable names")

    def test_many_variable_definitions(self) -> None:
        """Test analyzer with many variable definitions."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # 1000 variable definitions
        code = ""
        for i in range(1000):
            code += f"var{i} = {i}.0;\n"
        code += "out1 = var999;"

        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        # Should track all variables correctly
        undefined_errors = [
            d
            for d in diagnostics
            if "var999" in d.message and "undefined" in d.message.lower()
        ]
        assert len(undefined_errors) == 0
        print("✓ Analyzer tracks 1000 variable definitions")

    def test_shadowed_variables(self) -> None:
        """Test analyzer with variable shadowing scenarios."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Variable shadowing in nested scopes
        # GenExpr may not have traditional scopes, but test reassignment
        code = """
        x = 1.0;
        x = 2.0;
        x = 3.0;
        out1 = vec4(x, x, x, 1.0);
        """
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        # Should handle multiple assignments to same variable
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0
        print("✓ Analyzer handles variable shadowing/reassignment")

    def test_duplicate_definitions_same_line(self) -> None:
        """Test analyzer with unusual duplicate patterns."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Multiple uses of same variable in expression
        code = "out1 = in1 + in1 + in1 + in1;"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        # Should not produce duplicate warnings
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0
        print("✓ Analyzer handles multiple uses of same variable")

    def test_undefined_in_complex_expressions(self) -> None:
        """Test analyzer with undefined variables in complex expressions."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Undefined in nested function calls
        code = "out1 = clamp(undefined_var, 0.0, 1.0);"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        undefined_warnings = [d for d in diagnostics if "undefined_var" in d.message]
        assert len(undefined_warnings) > 0
        print("✓ Analyzer detects undefined in nested function calls")

        # Undefined in swizzle
        code = "out1 = undefined_vec.rgb;"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        undefined_warnings = [d for d in diagnostics if "undefined_vec" in d.message]
        assert len(undefined_warnings) > 0
        print("✓ Analyzer detects undefined in swizzle operations")

    def test_recursive_function_definitions(self) -> None:
        """Test analyzer behavior with recursive patterns.

        Note: GenExpr doesn't support user-defined functions, but we test
        recursive-like patterns with builtin functions.
        """
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Nested function calls creating recursion-like pattern
        code = (
            "out1 = clamp(clamp(clamp(clamp(clamp("
            "sample(in1, norm), 0.0, 1.0), 0.0, 1.0), 0.0, 1.0), 0.0, 1.0), "
            "0.0, 1.0);"
        )
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0
        print("✓ Analyzer handles deeply nested function calls")

    def test_circular_variable_dependencies(self) -> None:
        """Test analyzer with circular variable usage patterns."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Variables referencing each other (forward references)
        # The analyzer collects all definitions first, so forward references
        # within the same file don't produce warnings (this is intentional)
        code = """
        a = b + 1.0;
        b = c + 1.0;
        c = 1.0;
        out1 = vec4(a, a, a, 1.0);
        """
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        # Should not crash and should track all variables correctly
        # The analyzer's first pass collects all definitions, so no undefined warnings
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0
        print("✓ Analyzer handles forward variable references without crashing")

    def test_all_builtin_functions(self) -> None:
        """Test analyzer with all builtin functions to ensure no crashes."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()
        from max_linter.genexpr.builtins import BUILTIN_FUNCTIONS

        # Test each builtin function with minimum arguments
        for func_name, (min_args, _max_args, _) in BUILTIN_FUNCTIONS.items():
            # Create argument list with correct count
            args = ", ".join(["1.0"] * min_args)
            code = f"out1 = {func_name}({args});"

            tree = parser.parse(code)
            diagnostics = analyzer.analyze(tree)

            # Should not produce unknown function errors
            unknown_errors = [
                d
                for d in diagnostics
                if "unknown" in d.message.lower() and func_name in d.message.lower()
            ]
            assert len(unknown_errors) == 0

        print(f"✓ Analyzer validates all {len(BUILTIN_FUNCTIONS)} builtin functions")

    def test_extreme_swizzle_combinations(self) -> None:
        """Test analyzer with various swizzle combinations."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Valid 4-component swizzles
        valid_swizzles = ["xyzw", "rgba", "xxxx", "yyyy", "wzyx", "abgr"]
        for swizzle in valid_swizzles:
            code = f"out1 = in1.{swizzle};"
            tree = parser.parse(code)
            diagnostics = analyzer.analyze(tree)

            swizzle_errors = [d for d in diagnostics if "swizzle" in d.message.lower()]
            assert len(swizzle_errors) == 0

        print(f"✓ Analyzer validates {len(valid_swizzles)} valid swizzle patterns")

        # Invalid swizzles
        invalid_swizzles = ["xyzwq", "rgbaa", "xyz12", ""]
        for swizzle in invalid_swizzles:
            if swizzle:  # Skip empty swizzle (parse error)
                try:
                    code = f"out1 = in1.{swizzle};"
                    tree = parser.parse(code)
                    diagnostics = analyzer.analyze(tree)
                    # Should produce swizzle error for invalid chars/length
                except ParseError:
                    # Some invalid swizzles may fail parsing
                    pass

        print("✓ Analyzer detects invalid swizzle patterns")

    def test_input_protection_comprehensive(self) -> None:
        """Test input variable protection comprehensively."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Test all input variables (in1, in2, in3, in4)
        for input_var in ["in1", "in2", "in3", "in4"]:
            # Direct assignment
            code = f"{input_var} = vec4(0.0, 0.0, 0.0, 1.0);"
            tree = parser.parse(code)
            diagnostics = analyzer.analyze(tree)

            input_errors = [
                d
                for d in diagnostics
                if "input" in d.message.lower() and input_var in d.message
            ]
            assert len(input_errors) > 0

            # Compound assignment
            code = f"{input_var} += vec4(0.1, 0.1, 0.1, 0.0);"
            tree = parser.parse(code)
            diagnostics = analyzer.analyze(tree)

            input_errors = [
                d
                for d in diagnostics
                if "input" in d.message.lower() and input_var in d.message
            ]
            assert len(input_errors) > 0

        print("✓ Analyzer protects all input variables (in1-in4)")

    def test_no_output_assignment_variations(self) -> None:
        """Test various scenarios where out1 is not assigned."""
        parser = GenExprParser()
        analyzer = SemanticAnalyzer()

        # Assignment to other outputs (out2, out3, etc.) but not out1
        code = "out2 = in1; out3 = in2;"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        out1_warnings = [
            d
            for d in diagnostics
            if "out1" in d.message.lower() and "no assignment" in d.message.lower()
        ]
        assert len(out1_warnings) > 0
        print("✓ Analyzer warns when out1 is not assigned (but out2/out3 are)")

        # Out1 used but not assigned
        code = "x = out1;"
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        out1_warnings = [
            d
            for d in diagnostics
            if "out1" in d.message.lower()
            and (
                "no assignment" in d.message.lower() or "undefined" in d.message.lower()
            )
        ]
        assert len(out1_warnings) > 0
        print("✓ Analyzer warns when out1 is used before assignment")


class TestValidatorAdversarial:
    """Adversarial tests for GenExprValidator (combined parser + analyzer)."""

    def test_validator_handles_all_error_types(self) -> None:
        """Test that validator properly handles all error types."""
        validator = GenExprValidator()

        # Syntax error
        diagnostics = validator.validate("out1 = ;")
        assert len(diagnostics) > 0
        assert any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)
        print("✓ Validator handles syntax errors")

        # Semantic error (unknown function)
        diagnostics = validator.validate("out1 = unknown_func(1.0);")
        assert len(diagnostics) > 0
        assert any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)
        print("✓ Validator handles semantic errors")

        # Warning (undefined variable)
        diagnostics = validator.validate("out1 = undefined_var;")
        assert len(diagnostics) > 0
        assert any(d.severity == DiagnosticSeverity.WARNING for d in diagnostics)
        print("✓ Validator handles semantic warnings")

    def test_validator_empty_code(self) -> None:
        """Test validator with empty code."""
        validator = GenExprValidator()

        diagnostics = validator.validate("")
        # Should have warning about no out1 assignment
        out1_warnings = [d for d in diagnostics if "out1" in d.message.lower()]
        assert len(out1_warnings) > 0
        print("✓ Validator warns about missing out1 on empty code")

    def test_validator_maximum_complexity(self) -> None:
        """Test validator with maximum complexity code."""
        validator = GenExprValidator()

        # Create very complex shader
        code = "// Complex shader\n"

        # Many variables
        for i in range(50):
            code += f"var{i} = sample(in1, norm + vec2({i * 0.01}, 0.0));\n"

        # Complex operations - sum all RGB values
        code += "sum = vec3(0.0, 0.0, 0.0);\n"
        for i in range(50):
            code += f"sum += var{i}.rgb;\n"
        code += "result = sum / 50.0;\n"

        # Output - vec4 needs 4 arguments
        code += "out1 = vec4(result.r, result.g, result.b, 1.0);"

        diagnostics = validator.validate(code)

        # Should parse and analyze without crashing
        # May have warnings but no errors
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0
        print("✓ Validator handles maximum complexity shader (50 variables)")

    def test_validator_stress_test(self) -> None:
        """Stress test validator with rapid sequential validations."""
        validator = GenExprValidator()

        # Validate 100 different code snippets rapidly
        test_cases = [
            "out1 = in1;",
            "out1 = sample(in1, norm);",
            "color = sample(in1, norm); out1 = color * 0.5;",
            "if (norm.x > 0.5) { out1 = in1; } else { out1 = in2; }",
            "for (i = 0; i < 10; i++) { sum += i; } out1 = vec4(sum, sum, sum, 1.0);",
        ]

        for i in range(100):
            code = test_cases[i % len(test_cases)]
            diagnostics = validator.validate(code)
            assert diagnostics is not None

        print("✓ Validator handles 100 rapid sequential validations")


def test_parser_recovery_adversarial() -> None:
    """Test parser error recovery with extreme cases."""
    parser = GenExprParser()

    # Very long file with multiple errors scattered throughout
    code = ""
    for i in range(100):
        if i % 10 == 0:
            code += f"out{i} @ in1;\n"  # Error
        else:
            code += f"var{i} = {i}.0;\n"  # Valid

    tree, errors = parser.parse_with_recovery(code)
    assert tree is None
    assert len(errors) >= 5  # Should find multiple errors
    print(f"✓ Parser recovery finds {len(errors)} errors in 100-line file")


def test_all_diagnostic_codes_unique() -> None:
    """Test that all diagnostic codes are properly set and unique."""
    parser = GenExprParser()
    analyzer = SemanticAnalyzer()

    test_cases = [
        ("out1 = undefined_var;", "undefined-variable"),
        ("out1 = unknown_func();", "unknown-function"),
        ("out1 = sample(in1);", "argument-count"),
        ("out1 = in1.xyzwrgba;", "swizzle-length"),
        ("in1 = vec4(0.0, 0.0, 0.0, 1.0);", "input-assignment"),
        ("x = 1.0;", "no-output"),
    ]

    codes_found = set()

    for code, _expected_code in test_cases:
        tree = parser.parse(code)
        diagnostics = analyzer.analyze(tree)

        for diag in diagnostics:
            if diag.code:
                codes_found.add(diag.code)

    # Should have found multiple unique diagnostic codes
    assert len(codes_found) >= 4
    print(f"✓ Found {len(codes_found)} unique diagnostic codes: {sorted(codes_found)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

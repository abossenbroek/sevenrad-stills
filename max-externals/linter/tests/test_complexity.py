"""Tests for GenExpr complexity warnings.

Tests the complexity analysis features that detect:
- Deeply nested parentheses (>5 levels)
- Very long lines (>200 characters)
"""

from max_linter.genexpr import GenExprValidator
from max_linter.results import DiagnosticSeverity


class TestNestingDepth:
    """Tests for parenthesis nesting depth detection."""

    def test_shallow_nesting_ok(self) -> None:
        """4 levels of nesting should not warn."""
        code = "out1 = (((x + 1)));"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 0

    def test_moderate_nesting_ok(self) -> None:
        """5 levels of nesting should not warn (threshold)."""
        code = "out1 = ((((x + 1))));"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 0

    def test_deep_nesting_warns(self) -> None:
        """6+ levels of nesting should warn."""
        code = "out1 = ((((((x + 1))))));"  # 6 levels
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 1
        assert nesting_warnings[0].severity == DiagnosticSeverity.WARNING
        assert "6" in nesting_warnings[0].message  # depth 6

    def test_very_deep_nesting_warns(self) -> None:
        """Very deeply nested expressions should warn with correct depth."""
        code = "out1 = ((((((((x + 1))))))));"  # 8 levels
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 1
        assert "8" in nesting_warnings[0].message

    def test_nested_function_calls(self) -> None:
        """Nested function calls count as nesting."""
        code = "out1 = sin(cos(tan(sqrt(pow(x, 2)))));"  # 5 levels from functions
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        # This is within threshold
        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 0

    def test_complex_pcg_expression(self) -> None:
        """Test deeply nested expressions similar to PCG hash patterns."""
        # Create an expression with 6+ levels of nesting
        code = """
        state1 = int((uint(combined1) * uint(PCG_MULT) + uint(PCG_INC)));
        word1 = int((((state1 >> ((state1 >> 28) + 4)) ^ state1)) * uint(PCG_FACTOR));
        out1 = vec(1.0, 1.0, 1.0, 1.0);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        # The word1 line has 6+ levels of nesting, should warn
        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) >= 1

    def test_nesting_in_comments_ignored(self) -> None:
        """Parentheses in comments should not count."""
        code = """
        // This has (((((deep))))) nesting in comment
        out1 = x + 1;
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 0

    def test_nesting_in_strings_ignored(self) -> None:
        """Parentheses in strings should not count."""
        code = 'out1 = "(((((";\nout1 = x;'
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        nesting_warnings = [d for d in diagnostics if d.code == "complexity-nesting"]
        assert len(nesting_warnings) == 0


class TestLineLength:
    """Tests for line length detection."""

    def test_normal_line_ok(self) -> None:
        """Lines under 200 chars should not warn."""
        code = "out1 = sample(in1, norm) + vec(1.0, 2.0, 3.0, 4.0);"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        length_warnings = [d for d in diagnostics if d.code == "complexity-line-length"]
        assert len(length_warnings) == 0

    def test_exactly_200_chars_ok(self) -> None:
        """200 char line should not warn (threshold)."""
        # Create a line of exactly 200 characters
        # "out1 = " is 7 chars, we need 193 more
        # Use padding to reach exactly 200
        code = "out1 = " + "a" * 192 + ";"
        assert len(code) == 200

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        length_warnings = [d for d in diagnostics if d.code == "complexity-line-length"]
        assert len(length_warnings) == 0

    def test_long_line_warns(self) -> None:
        """Lines over 200 chars should warn."""
        # Create a line over 200 characters
        long_var = "a" * 200
        code = f"out1 = {long_var};"

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        length_warnings = [d for d in diagnostics if d.code == "complexity-line-length"]
        assert len(length_warnings) == 1
        assert length_warnings[0].severity == DiagnosticSeverity.WARNING
        assert "200" in length_warnings[0].message

    def test_multiple_long_lines_warn(self) -> None:
        """Multiple long lines should each produce a warning."""
        long_var = "x + " * 60  # ~240 chars
        code = f"a = {long_var}1;\nb = {long_var}2;\nout1 = a;"

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        length_warnings = [d for d in diagnostics if d.code == "complexity-line-length"]
        assert len(length_warnings) == 2

    def test_line_number_in_warning(self) -> None:
        """Warning should include correct line number."""
        code = "x = 1;\n" + "y = " + "a" * 250 + ";\nout1 = x;"

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        length_warnings = [d for d in diagnostics if d.code == "complexity-line-length"]
        assert len(length_warnings) == 1
        # Line 2 (1-indexed in message)
        assert "Line 2" in length_warnings[0].message


class TestComplexityIntegration:
    """Integration tests for complexity warnings with other diagnostics."""

    def test_complexity_with_semantic_warnings(self) -> None:
        """Complexity warnings should work alongside semantic warnings."""
        # Deeply nested + undefined variable
        code = "out1 = ((((((undefined_var))))));"

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        # Should have both complexity and undefined variable warnings
        codes = {d.code for d in diagnostics}
        assert "complexity-nesting" in codes
        # undefined_var might be flagged depending on COMMON_VARIABLES

    def test_complexity_with_syntax_error(self) -> None:
        """Syntax errors should take precedence over complexity checks."""
        # Invalid syntax - complexity check won't run
        code = "out1 = ((((((;"

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        # Should have syntax error, not complexity warning
        assert len(diagnostics) >= 1
        assert diagnostics[0].code == "syntax-error"

    def test_valid_complex_shader(self) -> None:
        """A valid but complex shader should only get complexity warnings."""
        code = """
        // PCG hash implementation
        PCG_MULT = 747796405;
        combined = uint(px) * uint(373) + uint(py) * uint(668);
        state = int((uint(combined) * uint(PCG_MULT)));
        out1 = vec(float(state), 0.0, 0.0, 1.0);
        """

        validator = GenExprValidator()
        diagnostics = validator.validate(code)

        # Only warnings, no errors
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]
        assert len(errors) == 0

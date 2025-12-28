"""Tests for overflow detection in GenExpr linter."""

from max_linter.genexpr import GenExprValidator
from max_linter.results import DiagnosticSeverity


class TestOverflowDetection:
    """Test detection of large constants that may cause overflow."""

    def test_large_constant_in_multiplication_warns(self) -> None:
        """Constants >10M in multiplication should warn."""
        code = "out1 = px * 374761393;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) >= 1
        assert overflow[0].severity == DiagnosticSeverity.WARNING

    def test_safe_constant_no_warning(self) -> None:
        """Small constants should not trigger warning."""
        code = "out1 = px * 12.9898;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 0

    def test_division_with_large_constant_no_warning(self) -> None:
        """Division doesn't cause overflow - should NOT warn."""
        code = "out1 = hash / 4294967296.0;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 0

    def test_multiple_large_constants(self) -> None:
        """Multiple dangerous multiplications should each warn."""
        code = "out1 = a * 374761393 + b * 668265263;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) >= 2

    def test_threshold_boundary(self) -> None:
        """Test at exactly the threshold."""
        validator = GenExprValidator()

        # At 10M - no warning
        d1 = validator.validate("out1 = x * 10000000;")
        assert len([d for d in d1 if d.code == "overflow-risk"]) == 0

        # Just over - warning
        d2 = validator.validate("out1 = x * 10000001;")
        assert len([d for d in d2 if d.code == "overflow-risk"]) == 1

    def test_scientific_notation_large_constant(self) -> None:
        """Scientific notation large constants should warn."""
        code = "out1 = x * 1e8;"  # 100 million
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 1

    def test_negative_large_constant(self) -> None:
        """Negative large constants should warn based on absolute value."""
        code = "out1 = x * -50000000;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 1

    def test_modulo_with_large_constant_no_warning(self) -> None:
        """Modulo with large constants should NOT warn."""
        code = "out1 = x % 4294967296;"
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 0

    def test_function_with_large_constant_warns(self) -> None:
        """Large constants in function bodies should warn."""
        code = """
        pcg_rand(px, py, s) {
            combined = px * 374761393 + py * 668265263;
            return combined / 4294967296.0;
        }
        out1 = vec(pcg_rand(1, 1, 0), 0, 0, 1);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        # Should warn for the two multiplications, not the division
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 2

    def test_gpu_hash_no_warning(self) -> None:
        """GPU-safe hash with small constants should not warn."""
        code = """
        gpu_hash(px, py, s) {
            dot_val = px * 12.9898 + py * 78.233 + s * 43.758;
            return fract(sin(dot_val) * 43758.5453);
        }
        out1 = gpu_hash(1, 1, 0);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        overflow = [d for d in diagnostics if d.code == "overflow-risk"]
        assert len(overflow) == 0

"""Tests for uint->int arithmetic detection in GenExpr linter."""

from max_linter.genexpr import GenExprValidator
from max_linter.results import DiagnosticSeverity


class TestUintIntArithmetic:
    """Test detection of uint->int casts that break unsigned arithmetic."""

    def test_int_wrapping_uint_multiplication_warns(self) -> None:
        """int(uint() * uint()) should warn."""
        code = """
        state = int((uint(x) * uint(747796405) + uint(2891336453)));
        out1 = state;
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) >= 1
        assert uint_int[0].severity == DiagnosticSeverity.WARNING

    def test_int_wrapping_uint_shift_warns(self) -> None:
        """int((...) >> (...)) with uint() should warn."""
        code = """
        state = 12345;
        word = int(((state >> ((state >> 28) + 4)) ^ state) * uint(277803737));
        out1 = word;
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) >= 1

    def test_simple_int_cast_no_warning(self) -> None:
        """Simple int() casts without uint() should not warn."""
        code = """
        px = int(norm.x * dim.x);
        py = int(norm.y * dim.y);
        out1 = vec(px, py, 0, 1);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) == 0

    def test_uint_without_int_wrapper_no_warning(self) -> None:
        """uint() without int() wrapper should not warn."""
        code = """
        combined = uint(x) * uint(374761393);
        out1 = combined / 4294967296.0;
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) == 0

    def test_gpu_hash_pattern_no_warning(self) -> None:
        """GPU-safe hash pattern should not warn."""
        code = """
        gpu_hash(px, py, s) {
            dot_val = px * 12.9898 + py * 78.233 + s * 43.758;
            return fract(sin(dot_val) * 43758.5453);
        }
        r = gpu_hash(1, 1, 0);
        out1 = vec(r, r, r, 1);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) == 0

    def test_pcg_inline_pattern_warns(self) -> None:
        """Inline PCG pattern with int(uint()) should warn."""
        code = """
        x = 100;
        y = 200;
        seed = 42;
        combined = uint(x) * uint(374761393) + uint(y) * uint(668265263) + uint(seed);
        state = int((uint(combined) * uint(747796405) + uint(2891336453)));
        word = int(((state >> ((state >> 28) + 4)) ^ state) * uint(277803737));
        hash = (word >> 22) ^ word;
        r = float(uint(hash)) / 4294967296.0;
        out1 = vec(r, r, r, 1);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        # Should warn on state and word lines
        assert len(uint_int) >= 2

    def test_warning_message_mentions_rewrite(self) -> None:
        """Warning message should suggest rewriting."""
        code = """
        state = int((uint(x) * uint(747796405)));
        out1 = state;
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) >= 1
        assert "rewrite" in uint_int[0].message.lower()

    def test_float_uint_to_normalize_no_warning(self) -> None:
        """float(uint(hash)) / MAX pattern is ok - no int() wrapper."""
        code = """
        hash = 12345;
        r = float(uint(hash)) / 4294967296.0;
        out1 = vec(r, r, r, 1);
        """
        validator = GenExprValidator()
        diagnostics = validator.validate(code)
        uint_int = [d for d in diagnostics if d.code == "uint-int-arithmetic"]
        assert len(uint_int) == 0

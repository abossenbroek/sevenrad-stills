"""Tests for GenExpr strict block scoping.

GenExpr uses strict block scoping where:
- First assignment = declaration in current scope
- Variables in inner scopes (if/else/for/while) are NOT visible outside
- Even if assigned in both if AND else branches, variables are NOT visible outside
"""

from max_linter.genexpr import GenExprValidator


class TestBlockScoping:
    """Test strict block scoping rules."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.validator = GenExprValidator()

    def has_undefined_error(
        self, code: str, var_name: str, params: set[str] | None = None
    ) -> bool:
        """Check if code has undefined variable error for given variable."""
        diagnostics = self.validator.validate(code, declared_params=params)
        return any(
            var_name in d.message and d.code == "undefined-variable"
            for d in diagnostics
        )

    def has_no_undefined_errors(
        self, code: str, params: set[str] | None = None
    ) -> bool:
        """Check if code has no undefined variable errors."""
        diagnostics = self.validator.validate(code, declared_params=params)
        return not any(d.code == "undefined-variable" for d in diagnostics)

    # ===== Tests that SHOULD produce undefined-variable errors =====

    def test_if_only_scope(self) -> None:
        """Variable defined only in if-branch is not visible outside."""
        code = """
        if (c > 0) { x = 1; }
        y = x;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "x")

    def test_else_only_scope(self) -> None:
        """Variable defined only in else-branch is not visible outside."""
        code = """
        if (c > 0) { a = 1; } else { x = 2; }
        y = x;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "x")

    def test_both_branches_still_scoped(self) -> None:
        """Variable in BOTH branches is STILL not visible outside (strict)."""
        code = """
        if (c > 0) { x = 1; } else { x = 2; }
        y = x;
        out1 = vec(y, 0, 0, 1);
        """
        # This is the key test - GenExpr has strict scoping
        assert self.has_undefined_error(code, "x")

    def test_for_loop_body_scope(self) -> None:
        """Variable defined inside for loop body is not visible outside."""
        code = """
        for (i = 0; i < 10; i += 1) {
            temp = i * 2;
        }
        y = temp;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "temp")

    def test_while_loop_body_scope(self) -> None:
        """Variable defined inside while loop body is not visible outside."""
        code = """
        i = 0;
        while (i < 10) {
            temp = i * 2;
            i += 1;
        }
        y = temp;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "temp")

    def test_nested_scopes_outer_cannot_see_inner(self) -> None:
        """Outer scope cannot see variables from nested inner scopes."""
        code = """
        if (c > 0) {
            x = 1;
            if (d > 0) {
                y = x;
            }
        }
        z = x;
        out1 = vec(z, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "x")

    def test_function_params_scoped(self) -> None:
        """Function parameters are not visible outside function."""
        code = """
        foo(a) { return a + 1; }
        y = a;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "a")

    def test_use_before_assignment(self) -> None:
        """Variable used before any assignment is undefined."""
        code = """
        y = x;
        x = 1;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_undefined_error(code, "x")

    def test_sr_downscale_bug(self) -> None:
        """Reproduce the sr.downscale bug that should be caught."""
        code = """
        s = clamp(scale, 0.01, 1.0);
        pixel_size = 1.0 / s;

        if (pixelate > 0.5) {
            coord_x = floor(norm.x);
            coord_y = floor(norm.y);
            coord = vec(coord_x, coord_y);
        } else {
            coord = norm * s;
        }

        coord = clamp(coord, vec(0, 0), vec(1, 1));
        out1 = sample(in1, coord);
        """
        params = {"scale", "pixelate", "method"}
        assert self.has_undefined_error(code, "coord", params)

    # ===== Tests that should NOT produce undefined-variable errors =====

    def test_declared_before_if(self) -> None:
        """Variable declared before if is visible after."""
        code = """
        c = 1;
        x = 0;
        if (c > 0) { x = 1; } else { x = 2; }
        y = x;
        out1 = vec(y, 0, 0, 1);
        """
        assert self.has_no_undefined_errors(code)

    def test_for_loop_init_visible(self) -> None:
        """For loop init variable is visible in outer scope."""
        code = """
        for (i = 0; i < 10; i += 1) {
            temp = i * 2;
        }
        y = i;
        out1 = vec(y, 0, 0, 1);
        """
        # Note: i is defined in for_init which is outer scope
        assert self.has_no_undefined_errors(code)

    def test_shader_params_predeclared(self) -> None:
        """Shader parameters are pre-declared and visible."""
        code = """
        y = scale;
        out1 = vec(y, 0, 0, 1);
        """
        params = {"scale"}
        assert self.has_no_undefined_errors(code, params)

    def test_function_params_in_body(self) -> None:
        """Function parameters are visible inside function body."""
        code = """
        foo(a, b) { return a + b; }
        out1 = vec(foo(1, 2), 0, 0, 1);
        """
        assert self.has_no_undefined_errors(code)

    def test_nested_scopes_inner_sees_outer(self) -> None:
        """Inner scopes can see variables from outer scopes."""
        code = """
        c = 1;
        x = 1;
        if (c > 0) { y = x; }
        out1 = vec(x, 0, 0, 1);
        """
        assert self.has_no_undefined_errors(code)

    def test_sr_downscale_fixed(self) -> None:
        """Fixed sr.downscale code should pass."""
        code = """
        s = clamp(scale, 0.01, 1.0);
        pixel_size = 1.0 / s;

        coord = vec(0, 0);

        if (pixelate > 0.5) {
            coord_x = floor(norm.x);
            coord_y = floor(norm.y);
            coord = vec(coord_x, coord_y);
        } else {
            coord = norm * s;
        }

        coord = clamp(coord, vec(0, 0), vec(1, 1));
        out1 = sample(in1, coord);
        """
        params = {"scale", "pixelate", "method"}
        assert self.has_no_undefined_errors(code, params)

    def test_builtin_variables_visible(self) -> None:
        """Builtin variables (norm, dim, cell, in1) are always visible."""
        code = """
        x = norm.x;
        y = dim.y;
        z = cell.x;
        out1 = sample(in1, norm);
        """
        assert self.has_no_undefined_errors(code)

    def test_function_call_to_user_function(self) -> None:
        """Calling user-defined function should work."""
        code = """
        add(a, b) { return a + b; }
        result = add(1, 2);
        out1 = vec(result, 0, 0, 1);
        """
        assert self.has_no_undefined_errors(code)


class TestScopeTracker:
    """Test the ScopeTracker class directly."""

    def test_push_pop_scope(self) -> None:
        """Test scope push/pop."""
        from max_linter.genexpr.analyzer import ScopeTracker

        tracker = ScopeTracker()
        assert tracker.current_depth() == 1  # Global scope

        tracker.push_scope()
        assert tracker.current_depth() == 2

        tracker.push_scope()
        assert tracker.current_depth() == 3

        tracker.pop_scope()
        assert tracker.current_depth() == 2

        tracker.pop_scope()
        assert tracker.current_depth() == 1

        # Can't pop past global scope
        tracker.pop_scope()
        assert tracker.current_depth() == 1

    def test_declare_and_visible(self) -> None:
        """Test variable declaration and visibility."""
        from max_linter.genexpr.analyzer import ScopeTracker

        tracker = ScopeTracker()

        # Initially not visible
        assert not tracker.is_visible("x")

        # After declaration, visible
        tracker.declare("x")
        assert tracker.is_visible("x")

    def test_inner_scope_visibility(self) -> None:
        """Test that inner scopes can see outer variables."""
        from max_linter.genexpr.analyzer import ScopeTracker

        tracker = ScopeTracker()
        tracker.declare("outer")

        tracker.push_scope()
        # Inner scope can see outer variable
        assert tracker.is_visible("outer")

        # Inner declaration
        tracker.declare("inner")
        assert tracker.is_visible("inner")

        tracker.pop_scope()
        # Inner variable no longer visible
        assert not tracker.is_visible("inner")
        # Outer still visible
        assert tracker.is_visible("outer")

    def test_params_and_builtins_preloaded(self) -> None:
        """Test that params and builtins are preloaded."""
        from max_linter.genexpr.analyzer import ScopeTracker

        tracker = ScopeTracker(
            declared_params={"scale", "offset"},
            builtins={"sin", "cos"},
        )

        assert tracker.is_visible("scale")
        assert tracker.is_visible("offset")
        assert tracker.is_visible("sin")
        assert tracker.is_visible("cos")
        assert not tracker.is_visible("undefined")

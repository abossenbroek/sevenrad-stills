"""
Test user-defined functions and return statements in GenExpr grammar.
"""

from __future__ import annotations

import pytest
from lark import Lark
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedToken


def load_parser() -> Lark:
    """Load the GenExpr grammar."""
    with open("grammars/genexpr.lark") as f:
        grammar = f.read()
    return Lark(grammar)


def test_simple_function_no_params() -> None:
    """Test function definition with no parameters."""
    parser = load_parser()
    code = """
    init() {
        x = 0;
        y = 0;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_function_with_single_param() -> None:
    """Test function definition with one parameter."""
    parser = load_parser()
    code = """
    square(x) {
        return x * x;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_function_with_multiple_params() -> None:
    """Test function definition with multiple parameters."""
    parser = load_parser()
    code = """
    add(a, b, c) {
        result = a + b + c;
        return result;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_pcg_random_function() -> None:
    """Test realistic PCG random number generator function."""
    parser = load_parser()
    code = """
    pcg_rand(px, py, s) {
        combined = px * 374761393 + py * 668265263 + s;
        state = combined * 747796405 + 2891336453;
        shift_amt = (state >> 28) + 4;
        word = ((state >> shift_amt) ^ state) * 277803737;
        hash = (word >> 22) ^ word;
        return hash / 4294967296.0;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_return_with_expression() -> None:
    """Test return statement with various expressions."""
    parser = load_parser()
    code = """
    calc() {
        return 1 + 2 * 3;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_return_with_function_call() -> None:
    """Test return statement with function call."""
    parser = load_parser()
    code = """
    wrapper() {
        return abs(-5);
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_return_with_ternary() -> None:
    """Test return statement with ternary operator."""
    parser = load_parser()
    code = """
    max(a, b) {
        return a > b ? a : b;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_multiple_functions() -> None:
    """Test multiple function definitions in same file."""
    parser = load_parser()
    code = """
    square(x) {
        return x * x;
    }

    distance(x1, y1, x2, y2) {
        dx = x2 - x1;
        dy = y2 - y1;
        return square(dx) + square(dy);
    }

    normalize(x, y) {
        d = distance(0, 0, x, y);
        return d > 0 ? 1.0 / d : 0.0;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_function_with_control_flow() -> None:
    """Test function with if statements and loops."""
    parser = load_parser()
    code = """
    clamp(value, min_val, max_val) {
        if (value < min_val) {
            return min_val;
        } else if (value > max_val) {
            return max_val;
        }
        return value;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_function_with_loop() -> None:
    """Test function containing a loop."""
    parser = load_parser()
    code = """
    factorial(n) {
        result = 1;
        for (i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_nested_function_calls() -> None:
    """Test nested function calls in return."""
    parser = load_parser()
    code = """
    process(x) {
        return abs(sin(x * 2.0));
    }
    """
    tree = parser.parse(code)
    assert tree is not None


def test_return_cannot_be_variable_name() -> None:
    """Test that 'return' cannot be used as a variable name."""
    parser = load_parser()
    code = """
    return = 5;
    """
    with pytest.raises((UnexpectedToken, UnexpectedCharacters, LarkError)):
        parser.parse(code)


def test_return_cannot_be_function_name() -> None:
    """Test that 'return' cannot be used as a function name."""
    parser = load_parser()
    code = """
    return() {
        x = 1;
    }
    """
    with pytest.raises((UnexpectedToken, UnexpectedCharacters, LarkError)):
        parser.parse(code)


def test_return_cannot_be_parameter_name() -> None:
    """Test that 'return' cannot be used as a parameter name."""
    parser = load_parser()
    code = """
    foo(return) {
        return return * 2;
    }
    """
    with pytest.raises((UnexpectedToken, UnexpectedCharacters, LarkError)):
        parser.parse(code)


def test_mixed_statements_and_functions() -> None:
    """Test mixing regular statements with functions."""
    parser = load_parser()
    code = """
    helper(x) {
        return x * 2;
    }

    value = helper(5);
    result = value + 10;
    """
    tree = parser.parse(code)
    assert tree is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

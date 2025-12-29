#!/usr/bin/env python3
"""Test edge cases for the GenExpr grammar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark


def test_case(parser: Any, name: str, code: str, *, should_pass: bool = True) -> bool:
    """Test a single code snippet."""
    try:
        parser.parse(code)
        if should_pass:
            print(f"✓ {name}")
            return True
        else:
            print(f"✗ {name} - Expected to fail but passed")
            return False
    except Exception as e:
        if not should_pass:
            print(f"✓ {name} - Failed as expected")
            return True
        else:
            print(f"✗ {name}")
            print(f"  Error: {e}")
            return False


def main() -> int:
    # Load grammar
    grammar_path = Path(__file__).parent / "grammars" / "genexpr.lark"
    parser = Lark.open(str(grammar_path), parser="lalr")

    test_cases = [
        # Basic assignments
        ("simple assignment", "x = 5;", True),
        ("float assignment", "y = 3.14;", True),
        ("negative number", "z = -1.5;", True),
        ("scientific notation", "a = 1e-6;", True),
        # Compound assignments
        ("plus equals", "x += 5;", True),
        ("minus equals", "y -= 2.0;", True),
        ("times equals", "z *= 3;", True),
        ("divide equals", "w /= 4.0;", True),
        # Expressions
        ("arithmetic", "result = a + b * c - d / e;", True),
        ("parentheses", "result = (a + b) * (c - d);", True),
        ("ternary", "result = (x > 0) ? a : b;", True),
        ("nested ternary", "result = (x > 0) ? (y > 0 ? a : b) : c;", True),
        ("logical and", "result = (a > 0) && (b < 10);", True),
        ("logical or", "result = (a == 0) || (b != 0);", True),
        ("bitwise ops", "result = (a & b) | (c ^ d);", True),
        ("shift ops", "result = (a << 2) >> 1;", True),
        # Function calls
        ("simple call", "result = func();", True),
        ("call with args", "result = func(a, b, c);", True),
        ("nested calls", "result = func1(func2(x), y);", True),
        ("sample call", "color = sample(in1, norm);", True),
        # Member access (swizzle)
        ("single component", "r = color.r;", True),
        ("rgb swizzle", "rgb = color.rgb;", True),
        ("xyzw swizzle", "pos = vec.xyzw;", True),
        ("mixed swizzle", "val = vec.xy;", True),
        # Array indexing
        ("array index", "val = arr[0];", True),
        ("expr index", "val = arr[i + 1];", True),
        # If statements
        ("simple if", "if (x > 0) { y = 1; }", True),
        ("if else", "if (x > 0) { y = 1; } else { y = 0; }", True),
        (
            "if else if",
            "if (x > 0) { y = 1; } else if (x < 0) { y = -1; } else { y = 0; }",
            True,
        ),
        # Loops
        ("for loop", "for (i = 0; i < 10; i += 1) { sum += i; }", True),
        (
            "nested for",
            "for (i = 0; i < 10; i += 1) { for (j = 0; j < 10; j += 1) { sum += 1; } }",
            True,
        ),
        ("while loop", "while (x < 100) { x *= 2; }", True),
        ("break", "for (i = 0; i < 10; i += 1) { break; }", True),
        ("continue", "for (i = 0; i < 10; i += 1) { continue; }", True),
        # Comments
        ("single line comment", "// This is a comment\nx = 5;", True),
        ("multi-line comment", "/* This is a\nmulti-line comment */\nx = 5;", True),
        (
            "comment between else",
            "if (x) { y = 1; }\n// comment\nelse { y = 0; }",
            True,
        ),
        # Edge cases
        ("multiple statements", "x = 1; y = 2; z = 3;", True),
        ("empty block", "if (x) { }", True),
        ("nested blocks", "if (x) { if (y) { z = 1; } }", True),
        # Vector construction
        ("vec2", "v = vec(x, y);", True),
        ("vec3", "v = vec(x, y, z);", True),
        ("vec4", "v = vec(r, g, b, a);", True),
        # Complex expression
        ("complex", "out1 = vec(r_out, g_out, b_out, in1.a);", True),
        # Invalid syntax (should fail)
        ("missing semicolon", "x = 5", False),
        ("invalid keyword", "foo = 5;", True),  # 'foo' is not a keyword, should pass
    ]

    print(f"\nTesting {len(test_cases)} edge cases...\n")

    passed = sum(
        test_case(parser, name, code, should_pass=should_pass)
        for name, code, should_pass in test_cases
    )
    failed = len(test_cases) - passed

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

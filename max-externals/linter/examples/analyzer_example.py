"""Example usage of the GenExpr semantic analyzer.

This script demonstrates how to use the GenExprParser and SemanticAnalyzer
to validate GenExpr shader code.
"""

from max_linter.genexpr import GenExprParser, ParseError, SemanticAnalyzer


def analyze_shader(code: str, description: str) -> None:
    """Analyze a GenExpr shader and display diagnostics.

    Args:
        code: GenExpr shader source code
        description: Description of what this example demonstrates
    """
    print(f"\n{'=' * 70}")
    print(f"Example: {description}")
    print(f"{'=' * 70}")
    print("\nCode:")
    print(code)
    print("\nAnalysis:")

    try:
        # Parse the code
        parser = GenExprParser()
        tree = parser.parse(code)
        print("✓ Syntax is valid")

        # Analyze semantics
        analyzer = SemanticAnalyzer()
        diagnostics = analyzer.analyze(tree)

        if not diagnostics:
            print("✓ No semantic issues found")
        else:
            print(f"Found {len(diagnostics)} diagnostic(s):\n")
            for diagnostic in diagnostics:
                severity = diagnostic.severity.name
                line = diagnostic.range.start.line + 1
                char = diagnostic.range.start.character + 1
                pos = f"{line}:{char}"
                code_str = f"[{diagnostic.code}]" if diagnostic.code else ""
                print(f"  {severity} at {pos} {code_str}")
                print(f"  → {diagnostic.message}\n")

    except ParseError as e:
        print(f"✗ Syntax error at line {e.line}, column {e.column}:")
        print(f"  → {e.message}")


def main() -> None:
    """Run example analyses."""
    print("GenExpr Semantic Analyzer - Examples")

    # Example 1: Valid shader
    analyze_shader(
        """
        // Basic texture sampling
        color = sample(in1, norm);
        out1 = color;
        """,
        "Valid shader with texture sampling",
    )

    # Example 2: Undefined variable
    analyze_shader(
        """
        // Using undefined variable
        out1 = undefined_variable;
        """,
        "Undefined variable usage",
    )

    # Example 3: Unknown function
    analyze_shader(
        """
        // Calling unknown function
        out1 = unknown_function(in1, norm);
        """,
        "Unknown function call",
    )

    # Example 4: Wrong argument count
    analyze_shader(
        """
        // sample() requires 2-3 arguments
        out1 = sample(in1);
        """,
        "Incorrect argument count",
    )

    # Example 5: Invalid swizzle
    analyze_shader(
        """
        // Swizzle too long (max 4 components)
        out1 = in1.xyzwrgba;
        """,
        "Invalid swizzle (too many components)",
    )

    # Example 6: Missing output
    analyze_shader(
        """
        // No assignment to out1
        color = sample(in1, norm);
        x = color.r;
        """,
        "Missing output assignment",
    )

    # Example 7: Input assignment protection
    analyze_shader(
        """
        // Cannot assign to input textures
        in1 = vec4(0, 0, 0, 1);
        out1 = in1;
        """,
        "Illegal assignment to input variable",
    )

    # Example 8: Complex valid shader
    analyze_shader(
        """
        // Complex shader with multiple effects
        color = sample(in1, norm);

        // Extract channels
        r = color.r;
        g = color.g;
        b = color.b;

        // Apply processing
        r = clamp(r * 2.0, 0.0, 1.0);
        g = pow(g, 0.5);
        b = smoothstep(0.0, 1.0, b);

        // Recombine
        out1 = vec4(r, g, b, color.a);
        """,
        "Complex shader with channel processing",
    )

    # Example 9: Using common variables
    analyze_shader(
        """
        // Loop with common variable names
        sum = vec4(0, 0, 0, 0);
        for (i = 0; i < 10; i++) {
            offset = vec2(i, i) / dim;
            sum += sample(in1, norm + offset);
        }
        out1 = sum / 10.0;
        """,
        "Loop using common variables (i, sum, offset)",
    )

    print(f"\n{'=' * 70}")
    print("Examples complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

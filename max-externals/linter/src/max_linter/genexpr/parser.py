"""GenExpr parser wrapper using Lark.

This module provides a parser for GenExpr shader language code using the Lark
parsing library. The parser validates syntax and produces an abstract syntax
tree (AST) for further analysis.

Example:
    >>> from max_linter.genexpr.parser import GenExprParser, ParseError
    >>> parser = GenExprParser()
    >>> tree = parser.parse("out1 = in1;")
    >>> print(tree.pretty())

    >>> try:
    ...     parser.parse("out1 = ;")  # Invalid syntax
    ... except ParseError as e:
    ...     print(f"Error at line {e.line}, column {e.column}: {e.message}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lark import Lark, LarkError, UnexpectedInput

if TYPE_CHECKING:
    from lark import Tree


class ParseError(Exception):
    """Exception raised when GenExpr code fails to parse.

    Attributes:
        message: Human-readable error message describing the syntax error
        line: Line number where the error occurred (1-indexed)
        column: Column number where the error occurred (1-indexed)
    """

    def __init__(self, message: str, line: int, column: int) -> None:
        """Initialize the ParseError.

        Args:
            message: Description of the parsing error
            line: Line number where the error occurred (1-indexed)
            column: Column number where the error occurred (1-indexed)
        """
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, column {column}: {message}")

    def __repr__(self) -> str:
        """Return a detailed representation of the error."""
        return (
            f"ParseError(message={self.message!r}, "
            f"line={self.line}, column={self.column})"
        )


class GenExprParser:
    """Parser for GenExpr shader language.

    This class wraps the Lark parser and provides a clean interface for parsing
    GenExpr code. The parser is initialized once and cached for performance, as
    creating a Lark parser from a grammar file is expensive.

    Attributes:
        _parser: Cached Lark parser instance
    """

    def __init__(self) -> None:
        """Initialize the GenExpr parser.

        Loads the grammar file and creates a Lark parser in LALR mode for
        optimal parsing speed. The parser instance is cached for reuse.
        """
        # Load grammar from package resources
        # The grammar file is at max-externals/linter/grammars/genexpr.lark
        # We use Path to locate it relative to the package root
        from pathlib import Path

        # Find the grammar file: go up from src/max_linter to linter root
        package_dir = Path(__file__).parent.parent.parent.parent
        grammar_path = package_dir / "grammars" / "genexpr.lark"

        if not grammar_path.exists():
            raise FileNotFoundError(f"GenExpr grammar file not found at {grammar_path}")

        grammar_text = grammar_path.read_text(encoding="utf-8")

        # Create LALR parser for speed (faster than Earley, works for our grammar)
        self._parser: Lark = Lark(
            grammar_text,
            parser="lalr",
            start="start",
            propagate_positions=True,  # Track line/column info
            maybe_placeholders=False,  # Don't allow ambiguous parses
        )

    def parse(self, code: str) -> Tree:
        """Parse GenExpr code and return an abstract syntax tree.

        Args:
            code: GenExpr shader source code to parse

        Returns:
            Lark Tree representing the parsed code structure

        Raises:
            ParseError: If the code contains syntax errors

        Example:
            >>> parser = GenExprParser()
            >>> tree = parser.parse("out1 = sample(in1, norm);")
            >>> print(tree.data)  # 'start'
        """
        try:
            return self._parser.parse(code)
        except UnexpectedInput as e:
            # Lark provides detailed error information with line/column
            raise ParseError(
                message=str(e.get_context(code)),
                line=e.line,
                column=e.column,
            ) from e
        except LarkError as e:
            # Generic Lark error - provide best-effort location info
            raise ParseError(
                message=str(e),
                line=1,
                column=1,
            ) from e

    def parse_with_recovery(self, code: str) -> tuple[Tree | None, list[ParseError]]:
        """Parse GenExpr code with error recovery to collect multiple errors.

        This method attempts to continue parsing after encountering syntax errors,
        collecting all errors found in the code. It's useful for IDE-like tools that
        want to show all syntax errors at once rather than stopping at the first one.

        Args:
            code: GenExpr shader source code to parse

        Returns:
            A tuple containing:
                - Tree | None: The parsed AST if successful, None if parsing failed
                - list[ParseError]: List of all syntax errors found (empty if no errors)

        Example:
            >>> parser = GenExprParser()
            >>> tree, errors = parser.parse_with_recovery("out1 = in1;")
            >>> assert tree is not None and len(errors) == 0
            >>>
            >>> tree, errors = parser.parse_with_recovery("out1 = ;\\nout2 @ in2;")
            >>> assert tree is None and len(errors) == 2
        """
        errors: list[ParseError] = []

        # First, try to parse normally
        try:
            tree = self._parser.parse(code)
            return tree, errors  # Success - no errors
        except UnexpectedInput as e:
            # Collect the first error
            errors.append(
                ParseError(
                    message=str(e.get_context(code)),
                    line=e.line,
                    column=e.column,
                )
            )
        except LarkError as e:
            # Generic Lark error
            errors.append(
                ParseError(
                    message=str(e),
                    line=1,
                    column=1,
                )
            )
            return None, errors

        # Now attempt error recovery by parsing line-by-line
        # This allows us to find multiple syntax errors
        lines = code.split("\n")
        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines and lines with only whitespace
            if not line.strip():
                continue

            # Skip lines we already know have errors from the first parse attempt
            if any(error.line == line_num for error in errors):
                continue

            # Try to parse this line as a complete statement
            try:
                # Add semicolon if missing for single-line testing
                test_line = line.strip()
                if test_line and not test_line.endswith((";", "{", "}")):
                    test_line += ";"
                self._parser.parse(test_line)
            except UnexpectedInput as e:
                # Found another error - add it with correct line number
                # Only add if we don't already have an error for this line
                if not any(error.line == line_num for error in errors):
                    errors.append(
                        ParseError(
                            message=str(e.get_context(test_line)),
                            line=line_num,
                            column=e.column,
                        )
                    )
            except LarkError:
                # Generic error - skip to avoid false positives
                pass

        # Sort errors by line number for consistency
        errors.sort(key=lambda e: (e.line, e.column))

        return None, errors

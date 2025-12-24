"""GenExpr validator combining parser and semantic analyzer.

This module provides a unified entry point for validating GenExpr shader code.
The GenExprValidator class combines syntax parsing and semantic analysis into
a single validate() method that returns all diagnostics.

Example:
    >>> from max_linter.genexpr import GenExprValidator
    >>> validator = GenExprValidator()
    >>> diagnostics = validator.validate("out1 = sample(in1, norm);")
    >>> if diagnostics:
    ...     for diag in diagnostics:
    ...         print(diag)
"""

from __future__ import annotations

from max_linter.genexpr.analyzer import SemanticAnalyzer
from max_linter.genexpr.parser import GenExprParser, ParseError
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range


class GenExprValidator:
    """Unified validator for GenExpr shader code.

    This class provides a single entry point for validating GenExpr code,
    combining both syntax parsing and semantic analysis. It handles parse
    errors and converts them to the Diagnostic format, then performs semantic
    analysis on successfully parsed code.

    The parser instance is cached for performance, as creating a Lark parser
    from a grammar file is expensive.

    Attributes:
        _parser: Cached GenExprParser instance for syntax validation.

    Example:
        >>> validator = GenExprValidator()
        >>> # Valid code
        >>> diagnostics = validator.validate("out1 = sample(in1, norm);")
        >>> len(diagnostics)
        0
        >>> # Invalid code
        >>> diagnostics = validator.validate("out1 = ;")
        >>> len(diagnostics) > 0
        True
    """

    def __init__(self) -> None:
        """Initialize the validator with cached parser instance."""
        self._parser = GenExprParser()

    def validate(self, code: str) -> list[Diagnostic]:
        """Validate GenExpr code and return all diagnostics.

        This method performs both syntax parsing and semantic analysis,
        returning a unified list of diagnostics. If parsing fails, a single
        parse error diagnostic is returned. If parsing succeeds, semantic
        analysis is performed and may return multiple diagnostics.

        Args:
            code: GenExpr shader source code to validate.

        Returns:
            List of Diagnostic objects representing errors and warnings.
            An empty list indicates the code is valid.

        Example:
            >>> validator = GenExprValidator()
            >>> # Syntax error
            >>> diagnostics = validator.validate("out1 = ;")
            >>> diagnostics[0].severity == DiagnosticSeverity.ERROR
            True
            >>> # Semantic warning
            >>> diagnostics = validator.validate("x = undefined_var;")
            >>> any(d.severity == DiagnosticSeverity.WARNING for d in diagnostics)
            True
            >>> # Valid code
            >>> diagnostics = validator.validate("out1 = sample(in1, norm);")
            >>> len(diagnostics)
            0
        """
        # Step 1: Parse the code
        try:
            tree = self._parser.parse(code)
        except ParseError as e:
            # Convert ParseError to Diagnostic format
            return [self._parse_error_to_diagnostic(e)]

        # Step 2: Perform semantic analysis
        analyzer = SemanticAnalyzer()
        diagnostics = analyzer.analyze(tree)

        return diagnostics

    def _parse_error_to_diagnostic(self, error: ParseError) -> Diagnostic:
        """Convert a ParseError to a Diagnostic object.

        Args:
            error: ParseError from the parser.

        Returns:
            Diagnostic object with ERROR severity and proper location info.
        """
        # ParseError uses 1-indexed line/column, Diagnostic uses 0-indexed
        line = error.line - 1
        column = error.column - 1

        return Diagnostic(
            range=Range(
                start=Position(line, column),
                end=Position(line, column + 1),
            ),
            severity=DiagnosticSeverity.ERROR,
            message=error.message,
            source="genexpr-parser",
            code="syntax-error",
        )

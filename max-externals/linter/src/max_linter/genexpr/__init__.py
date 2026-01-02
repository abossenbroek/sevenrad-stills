"""GenExpr shader language support.

This package provides parsing, validation, and type checking for GenExpr shaders.
GenExpr is Max/MSP's shader language based on C syntax with built-in functions
for texture sampling, vector operations, and mathematical computations.
"""

from __future__ import annotations

from max_linter.genexpr.analyzer import SemanticAnalyzer
from max_linter.genexpr.builtins import (
    ALL_BUILTINS,
    BUILTIN_FUNCTIONS,
    BUILTIN_VARIABLES,
    COMMON_VARIABLES,
)
from max_linter.genexpr.parser import GenExprParser, ParseError
from max_linter.genexpr.validator import GenExprValidator

__all__ = [
    "BUILTIN_FUNCTIONS",
    "BUILTIN_VARIABLES",
    "COMMON_VARIABLES",
    "ALL_BUILTINS",
    "GenExprParser",
    "ParseError",
    "SemanticAnalyzer",
    "GenExprValidator",
]

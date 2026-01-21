"""
Parameter expression validation for TouchDesigner .parm files.

Validates Python expressions in parameter fields (mode 49/17) using
ast.parse(mode='eval').
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from td_linter.embedded.constants import EXPRESSION_GLOBALS
from td_linter.rules.base import Violation

if TYPE_CHECKING:
    from td_linter.parsers.parm_parser import ParsedParmFile


# Expression modes in .parm files
EXPRESSION_MODE = 49  # Pure expression mode
STRING_EXPRESSION_MODE = 17  # String with expression


@dataclass
class Expression:
    """A parameter expression extracted from .parm file."""

    param_name: str
    expression: str
    mode: int
    line: Optional[int]
    source_file: Optional[Path]


# Python built-in names available in expressions
PYTHON_BUILTINS_FOR_EXPRESSIONS: set[str] = {
    # Type conversions
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "tuple",
    "set",
    # Math
    "abs",
    "min",
    "max",
    "pow",
    "round",
    "sum",
    # Other common
    "len",
    "range",
    "True",
    "False",
    "None",
}


class ExpressionValidator:
    """
    Validate parameter expressions from .parm files.

    Uses ast.parse(mode='eval') since expressions are single values,
    not full Python statements.
    """

    def extract_expressions(
        self,
        parm_file: "ParsedParmFile",
    ) -> Iterator[Expression]:
        """
        Extract expressions from parsed .parm file.

        Args:
            parm_file: Parsed .parm file with parameters.

        Yields:
            Expression objects for each expression-mode parameter.

        """
        for i, param in enumerate(parm_file.parameters):
            if param.mode in (EXPRESSION_MODE, STRING_EXPRESSION_MODE):
                expr_text = param.expression
                if expr_text:
                    yield Expression(
                        param_name=param.name,
                        expression=expr_text,
                        mode=param.mode,
                        line=i + 1,  # Approximate line number
                        source_file=parm_file.source,
                    )

    def validate_expression(
        self,
        expr: Expression,
        extra_globals: Optional[set[str]] = None,
    ) -> Iterator[Violation]:
        """
        Validate a single expression and yield violations.

        Args:
            expr: The expression to validate.
            extra_globals: Additional names to consider as defined.

        Yields:
            Violation objects for errors found.

        """
        file_path_str = str(expr.source_file) if expr.source_file else "<unknown>"

        # Try to parse as expression
        try:
            tree = ast.parse(expr.expression, mode="eval")
        except SyntaxError as e:
            yield Violation(
                rule="expression-syntax-error",
                message=f"Expression syntax error in '{expr.param_name}': {e.msg}",
                path=file_path_str,
                severity="error",
                source_file=expr.source_file,
                line=expr.line,
                context={"param": expr.param_name, "expression": expr.expression},
            )
            return

        # Check for undefined names
        all_globals = EXPRESSION_GLOBALS | PYTHON_BUILTINS_FOR_EXPRESSIONS
        if extra_globals:
            all_globals = all_globals | extra_globals

        undefined = self._find_undefined_in_expression(tree, all_globals)
        for name, _ in undefined:
            yield Violation(
                rule="expression-undefined-name",
                message=f"Undefined name in expression '{expr.param_name}': '{name}'",
                path=file_path_str,
                severity="warning",
                source_file=expr.source_file,
                line=expr.line,
                context={"param": expr.param_name, "name": name},
            )

    def validate_parm_file(
        self,
        parm_file: "ParsedParmFile",
        extra_globals: Optional[set[str]] = None,
    ) -> Iterator[Violation]:
        """
        Validate all expressions in a .parm file.

        Args:
            parm_file: Parsed .parm file with parameters.
            extra_globals: Additional names to consider as defined.

        Yields:
            Violation objects for errors found.

        """
        for expr in self.extract_expressions(parm_file):
            yield from self.validate_expression(expr, extra_globals)

    def _find_undefined_in_expression(
        self,
        tree: ast.Expression,
        known_globals: set[str],
    ) -> list[tuple[str, int]]:
        """
        Find undefined names in expression AST.

        Args:
            tree: Parsed expression AST.
            known_globals: Set of names to consider as defined.

        Returns:
            List of (name, lineno) tuples for undefined names.

        """
        undefined: list[tuple[str, int]] = []

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in known_globals
            ):
                undefined.append((node.id, node.lineno))

        return undefined

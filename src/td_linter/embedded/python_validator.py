"""
Python AST validation for TouchDesigner .text files.

Validates Python scripts using ast.parse() with a whitelist of
TouchDesigner builtins to avoid false positives.
"""

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from td_linter.embedded.constants import EXECUTE_CALLBACKS, TD_BUILTINS
from td_linter.rules.base import Violation


@dataclass
class UndefinedName:
    """An undefined name reference found in the AST."""

    name: str
    line: int
    column: int


@dataclass
class PythonValidationResult:
    """Result of Python validation."""

    syntax_errors: list[Violation] = field(default_factory=list)
    undefined_names: list[UndefinedName] = field(default_factory=list)
    defined_functions: set[str] = field(default_factory=set)
    missing_callbacks: list[str] = field(default_factory=list)


# Python built-in names (from builtins module)
PYTHON_BUILTINS: set[str] = set(dir(builtins))


class PythonValidator:
    """
    Validate Python scripts using AST analysis.

    Catches syntax errors and optionally detects undefined names
    while avoiding false positives for TouchDesigner builtins.
    """

    def validate(
        self,
        content: str,
        source_file: Optional[Path] = None,
        check_undefined: bool = True,
        check_callbacks: bool = False,
        extra_globals: Optional[set[str]] = None,
    ) -> Iterator[Violation]:
        """
        Validate Python content and yield violations.

        Args:
            content: Python script content (without .text header).
            source_file: Optional source file path for error reporting.
            check_undefined: Whether to check for undefined names.
            check_callbacks: Whether to check for missing callbacks.
            extra_globals: Additional names to consider as defined.

        Yields:
            Violation objects for errors found.

        """
        file_path_str = str(source_file) if source_file else "<unknown>"

        # Try to parse the content
        try:
            tree = ast.parse(content, filename=file_path_str)
        except SyntaxError as e:
            yield Violation(
                rule="python-syntax-error",
                message=f"Syntax error: {e.msg}",
                path=file_path_str,
                severity="error",
                source_file=source_file,
                line=e.lineno,
                context={"offset": e.offset},
            )
            return

        # Check for undefined names
        if check_undefined:
            all_globals = TD_BUILTINS | PYTHON_BUILTINS
            if extra_globals:
                all_globals = all_globals | extra_globals

            undefined = self.find_undefined_names(tree, all_globals)
            for undef in undefined:
                yield Violation(
                    rule="python-undefined-name",
                    message=f"Undefined name: '{undef.name}'",
                    path=file_path_str,
                    severity="warning",
                    source_file=source_file,
                    line=undef.line,
                    context={"name": undef.name, "column": undef.column},
                )

        # Check for missing callbacks
        if check_callbacks:
            defined_funcs = self.find_defined_functions(tree)
            missing = self.check_callback_completeness(defined_funcs)
            for callback in missing:
                yield Violation(
                    rule="python-missing-callback",
                    message=f"Missing callback: '{callback}'",
                    path=file_path_str,
                    severity="info",
                    source_file=source_file,
                )

    def parse_syntax(
        self,
        content: str,
        filename: str = "<string>",
    ) -> ast.Module | SyntaxError:
        """
        Parse content and return AST or SyntaxError.

        Args:
            content: Python source code.
            filename: Filename for error messages.

        Returns:
            Parsed AST module or SyntaxError if parsing fails.

        """
        try:
            return ast.parse(content, filename=filename)
        except SyntaxError as e:
            return e

    def find_undefined_names(
        self,
        tree: ast.Module,
        known_globals: Optional[set[str]] = None,
    ) -> list[UndefinedName]:
        """
        Find names used but not defined.

        Uses a simplified scope analysis that tracks:
        - Function/class definitions
        - Assignments
        - Imports
        - Comprehension variables
        - Exception handlers

        Args:
            tree: Parsed AST module.
            known_globals: Set of names to consider as pre-defined.

        Returns:
            List of undefined names with their locations.

        """
        effective_globals: set[str]
        if known_globals is None:
            effective_globals = TD_BUILTINS | PYTHON_BUILTINS
        else:
            effective_globals = known_globals

        # Collect defined names
        defined: set[str] = set(effective_globals)

        # First pass: collect all definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                defined.add(node.name)
                # Add parameters to defined names
                for arg in node.args.args:
                    defined.add(arg.arg)
                for arg in node.args.posonlyargs:
                    defined.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split(".")[0]
                    defined.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    defined.add(name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.comprehension):
                # Comprehension variables
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                elif isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
            elif isinstance(node, ast.For):
                # For loop variables
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
                elif isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
            elif isinstance(node, ast.With):
                # With statement variables
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)

        # Second pass: find undefined names (Load context)
        undefined: list[UndefinedName] = []

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
            ):
                undefined.append(
                    UndefinedName(
                        name=node.id,
                        line=node.lineno,
                        column=node.col_offset,
                    )
                )

        return undefined

    def find_defined_functions(self, tree: ast.Module) -> set[str]:
        """
        Find all function definitions in the tree.

        Args:
            tree: Parsed AST module.

        Returns:
            Set of function names defined in the module.

        """
        functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                functions.add(node.name)
        return functions

    def check_callback_completeness(
        self,
        defined_functions: set[str],
        expected_callbacks: Optional[set[str]] = None,
    ) -> list[str]:
        """
        Return list of missing callbacks.

        Only reports missing callbacks if at least one callback is defined.
        This avoids flagging scripts that aren't meant to be Execute DATs.

        Args:
            defined_functions: Set of function names defined in the script.
            expected_callbacks: Set of expected callback names.

        Returns:
            List of missing callback names (INFO level, not ERROR).

        """
        callbacks: set[str] = (
            expected_callbacks if expected_callbacks is not None else EXECUTE_CALLBACKS
        )

        # Check if any callbacks are defined
        defined_callbacks = defined_functions & callbacks

        if not defined_callbacks:
            # No callbacks defined, this isn't an Execute DAT script
            return []

        # Return missing callbacks
        return sorted(callbacks - defined_functions)

    def strip_text_header(self, content: str) -> tuple[str, int]:
        """
        Remove .text file header (version line).

        Same format as language detector.

        Args:
            content: Raw file content.

        Returns:
            Tuple of (stripped_content, lines_stripped).

        """
        if not content:
            return content, 0

        lines = content.split("\n")

        if len(lines) < 1:
            return content, 0

        first_line = lines[0].strip()
        if not first_line.isdigit():
            return content, 0

        if len(lines) >= 2 and lines[1].startswith("*"):
            return "\n".join(lines[2:]), 2

        return "\n".join(lines[1:]), 1

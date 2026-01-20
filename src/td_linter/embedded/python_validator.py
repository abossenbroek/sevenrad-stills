"""
Python AST validation for TouchDesigner .text files.

Validates Python scripts using ast.parse() and pyflakes for undefined name
detection, with a whitelist of TouchDesigner builtins to avoid false positives.
"""

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from pyflakes import checker as pyflakes_checker
from pyflakes import messages as pyflakes_messages

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
            # Extract the source line for better error display
            source_line = None
            if e.lineno is not None:
                lines = content.split("\n")
                if 1 <= e.lineno <= len(lines):
                    source_line = lines[e.lineno - 1]

            yield Violation(
                rule="python-syntax-error",
                message=f"Syntax error: {e.msg}",
                path=file_path_str,
                severity="error",
                source_file=source_file,
                line=e.lineno,
                context={
                    "offset": e.offset,
                    "source_line": source_line,
                    "language": "Python",
                },
            )
            return

        # Check for undefined names
        if check_undefined:
            all_globals = TD_BUILTINS | PYTHON_BUILTINS
            if extra_globals:
                all_globals = all_globals | extra_globals

            undefined = self.find_undefined_names(tree, all_globals)
            lines = content.split("\n")
            for undef in undefined:
                # Extract source line for better error display
                source_line = None
                if 1 <= undef.line <= len(lines):
                    source_line = lines[undef.line - 1]

                yield Violation(
                    rule="python-undefined-name",
                    message=f"Undefined name: '{undef.name}'",
                    path=file_path_str,
                    severity="warning",
                    source_file=source_file,
                    line=undef.line,
                    context={
                        "name": undef.name,
                        "column": undef.column,
                        "source_line": source_line,
                        "language": "Python",
                    },
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
        Find names used but not defined using pyflakes.

        Uses pyflakes for accurate scope analysis that handles:
        - Nested scopes and closures
        - Comprehension variables
        - Exception handlers
        - Complex assignment patterns
        - All Python scoping rules

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

        # Run pyflakes checker
        # Pass builtins as additional scope to avoid false positives
        checker = pyflakes_checker.Checker(
            tree,
            filename="<string>",
            builtins=list(effective_globals),
        )

        # Extract undefined name messages
        undefined: list[UndefinedName] = []
        for message in checker.messages:
            if isinstance(message, pyflakes_messages.UndefinedName):
                # pyflakes message format: message.message_args contains the name
                name = message.message_args[0]
                undefined.append(
                    UndefinedName(
                        name=name,
                        line=message.lineno,
                        column=message.col,
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
        Remove .text file header (version line and binary metadata).

        Same format as language detector. See LanguageDetector.strip_text_header
        for detailed format documentation.

        Args:
            content: Raw file content.

        Returns:
            Tuple of (stripped_content, lines_stripped).

        """
        if not content:
            return content, 0

        # Handle binary content after the '*' marker
        # Binary format: "2\n*" + 24 bytes binary + actual content
        if len(content) >= 27 and content.startswith("2\n*"):
            header_region = content[3:25]
            if any(ord(c) < 32 and ord(c) != ord('\t') for c in header_region):
                # Binary format - skip exactly 27 bytes
                stripped = content[27:]
                lines_stripped = content[:27].count("\n")
                return stripped, lines_stripped

        # Fallback: line-based stripping
        lines = content.split("\n")

        if len(lines) < 1:
            return content, 0

        first_line = lines[0].strip()
        if not first_line.isdigit():
            return content, 0

        if len(lines) >= 2 and lines[1].startswith("*"):
            return "\n".join(lines[2:]), 2

        return "\n".join(lines[1:]), 1

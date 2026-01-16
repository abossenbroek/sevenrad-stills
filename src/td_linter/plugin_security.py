"""Security checker for plugin code using AST analysis.

This module provides static analysis to detect potentially dangerous
patterns in plugin code before execution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


# Dangerous built-in functions that should be blocked
DANGEROUS_BUILTINS = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",  # File I/O should be controlled
    "input",  # Interactive input
    "breakpoint",  # Debugger access
    "globals",  # Global namespace access
    "locals",  # Local namespace access
    "vars",  # Namespace access
    "delattr",  # Attribute deletion
    "setattr",  # Can modify arbitrary objects
})

# Dangerous modules that should not be imported
DANGEROUS_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",  # File system access
    "io",  # Raw I/O
    "socket",  # Network access
    "http",
    "urllib",
    "requests",  # Network requests
    "ftplib",
    "smtplib",
    "telnetlib",
    "ctypes",  # Foreign function interface
    "multiprocessing",  # Process spawning
    "threading",  # Thread spawning (potential for DoS)
    "pickle",  # Unsafe deserialization
    "marshal",  # Unsafe serialization
    "shelve",
    "importlib",  # Dynamic imports
    "builtins",  # Access to built-in namespace
    "code",  # Interactive interpreter
    "codeop",
    "pty",  # Pseudo-terminal
    "tty",
    "termios",
    "resource",  # System resource access
    "sysconfig",
    "platform",  # System information
    "tempfile",  # Temp file creation
    "glob",  # File pattern matching
    "fnmatch",
    "linecache",  # File reading
    "traceback",  # Can leak information
    "gc",  # Garbage collector manipulation
    "inspect",  # Runtime introspection
    "dis",  # Bytecode disassembly
    "asyncio",  # Async execution
    "concurrent",
    "signal",  # Signal handling
    "mmap",  # Memory mapping
    "sqlite3",  # Database access
    "dbm",
})

# Module prefixes that should be blocked (for submodule imports)
DANGEROUS_MODULE_PREFIXES = frozenset({
    "os.",
    "sys.",
    "subprocess.",
    "http.",
    "urllib.",
    "socket.",
    "ctypes.",
    "multiprocessing.",
    "importlib.",
    "asyncio.",
    "concurrent.",
})

# Allowed modules that plugins CAN use
ALLOWED_MODULES = frozenset({
    # Standard library safe modules
    "abc",
    "collections",
    "copy",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "itertools",
    "json",
    "math",
    "numbers",
    "operator",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
    "types",
    "uuid",
    "warnings",
    "weakref",
    # Third-party safe modules commonly used
    "networkx",
    "nx",
    "lark",
    # td_linter modules that plugins need
    "td_linter",
    "td_linter.rules",
    "td_linter.rules.base",
})

# Dangerous attribute access patterns
DANGEROUS_ATTRIBUTES = frozenset({
    "__code__",
    "__globals__",
    "__builtins__",
    "__subclasses__",
    "__mro__",
    "__class__",
    "__bases__",
    "__dict__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "__init_subclass__",
    "__set_name__",
    "func_code",
    "func_globals",
    "gi_frame",
    "gi_code",
    "f_locals",
    "f_globals",
    "f_code",
    "f_builtins",
    "co_code",
})


@dataclass(frozen=True)
class SecurityViolation:
    """A security violation found in plugin code."""

    message: str
    line: int
    col: int
    severity: str  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] Line {self.line}: {self.message}"


class PluginSecurityChecker:
    """Static analyzer to detect dangerous patterns in plugin code."""

    def __init__(
        self,
        allow_file_read: bool = False,
        additional_allowed_modules: frozenset[str] | None = None,
    ) -> None:
        """Initialize the security checker.

        Args:
            allow_file_read: If True, allow 'open' for reading (still blocked for write).
            additional_allowed_modules: Extra modules to allow beyond defaults.
        """
        self.allow_file_read = allow_file_read
        self.allowed_modules = ALLOWED_MODULES
        if additional_allowed_modules:
            self.allowed_modules = self.allowed_modules | additional_allowed_modules

    def check_source(self, source: str, filename: str = "<plugin>") -> list[SecurityViolation]:
        """Check source code for security violations.

        Args:
            source: Python source code to check.
            filename: Filename for error messages.

        Returns:
            List of security violations found.
        """
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return [
                SecurityViolation(
                    message=f"Syntax error: {e.msg}",
                    line=e.lineno or 1,
                    col=e.offset or 0,
                    severity="error",
                )
            ]

        return list(self._check_ast(tree))

    def check_file(self, path: Path) -> list[SecurityViolation]:
        """Check a Python file for security violations.

        Args:
            path: Path to the Python file.

        Returns:
            List of security violations found.
        """
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as e:
            return [
                SecurityViolation(
                    message=f"Cannot read file: {e}",
                    line=1,
                    col=0,
                    severity="error",
                )
            ]

        return self.check_source(source, filename=str(path))

    def _check_ast(self, tree: ast.AST) -> Iterator[SecurityViolation]:
        """Walk the AST and yield security violations."""
        for node in ast.walk(tree):
            yield from self._check_node(node)

    def _check_node(self, node: ast.AST) -> Iterator[SecurityViolation]:
        """Check a single AST node for violations."""
        # Check imports
        if isinstance(node, ast.Import):
            yield from self._check_import(node)
        elif isinstance(node, ast.ImportFrom):
            yield from self._check_import_from(node)
        # Check function calls
        elif isinstance(node, ast.Call):
            yield from self._check_call(node)
        # Check attribute access
        elif isinstance(node, ast.Attribute):
            yield from self._check_attribute(node)
        # Check name access (for built-ins)
        elif isinstance(node, ast.Name):
            yield from self._check_name(node)

    def _check_import(self, node: ast.Import) -> Iterator[SecurityViolation]:
        """Check regular import statements."""
        for alias in node.names:
            module = alias.name
            if not self._is_module_allowed(module):
                yield SecurityViolation(
                    message=f"Dangerous import: '{module}' is not allowed in plugins",
                    line=node.lineno,
                    col=node.col_offset,
                    severity="error",
                )

    def _check_import_from(self, node: ast.ImportFrom) -> Iterator[SecurityViolation]:
        """Check 'from X import Y' statements."""
        module = node.module or ""

        # Check the module itself
        if not self._is_module_allowed(module):
            yield SecurityViolation(
                message=f"Dangerous import: 'from {module}' is not allowed in plugins",
                line=node.lineno,
                col=node.col_offset,
                severity="error",
            )
            return

        # Check what's being imported from the module
        for alias in node.names:
            name = alias.name
            # Block import of dangerous items even from allowed modules
            if name in DANGEROUS_BUILTINS:
                yield SecurityViolation(
                    message=f"Dangerous import: '{name}' from '{module}' is not allowed",
                    line=node.lineno,
                    col=node.col_offset,
                    severity="error",
                )

    def _is_module_allowed(self, module: str) -> bool:
        """Check if a module import is allowed."""
        # Empty module (relative import) - allow but warn
        if not module:
            return True

        # Check if it's in the allowed list
        if module in self.allowed_modules:
            return True

        # Check if it starts with an allowed module prefix
        for allowed in self.allowed_modules:
            if module.startswith(allowed + "."):
                return True

        # Check if it's explicitly dangerous
        if module in DANGEROUS_MODULES:
            return False

        # Check dangerous prefixes
        for prefix in DANGEROUS_MODULE_PREFIXES:
            if module.startswith(prefix):
                return False

        # Default: block unknown modules for safety
        # Plugins should only use explicitly allowed modules
        return False

    def _check_call(self, node: ast.Call) -> Iterator[SecurityViolation]:
        """Check function calls for dangerous patterns."""
        # Direct call to dangerous built-in
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in DANGEROUS_BUILTINS:
                # Special case for 'open' if file read is allowed
                if name == "open" and self.allow_file_read:
                    # Check if it's being used for writing
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = str(kw.value.value)
                            if any(c in mode for c in "wax+"):
                                yield SecurityViolation(
                                    message="File writing is not allowed in plugins",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    severity="error",
                                )
                                return
                    # Read-only open is allowed
                    return
                yield SecurityViolation(
                    message=f"Dangerous call: '{name}()' is not allowed in plugins",
                    line=node.lineno,
                    col=node.col_offset,
                    severity="error",
                )

        # Call via getattr pattern: getattr(obj, '__code__')
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                attr_name = str(node.args[1].value)
                if attr_name in DANGEROUS_ATTRIBUTES:
                    yield SecurityViolation(
                        message=f"Dangerous getattr: accessing '{attr_name}' is not allowed",
                        line=node.lineno,
                        col=node.col_offset,
                        severity="error",
                    )

    def _check_attribute(self, node: ast.Attribute) -> Iterator[SecurityViolation]:
        """Check attribute access for dangerous patterns."""
        attr = node.attr

        # Check for dangerous dunder attributes
        if attr in DANGEROUS_ATTRIBUTES:
            yield SecurityViolation(
                message=f"Dangerous attribute access: '{attr}' is not allowed",
                line=node.lineno,
                col=node.col_offset,
                severity="error",
            )

    def _check_name(self, node: ast.Name) -> Iterator[SecurityViolation]:
        """Check name lookups for dangerous patterns."""
        # Only check in Load context (reading the name)
        if not isinstance(node.ctx, ast.Load):
            return

        name = node.id

        # Check for direct access to dangerous built-ins
        # Note: This catches attempts to reference them even without calling
        if name in {"__builtins__", "__loader__", "__spec__", "__cached__", "__file__"}:
            yield SecurityViolation(
                message=f"Dangerous name access: '{name}' is not allowed in plugins",
                line=node.lineno,
                col=node.col_offset,
                severity="warning",
            )


class PluginSecurityError(Exception):
    """Raised when plugin code fails security validation."""

    def __init__(self, violations: list[SecurityViolation]) -> None:
        self.violations = violations
        messages = "\n".join(str(v) for v in violations if v.severity == "error")
        super().__init__(f"Plugin security validation failed:\n{messages}")


def validate_plugin_security(
    source: str,
    filename: str = "<plugin>",
    allow_file_read: bool = False,
) -> None:
    """Validate plugin source code and raise on security violations.

    Args:
        source: Python source code to validate.
        filename: Filename for error messages.
        allow_file_read: If True, allow file reading operations.

    Raises:
        PluginSecurityError: If the code contains security violations.
    """
    checker = PluginSecurityChecker(allow_file_read=allow_file_read)
    violations = checker.check_source(source, filename)

    # Filter to only errors (warnings are informational)
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        raise PluginSecurityError(errors)

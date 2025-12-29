"""libclang-based semantic validator for Max/MSP C externals.

This module provides comprehensive semantic analysis using libclang:
- Macro expansion (sees real code, not macro names)
- Type resolution (knows what types variables have)
- Symbol tracking (follows aliases through assignments)
- Interprocedural analysis (tracks calls to helper functions)

Replaces tree-sitter structural checks with proper semantic analysis.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from max_linter.results import (
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Optional libclang import with graceful fallback
try:
    from clang.cindex import (
        Config,
        Cursor,
        CursorKind,
        Index,
        TranslationUnit,
        TranslationUnitLoadError,
    )

    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    Index = None
    Cursor = None
    CursorKind = None

logger = logging.getLogger(__name__)


def _find_libclang() -> str | None:
    """Find libclang library path on macOS."""
    # Try Xcode command line tools first
    try:
        result = subprocess.run(
            ["xcrun", "--find", "clang"],
            capture_output=True,
            text=True,
            check=True,
        )
        clang_path = Path(result.stdout.strip())
        # libclang is typically in ../lib/ relative to clang binary
        lib_dir = clang_path.parent.parent / "lib"
        libclang = lib_dir / "libclang.dylib"
        if libclang.exists():
            return str(libclang)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try Homebrew LLVM
    brew_paths = [
        "/opt/homebrew/opt/llvm/lib/libclang.dylib",  # Apple Silicon
        "/usr/local/opt/llvm/lib/libclang.dylib",  # Intel
    ]
    for path in brew_paths:
        if Path(path).exists():
            return path

    return None


def _make_diagnostic(
    line: int,
    col: int,
    severity: DiagnosticSeverity,
    code: str,
    message: str,
    source: str = "c-semantic",
) -> Diagnostic:
    """Create a diagnostic at given location."""
    return Diagnostic(
        range=Range(
            start=Position(line, col),
            end=Position(line, col),
        ),
        severity=severity,
        message=message,
        source=source,
        code=code,
    )


@dataclass
class MatrixCreation:
    """Tracks a jit_matrix creation with name argument."""

    var_name: str
    line: int
    col: int
    has_name_arg: bool
    cursor: Cursor
    has_late_config: bool = False  # Set when jit_attr_set* called on it
    late_config_line: int = 0  # Line where late config occurred


@dataclass
class FunctionAnalysis:
    """Analysis state for a single function."""

    name: str
    matrices: dict[str, MatrixCreation] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> original
    calls_jit_attr_set: bool = False


class CSemanticValidator:
    """libclang-based semantic validator for Max/MSP C externals.

    Provides comprehensive analysis with:
    - Automatic macro expansion
    - Full type resolution
    - Symbol and alias tracking
    - Interprocedural call analysis
    """

    def __init__(self, max_sdk_path: Path | None = None) -> None:
        """Initialize the validator.

        Args:
            max_sdk_path: Path to Max SDK (auto-detected if not provided)
        """
        self._available = LIBCLANG_AVAILABLE
        self._index: Index | None = None
        self._max_sdk_path = max_sdk_path or self._find_max_sdk()

        if LIBCLANG_AVAILABLE:
            try:
                # Configure libclang path
                libclang_path = _find_libclang()
                if libclang_path:
                    Config.set_library_file(libclang_path)
                self._index = Index.create()
            except Exception as e:
                logger.warning("Failed to initialize libclang: %s", e)
                self._available = False

    def _find_max_sdk(self) -> Path | None:
        """Auto-detect Max SDK path from environment or common locations."""
        # Check environment variable
        sdk_env = os.environ.get("MAX_SDK_PATH")
        if sdk_env:
            return Path(sdk_env)

        # Check relative to linter (assuming in max-externals/linter)
        linter_dir = Path(__file__).parent.parent.parent.parent
        sdk_path = linter_dir.parent / "max-sdk"
        if sdk_path.exists():
            return sdk_path

        return None

    def is_available(self) -> bool:
        """Check if libclang is available."""
        return self._available and self._index is not None

    def _get_compile_args(self, filepath: Path) -> list[str]:
        """Get compilation arguments for parsing."""
        args = [
            "-x",
            "c",
            "-std=c11",
            "-DMAC_VERSION",
            "-Wno-everything",  # Suppress warnings, we only want our checks
        ]

        # Add Max SDK includes if available
        if self._max_sdk_path:
            # Try both SDK structures (direct and max-sdk-base subdirectory)
            base_paths = [
                self._max_sdk_path / "source" / "c74support",
                self._max_sdk_path / "source" / "max-sdk-base" / "c74support",
            ]
            for base in base_paths:
                max_includes = base / "max-includes"
                jit_includes = base / "jit-includes"
                if max_includes.exists():
                    args.extend(["-I", str(max_includes)])
                if jit_includes.exists():
                    args.extend(["-I", str(jit_includes)])

        # Add source directory includes (for common headers)
        source_dir = filepath.parent
        args.extend(["-I", str(source_dir)])
        common_dir = source_dir.parent / "common"
        if common_dir.exists():
            args.extend(["-I", str(common_dir)])

        return args

    def validate(self, source: str, filepath: Path | None = None) -> list[Diagnostic]:
        """Validate C source code.

        Args:
            source: C source code
            filepath: Optional path for include resolution

        Returns:
            List of diagnostics
        """
        if not self.is_available():
            return [
                _make_diagnostic(
                    0,
                    0,
                    DiagnosticSeverity.WARNING,
                    "libclang-unavailable",
                    "libclang not available. Install with: pip install libclang",
                )
            ]

        # Create temporary file for parsing if needed
        if filepath is None:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
                f.write(source)
                filepath = Path(f.name)
            cleanup = True
        else:
            # Write source to filepath for parsing
            filepath.write_text(source)
            cleanup = False

        try:
            return self._validate_file(filepath)
        finally:
            if cleanup:
                filepath.unlink()

    def validate_file(self, filepath: Path) -> list[Diagnostic]:
        """Validate a C source file.

        Args:
            filepath: Path to C source file

        Returns:
            List of diagnostics
        """
        if not self.is_available():
            return [
                _make_diagnostic(
                    0,
                    0,
                    DiagnosticSeverity.WARNING,
                    "libclang-unavailable",
                    "libclang not available. Install with: pip install libclang",
                )
            ]

        return self._validate_file(filepath)

    def _validate_file(self, filepath: Path) -> list[Diagnostic]:
        """Internal file validation."""
        assert self._index is not None

        args = self._get_compile_args(filepath)

        try:
            tu = self._index.parse(
                str(filepath),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except TranslationUnitLoadError as e:
            return [
                _make_diagnostic(
                    0,
                    0,
                    DiagnosticSeverity.ERROR,
                    "parse-error",
                    f"Failed to parse file: {e}",
                )
            ]

        diagnostics: list[Diagnostic] = []

        # Collect analysis data
        max_object_types = self._find_max_object_types(tu.cursor)
        dangerous_functions = self._find_dangerous_functions(tu.cursor)

        # Run checks
        diagnostics.extend(
            self._check_struct_first_member(tu.cursor, max_object_types, filepath)
        )
        diagnostics.extend(self._check_ext_main(tu.cursor, filepath))
        diagnostics.extend(self._check_class_register(tu.cursor, filepath))
        diagnostics.extend(
            self._check_jit_matrix_late_config(tu.cursor, dangerous_functions, filepath)
        )

        return diagnostics

    def _find_max_object_types(self, root: Cursor) -> set[str]:
        """Find types used in class_new() or object_alloc() calls.

        These are the actual Max object types that need t_object first.
        """
        max_types: set[str] = set()

        for cursor in root.walk_preorder():
            if cursor.kind != CursorKind.CALL_EXPR:
                continue

            func_name = cursor.spelling
            if func_name not in ("class_new", "object_alloc"):
                continue

            # Find sizeof() argument to get the type
            # Look through all children for type references
            for child in cursor.walk_preorder():
                # Check for type references in sizeof expressions
                if child.kind == CursorKind.TYPE_REF:
                    type_spelling = child.spelling
                    if type_spelling:
                        max_types.add(type_spelling)
                        # Also track underlying type
                        referenced = child.referenced
                        if referenced:
                            max_types.add(referenced.spelling)

                # Also check tokens for sizeof patterns
                tokens = list(child.get_tokens())
                for i, token in enumerate(tokens):
                    if token.spelling == "sizeof" and i + 1 < len(tokens):
                        # Next tokens might contain the type
                        for j in range(i + 1, min(i + 5, len(tokens))):
                            type_name = tokens[j].spelling
                            if type_name.startswith("t_") or type_name.startswith("_"):
                                max_types.add(type_name)

        return max_types

    def _find_dangerous_functions(self, root: Cursor) -> set[str]:
        """Find functions that call jit_attr_set* (directly or transitively)."""
        # Build call graph
        call_graph: dict[str, set[str]] = {}
        direct_callers: set[str] = set()
        current_function: str | None = None

        for cursor in root.walk_preorder():
            if cursor.kind == CursorKind.FUNCTION_DECL:
                func_name = cursor.spelling
                current_function = func_name
                if func_name not in call_graph:
                    call_graph[func_name] = set()

            elif cursor.kind == CursorKind.CALL_EXPR and current_function:
                callee = cursor.spelling
                call_graph[current_function].add(callee)

                if callee.startswith("jit_attr_set"):
                    direct_callers.add(current_function)

        # Compute transitive closure
        dangerous = set(direct_callers)
        changed = True
        while changed:
            changed = False
            for func, callees in call_graph.items():
                if func not in dangerous and any(
                    callee in dangerous for callee in callees
                ):
                    dangerous.add(func)
                    changed = True

        return dangerous

    def _check_struct_first_member(
        self,
        root: Cursor,
        max_object_types: set[str],
        filepath: Path,
    ) -> Iterator[Diagnostic]:
        """Check that Max object structs have t_object as first member."""
        for cursor in root.walk_preorder():
            # Only check struct declarations
            if cursor.kind != CursorKind.STRUCT_DECL:
                continue

            # Skip if not in our file
            if cursor.location.file and Path(cursor.location.file.name) != filepath:
                continue

            struct_name = cursor.spelling or cursor.type.spelling

            # Check if this struct is used as a Max object
            is_max_object = False
            for max_type in max_object_types:
                if struct_name in max_type or f"_{struct_name}" in max_type:
                    is_max_object = True
                    break

            if not is_max_object:
                continue

            # Get first field
            fields = [
                c for c in cursor.get_children() if c.kind == CursorKind.FIELD_DECL
            ]
            if not fields:
                continue

            first_field = fields[0]
            first_type = first_field.type.spelling

            if first_type != "t_object":
                yield _make_diagnostic(
                    cursor.location.line - 1,
                    cursor.location.column - 1,
                    DiagnosticSeverity.ERROR,
                    "struct-first-member",
                    f"Max object struct '{struct_name}' must have t_object as first "
                    f"member for inheritance. Found: {first_type}",
                )

    def _check_ext_main(self, root: Cursor, filepath: Path) -> Iterator[Diagnostic]:
        """Check that ext_main() exists."""
        for cursor in root.walk_preorder():
            if (
                cursor.kind == CursorKind.FUNCTION_DECL
                and cursor.spelling == "ext_main"
            ):
                return  # Found it

        yield _make_diagnostic(
            0,
            0,
            DiagnosticSeverity.WARNING,
            "no-ext-main",
            "No ext_main() function found. Max externals require ext_main(void *r).",
        )

    def _check_class_register(
        self, root: Cursor, filepath: Path
    ) -> Iterator[Diagnostic]:
        """Check that class_register() is called."""
        for cursor in root.walk_preorder():
            if (
                cursor.kind == CursorKind.CALL_EXPR
                and cursor.spelling == "class_register"
            ):
                return  # Found it

        yield _make_diagnostic(
            0,
            0,
            DiagnosticSeverity.WARNING,
            "no-class-register",
            "No class_register() call found. The object won't be available in Max.",
        )

    def _check_jit_matrix_late_config(
        self,
        root: Cursor,
        dangerous_functions: set[str],
        filepath: Path,
    ) -> Iterator[Diagnostic]:
        """Check for jit_matrix created with name, then configured via jit_attr_set*.

        This catches:
        - Direct: jit_attr_set on same variable
        - Aliased: jit_attr_set on aliased local variable
        - Interprocedural: passing matrix to helper that calls jit_attr_set
        """
        for cursor in root.walk_preorder():
            if cursor.kind != CursorKind.FUNCTION_DECL:
                continue

            # Skip if not in our file
            if cursor.location.file and Path(cursor.location.file.name) != filepath:
                continue

            # Analyze this function
            yield from self._analyze_function_for_late_config(
                cursor, dangerous_functions
            )

    def _analyze_function_for_late_config(
        self,
        func_cursor: Cursor,
        dangerous_functions: set[str],
    ) -> Iterator[Diagnostic]:
        """Analyze a single function for jit_matrix late config patterns."""
        analysis = FunctionAnalysis(name=func_cursor.spelling)

        # First pass: collect matrix creations and aliases
        for cursor in func_cursor.walk_preorder():
            if cursor.kind == CursorKind.CALL_EXPR:
                self._handle_call_expr(
                    cursor, analysis, dangerous_functions, func_cursor
                )

            elif cursor.kind == CursorKind.BINARY_OPERATOR:
                self._handle_assignment(cursor, analysis)

            elif cursor.kind == CursorKind.VAR_DECL:
                # Track var decls with initialization for aliasing
                self._handle_var_decl(cursor, analysis)

        # Check for violations - matrix created with name AND had late config
        for var_name, creation in analysis.matrices.items():
            if creation.has_name_arg and creation.has_late_config:
                yield _make_diagnostic(
                    creation.line - 1,
                    creation.col - 1,
                    DiagnosticSeverity.ERROR,
                    "jit-matrix-late-config",
                    f"jit.matrix '{var_name}' created with name, then "
                    f"type/planecount set via jit_attr_set* (line "
                    f"{creation.late_config_line}). Max 9 requires properties "
                    "set at construction. Use jit_object_new() without name.",
                )

    def _handle_call_expr(
        self,
        cursor: Cursor,
        analysis: FunctionAnalysis,
        dangerous_functions: set[str],
        func_cursor: Cursor,
    ) -> None:
        """Handle a call expression during analysis."""
        func_name = cursor.spelling

        # jit_object_new may be a macro that expands to jit_object_new_imp
        if func_name in ("jit_object_new", "jit_object_new_imp"):
            self._track_matrix_creation(cursor, analysis, func_cursor)

        elif func_name.startswith("jit_attr_set"):
            self._check_attr_set_on_matrix(cursor, analysis)

        elif func_name in dangerous_functions:
            # Helper function that calls jit_attr_set
            self._check_dangerous_call(cursor, analysis)

    def _track_matrix_creation(
        self,
        cursor: Cursor,
        analysis: FunctionAnalysis,
        func_cursor: Cursor,
    ) -> None:
        """Track jit_object_new call for jit_matrix."""
        args = list(cursor.get_arguments())
        if not args:
            return

        # Check if first arg is gensym("jit_matrix")
        first_arg = args[0]
        is_jit_matrix = False

        for child in first_arg.walk_preorder():
            if child.kind == CursorKind.STRING_LITERAL:
                literal = child.spelling.strip('"')
                if literal == "jit_matrix":
                    is_jit_matrix = True
                    break

        if not is_jit_matrix:
            return

        # Determine if name argument was passed by analyzing source tokens
        # (can't use cursor.get_arguments() as macro expansion adds padding args)
        has_name_arg = self._has_name_argument(cursor, func_cursor)

        # Find the assignment target using the function context
        target = self._find_assignment_target(cursor, func_cursor)
        if target:
            analysis.matrices[target] = MatrixCreation(
                var_name=target,
                line=cursor.location.line,
                col=cursor.location.column,
                has_name_arg=has_name_arg,
                cursor=cursor,
            )

    def _has_name_argument(self, call_cursor: Cursor, func_cursor: Cursor) -> bool:
        """Check if jit_object_new was called with a name argument.

        Uses source token analysis because macro expansion adds padding arguments
        that would make cursor.get_arguments() unreliable.
        """
        # Find the BINARY_OPERATOR containing this call
        for child in func_cursor.walk_preorder():
            if child.kind == CursorKind.BINARY_OPERATOR:
                children = list(child.get_children())
                if len(children) == 2:
                    # Check if RHS contains our call
                    for rhs_child in children[1].walk_preorder():
                        if (
                            rhs_child.kind == CursorKind.CALL_EXPR
                            and rhs_child.location.line == call_cursor.location.line
                            and rhs_child.location.column == call_cursor.location.column
                        ):
                            # Found the assignment, analyze tokens
                            tokens = [t.spelling for t in child.get_tokens()]
                            return self._count_jit_object_new_args(tokens) >= 2

        return False

    def _count_jit_object_new_args(self, tokens: list[str]) -> int:
        """Count arguments to jit_object_new from source tokens."""
        in_call = False
        paren_depth = 0
        comma_count = 0

        for tok in tokens:
            if tok == "jit_object_new":
                in_call = True
            elif in_call:
                if tok == "(":
                    paren_depth += 1
                elif tok == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        break
                elif tok == "," and paren_depth == 1:
                    comma_count += 1

        return comma_count + 1 if in_call else 0

    def _check_attr_set_on_matrix(
        self, cursor: Cursor, analysis: FunctionAnalysis
    ) -> None:
        """Check if jit_attr_set is called on a tracked matrix."""
        args = list(cursor.get_arguments())
        if len(args) < 2:
            return

        # Get target variable
        target_expr = args[0]
        target = self._expr_to_string(target_expr)

        # Resolve aliases
        resolved = self._resolve_alias(target, analysis.aliases)

        # Check if setting type or planecount
        attr_name = None
        second_arg = args[1]
        for child in second_arg.walk_preorder():
            if child.kind == CursorKind.STRING_LITERAL:
                attr_name = child.spelling.strip('"')
                break

        if attr_name in ("type", "planecount") and resolved in analysis.matrices:
            creation = analysis.matrices[resolved]
            if creation.has_name_arg:
                creation.has_late_config = True
                creation.late_config_line = cursor.location.line

    def _check_dangerous_call(self, cursor: Cursor, analysis: FunctionAnalysis) -> None:
        """Check if a matrix is passed to a function that calls jit_attr_set."""
        for arg in cursor.get_arguments():
            arg_str = self._expr_to_string(arg)
            resolved = self._resolve_alias(arg_str, analysis.aliases)

            if resolved in analysis.matrices:
                creation = analysis.matrices[resolved]
                if creation.has_name_arg:
                    # Matrix with name passed to dangerous function
                    creation.has_late_config = True
                    creation.late_config_line = cursor.location.line

    def _handle_assignment(self, cursor: Cursor, analysis: FunctionAnalysis) -> None:
        """Track variable assignments for alias resolution."""
        children = list(cursor.get_children())
        if len(children) != 2:
            return

        # Check if this is a simple assignment (not +=, etc)
        # Binary operator with =
        lhs = self._expr_to_string(children[0])
        rhs = self._expr_to_string(children[1])

        # Track alias if RHS is a known matrix or alias
        resolved_rhs = self._resolve_alias(rhs, analysis.aliases)
        if resolved_rhs in analysis.matrices or rhs in analysis.aliases:
            analysis.aliases[lhs] = rhs

    def _handle_var_decl(self, cursor: Cursor, analysis: FunctionAnalysis) -> None:
        """Track variable declarations with initialization for alias resolution.

        Handles patterns like: void *m = x->matrix
        """
        var_name = cursor.spelling
        if not var_name:
            return

        # Get children - first child is type, subsequent are initializers
        children = list(cursor.get_children())
        if not children:
            return

        # The last non-type child is usually the initializer expression
        init_expr = None
        for child in children:
            # Skip type-related cursors
            if child.kind not in (CursorKind.TYPE_REF,):
                init_expr = child

        if init_expr:
            init_str = self._expr_to_string(init_expr)
            # Track alias if initializer is a known matrix or alias
            resolved_init = self._resolve_alias(init_str, analysis.aliases)
            if resolved_init in analysis.matrices or init_str in analysis.aliases:
                analysis.aliases[var_name] = init_str

    def _find_assignment_target(
        self, call_cursor: Cursor, func_cursor: Cursor
    ) -> str | None:
        """Find the variable being assigned from a call expression.

        Args:
            call_cursor: The call expression cursor
            func_cursor: The containing function's cursor (passed explicitly
                because semantic_parent is often None for macro-expanded calls)

        Returns:
            The assignment target as a string (e.g., "x->matrix"), or None
        """
        # Scan the function body for assignments containing this call
        for child in func_cursor.walk_preorder():
            if child.kind == CursorKind.BINARY_OPERATOR:
                children = list(child.get_children())
                if len(children) == 2:
                    # Check if RHS contains our call (match by location)
                    for rhs_child in children[1].walk_preorder():
                        if (
                            rhs_child.kind == CursorKind.CALL_EXPR
                            and rhs_child.location.line == call_cursor.location.line
                            and rhs_child.location.column == call_cursor.location.column
                        ):
                            return self._expr_to_string(children[0])

        return None

    def _expr_to_string(self, cursor: Cursor) -> str:
        """Convert an expression cursor to a string representation."""
        # Get tokens for this expression
        tokens = list(cursor.get_tokens())
        if tokens:
            return "".join(t.spelling for t in tokens)
        return cursor.spelling or ""

    def _resolve_alias(self, var: str, aliases: dict[str, str]) -> str:
        """Resolve a variable through alias chain."""
        seen: set[str] = set()
        while var in aliases and var not in seen:
            seen.add(var)
            var = aliases[var]
        return var

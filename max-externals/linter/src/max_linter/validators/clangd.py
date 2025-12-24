"""Clangd validator for GenExpr (C-like) code."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from max_linter.lsp_client import LSPClient
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range

logger = logging.getLogger(__name__)


# GenExpr built-in functions and variables that clangd won't know about
GENEXPR_BUILTINS = {
    # Input/output
    "in1",
    "in2",
    "in3",
    "in4",
    "out1",
    "out2",
    "out3",
    "out4",
    # Coordinates
    "norm",
    "cell",
    "dim",
    "snorm",
    # Functions
    "sample",
    "nearest",
    "vec",
    "swiz",
    "param",
    # Math functions (subset)
    "abs",
    "ceil",
    "floor",
    "round",
    "trunc",
    "fract",
    "sign",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "pow",
    "sqrt",
    "rsqrt",
    "min",
    "max",
    "clamp",
    "mix",
    "step",
    "smoothstep",
    "length",
    "distance",
    "dot",
    "cross",
    "normalize",
    "reflect",
    "refract",
    "noise",
    "pnoise",
    # Type conversions
    "int",
    "float",
    "uint",
}


class ClangdValidator:
    """Validates GenExpr/C-like code using clangd LSP.

    clangd is the official C/C++ language server from LLVM.
    We use it to validate GenExpr syntax by wrapping code in a C context.

    Source: https://clangd.llvm.org/
    """

    COMMAND = ["clangd", "--log=error", "--enable-config=false"]

    def __init__(self) -> None:
        """Initialize clangd validator."""
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if clangd is installed and available."""
        if self._available is None:
            import subprocess

            try:
                result = subprocess.run(
                    ["clangd", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                self._available = False
                logger.warning("clangd not found. Install with: xcode-select --install")

        return self._available

    def _wrap_genexpr(self, code: str) -> str:
        """Wrap GenExpr code in C context for clangd parsing.

        Creates declarations for GenExpr built-ins and wraps the code
        in a function body so clangd can parse it.

        Args:
            code: GenExpr shader code

        Returns:
            Wrapped C code
        """
        # Create typedefs and declarations for GenExpr types
        header = """
// GenExpr type stubs for clangd
typedef struct { float x, y, z, w; float r, g, b, a; } vec4;
typedef struct { float x, y, z; float r, g, b; } vec3;
typedef struct { float x, y; } vec2;
typedef vec4 (*sample_fn)(int, vec2);
typedef vec4 (*vec_fn)(float, float, float, float);

// Built-in variables
extern vec2 norm;
extern vec2 cell;
extern vec2 dim;
extern vec2 snorm;
extern vec4 in1;
extern vec4 in2;
extern vec4 in3;
extern vec4 in4;
vec4 out1;
vec4 out2;

// Built-in functions
vec4 sample(int input, vec2 coord);
vec4 nearest(int input, vec2 coord);
vec4 vec(float r, float g, float b, float a);
float clamp(float x, float lo, float hi);
float mix(float a, float b, float t);
float step(float edge, float x);
float smoothstep(float e0, float e1, float x);
float length(vec2 v);
float distance(vec2 a, vec2 b);
float dot(vec2 a, vec2 b);
vec2 normalize(vec2 v);

// Math functions
float abs(float x);
float floor(float x);
float ceil(float x);
float round(float x);
float fract(float x);
float sign(float x);
float sin(float x);
float cos(float x);
float tan(float x);
float pow(float x, float y);
float sqrt(float x);
float exp(float x);
float log(float x);
float min(float a, float b);
float max(float a, float b);

// Entry point
void genexpr_main() {
"""
        footer = "\n}\n"

        return header + code + footer

    def _filter_diagnostics(
        self, diagnostics: list[Diagnostic], code_offset: int
    ) -> list[Diagnostic]:
        """Filter and adjust diagnostics for GenExpr context.

        Removes diagnostics about GenExpr built-ins and adjusts line numbers
        to account for the wrapper header.

        Args:
            diagnostics: Raw diagnostics from clangd
            code_offset: Number of header lines added

        Returns:
            Filtered and adjusted diagnostics
        """
        filtered = []

        for diag in diagnostics:
            msg_lower = diag.message.lower()

            # Skip diagnostics about our stubs
            if any(
                builtin in msg_lower
                for builtin in ["genexpr_main", "extern vec", "typedef struct"]
            ):
                continue

            # Skip "too many errors" diagnostic
            if "too many errors" in msg_lower:
                continue

            # Skip incompatible library redeclaration warnings (our math function stubs)
            if "incompatible" in msg_lower and "library function" in msg_lower:
                continue

            # Skip "undeclared identifier" for GenExpr built-ins
            if "undeclared identifier" in msg_lower:
                # Extract identifier from message like:
                # "use of undeclared identifier 'param'"
                skip = False
                for builtin in GENEXPR_BUILTINS:
                    if f"'{builtin}'" in diag.message:
                        skip = True
                        break
                if skip:
                    continue
                # GenExpr allows implicit variable declarations - filter all undeclared
                # identifier errors as these are typically GenExpr variables
                # This is aggressive but necessary since GenExpr is not C
                continue

            # Skip implicit declaration warnings (GenExpr functions)
            if "implicit declaration" in msg_lower:
                continue

            # Skip "expected expression" errors (GenExpr has different syntax)
            if "expected expression" in msg_lower:
                continue

            # Skip type incompatibility errors (GenExpr has implicit conversions)
            if "incompatible type" in msg_lower:
                continue

            # Skip "call to undeclared function" for GenExpr built-ins
            if "undeclared function" in msg_lower:
                continue

            # Adjust line numbers to account for header
            adjusted_start = Position(
                line=max(0, diag.range.start.line - code_offset),
                character=diag.range.start.character,
            )
            adjusted_end = Position(
                line=max(0, diag.range.end.line - code_offset),
                character=diag.range.end.character,
            )

            filtered.append(
                Diagnostic(
                    range=Range(start=adjusted_start, end=adjusted_end),
                    severity=diag.severity,
                    message=diag.message,
                    source="clangd",
                    code=diag.code,
                )
            )

        return filtered

    def validate(self, code: str, filename: str = "genexpr.c") -> list[Diagnostic]:
        """Validate GenExpr code using clangd.

        Args:
            code: GenExpr shader code
            filename: Virtual filename

        Returns:
            List of diagnostics from the language server
        """
        if not self.is_available():
            return [
                Diagnostic(
                    range=_zero_range(),
                    severity=DiagnosticSeverity.WARNING,
                    message="clangd not available - skipping GenExpr validation",
                    source="max-linter",
                )
            ]

        # Wrap code and track header size
        wrapped = self._wrap_genexpr(code)
        header_lines = wrapped.count("\n", 0, wrapped.find("void genexpr_main()")) + 2

        # Create temp file for clangd
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            filepath.write_text(wrapped)
            file_uri = f"file://{filepath}"

            # Create compile_commands.json for clangd
            compile_commands = Path(tmpdir) / "compile_commands.json"
            cc_entry = {
                "directory": tmpdir,
                "file": str(filepath),
                "command": f"clang -c {filepath}",
            }
            import json

            compile_commands.write_text(json.dumps([cc_entry]))

            client = LSPClient(self.COMMAND, source_name="clangd")

            try:
                if not client.start():
                    return [
                        Diagnostic(
                            range=_zero_range(),
                            severity=DiagnosticSeverity.WARNING,
                            message="Failed to start clangd",
                            source="max-linter",
                        )
                    ]

                client.initialize(f"file://{tmpdir}")
                client.open_document(file_uri, wrapped, "c")

                # Wait for diagnostics
                diagnostics = client.get_diagnostics(file_uri, timeout=10.0)

                client.close_document(file_uri)

                # Filter and adjust diagnostics
                return self._filter_diagnostics(diagnostics, header_lines)

            except Exception as e:
                logger.error(f"GenExpr validation error: {e}")
                return [
                    Diagnostic(
                        range=_zero_range(),
                        severity=DiagnosticSeverity.ERROR,
                        message=f"GenExpr validation failed: {e}",
                        source="max-linter",
                    )
                ]
            finally:
                client.shutdown()


def _zero_range() -> Range:
    """Create a zero range for synthetic diagnostics."""
    zero_pos = Position(line=0, character=0)
    return Range(start=zero_pos, end=zero_pos)

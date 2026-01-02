"""Command-line interface for max-linter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from max_linter.extractors.genjit import GenjitExtractor
from max_linter.formatters import FORMATTERS, get_formatter
from max_linter.genexpr import GenExprValidator
from max_linter.results import (
    Diagnostic,
    DiagnosticSeverity,
    LintResult,
    Position,
    Range,
)
from max_linter.validators.c_semantic import CSemanticValidator
from max_linter.validators.clangd import ClangdValidator
from max_linter.validators.glsl import GLSLValidator
from max_linter.validators.maxhelp import MaxhelpLinter


def _make_param_diagnostic(param_name: str, error_msg: str) -> Diagnostic:
    """Create a diagnostic for param range validation errors.

    Args:
        param_name: Name of the parameter with the error
        error_msg: Description of the range error

    Returns:
        Diagnostic object
    """
    return Diagnostic(
        range=Range(start=Position(0, 0), end=Position(0, 0)),
        severity=DiagnosticSeverity.WARNING,
        message=f"Parameter '{param_name}': {error_msg}",
        source="genexpr-params",
        code="param-range",
    )


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def check_lsp_servers() -> None:
    """Check availability of LSP servers."""
    print("Checking LSP server availability...\n")

    glsl = GLSLValidator()
    clangd = ClangdValidator()

    print(f"glsl_analyzer: {'available' if glsl.is_available() else 'NOT FOUND'}")
    if not glsl.is_available():
        print("  Install with: brew install glsl_analyzer")
        print("  Or download from: https://github.com/nolanderc/glsl_analyzer")

    print(f"clangd:        {'available' if clangd.is_available() else 'NOT FOUND'}")
    if not clangd.is_available():
        print("  Install with: xcode-select --install")
        print("  Or: brew install llvm")


def lint_file(
    filepath: Path,
    glsl_validator: GLSLValidator,
    clangd_validator: ClangdValidator,
    genexpr_validator: GenExprValidator,
    fallback: bool = False,
) -> LintResult:
    """Lint a single .genjit file.

    Args:
        filepath: Path to the file
        glsl_validator: GLSL validator instance
        clangd_validator: Clangd validator instance
        genexpr_validator: GenExpr validator instance
        fallback: If True, skip LSP validation if servers unavailable

    Returns:
        Lint result with diagnostics
    """
    extractor = GenjitExtractor()
    shaders = extractor.extract(filepath)

    if not shaders:
        return LintResult(
            filepath=str(filepath),
            diagnostics=[],
            success=True,
        )

    all_diagnostics = []

    for shader in shaders:
        if shader.language == "glsl":
            if glsl_validator.is_available() or not fallback:
                diagnostics = glsl_validator.validate(
                    shader.code,
                    filename=f"{filepath.stem}.frag",
                )
                all_diagnostics.extend(diagnostics)
        else:  # genexpr
            # Extract param names for semantic analysis
            param_names = {p.name for p in shader.params}

            # Use GenExprValidator for GenExpr shaders with param context
            diagnostics = genexpr_validator.validate(
                shader.code, declared_params=param_names
            )
            all_diagnostics.extend(diagnostics)

            # Validate param ranges
            range_errors = extractor.validate_param_ranges(shader.params)
            for param_name, error_msg in range_errors:
                all_diagnostics.append(_make_param_diagnostic(param_name, error_msg))

    # Filter out "not available" warnings in fallback mode
    if fallback:
        all_diagnostics = [
            d for d in all_diagnostics if "not available" not in d.message.lower()
        ]

    has_errors = any(d.severity == DiagnosticSeverity.ERROR for d in all_diagnostics)

    return LintResult(
        filepath=str(filepath),
        diagnostics=all_diagnostics,
        success=not has_errors,
    )


def lint_c_file(
    filepath: Path,
    c_validator: CSemanticValidator,
    strict: bool = False,
) -> LintResult:
    """Lint a single C source file for Max/MSP external patterns.

    Args:
        filepath: Path to the C file
        c_validator: CSemanticValidator instance
        strict: If True, treat warnings as errors

    Returns:
        Lint result with diagnostics
    """
    diagnostics = c_validator.validate_file(filepath)

    if strict:
        # Treat warnings as errors in strict mode
        has_errors = any(
            d.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.WARNING)
            for d in diagnostics
        )
    else:
        has_errors = any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics)

    return LintResult(
        filepath=str(filepath),
        diagnostics=diagnostics,
        success=not has_errors,
    )


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description=(
            "LSP-based linter for Max/MSP GenExpr, GLSL shaders, and help patchers"
        ),
        prog="max-lint",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Files or directories to lint (.genjit, .maxhelp, and/or .maxpat)",
    )
    parser.add_argument(
        "--check-lsp",
        action="store_true",
        help="Check if LSP servers are available",
    )
    parser.add_argument(
        "--glsl",
        action="store_true",
        help="Force GLSL validation only",
    )
    parser.add_argument(
        "--clangd",
        action="store_true",
        help="Force clangd validation only",
    )
    parser.add_argument(
        "--maxhelp",
        action="store_true",
        help="Lint only .maxhelp files (skip .genjit)",
    )
    parser.add_argument(
        "--maxpat",
        action="store_true",
        help="Lint only .maxpat files (skip .genjit and .maxhelp)",
    )
    parser.add_argument(
        "--c-external",
        action="store_true",
        help="Lint C external source files (.c) for Max/MSP patterns",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Skip validation if LSP servers unavailable",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="max-linter 0.1.0",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        choices=list(FORMATTERS.keys()),
        default="text",
        help="Output format: text (default), json, yaml, or github (CI annotations)",
    )

    parsed = parser.parse_args(args)
    setup_logging(parsed.verbose)

    if parsed.check_lsp:
        check_lsp_servers()
        return 0

    if not parsed.files:
        parser.print_help()
        return 1

    # Collect files to lint
    genjit_files: list[Path] = []
    maxhelp_files: list[Path] = []
    c_files: list[Path] = []

    for path in parsed.files:
        if path.is_dir():
            if not parsed.maxhelp and not parsed.maxpat and not parsed.c_external:
                genjit_files.extend(path.glob("*.genjit"))
            if not parsed.maxpat and not parsed.c_external:
                maxhelp_files.extend(path.glob("*.maxhelp"))
            if not parsed.maxhelp and not parsed.c_external:
                maxhelp_files.extend(path.glob("*.maxpat"))
            if parsed.c_external:
                c_files.extend(path.rglob("*.c"))
        elif (
            path.suffix == ".genjit"
            and not parsed.maxhelp
            and not parsed.maxpat
            and not parsed.c_external
        ):
            genjit_files.append(path)
        elif path.suffix in (".maxhelp", ".maxpat") and not parsed.c_external:
            # Handle both .maxhelp and .maxpat with proper flag filtering
            is_maxhelp = path.suffix == ".maxhelp" and not parsed.maxpat
            is_maxpat = path.suffix == ".maxpat" and not parsed.maxhelp
            if is_maxhelp or is_maxpat:
                maxhelp_files.append(path)
        elif path.suffix == ".c" and parsed.c_external:
            c_files.append(path)
        else:
            print(f"Warning: Skipping unsupported file: {path}", file=sys.stderr)

    if not genjit_files and not maxhelp_files and not c_files:
        print("No .genjit, .maxhelp, .maxpat, or .c files found", file=sys.stderr)
        return 1

    # Initialize validators
    glsl_validator = GLSLValidator()
    clangd_validator = ClangdValidator()
    genexpr_validator = GenExprValidator()
    c_validator = CSemanticValidator()

    maxhelp_linter = MaxhelpLinter(strict=parsed.strict, verbose=parsed.verbose)

    # Initialize output formatter
    formatter = get_formatter(parsed.output_format)

    # Lint files
    results: list[LintResult] = []

    # Lint .genjit files
    for filepath in sorted(genjit_files):
        result = lint_file(
            filepath,
            glsl_validator,
            clangd_validator,
            genexpr_validator,
            fallback=parsed.fallback,
        )
        results.append(result)

        # Convert diagnostics to LintError format for formatter
        from max_linter.lint_error import LintError
        from max_linter.types import Severity

        errors = [
            LintError(
                severity=Severity.ERROR,
                rule=str(d.code or d.source or "genjit"),
                message=d.message,
            )
            for d in result.diagnostics
            if d.severity == DiagnosticSeverity.ERROR
        ]
        warnings = [
            LintError(
                severity=Severity.WARNING,
                rule=str(d.code or d.source or "genjit"),
                message=d.message,
            )
            for d in result.diagnostics
            if d.severity == DiagnosticSeverity.WARNING
        ]
        formatter.format_file_result(filepath, errors, warnings, parsed.verbose)

    # Lint .maxhelp/.maxpat files
    for filepath in sorted(maxhelp_files):
        maxhelp_linter.validate_file(filepath)

        # Use formatter instead of print_results for non-text formats
        if parsed.output_format == "text":
            maxhelp_linter.print_results(filepath)
        else:
            formatter.format_file_result(
                filepath,
                maxhelp_linter.errors,
                maxhelp_linter.warnings,
                parsed.verbose,
            )

        results.append(
            LintResult(
                filepath=str(filepath),
                diagnostics=[],
                success=not maxhelp_linter.has_errors(),
            )
        )

    # Lint C external source files
    for filepath in sorted(c_files):
        result = lint_c_file(
            filepath,
            c_validator,
            strict=parsed.strict,
        )
        results.append(result)

        # Convert diagnostics to LintError format for formatter
        from max_linter.lint_error import LintError
        from max_linter.types import Severity

        errors = [
            LintError(
                severity=Severity.ERROR,
                rule=str(d.code or d.source or "c-external"),
                message=d.message,
            )
            for d in result.diagnostics
            if d.severity == DiagnosticSeverity.ERROR
        ]
        warnings = [
            LintError(
                severity=Severity.WARNING,
                rule=str(d.code or d.source or "c-external"),
                message=d.message,
            )
            for d in result.diagnostics
            if d.severity == DiagnosticSeverity.WARNING
        ]
        formatter.format_file_result(filepath, errors, warnings, parsed.verbose)

    # Summary
    total = len(results)
    failed = sum(1 for r in results if not r.success)
    warnings_count = sum(1 for r in results if r.has_warnings and not r.has_errors)

    # Build file counts
    file_counts: dict[str, int] = {}
    if genjit_files:
        file_counts[".genjit"] = len(genjit_files)
    if maxhelp_files:
        maxhelp_count = len([f for f in maxhelp_files if f.suffix == ".maxhelp"])
        maxpat_count = len([f for f in maxhelp_files if f.suffix == ".maxpat"])
        if maxhelp_count:
            file_counts[".maxhelp"] = maxhelp_count
        if maxpat_count:
            file_counts[".maxpat"] = maxpat_count
    if c_files:
        file_counts[".c"] = len(c_files)

    formatter.format_summary(total, failed, warnings_count, file_counts)
    formatter.finalize()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

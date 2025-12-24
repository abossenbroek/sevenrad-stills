"""Command-line interface for max-linter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from max_linter.extractors.genjit import GenjitExtractor
from max_linter.results import DiagnosticSeverity, LintResult
from max_linter.validators.clangd import ClangdValidator
from max_linter.validators.glsl import GLSLValidator


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
    fallback: bool = False,
) -> LintResult:
    """Lint a single .genjit file.

    Args:
        filepath: Path to the file
        glsl_validator: GLSL validator instance
        clangd_validator: Clangd validator instance
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
            if clangd_validator.is_available() or not fallback:
                diagnostics = clangd_validator.validate(
                    shader.code,
                    filename=f"{filepath.stem}.c",
                )
                all_diagnostics.extend(diagnostics)

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


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description="LSP-based linter for Max/MSP GenExpr and GLSL shaders",
        prog="max-lint",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Files or directories to lint",
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

    parsed = parser.parse_args(args)
    setup_logging(parsed.verbose)

    if parsed.check_lsp:
        check_lsp_servers()
        return 0

    if not parsed.files:
        parser.print_help()
        return 1

    # Collect all .genjit files
    files_to_lint: list[Path] = []
    for path in parsed.files:
        if path.is_dir():
            files_to_lint.extend(path.glob("*.genjit"))
        elif path.suffix == ".genjit":
            files_to_lint.append(path)
        else:
            print(f"Warning: Skipping non-.genjit file: {path}", file=sys.stderr)

    if not files_to_lint:
        print("No .genjit files found", file=sys.stderr)
        return 1

    # Initialize validators
    glsl_validator = GLSLValidator()
    clangd_validator = ClangdValidator()

    # Lint files
    results: list[LintResult] = []
    for filepath in sorted(files_to_lint):
        result = lint_file(
            filepath,
            glsl_validator,
            clangd_validator,
            fallback=parsed.fallback,
        )
        results.append(result)

        # Print results
        if result.diagnostics:
            print(f"\n{filepath}:")
            for diagnostic in result.diagnostics:
                print(f"  {diagnostic}")
        elif parsed.verbose:
            print(f"{filepath}: OK")

    # Summary
    total = len(results)
    failed = sum(1 for r in results if not r.success)
    warnings = sum(1 for r in results if r.has_warnings and not r.has_errors)

    print(f"\nValidated {total} file(s)")
    if failed:
        print(f"  {failed} with errors")
    if warnings:
        print(f"  {warnings} with warnings")
    if failed == 0 and warnings == 0:
        print("  All files passed")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

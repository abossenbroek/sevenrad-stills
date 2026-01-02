"""Output formatters for linter results.

This module provides different output formats for linter results:
- text: Human-readable text format (default)
- json: Machine-parseable JSON format
- yaml: Structured YAML format
- github: GitHub Actions workflow commands for annotations

Usage:
    formatter = get_formatter("json")
    formatter.format_file_result(filepath, errors, warnings)
    formatter.format_summary(total, failed, warnings_count)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max_linter.lint_error import LintError


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format_file_result(
        self,
        filepath: Path,
        errors: list[LintError],
        warnings: list[LintError],
        verbose: bool = False,
    ) -> None:
        """Format and output results for a single file."""
        ...

    @abstractmethod
    def format_summary(
        self,
        total: int,
        failed: int,
        warnings_count: int,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        """Format and output the summary."""
        ...

    def finalize(self) -> None:  # noqa: B027
        """Called after all results have been formatted.

        Override in subclasses that need to output collected data at the end
        (e.g., JSON array output).

        Note: This is intentionally not abstract - default is no-op.
        """


class TextFormatter(OutputFormatter):
    """Human-readable text output format."""

    def format_file_result(
        self,
        filepath: Path,
        errors: list[LintError],
        warnings: list[LintError],
        verbose: bool = False,
    ) -> None:
        """Format and output results for a single file."""
        if errors or warnings:
            print(f"\n{filepath}:")
            for err in errors:
                print(err)
            for warn in warnings:
                print(warn)
        elif verbose:
            print(f"\n{filepath}: OK")

    def format_summary(
        self,
        total: int,
        failed: int,
        warnings_count: int,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        """Format and output the summary."""
        print(f"\nValidated {total} file(s)")
        if file_counts:
            for ext, count in file_counts.items():
                if count:
                    print(f"  {count} {ext} file(s)")
        if failed:
            print(f"  {failed} with errors")
        if warnings_count:
            print(f"  {warnings_count} with warnings")
        if failed == 0 and warnings_count == 0:
            print("  All files passed")


class JSONFormatter(OutputFormatter):
    """Machine-parseable JSON output format."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def format_file_result(
        self,
        filepath: Path,
        errors: list[LintError],
        warnings: list[LintError],
        verbose: bool = False,
    ) -> None:
        """Collect results for JSON output."""
        if errors or warnings or verbose:
            self.results.append(
                {
                    "file": str(filepath),
                    "errors": [e.to_dict() for e in errors],
                    "warnings": [w.to_dict() for w in warnings],
                    "status": "error" if errors else ("warning" if warnings else "ok"),
                }
            )

    def format_summary(
        self,
        total: int,
        failed: int,
        warnings_count: int,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        """Add summary to results."""
        self.summary = {
            "total_files": total,
            "files_with_errors": failed,
            "files_with_warnings": warnings_count,
            "file_counts": file_counts or {},
            "success": failed == 0,
        }

    def finalize(self) -> None:
        """Output collected JSON."""
        output = {
            "results": self.results,
            "summary": getattr(self, "summary", {}),
        }
        print(json.dumps(output, indent=2))


class YAMLFormatter(OutputFormatter):
    """Structured YAML output format."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def format_file_result(
        self,
        filepath: Path,
        errors: list[LintError],
        warnings: list[LintError],
        verbose: bool = False,
    ) -> None:
        """Collect results for YAML output."""
        if errors or warnings or verbose:
            self.results.append(
                {
                    "file": str(filepath),
                    "errors": [e.to_dict() for e in errors],
                    "warnings": [w.to_dict() for w in warnings],
                    "status": "error" if errors else ("warning" if warnings else "ok"),
                }
            )

    def format_summary(
        self,
        total: int,
        failed: int,
        warnings_count: int,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        """Add summary to results."""
        self.summary = {
            "total_files": total,
            "files_with_errors": failed,
            "files_with_warnings": warnings_count,
            "file_counts": file_counts or {},
            "success": failed == 0,
        }

    def finalize(self) -> None:
        """Output collected YAML."""
        # Simple YAML output without external dependency
        print("results:")
        for result in self.results:
            print(f"  - file: {result['file']}")
            print(f"    status: {result['status']}")
            if result["errors"]:
                print("    errors:")
                for err in result["errors"]:
                    print(f"      - severity: {err['severity']}")
                    print(f"        rule: {err['rule']}")
                    print(f'        message: "{err["message"]}"')
                    if err.get("object_id"):
                        print(f"        object_id: {err['object_id']}")
            if result["warnings"]:
                print("    warnings:")
                for warn in result["warnings"]:
                    print(f"      - severity: {warn['severity']}")
                    print(f"        rule: {warn['rule']}")
                    print(f'        message: "{warn["message"]}"')
                    if warn.get("object_id"):
                        print(f"        object_id: {warn['object_id']}")

        print("\nsummary:")
        summary = getattr(self, "summary", {})
        print(f"  total_files: {summary.get('total_files', 0)}")
        print(f"  files_with_errors: {summary.get('files_with_errors', 0)}")
        print(f"  files_with_warnings: {summary.get('files_with_warnings', 0)}")
        print(f"  success: {summary.get('success', True)}")
        if summary.get("file_counts"):
            print("  file_counts:")
            for ext, count in summary["file_counts"].items():
                print(f"    {ext}: {count}")


class GitHubFormatter(OutputFormatter):
    """GitHub Actions workflow command format for annotations.

    Outputs errors and warnings as workflow commands that GitHub Actions
    will display as annotations on the code.

    See: https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions
    """

    def format_file_result(
        self,
        filepath: Path,
        errors: list[LintError],
        warnings: list[LintError],
        verbose: bool = False,
    ) -> None:
        """Output GitHub Actions workflow commands."""
        for err in errors:
            print(err.to_github_annotation(str(filepath)))
        for warn in warnings:
            print(warn.to_github_annotation(str(filepath)))

    def format_summary(
        self,
        total: int,
        failed: int,
        warnings_count: int,
        file_counts: dict[str, int] | None = None,
    ) -> None:
        """Output summary as a notice."""
        if failed > 0:
            print(f"::error::Linting failed: {failed}/{total} files have errors")
        elif warnings_count > 0:
            print(
                f"::warning::Linting completed with warnings: "
                f"{warnings_count}/{total} files have warnings"
            )
        else:
            print(f"::notice::Linting passed: {total} files validated successfully")


# Registry of available formatters
FORMATTERS: dict[str, type[OutputFormatter]] = {
    "text": TextFormatter,
    "json": JSONFormatter,
    "yaml": YAMLFormatter,
    "github": GitHubFormatter,
}


def get_formatter(format_name: str) -> OutputFormatter:
    """Get an output formatter by name.

    Args:
        format_name: One of 'text', 'json', 'yaml', 'github'

    Returns:
        OutputFormatter instance

    Raises:
        ValueError: If format_name is not recognized
    """
    if format_name not in FORMATTERS:
        valid = ", ".join(FORMATTERS.keys())
        raise ValueError(
            f"Unknown output format: {format_name}. Valid formats: {valid}"
        )
    return FORMATTERS[format_name]()

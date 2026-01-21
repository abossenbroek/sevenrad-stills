"""Tests for TextFormatter."""

from pathlib import Path

import pytest

from td_linter.output.text import TextFormatter
from td_linter.rules.base import Violation


class TestTextFormatterBasics:
    """Basic TextFormatter tests."""

    def test_name_property(self) -> None:
        """Test that formatter name is 'text'."""
        formatter = TextFormatter()
        assert formatter.name == "text"

    def test_supports_color(self) -> None:
        """Test that text formatter supports color."""
        formatter = TextFormatter()
        assert formatter.supports_color is True

    def test_empty_violations_with_project_path(self) -> None:
        """Test empty violations list shows OK message."""
        formatter = TextFormatter(no_color=True)
        output = formatter.format([], project_path="test.toe.dir")
        assert "OK" in output
        assert "test.toe.dir" in output

    def test_empty_violations_no_project_path(self) -> None:
        """Test empty violations without project path returns empty string."""
        formatter = TextFormatter(no_color=True)
        output = formatter.format([])
        assert output == ""


class TestTextFormatterViolations:
    """Test violation formatting."""

    def test_single_error(self) -> None:
        """Test single error violation formatting."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(
                rule="C001",
                message="Test error message",
                path="/project/op1",
                severity="error",
            )
        ]
        output = formatter.format(violations)
        assert "ERROR" in output
        assert "C001" in output
        assert "Test error message" in output

    def test_single_warning(self) -> None:
        """Test single warning violation formatting."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(
                rule="C002",
                message="Test warning",
                path="/project/op1",
                severity="warning",
            )
        ]
        output = formatter.format(violations)
        assert "WARNING" in output
        assert "C002" in output

    def test_single_info(self) -> None:
        """Test single info violation formatting."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(
                rule="P003",
                message="Test info",
                path="/project/op1",
                severity="info",
            )
        ]
        output = formatter.format(violations)
        assert "INFO" in output
        assert "P003" in output

    def test_violation_with_line_number(self) -> None:
        """Test violation with line number shows line in output."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(
                rule="G001",
                message="GLSL error",
                path="/project/shader1",
                severity="error",
                source_file=Path("/project/shader1.text"),
                line=42,
            )
        ]
        output = formatter.format(violations)
        assert ":42" in output
        assert "G001" in output

    def test_multiple_violations_grouped_by_file(self) -> None:
        """Test multiple violations are grouped by source file."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(
                rule="C001",
                message="Error 1",
                path="/project/op1",
                severity="error",
                source_file=Path("/project/file1.n"),
            ),
            Violation(
                rule="C002",
                message="Warning 1",
                path="/project/op2",
                severity="warning",
                source_file=Path("/project/file2.n"),
            ),
            Violation(
                rule="S001",
                message="Error 2",
                path="/project/op3",
                severity="error",
                source_file=Path("/project/file1.n"),
            ),
        ]
        output = formatter.format(violations)
        # Both file paths should appear
        assert "file1.n" in output
        assert "file2.n" in output


class TestTextFormatterSummary:
    """Test summary output."""

    def test_summary_counts(self) -> None:
        """Test summary shows correct counts."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(rule="C001", message="E1", path="/p/1", severity="error"),
            Violation(rule="C002", message="W1", path="/p/2", severity="warning"),
            Violation(rule="C003", message="W2", path="/p/3", severity="warning"),
            Violation(rule="P003", message="I1", path="/p/4", severity="info"),
        ]
        output = formatter.format(violations)
        assert "1 errors" in output
        assert "2 warnings" in output
        assert "1 info" in output


class TestTextFormatterNoColor:
    """Test no_color option."""

    def test_no_color_option(self) -> None:
        """Test that no_color option works."""
        formatter = TextFormatter(no_color=True)
        violations = [
            Violation(rule="C001", message="Test", path="/p/1", severity="error"),
        ]
        output = formatter.format(violations)
        # Output should not contain ANSI escape codes
        assert "\x1b[" not in output

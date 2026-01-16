"""Tests for td-linter programmatic API."""

from pathlib import Path

import pytest

from td_linter import Severity, Violation, lint, lint_and_check


class TestLintFunction:
    """Tests for the lint() function."""

    def test_lint_nonexistent_path_raises(self) -> None:
        """Test that lint raises FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            lint("/nonexistent/path.toe.dir")

    def test_lint_returns_list(self, tmp_path: Path) -> None:
        """Test that lint returns a list of violations."""
        # Create a minimal .toe.dir structure
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("")

        result = lint(toe_dir)
        assert isinstance(result, list)


class TestLintAndCheckFunction:
    """Tests for the lint_and_check() function."""

    def test_lint_and_check_returns_tuple(self, tmp_path: Path) -> None:
        """Test that lint_and_check returns a tuple of (bool, list)."""
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("")

        passed, violations = lint_and_check(toe_dir)
        assert isinstance(passed, bool)
        assert isinstance(violations, list)


class TestSeverityEnum:
    """Tests for the Severity enum."""

    def test_severity_values(self) -> None:
        """Test Severity enum has expected values."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_severity_is_string(self) -> None:
        """Test Severity enum values can be used as strings."""
        assert Severity.ERROR == "error"
        assert Severity.WARNING == "warning"
        assert Severity.INFO == "info"


class TestViolationExport:
    """Tests for Violation class export."""

    def test_violation_can_be_imported(self) -> None:
        """Test that Violation can be imported from td_linter."""
        from td_linter import Violation

        v = Violation(
            rule="C001",
            message="Test",
            path="/test",
            severity="error",
        )
        assert v.rule == "C001"
        assert v.message == "Test"
        assert v.severity == "error"

"""
Unit tests for the fix command CLI.

These tests verify that the td-linter fix command correctly handles:
- Help output with all documented options
- Error cases (missing paths, invalid paths)
- Dry-run mode
"""

from td_linter.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class TestFixHelpOutput:
    """Tests for fix command help output."""

    def test_fix_help_returns_zero(self) -> None:
        """td-linter fix --help should return exit code 0."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "fix" in output_lower

    def test_fix_help_shows_path_argument(self) -> None:
        """Fix help should show path argument."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output.lower()

    def test_fix_help_shows_dry_run_option(self) -> None:
        """Fix help should show --dry-run option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output or "-n" in result.output

    def test_fix_help_shows_config_option(self) -> None:
        """Fix help should show --config option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output or "-c" in result.output

    def test_fix_help_shows_select_option(self) -> None:
        """Fix help should show --select option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--select" in result.output or "-s" in result.output

    def test_fix_help_shows_ignore_option(self) -> None:
        """Fix help should show --ignore option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--ignore" in result.output or "-i" in result.output

    def test_fix_help_shows_verbose_option(self) -> None:
        """Fix help should show --verbose option."""
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output or "-v" in result.output


class TestFixNegativeCases:
    """Tests that should fail with non-zero exit code."""

    def test_fix_without_path_fails(self) -> None:
        """td-linter fix with no path argument should fail."""
        result = runner.invoke(app, ["fix"])
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "missing" in output_lower or "error" in output_lower

    def test_fix_nonexistent_path_fails(self) -> None:
        """td-linter fix /nonexistent/path.toe.dir should fail."""
        result = runner.invoke(app, ["fix", "/nonexistent/path.toe.dir"])
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "not found" in output_lower or "error" in output_lower


class TestMainHelpIncludesFix:
    """Tests that main help includes fix command."""

    def test_main_help_shows_fix_command(self) -> None:
        """Main help should list fix command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fix" in result.output.lower()

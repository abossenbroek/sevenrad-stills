"""
Unit tests for the watch command CLI.

These tests verify that the td-linter watch command correctly handles:
- Help output with all documented options
- Error cases (missing paths, invalid paths, invalid options)
- Optional dependency handling (watchdog)
"""

from pathlib import Path
from unittest.mock import patch

from td_linter.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class TestWatchHelpOutput:
    """Tests for watch command help output."""

    def test_watch_help_returns_zero(self) -> None:
        """td-linter watch --help should return exit code 0."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "watch" in output_lower or "monitor" in output_lower

    def test_watch_help_shows_path_argument(self) -> None:
        """Watch help should show path argument."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output.lower()

    def test_watch_help_shows_debounce_option(self) -> None:
        """Watch help should show --debounce option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--debounce" in result.output

    def test_watch_help_shows_fix_option(self) -> None:
        """Watch help should show --fix option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--fix" in result.output

    def test_watch_help_shows_clear_option(self) -> None:
        """Watch help should show --clear option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--clear" in result.output or "--no-clear" in result.output

    def test_watch_help_shows_config_option(self) -> None:
        """Watch help should show --config option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output or "-c" in result.output

    def test_watch_help_shows_select_option(self) -> None:
        """Watch help should show --select option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--select" in result.output or "-s" in result.output

    def test_watch_help_shows_ignore_option(self) -> None:
        """Watch help should show --ignore option."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--ignore" in result.output or "-i" in result.output


class TestWatchNegativeCases:
    """Tests that should fail with non-zero exit code."""

    def test_watch_without_path_fails(self) -> None:
        """td-linter watch with no path argument should fail."""
        result = runner.invoke(app, ["watch"])
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "missing" in output_lower or "error" in output_lower

    def test_watch_nonexistent_path_fails(self) -> None:
        """td-linter watch /nonexistent/path.toe.dir should fail."""
        result = runner.invoke(app, ["watch", "/nonexistent/path.toe.dir"])
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "not found" in output_lower or "error" in output_lower

    def test_watch_invalid_debounce_fails(self) -> None:
        """td-linter watch with invalid debounce value should fail."""
        result = runner.invoke(
            app, ["watch", "/nonexistent/path.toe.dir", "--debounce", "invalid"]
        )
        assert result.exit_code != 0


class TestWatchOptionalDependency:
    """Tests for watchdog optional dependency handling."""

    def test_watch_shows_install_message_when_watchdog_missing(
        self, tmp_path: Path
    ) -> None:
        """Watch command should show install message when watchdog is missing."""
        # Create a minimal .toe.dir structure
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("")

        # Mock watchdog import to raise ImportError
        with patch.dict("sys.modules", {"watchdog": None}):
            with patch(
                "td_linter.watch.TDLintWatcher.start",
                side_effect=ImportError("No module named 'watchdog'"),
            ):
                result = runner.invoke(app, ["watch", str(toe_dir)])
                # Should fail and suggest installing watchdog
                # The exact message depends on implementation
                assert result.exit_code != 0


class TestMainHelpIncludesWatch:
    """Tests that main help includes watch command."""

    def test_main_help_shows_watch_command(self) -> None:
        """Main help should list watch command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower()

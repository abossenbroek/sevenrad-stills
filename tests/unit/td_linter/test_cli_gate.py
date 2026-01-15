"""
Unit tests for CLI gate criteria (G1.3).

These tests verify that the td-linter CLI correctly handles:
- Help commands for all subcommands
- Version command
- Rules listing
- Error cases (missing paths, invalid flags, unknown commands)
"""

import re

from td_linter.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class TestPositiveCases:
    """Tests that should succeed with exit code 0."""

    def test_main_help_returns_zero(self) -> None:
        """td-linter --help should return exit code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "td-linter" in output_lower or "validate" in output_lower

    def test_lint_help_returns_zero(self) -> None:
        """td-linter lint --help should return exit code 0."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output.lower()

    def test_rules_help_returns_zero(self) -> None:
        """td-linter rules --help should return exit code 0."""
        result = runner.invoke(app, ["rules", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "rules" in output_lower or "lint" in output_lower

    def test_init_help_returns_zero(self) -> None:
        """td-linter init --help should return exit code 0."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_version_returns_zero(self) -> None:
        """td-linter version should return exit code 0."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "td-linter" in result.output.lower()
        # Should contain a version string (e.g., 0.1.0)
        assert any(char.isdigit() for char in result.output)

    def test_rules_lists_available_rules(self) -> None:
        """td-linter rules should list available rules with exit code 0."""
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        # Should contain rule IDs from the registered rules
        # These are the rules registered in linter.py:get_all_rules()
        output_lower = result.output.lower()
        assert "no-invalid-cycles" in output_lower or "cycles" in output_lower


class TestNegativeCases:
    """Tests that should fail with non-zero exit code."""

    def test_lint_without_path_fails(self) -> None:
        """td-linter lint with no path argument should fail."""
        result = runner.invoke(app, ["lint"])
        assert result.exit_code != 0
        # Typer typically shows "Missing argument" for required args
        output_lower = result.output.lower()
        assert "missing" in output_lower or "error" in output_lower

    def test_lint_nonexistent_path_fails(self) -> None:
        """td-linter lint /nonexistent/path.toe.dir should fail."""
        result = runner.invoke(app, ["lint", "/nonexistent/path.toe.dir"])
        assert result.exit_code != 0
        # Should indicate path not found
        output_lower = result.output.lower()
        assert "not found" in output_lower or "error" in output_lower

    def test_invalid_flag_fails(self) -> None:
        """td-linter --invalid-flag should fail."""
        result = runner.invoke(app, ["--invalid-flag"])
        assert result.exit_code != 0
        # Typer typically shows "No such option" for invalid flags
        output_lower = result.output.lower()
        assert "no such option" in output_lower or "error" in output_lower

    def test_unknown_command_fails(self) -> None:
        """td-linter unknowncommand should fail."""
        result = runner.invoke(app, ["unknowncommand"])
        assert result.exit_code != 0
        # Typer shows available commands or error message
        output_lower = result.output.lower()
        assert (
            "no such command" in output_lower
            or "error" in output_lower
            or "usage" in output_lower
        )


class TestHelpOutput:
    """Additional tests for help output content."""

    def test_main_help_shows_all_commands(self) -> None:
        """Main help should list all available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Should mention all subcommands
        output_lower = result.output.lower()
        assert "lint" in output_lower
        assert "rules" in output_lower
        assert "init" in output_lower
        assert "version" in output_lower

    def test_lint_help_shows_options(self) -> None:
        """Lint help should show available options."""
        result = runner.invoke(app, ["lint", "--help"])
        assert result.exit_code == 0
        # Should mention key options
        assert "--config" in result.output or "-c" in result.output
        assert "--format" in result.output or "-f" in result.output
        assert "--quiet" in result.output or "-q" in result.output
        assert "--verbose" in result.output or "-v" in result.output

    def test_init_help_shows_force_option(self) -> None:
        """Init help should show --force option."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output


class TestVersionOutput:
    """Tests for version command output format."""

    def test_version_contains_semver(self) -> None:
        """Version output should contain semantic version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        # Should match pattern like "0.1.0" or similar
        version_pattern = r"\d+\.\d+\.\d+"
        assert re.search(
            version_pattern, result.output
        ), f"No semver found in: {result.output}"


class TestRulesOutput:
    """Tests for rules command output format."""

    def test_rules_shows_table_headers(self) -> None:
        """Rules output should show table with ID, Description, Severity columns."""
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        # Check for table headers (may be styled/formatted)
        assert "id" in output_lower or "rule" in output_lower
        assert "description" in output_lower or "desc" in output_lower
        assert "severity" in output_lower

    def test_rules_shows_registered_rules(self) -> None:
        """Rules output should include all registered rules."""
        result = runner.invoke(app, ["rules"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        # Check for known rule IDs (from linter.py get_all_rules())
        expected_rules = [
            "no-invalid-cycles",
            "no-dangling-inputs",
            "valid-operator-references",
            "type-compatibility",
        ]
        for rule_id in expected_rules:
            assert (
                rule_id in output_lower
            ), f"Expected rule '{rule_id}' not found in output"

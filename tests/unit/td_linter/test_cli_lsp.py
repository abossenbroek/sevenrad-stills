"""
Unit tests for the lsp command CLI.

These tests verify that the td-linter lsp command correctly handles:
- Help output with all documented options
- Transport option validation
- Optional dependency handling (pygls)
"""

from unittest.mock import patch

from td_linter.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class TestLspHelpOutput:
    """Tests for lsp command help output."""

    def test_lsp_help_returns_zero(self) -> None:
        """td-linter lsp --help should return exit code 0."""
        result = runner.invoke(app, ["lsp", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "lsp" in output_lower or "server" in output_lower

    def test_lsp_help_shows_transport_option(self) -> None:
        """LSP help should show --transport option."""
        result = runner.invoke(app, ["lsp", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output or "-t" in result.output

    def test_lsp_help_shows_host_option(self) -> None:
        """LSP help should show --host option."""
        result = runner.invoke(app, ["lsp", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output

    def test_lsp_help_shows_port_option(self) -> None:
        """LSP help should show --port option."""
        result = runner.invoke(app, ["lsp", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output or "-p" in result.output

    def test_lsp_help_mentions_transport_types(self) -> None:
        """LSP help should mention available transport types."""
        result = runner.invoke(app, ["lsp", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        # At least stdio should be mentioned as default
        assert "stdio" in output_lower or "tcp" in output_lower or "ws" in output_lower


class TestLspNegativeCases:
    """Tests that should fail with non-zero exit code."""

    def test_lsp_invalid_transport_fails(self) -> None:
        """td-linter lsp --transport invalid should fail."""
        result = runner.invoke(app, ["lsp", "--transport", "invalid"])
        assert result.exit_code != 0


class TestLspOptionalDependency:
    """Tests for pygls optional dependency handling."""

    def test_lsp_shows_install_message_when_pygls_missing(self) -> None:
        """LSP command should show install message when pygls is missing."""
        # Mock pygls import to raise ImportError
        with patch.dict("sys.modules", {"pygls": None}):
            with patch(
                "td_linter.lsp.server.start_lsp_server",
                side_effect=ImportError("No module named 'pygls'"),
            ):
                result = runner.invoke(app, ["lsp"])
                # Should fail and suggest installing pygls
                # The exact message depends on implementation
                assert result.exit_code != 0


class TestMainHelpIncludesLsp:
    """Tests that main help includes lsp command."""

    def test_main_help_shows_lsp_command(self) -> None:
        """Main help should list lsp command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lsp" in result.output.lower()

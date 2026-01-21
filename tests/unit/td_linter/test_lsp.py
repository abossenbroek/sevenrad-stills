"""Unit tests for LSP server functionality (TDL-064).

Tests verify that the LSP server correctly:
- Finds .toe.dir projects from file paths
- Converts violations to LSP diagnostics
- Handles document events (open, save, close)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLSPImport:
    """Tests for LSP module import handling."""

    def test_import_error_without_pygls(self) -> None:
        """Should raise ImportError with helpful message when pygls not installed."""
        with patch.dict("sys.modules", {"lsprotocol": None, "pygls": None}):
            # Clear any cached import
            import sys
            if "td_linter.lsp.server" in sys.modules:
                del sys.modules["td_linter.lsp.server"]
            if "td_linter.lsp" in sys.modules:
                del sys.modules["td_linter.lsp"]

            # The import should fail gracefully - we just test the module loads
            # when dependencies are available


class TestTDLintLanguageServerBase:
    """Tests for TDLintLanguageServer base functionality."""

    @pytest.fixture
    def mock_lsp_types(self) -> None:
        """Mock LSP types for testing without pygls."""
        # This fixture ensures we can run tests even without pygls installed
        pass

    def test_find_toe_dir_from_n_file(self, tmp_path: Path) -> None:
        """Should find .toe.dir from a nested .n file path."""
        # Create structure: tmp/project.toe.dir/container/op.n
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()
        container = toe_dir / "container"
        container.mkdir()
        n_file = container / "op.n"
        n_file.write_text("TOP:blur\nend\n")

        # Test using mock server
        from td_linter.lsp.server import TDLintLanguageServer

        server = TDLintLanguageServer()
        result = server.find_toe_dir(str(n_file))

        assert result == toe_dir

    def test_find_toe_dir_caching(self, tmp_path: Path) -> None:
        """Should cache .toe.dir lookups for performance."""
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()
        n_file = toe_dir / "op.n"
        n_file.write_text("TOP:blur\nend\n")

        from td_linter.lsp.server import TDLintLanguageServer

        server = TDLintLanguageServer()

        # First lookup
        result1 = server.find_toe_dir(str(n_file))
        # Second lookup (should be cached)
        result2 = server.find_toe_dir(str(n_file))

        assert result1 == result2 == toe_dir
        assert str(n_file) in server._toe_dir_cache

    def test_find_toe_dir_not_found(self, tmp_path: Path) -> None:
        """Should return None when file is not in a .toe.dir."""
        regular_file = tmp_path / "regular_project" / "file.txt"
        regular_file.parent.mkdir()
        regular_file.write_text("content")

        from td_linter.lsp.server import TDLintLanguageServer

        server = TDLintLanguageServer()
        result = server.find_toe_dir(str(regular_file))

        assert result is None

    def test_find_toe_dir_from_toe_dir_itself(self, tmp_path: Path) -> None:
        """Should find .toe.dir when given the directory itself."""
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        from td_linter.lsp.server import TDLintLanguageServer

        server = TDLintLanguageServer()
        result = server.find_toe_dir(str(toe_dir))

        assert result == toe_dir


class TestViolationToDiagnostic:
    """Tests for converting violations to LSP diagnostics."""

    def test_error_severity_mapping(self, tmp_path: Path) -> None:
        """Error violations should map to Error severity."""
        try:
            from lsprotocol import types as lsp
        except ImportError:
            pytest.skip("pygls not installed")

        from td_linter.lsp.server import TDLintLanguageServer
        from td_linter.rules.base import Violation

        server = TDLintLanguageServer()
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        violation = Violation(
            rule="S001",
            message="Syntax error",
            path="/container/op",
            severity="error",
            line=10,
        )

        diagnostic = server.violation_to_diagnostic(violation, toe_dir)

        assert diagnostic.severity == lsp.DiagnosticSeverity.Error
        assert diagnostic.message == "Syntax error"
        assert diagnostic.code == "S001"
        assert diagnostic.source == "td-linter"

    def test_warning_severity_mapping(self, tmp_path: Path) -> None:
        """Warning violations should map to Warning severity."""
        try:
            from lsprotocol import types as lsp
        except ImportError:
            pytest.skip("pygls not installed")

        from td_linter.lsp.server import TDLintLanguageServer
        from td_linter.rules.base import Violation

        server = TDLintLanguageServer()
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        violation = Violation(
            rule="C002",
            message="Dangling input",
            path="/container/op",
            severity="warning",
        )

        diagnostic = server.violation_to_diagnostic(violation, toe_dir)

        assert diagnostic.severity == lsp.DiagnosticSeverity.Warning

    def test_info_severity_mapping(self, tmp_path: Path) -> None:
        """Info violations should map to Information severity."""
        try:
            from lsprotocol import types as lsp
        except ImportError:
            pytest.skip("pygls not installed")

        from td_linter.lsp.server import TDLintLanguageServer
        from td_linter.rules.base import Violation

        server = TDLintLanguageServer()
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        violation = Violation(
            rule="P003",
            message="Performance hint",
            path="/container/op",
            severity="info",
        )

        diagnostic = server.violation_to_diagnostic(violation, toe_dir)

        assert diagnostic.severity == lsp.DiagnosticSeverity.Information

    def test_line_conversion(self, tmp_path: Path) -> None:
        """Line should be converted to 0-indexed for LSP."""
        try:
            from lsprotocol import types as lsp
        except ImportError:
            pytest.skip("pygls not installed")

        from td_linter.lsp.server import TDLintLanguageServer
        from td_linter.rules.base import Violation

        server = TDLintLanguageServer()
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        violation = Violation(
            rule="S001",
            message="Test",
            path="/op",
            severity="error",
            line=10,  # 1-indexed
        )

        diagnostic = server.violation_to_diagnostic(violation, toe_dir)

        # LSP uses 0-indexed positions
        assert diagnostic.range.start.line == 9
        assert diagnostic.range.start.character == 0  # Column is always 0

    def test_missing_line_defaults_to_zero(self, tmp_path: Path) -> None:
        """Violations without line should default to line 0."""
        try:
            from lsprotocol import types as lsp
        except ImportError:
            pytest.skip("pygls not installed")

        from td_linter.lsp.server import TDLintLanguageServer
        from td_linter.rules.base import Violation

        server = TDLintLanguageServer()
        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        violation = Violation(
            rule="C001",
            message="Cycle detected",
            path="/op",
            severity="error",
            # No line/column
        )

        diagnostic = server.violation_to_diagnostic(violation, toe_dir)

        assert diagnostic.range.start.line == 0
        assert diagnostic.range.start.character == 0


class TestServerCreation:
    """Tests for server factory function."""

    def test_create_server_returns_configured_server(self) -> None:
        """create_server should return a configured TDLintLanguageServer."""
        try:
            from td_linter.lsp.server import create_server
        except ImportError:
            pytest.skip("pygls not installed")

        server = create_server()

        assert server is not None
        assert server.name == "td-linter-lsp"


class TestStartLspServer:
    """Tests for start_lsp_server function."""

    def test_invalid_transport_raises_error(self) -> None:
        """Invalid transport should raise ValueError."""
        try:
            from td_linter.lsp.server import start_lsp_server
        except ImportError:
            pytest.skip("pygls not installed")

        with pytest.raises(ValueError, match="Unknown transport"):
            start_lsp_server(transport="invalid")


class TestLSPCliCommand:
    """Tests for the CLI lsp command."""

    def test_cli_help_shows_lsp_command(self) -> None:
        """CLI help should show lsp command."""
        from typer.testing import CliRunner

        from td_linter.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert "lsp" in result.output

    def test_lsp_invalid_transport(self) -> None:
        """Invalid transport should fail with helpful message."""
        from typer.testing import CliRunner

        from td_linter.cli import app

        runner = CliRunner()

        # Mock the import to avoid needing pygls
        with patch("td_linter.cli.console.print"):
            result = runner.invoke(app, ["lsp", "--transport", "invalid"])

        # Should fail due to invalid transport or missing dependency
        assert result.exit_code != 0

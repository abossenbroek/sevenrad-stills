"""LSP server implementation for td-linter.

Provides real-time linting feedback in editors supporting the
Language Server Protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from lsprotocol import types as lsp
    from pygls.lsp.server import LanguageServer
except ImportError as e:
    raise ImportError(
        "LSP dependencies not installed. Install with: pip install td-linter[td-linter-lsp]"
    ) from e

from td_linter.linter import run_lint
from td_linter.rules.base import Violation
from td_linter.rules.registry import RuleRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class TDLintLanguageServer(LanguageServer):
    """Language Server for td-linter.

    Provides diagnostic messages for TouchDesigner .toe.dir projects.
    """

    def __init__(self, name: str = "td-linter-lsp", version: str = "0.1.0") -> None:
        """Initialize the language server."""
        super().__init__(name, version)
        self._registry: RuleRegistry | None = None
        self._toe_dir_cache: dict[str, Path] = {}

    @property
    def registry(self) -> RuleRegistry:
        """Get or create the rule registry."""
        if self._registry is None:
            self._registry = RuleRegistry()
        return self._registry

    def find_toe_dir(self, file_path: str) -> Path | None:
        """Find the .toe.dir containing the given file.

        Args:
            file_path: Path to a file within a .toe.dir project.

        Returns:
            Path to the .toe.dir directory, or None if not found.
        """
        # Check cache first
        if file_path in self._toe_dir_cache:
            return self._toe_dir_cache[file_path]

        path = Path(file_path)

        # Walk up the directory tree looking for .toe.dir
        current = path.parent if path.is_file() else path
        while current != current.parent:
            if current.name.endswith(".toe.dir"):
                self._toe_dir_cache[file_path] = current
                return current
            current = current.parent

        return None

    def lint_toe_dir(self, toe_dir: Path) -> Sequence[Violation]:
        """Lint a .toe.dir project.

        Args:
            toe_dir: Path to the .toe.dir directory.

        Returns:
            List of violations found.
        """
        return run_lint(
            toe_dir,
            validate_expressions=False,
            validate_embedded=True,
            rules=self.registry.enabled(),
            config=self.registry.config,
        )

    def violation_to_diagnostic(
        self, violation: Violation, toe_dir: Path
    ) -> lsp.Diagnostic:
        """Convert a Violation to an LSP Diagnostic.

        Args:
            violation: The linter violation.
            toe_dir: Path to the .toe.dir for resolving file paths.

        Returns:
            LSP Diagnostic object.
        """
        # Map severity to LSP severity
        severity_map = {
            "error": lsp.DiagnosticSeverity.Error,
            "warning": lsp.DiagnosticSeverity.Warning,
            "info": lsp.DiagnosticSeverity.Information,
        }
        lsp_severity = severity_map.get(
            violation.severity, lsp.DiagnosticSeverity.Warning
        )

        # Use line if available, otherwise default to start (column always 0)
        line = max(0, (violation.line or 1) - 1)  # LSP is 0-indexed
        col = 0  # Column info not available in current Violation model

        return lsp.Diagnostic(
            range=lsp.Range(
                start=lsp.Position(line=line, character=col),
                end=lsp.Position(line=line, character=col + 1),
            ),
            message=violation.message,
            severity=lsp_severity,
            source="td-linter",
            code=violation.rule,
        )


def create_server() -> TDLintLanguageServer:
    """Create and configure a TDLintLanguageServer instance.

    Returns:
        Configured language server.
    """
    server = TDLintLanguageServer()

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
        """Handle textDocument/didOpen notification."""
        _lint_document(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
        """Handle textDocument/didSave notification."""
        _lint_document(server, params.text_document.uri)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
        """Handle textDocument/didClose notification."""
        # Clear diagnostics for the closed document
        server.publish_diagnostics(params.text_document.uri, [])

    @server.feature(lsp.INITIALIZE)
    def initialize(params: lsp.InitializeParams) -> lsp.InitializeResult:
        """Handle initialize request."""
        logger.info("td-linter LSP server initializing")
        return lsp.InitializeResult(
            capabilities=lsp.ServerCapabilities(
                text_document_sync=lsp.TextDocumentSyncOptions(
                    open_close=True,
                    change=lsp.TextDocumentSyncKind.None_,  # No incremental changes
                    save=lsp.SaveOptions(include_text=False),
                ),
            ),
            server_info=lsp.ServerInfo(
                name="td-linter-lsp",
                version="0.1.0",
            ),
        )

    return server


def _lint_document(server: TDLintLanguageServer, uri: str) -> None:
    """Lint the document at the given URI and publish diagnostics.

    Args:
        server: The language server instance.
        uri: The document URI to lint.
    """
    # Convert URI to file path
    if uri.startswith("file://"):
        file_path = uri[7:]
    else:
        file_path = uri

    # Find the .toe.dir containing this file
    toe_dir = server.find_toe_dir(file_path)
    if toe_dir is None:
        logger.debug(f"File {file_path} is not in a .toe.dir project")
        return

    # Lint the project
    try:
        violations = server.lint_toe_dir(toe_dir)
    except Exception as e:
        logger.error(f"Error linting {toe_dir}: {e}")
        return

    # Group violations by file
    diagnostics_by_file: dict[str, list[lsp.Diagnostic]] = {}

    for violation in violations:
        # Resolve violation path to full file path
        if violation.source_file:
            viol_path = str(violation.source_file)
        else:
            # Use operator path to derive file location
            viol_path = str(toe_dir / violation.path.lstrip("/")) + ".n"

        if viol_path not in diagnostics_by_file:
            diagnostics_by_file[viol_path] = []

        diagnostics_by_file[viol_path].append(
            server.violation_to_diagnostic(violation, toe_dir)
        )

    # Publish diagnostics for each file
    for viol_file_path, diagnostics in diagnostics_by_file.items():
        viol_uri = f"file://{viol_file_path}"
        server.publish_diagnostics(viol_uri, diagnostics)

    # Clear diagnostics for the requested file if no violations
    if file_path not in diagnostics_by_file:
        server.publish_diagnostics(uri, [])


def start_lsp_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 2087,
) -> None:
    """Start the LSP server.

    Args:
        transport: Transport method ('stdio', 'tcp', 'ws').
        host: Host address for TCP/WebSocket transport.
        port: Port number for TCP/WebSocket transport.
    """
    server = create_server()

    if transport == "stdio":
        logger.info("Starting td-linter LSP server on stdio")
        server.start_io()
    elif transport == "tcp":
        logger.info(f"Starting td-linter LSP server on tcp://{host}:{port}")
        server.start_tcp(host, port)
    elif transport == "ws":
        logger.info(f"Starting td-linter LSP server on ws://{host}:{port}")
        server.start_ws(host, port)
    else:
        msg = f"Unknown transport: {transport}"
        raise ValueError(msg)

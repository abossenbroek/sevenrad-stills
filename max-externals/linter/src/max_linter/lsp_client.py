"""LSP client for communicating with language servers."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Any

from pylsp_jsonrpc.streams import JsonRpcStreamReader, JsonRpcStreamWriter

from max_linter.results import Diagnostic

logger = logging.getLogger(__name__)


class LSPError(Exception):
    """Error from LSP communication."""

    pass


class LSPClient:
    """Generic LSP client using python-lsp-jsonrpc.

    Communicates with language servers over stdin/stdout using JSON-RPC.
    Supports the core LSP lifecycle: initialize, open document, get diagnostics,
    and shutdown.
    """

    def __init__(self, command: list[str], source_name: str = "lsp"):
        """Initialize LSP client.

        Args:
            command: Command to launch the language server
            source_name: Name to use as diagnostic source
        """
        self.command = command
        self.source_name = source_name
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: JsonRpcStreamReader | None = None
        self.writer: JsonRpcStreamWriter | None = None
        self.request_id = 0
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._response_events: dict[int, threading.Event] = {}
        self._responses: dict[int, Any] = {}
        self._reader_thread: threading.Thread | None = None
        self._running = False

    def _check_server_available(self) -> bool:
        """Check if the language server is available."""
        try:
            result = subprocess.run(
                [self.command[0], "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def start(self) -> bool:
        """Start the language server process.

        Returns:
            True if server started successfully, False otherwise
        """
        if not self._check_server_available():
            logger.warning(f"Language server not available: {self.command[0]}")
            return False

        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.reader = JsonRpcStreamReader(self.process.stdout)
            self.writer = JsonRpcStreamWriter(self.process.stdin)
            self._running = True

            # Start reader thread for async message handling
            self._reader_thread = threading.Thread(
                target=self._read_messages, daemon=True
            )
            self._reader_thread.start()

            return True
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to start language server: {e}")
            return False

    def _read_messages(self) -> None:
        """Read messages from the language server in a background thread.

        Uses the callback-based listen() method from python-lsp-jsonrpc.
        """
        if not self.reader:
            return

        def message_consumer(message: dict[str, Any]) -> None:
            """Callback that processes each incoming message."""
            if self._running:
                self._handle_message(message)

        try:
            # listen() is blocking and calls message_consumer for each message
            self.reader.listen(message_consumer)
        except Exception as e:
            if self._running:
                logger.debug(f"Error in message listener: {e}")

    def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming message from the server."""
        if "id" in message and ("result" in message or "error" in message):
            # Response to a request
            request_id = message["id"]
            if "error" in message:
                self._responses[request_id] = {"error": message["error"]}
            else:
                self._responses[request_id] = message.get("result")
            if request_id in self._response_events:
                self._response_events[request_id].set()
        elif "method" in message:
            # Notification from server
            method = message["method"]
            params = message.get("params", {})

            if method == "textDocument/publishDiagnostics":
                uri = params.get("uri", "")
                diagnostics = params.get("diagnostics", [])
                self._diagnostics[uri] = diagnostics
                logger.debug(f"Received {len(diagnostics)} diagnostics for {uri}")

    def _request(self, method: str, params: dict[str, Any] | None) -> Any:
        """Send a request and wait for response."""
        if not self.writer:
            raise LSPError("LSP client not started")

        self.request_id += 1
        request_id = self.request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        event = threading.Event()
        self._response_events[request_id] = event

        self.writer.write(message)

        # Wait for response with timeout
        if not event.wait(timeout=30):
            raise LSPError(f"Timeout waiting for response to {method}")

        return self._responses.get(request_id)

    def _notify(self, method: str, params: dict[str, Any] | None) -> None:
        """Send a notification (no response expected)."""
        if not self.writer:
            raise LSPError("LSP client not started")

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }

        self.writer.write(message)

    def initialize(self, root_uri: str) -> dict[str, Any]:
        """Send LSP initialize request.

        Args:
            root_uri: URI of the workspace root

        Returns:
            Server capabilities
        """
        result = self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {
                            "relatedInformation": True,
                        },
                    },
                },
            },
        )

        # Send initialized notification
        self._notify("initialized", {})

        return result or {}

    def open_document(self, uri: str, text: str, language_id: str) -> None:
        """Notify server of document open.

        Args:
            uri: Document URI
            text: Document content
            language_id: Language identifier (e.g., "c", "glsl")
        """
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def close_document(self, uri: str) -> None:
        """Notify server of document close."""
        self._notify(
            "textDocument/didClose",
            {
                "textDocument": {
                    "uri": uri,
                }
            },
        )

    def get_diagnostics(self, uri: str, timeout: float = 5.0) -> list[Diagnostic]:
        """Wait for and return diagnostics for a document.

        Args:
            uri: Document URI
            timeout: Maximum time to wait for diagnostics

        Returns:
            List of diagnostics
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            if uri in self._diagnostics:
                raw_diagnostics = self._diagnostics[uri]
                return [
                    Diagnostic.from_lsp(d, self.source_name) for d in raw_diagnostics
                ]
            time.sleep(0.1)

        # Return empty list if no diagnostics received
        return []

    def shutdown(self) -> None:
        """Graceful LSP shutdown."""
        self._running = False

        # Send shutdown request if writer is available
        if self.writer:
            try:
                # Use a short timeout for shutdown request
                self.request_id += 1
                request_id = self.request_id
                event = threading.Event()
                self._response_events[request_id] = event
                self.writer.write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "shutdown",
                        "params": None,
                    }
                )
                event.wait(timeout=2)  # Short timeout for shutdown
                self.writer.write(
                    {
                        "jsonrpc": "2.0",
                        "method": "exit",
                        "params": None,
                    }
                )
            except Exception:
                pass

        # Terminate process to unblock the reader thread
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
            except Exception:
                pass

        # Close reader after process terminated
        if self.reader:
            import contextlib

            with contextlib.suppress(Exception):
                self.reader.close()

        self.process = None
        self.reader = None
        self.writer = None

    def __enter__(self) -> LSPClient:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()

"""GLSL validator using glsl_analyzer LSP server."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from max_linter.lsp_client import LSPClient
from max_linter.results import Diagnostic, DiagnosticSeverity, Position, Range

logger = logging.getLogger(__name__)


class GLSLValidator:
    """Validates GLSL code using glsl_analyzer LSP.

    glsl_analyzer is a GLSL language server that provides diagnostics,
    code completion, and other features for GLSL shaders.

    Source: https://github.com/nolanderc/glsl_analyzer
    """

    COMMAND = ["glsl_analyzer", "--stdio"]

    def __init__(self) -> None:
        """Initialize GLSL validator."""
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if glsl_analyzer is installed and available."""
        if self._available is None:
            import subprocess

            try:
                result = subprocess.run(
                    ["glsl_analyzer", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                self._available = False
                logger.warning(
                    "glsl_analyzer not found. Install with: brew install glsl_analyzer"
                )

        return self._available

    def validate(self, code: str, filename: str = "shader.frag") -> list[Diagnostic]:
        """Validate GLSL code using glsl_analyzer.

        Args:
            code: GLSL shader code
            filename: Virtual filename (extension determines shader type)

        Returns:
            List of diagnostics from the language server
        """
        if not self.is_available():
            return [
                Diagnostic(
                    range=_zero_range(),
                    severity=DiagnosticSeverity.WARNING,
                    message="glsl_analyzer not available - skipping GLSL validation",
                    source="max-linter",
                )
            ]

        # Create temp file for glsl_analyzer (it needs real files)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            filepath.write_text(code)
            file_uri = f"file://{filepath}"

            client = LSPClient(self.COMMAND, source_name="glsl_analyzer")

            try:
                if not client.start():
                    return [
                        Diagnostic(
                            range=_zero_range(),
                            severity=DiagnosticSeverity.WARNING,
                            message="Failed to start glsl_analyzer",
                            source="max-linter",
                        )
                    ]

                client.initialize(f"file://{tmpdir}")
                client.open_document(file_uri, code, "glsl")

                # Wait for diagnostics
                diagnostics = client.get_diagnostics(file_uri, timeout=10.0)

                client.close_document(file_uri)
                return diagnostics

            except Exception as e:
                logger.error(f"GLSL validation error: {e}")
                return [
                    Diagnostic(
                        range=_zero_range(),
                        severity=DiagnosticSeverity.ERROR,
                        message=f"GLSL validation failed: {e}",
                        source="max-linter",
                    )
                ]
            finally:
                client.shutdown()

    def validate_file(self, filepath: Path) -> list[Diagnostic]:
        """Validate a GLSL file.

        Args:
            filepath: Path to the GLSL file

        Returns:
            List of diagnostics
        """
        code = filepath.read_text()
        return self.validate(code, filepath.name)


def _zero_range() -> Range:
    """Create a zero range for synthetic diagnostics."""
    zero_pos = Position(line=0, character=0)
    return Range(start=zero_pos, end=zero_pos)

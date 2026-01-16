"""LSP server for td-linter.

Provides Language Server Protocol integration for real-time linting
in code editors.

This module requires the 'pygls' optional dependency:
    pip install td-linter[td-linter-lsp]
"""

from td_linter.lsp.server import TDLintLanguageServer, start_lsp_server

__all__ = ["TDLintLanguageServer", "start_lsp_server"]

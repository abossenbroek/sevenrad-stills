"""LSP server for td-linter.

Provides Language Server Protocol integration for real-time linting
in code editors.

This module requires the 'pygls' optional dependency:
    pip install td-linter[td-linter-lsp]
"""

# uri_utils is always available (no external deps)
from td_linter.lsp.uri_utils import (
    InvalidURIError,
    is_file_uri,
    normalize_uri,
    path_to_uri,
    uri_to_path,
)

__all__ = [
    "InvalidURIError",
    "is_file_uri",
    "normalize_uri",
    "path_to_uri",
    "uri_to_path",
]

# Server components require pygls - import lazily
try:
    from td_linter.lsp.server import TDLintLanguageServer, start_lsp_server

    __all__.extend(["TDLintLanguageServer", "start_lsp_server"])
except ImportError:
    # pygls not installed - LSP server not available
    pass

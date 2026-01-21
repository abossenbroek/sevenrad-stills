"""URI parsing utilities for the LSP server.

This module provides secure URI-to-file-path conversion that properly
handles URL encoding, Windows paths, and validates URI structure.
"""

from __future__ import annotations

import platform
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


class InvalidURIError(Exception):
    """Raised when a URI is malformed or has an unsupported scheme."""

    def __init__(self, uri: str, reason: str) -> None:
        self.uri = uri
        self.reason = reason
        super().__init__(f"Invalid URI '{uri}': {reason}")


def uri_to_path(uri: str) -> Path:
    """Convert a file:// URI to a filesystem path.

    This function properly handles:
    - URL encoding (e.g., %20 for spaces, %2F for slashes)
    - Windows drive letters (file:///C:/path)
    - UNC paths on Windows (file://server/share)
    - Unix absolute paths (file:///path)

    Args:
        uri: A file:// URI string.

    Returns:
        Path object representing the local filesystem path.

    Raises:
        InvalidURIError: If the URI is malformed or not a file:// URI.

    Examples:
        >>> uri_to_path("file:///home/user/file.txt")
        PosixPath('/home/user/file.txt')

        >>> uri_to_path("file:///C:/Users/name/file.txt")  # Windows
        WindowsPath('C:/Users/name/file.txt')

        >>> uri_to_path("file:///path%20with%20spaces/file.txt")
        PosixPath('/path with spaces/file.txt')
    """
    if not uri:
        raise InvalidURIError(uri, "URI is empty")

    # Parse the URI
    parsed = urlparse(uri)

    # Validate scheme
    if parsed.scheme != "file":
        raise InvalidURIError(
            uri,
            f"Expected 'file' scheme, got '{parsed.scheme}'"
        )

    # URL decode the path component
    path_str = unquote(parsed.path)

    # Handle Windows-specific cases
    if platform.system() == "Windows":
        # Windows drive letters: file:///C:/path -> C:/path
        # The path will be /C:/path, we need to strip the leading /
        if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
            path_str = path_str[1:]

        # UNC paths: file://server/share -> \\server\share
        if parsed.netloc:
            path_str = f"\\\\{parsed.netloc}{path_str.replace('/', '\\')}"
    else:
        # Unix: file:///path -> /path
        # netloc should be empty for local files
        if parsed.netloc and parsed.netloc != "localhost":
            raise InvalidURIError(
                uri,
                f"Non-local URIs not supported on Unix: netloc='{parsed.netloc}'"
            )

    # Validate the path is not empty after processing
    if not path_str or path_str == "/":
        raise InvalidURIError(uri, "Path component is empty")

    return Path(path_str)


def path_to_uri(path: Path) -> str:
    """Convert a filesystem path to a file:// URI.

    This function properly handles:
    - Special characters (spaces, unicode, etc.)
    - Windows drive letters
    - Absolute and relative paths (relative paths are made absolute)

    Args:
        path: A filesystem path.

    Returns:
        A properly encoded file:// URI string.

    Examples:
        >>> path_to_uri(Path("/home/user/file.txt"))
        'file:///home/user/file.txt'

        >>> path_to_uri(Path("/path with spaces/file.txt"))
        'file:///path%20with%20spaces/file.txt'
    """
    # Resolve to absolute path
    abs_path = path.resolve()

    # Convert to URI string
    path_str = str(abs_path)

    # On Windows, convert backslashes to forward slashes
    if platform.system() == "Windows":
        path_str = path_str.replace("\\", "/")
        # Add leading slash for drive letters
        if not path_str.startswith("/"):
            path_str = "/" + path_str

    # URL encode special characters (but not slashes or colons)
    # Only encode truly special characters
    encoded = ""
    for char in path_str:
        if char in "/:@":
            encoded += char
        elif char.isalnum() or char in "-_.~":
            encoded += char
        else:
            # Encode as UTF-8 bytes
            for byte in char.encode("utf-8"):
                encoded += f"%{byte:02X}"

    return f"file://{encoded}"


def is_file_uri(uri: str) -> bool:
    """Check if a string is a valid file:// URI.

    Args:
        uri: String to check.

    Returns:
        True if the string is a file:// URI, False otherwise.
    """
    try:
        parsed = urlparse(uri)
        return parsed.scheme == "file"
    except Exception:
        return False


def normalize_uri(uri: str) -> str:
    """Normalize a file:// URI to canonical form.

    This ensures consistent URI representation by:
    - Decoding and re-encoding the path
    - Normalizing path separators
    - Handling case sensitivity appropriately

    Args:
        uri: A file:// URI string.

    Returns:
        Normalized URI string.

    Raises:
        InvalidURIError: If the URI is malformed.
    """
    # Convert to path and back to normalize
    path = uri_to_path(uri)
    return path_to_uri(path)

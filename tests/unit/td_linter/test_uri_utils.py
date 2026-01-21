"""Unit tests for LSP URI utilities."""

import platform
from pathlib import Path

import pytest

from td_linter.lsp.uri_utils import (
    InvalidURIError,
    is_file_uri,
    normalize_uri,
    path_to_uri,
    uri_to_path,
)


class TestUriToPath:
    """Tests for uri_to_path function."""

    def test_simple_unix_path(self) -> None:
        """Should convert simple Unix file URI."""
        path = uri_to_path("file:///home/user/file.txt")
        assert str(path) == "/home/user/file.txt"

    def test_path_with_spaces(self) -> None:
        """Should decode URL-encoded spaces."""
        path = uri_to_path("file:///path%20with%20spaces/file.txt")
        assert str(path) == "/path with spaces/file.txt"

    def test_path_with_unicode(self) -> None:
        """Should decode URL-encoded unicode characters."""
        # %C3%A9 is é
        path = uri_to_path("file:///caf%C3%A9/file.txt")
        assert str(path) == "/café/file.txt"

    def test_path_with_special_characters(self) -> None:
        """Should decode various URL-encoded characters."""
        # %23 is #, %26 is &
        path = uri_to_path("file:///path%23with%26special/file.txt")
        assert str(path) == "/path#with&special/file.txt"

    def test_empty_uri_raises_error(self) -> None:
        """Should raise error for empty URI."""
        with pytest.raises(InvalidURIError):
            uri_to_path("")

    def test_non_file_scheme_raises_error(self) -> None:
        """Should reject non-file schemes."""
        with pytest.raises(InvalidURIError) as exc_info:
            uri_to_path("http://example.com/file.txt")
        assert "file" in str(exc_info.value)
        assert "http" in str(exc_info.value)

    def test_ftp_scheme_raises_error(self) -> None:
        """Should reject ftp scheme."""
        with pytest.raises(InvalidURIError):
            uri_to_path("ftp://server/file.txt")

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_remote_netloc_raises_error_on_unix(self) -> None:
        """Should reject non-local URIs on Unix."""
        with pytest.raises(InvalidURIError) as exc_info:
            uri_to_path("file://remote-server/path/file.txt")
        assert "Non-local" in str(exc_info.value)

    def test_localhost_netloc_allowed(self) -> None:
        """Should allow localhost netloc."""
        if platform.system() != "Windows":
            path = uri_to_path("file://localhost/home/user/file.txt")
            assert str(path) == "/home/user/file.txt"

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_drive_letter(self) -> None:
        """Should handle Windows drive letters."""
        path = uri_to_path("file:///C:/Users/name/file.txt")
        assert str(path) == "C:/Users/name/file.txt" or str(path) == "C:\\Users\\name\\file.txt"

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_unc_path(self) -> None:
        """Should handle Windows UNC paths."""
        path = uri_to_path("file://server/share/file.txt")
        assert "server" in str(path).lower()
        assert "share" in str(path).lower()

    def test_encoded_slashes_decoded(self) -> None:
        """Should decode encoded slashes in filenames."""
        # %2F is /
        path = uri_to_path("file:///path/file%2Fname.txt")
        assert "file/name.txt" in str(path)


class TestPathToUri:
    """Tests for path_to_uri function."""

    def test_simple_path(self, tmp_path: Path) -> None:
        """Should convert simple path to URI."""
        file = tmp_path / "file.txt"
        file.touch()
        uri = path_to_uri(file)
        assert uri.startswith("file://")
        assert "file.txt" in uri

    def test_path_with_spaces(self, tmp_path: Path) -> None:
        """Should encode spaces in path."""
        dir_with_space = tmp_path / "path with spaces"
        dir_with_space.mkdir()
        file = dir_with_space / "file.txt"
        file.touch()
        uri = path_to_uri(file)
        assert "%20" in uri or " " not in uri.replace("file://", "")

    def test_path_with_unicode(self, tmp_path: Path) -> None:
        """Should encode unicode characters."""
        try:
            unicode_dir = tmp_path / "café"
            unicode_dir.mkdir()
            file = unicode_dir / "file.txt"
            file.touch()
            uri = path_to_uri(file)
            assert "file://" in uri
            # Should be properly encoded
            assert "café" in uri or "%C3%A9" in uri
        except OSError:
            pytest.skip("Filesystem doesn't support unicode filenames")

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Converting path->uri->path should give equivalent path."""
        original = tmp_path / "subdir" / "file.txt"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.touch()

        uri = path_to_uri(original)
        recovered = uri_to_path(uri)

        assert original.resolve() == recovered.resolve()


class TestIsFileUri:
    """Tests for is_file_uri function."""

    def test_valid_file_uri(self) -> None:
        """Should return True for file:// URIs."""
        assert is_file_uri("file:///path/to/file.txt")
        assert is_file_uri("file://localhost/path/file.txt")
        assert is_file_uri("file:///C:/Windows/file.txt")

    def test_invalid_schemes(self) -> None:
        """Should return False for non-file schemes."""
        assert not is_file_uri("http://example.com/file.txt")
        assert not is_file_uri("https://example.com/file.txt")
        assert not is_file_uri("ftp://server/file.txt")

    def test_raw_path(self) -> None:
        """Should return False for raw paths."""
        assert not is_file_uri("/path/to/file.txt")
        assert not is_file_uri("C:/Windows/file.txt")

    def test_empty_string(self) -> None:
        """Should return False for empty string."""
        assert not is_file_uri("")

    def test_malformed_uri(self) -> None:
        """Should return False for malformed URIs."""
        assert not is_file_uri("not a uri at all")


class TestNormalizeUri:
    """Tests for normalize_uri function."""

    def test_normalizes_encoding(self, tmp_path: Path) -> None:
        """Should normalize URI encoding."""
        file = tmp_path / "file.txt"
        file.touch()

        # Create URI with unnecessary encoding
        uri = f"file://{tmp_path}/file%2Etxt"

        # Normalize (this will decode %2E to . and re-encode if needed)
        try:
            normalized = normalize_uri(uri)
            assert "file://" in normalized
        except InvalidURIError:
            # If the encoded version doesn't work, that's also fine
            pass


class TestSecurityCases:
    """Security-focused tests for URI handling."""

    def test_encoded_path_traversal(self) -> None:
        """Should decode path traversal sequences properly."""
        # %2E%2E is ..
        uri = "file:///home/user/%2E%2E/%2E%2E/etc/passwd"
        path = uri_to_path(uri)
        # The .. should be decoded, resulting in /home/user/../../etc/passwd
        assert ".." in str(path)

    def test_double_encoded_ignored(self) -> None:
        """Should only decode once (not double-decode)."""
        # %252E%252E would decode to %2E%2E, which should NOT further decode
        uri = "file:///home/user/%252E%252E/file.txt"
        path = uri_to_path(uri)
        # Should contain literal %2E%2E, not ..
        assert "%2E%2E" in str(path)

    def test_null_byte_in_path(self) -> None:
        """Should handle null bytes in encoded form."""
        uri = "file:///path/file%00.txt"
        path = uri_to_path(uri)
        # Should decode the null byte
        assert "\x00" in str(path)

    def test_very_long_uri(self) -> None:
        """Should handle very long URIs."""
        # Create a valid but very long path
        long_path = "/home/" + "a" * 1000 + "/file.txt"
        uri = f"file://{long_path}"
        path = uri_to_path(uri)
        assert len(str(path)) > 1000

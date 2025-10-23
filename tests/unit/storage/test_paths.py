"""Tests for storage.paths module."""

from pathlib import Path

import pytest
from sevenrad_stills.storage.paths import (
    MAX_FILENAME_LENGTH,
    ensure_directory,
    format_frame_filename,
    get_incremental_path,
    sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_sanitization(self) -> None:
        """Test removal of invalid characters."""
        assert sanitize_filename("test<file>name") == "test_file_name"
        assert sanitize_filename('test:file"name') == "test_file_name"
        assert sanitize_filename("test|file?name") == "test_file_name"

    def test_control_characters(self) -> None:
        """Test removal of control characters."""
        assert sanitize_filename("test\x00file") == "testfile"
        assert sanitize_filename("test\x1fname") == "testname"

    def test_multiple_underscores(self) -> None:
        """Test collapsing of multiple underscores."""
        assert sanitize_filename("test___file___name") == "test_file_name"

    def test_leading_trailing_chars(self) -> None:
        """Test removal of leading/trailing spaces and dots."""
        assert sanitize_filename("  test.txt  ") == "test.txt"
        assert sanitize_filename("..test.txt..") == "test.txt"

    def test_length_limit(self) -> None:
        """Test filename length limiting."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) == MAX_FILENAME_LENGTH

    def test_length_limit_with_extension(self) -> None:
        """Test filename length limiting preserves extension."""
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= MAX_FILENAME_LENGTH
        assert result.endswith(".txt")

    def test_empty_filename(self) -> None:
        """Test empty filename fallback."""
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"
        assert sanitize_filename("...") == "unnamed"


class TestGetIncrementalPath:
    """Tests for get_incremental_path function."""

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        """Test path that doesn't exist returns as-is."""
        base = tmp_path / "test.txt"
        result = get_incremental_path(base)
        assert result == base

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test incremental naming for existing file."""
        base = tmp_path / "test.txt"
        base.touch()
        result = get_incremental_path(base)
        assert result == tmp_path / "test_000.txt"

    def test_multiple_existing_files(self, tmp_path: Path) -> None:
        """Test incremental naming skips existing numbered files."""
        base = tmp_path / "test.txt"
        base.touch()
        (tmp_path / "test_000.txt").touch()
        (tmp_path / "test_001.txt").touch()
        result = get_incremental_path(base)
        assert result == tmp_path / "test_002.txt"

    def test_existing_directory(self, tmp_path: Path) -> None:
        """Test incremental naming for directory."""
        base = tmp_path / "testdir"
        base.mkdir()
        result = get_incremental_path(base)
        assert result == tmp_path / "testdir_000"

    def test_start_index(self, tmp_path: Path) -> None:
        """Test custom start index."""
        base = tmp_path / "test.txt"
        base.touch()
        result = get_incremental_path(base, start_index=5)
        assert result == tmp_path / "test_005.txt"


class TestFormatFrameFilename:
    """Tests for format_frame_filename function."""

    def test_basic_formatting(self) -> None:
        """Test basic frame filename formatting."""
        result = format_frame_filename(
            pattern="{video_id}_frame_{index:06d}.{ext}",
            video_id="abc123",
            index=42,
            ext="jpg",
        )
        assert result == "abc123_frame_000042.jpg"

    def test_sanitizes_video_id(self) -> None:
        """Test that video_id is sanitized."""
        result = format_frame_filename(
            pattern="{video_id}_frame_{index:06d}.{ext}",
            video_id="test<>video",
            index=1,
            ext="jpg",
        )
        # Invalid characters are replaced with underscores and collapsed
        assert "test_video" in result
        assert "<" not in result
        assert ">" not in result

    def test_with_timestamp(self) -> None:
        """Test formatting with timestamp."""
        result = format_frame_filename(
            pattern="{video_id}_{timestamp:.2f}_{index}.{ext}",
            video_id="abc123",
            index=1,
            timestamp=12.5,
            ext="jpg",
        )
        assert "12.50" in result

    def test_fallback_on_invalid_pattern(self) -> None:
        """Test fallback when pattern fails."""
        result = format_frame_filename(
            pattern="{invalid_placeholder}",
            video_id="abc123",
            index=1,
            ext="jpg",
        )
        # Should fall back to default pattern
        assert "abc123" in result
        assert "000001" in result
        assert result.endswith(".jpg")


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Test directory creation."""
        test_dir = tmp_path / "new_dir"
        result = ensure_directory(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test nested directory creation."""
        test_dir = tmp_path / "a" / "b" / "c"
        result = ensure_directory(test_dir)
        assert test_dir.exists()
        assert result == test_dir

    def test_existing_directory(self, tmp_path: Path) -> None:
        """Test with existing directory."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()
        result = ensure_directory(test_dir)
        assert result == test_dir

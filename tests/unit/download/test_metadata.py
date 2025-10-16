"""Tests for download.metadata module."""

from pathlib import Path

import pytest
from sevenrad_stills.download.metadata import VideoInfo


class TestVideoInfo:
    """Tests for VideoInfo model."""

    def test_basic_creation(self, tmp_path: Path) -> None:
        """Test creating VideoInfo with basic fields."""
        file_path = tmp_path / "test.mp4"
        video = VideoInfo(
            video_id="abc123",
            title="Test Video",
            duration=120.5,
            format="mp4",
            file_path=file_path,
        )
        assert video.video_id == "abc123"
        assert video.title == "Test Video"
        assert video.duration == 120.5
        assert video.format == "mp4"
        assert video.file_path == file_path

    def test_with_dimensions(self, tmp_path: Path) -> None:
        """Test VideoInfo with width and height."""
        video = VideoInfo(
            video_id="abc123",
            title="Test Video",
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video.width == 1920
        assert video.height == 1080
        assert video.fps == 30.0

    def test_resolution_property(self, tmp_path: Path) -> None:
        """Test resolution property formatting."""
        video = VideoInfo(
            video_id="abc123",
            title="Test",
            duration=60.0,
            width=1920,
            height=1080,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video.resolution == "1920x1080"

    def test_resolution_none_when_missing_dimensions(self, tmp_path: Path) -> None:
        """Test resolution is None when dimensions missing."""
        video = VideoInfo(
            video_id="abc123",
            title="Test",
            duration=60.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video.resolution is None

        video_partial = VideoInfo(
            video_id="abc123",
            title="Test",
            duration=60.0,
            width=1920,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video_partial.resolution is None

    def test_duration_str_property(self, tmp_path: Path) -> None:
        """Test duration_str property formatting."""
        # Less than a minute
        video1 = VideoInfo(
            video_id="abc",
            title="Test",
            duration=45.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video1.duration_str == "00:00:45"

        # Over a minute
        video2 = VideoInfo(
            video_id="abc",
            title="Test",
            duration=125.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video2.duration_str == "00:02:05"

        # Over an hour
        video3 = VideoInfo(
            video_id="abc",
            title="Test",
            duration=3665.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
        )
        assert video3.duration_str == "01:01:05"

    def test_optional_fields(self, tmp_path: Path) -> None:
        """Test optional fields can be None."""
        video = VideoInfo(
            video_id="abc123",
            title="Test",
            duration=60.0,
            format="mp4",
            file_path=tmp_path / "test.mp4",
            thumbnail_url=None,
            uploader=None,
            upload_date=None,
        )
        assert video.thumbnail_url is None
        assert video.uploader is None
        assert video.upload_date is None

    def test_with_all_fields(self, tmp_path: Path) -> None:
        """Test VideoInfo with all fields populated."""
        video = VideoInfo(
            video_id="abc123",
            title="Test Video",
            duration=120.5,
            width=1920,
            height=1080,
            fps=30.0,
            format="mp4",
            file_size=1024000,
            file_path=tmp_path / "test.mp4",
            thumbnail_url="https://example.com/thumb.jpg",
            uploader="Test Channel",
            upload_date="20240101",
        )
        assert video.format == "mp4"
        assert video.file_size == 1024000
        assert video.thumbnail_url == "https://example.com/thumb.jpg"
        assert video.uploader == "Test Channel"
        assert video.upload_date == "20240101"

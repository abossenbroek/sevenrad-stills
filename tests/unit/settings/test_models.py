"""Tests for settings.models module."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from sevenrad_stills.settings.models import (
    AppSettings,
    CacheSettings,
    ClearStrategy,
    DownloadSettings,
    ExtractionSettings,
    LoggingSettings,
)


class TestClearStrategy:
    """Tests for ClearStrategy enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert ClearStrategy.ON_START.value == "on_start"
        assert ClearStrategy.ON_EXIT.value == "on_exit"
        assert ClearStrategy.MANUAL.value == "manual"
        assert ClearStrategy.NEVER.value == "never"


class TestCacheSettings:
    """Tests for CacheSettings model."""

    def test_default_values(self) -> None:
        """Test default cache settings."""
        settings = CacheSettings()
        assert settings.cache_dir == Path.home() / ".cache" / "sevenrad_stills"
        assert settings.clear_strategy == ClearStrategy.ON_EXIT
        assert settings.max_size_gb == 10.0
        assert settings.retention_days == 7

    def test_custom_values(self) -> None:
        """Test custom cache settings."""
        settings = CacheSettings(
            cache_dir="/tmp/cache",
            clear_strategy=ClearStrategy.MANUAL,
            max_size_gb=20.0,
            retention_days=14,
        )
        assert settings.cache_dir == Path("/tmp/cache").resolve()
        assert settings.clear_strategy == ClearStrategy.MANUAL
        assert settings.max_size_gb == 20.0
        assert settings.retention_days == 14

    def test_expands_user_dir(self) -> None:
        """Test that ~ is expanded in cache_dir."""
        settings = CacheSettings(cache_dir="~/mycache")
        assert "~" not in str(settings.cache_dir)
        assert settings.cache_dir.is_absolute()

    def test_validation_max_size(self) -> None:
        """Test validation of max_size_gb."""
        with pytest.raises(ValidationError):
            CacheSettings(max_size_gb=0)
        with pytest.raises(ValidationError):
            CacheSettings(max_size_gb=-1)

    def test_validation_retention_days(self) -> None:
        """Test validation of retention_days."""
        with pytest.raises(ValidationError):
            CacheSettings(retention_days=0)
        with pytest.raises(ValidationError):
            CacheSettings(retention_days=-1)


class TestDownloadSettings:
    """Tests for DownloadSettings model."""

    def test_default_values(self) -> None:
        """Test default download settings."""
        settings = DownloadSettings()
        assert settings.format == "mp4"
        assert settings.quality == "best"
        assert settings.output_dir.name == "videos"
        assert settings.temp_dir is None

    def test_custom_values(self) -> None:
        """Test custom download settings."""
        settings = DownloadSettings(
            format="webm",
            quality="720p",
            output_dir="/tmp/videos",
            temp_dir="/tmp/temp",
        )
        assert settings.format == "webm"
        assert settings.quality == "720p"
        assert settings.output_dir.name == "videos"
        assert "/tmp" in str(settings.output_dir)
        assert settings.temp_dir.name == "temp"
        assert "/tmp" in str(settings.temp_dir)

    def test_valid_formats(self) -> None:
        """Test valid format values."""
        for fmt in ["mp4", "webm", "mkv"]:
            settings = DownloadSettings(format=fmt)
            assert settings.format == fmt

    def test_invalid_format(self) -> None:
        """Test invalid format raises error."""
        with pytest.raises(ValidationError):
            DownloadSettings(format="avi")


class TestExtractionSettings:
    """Tests for ExtractionSettings model."""

    def test_default_values(self) -> None:
        """Test default extraction settings."""
        settings = ExtractionSettings()
        assert settings.fps == 1.0
        assert settings.frame_interval is None
        assert settings.output_format == "jpg"
        assert settings.jpeg_quality == 95

    def test_fps_mode(self) -> None:
        """Test FPS extraction mode."""
        settings = ExtractionSettings(fps=2.0, frame_interval=None)
        assert settings.fps == 2.0
        assert settings.frame_interval is None

    def test_interval_mode(self) -> None:
        """Test interval extraction mode."""
        settings = ExtractionSettings(fps=None, frame_interval=30)
        assert settings.fps is None
        assert settings.frame_interval == 30

    def test_validation_fps_positive(self) -> None:
        """Test fps must be positive."""
        with pytest.raises(ValidationError):
            ExtractionSettings(fps=0, frame_interval=None)
        with pytest.raises(ValidationError):
            ExtractionSettings(fps=-1, frame_interval=None)

    def test_validation_interval_positive(self) -> None:
        """Test frame_interval must be positive."""
        with pytest.raises(ValidationError):
            ExtractionSettings(fps=None, frame_interval=0)
        with pytest.raises(ValidationError):
            ExtractionSettings(fps=None, frame_interval=-1)

    def test_jpeg_quality_range(self) -> None:
        """Test JPEG quality validation."""
        settings = ExtractionSettings(jpeg_quality=50)
        assert settings.jpeg_quality == 50

        with pytest.raises(ValidationError):
            ExtractionSettings(jpeg_quality=0)
        with pytest.raises(ValidationError):
            ExtractionSettings(jpeg_quality=101)


class TestLoggingSettings:
    """Tests for LoggingSettings model."""

    def test_default_values(self) -> None:
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.log_file is None
        assert settings.show_progress is True

    def test_valid_levels(self) -> None:
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = LoggingSettings(level=level)
            assert settings.level == level

    def test_invalid_level(self) -> None:
        """Test invalid log level raises error."""
        with pytest.raises(ValidationError):
            LoggingSettings(level="INVALID")


class TestAppSettings:
    """Tests for AppSettings root model."""

    def test_default_values(self) -> None:
        """Test default app settings."""
        settings = AppSettings()
        assert isinstance(settings.cache, CacheSettings)
        assert isinstance(settings.download, DownloadSettings)
        assert isinstance(settings.extraction, ExtractionSettings)
        assert isinstance(settings.logging, LoggingSettings)

    def test_validates_extraction_mutual_exclusion(self) -> None:
        """Test that fps and frame_interval cannot both be set."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            AppSettings(extraction={"fps": 1.0, "frame_interval": 30})

    def test_validates_extraction_requires_one(self) -> None:
        """Test that either fps or frame_interval must be set."""
        with pytest.raises(ValidationError, match="Must specify either"):
            AppSettings(extraction={"fps": None, "frame_interval": None})

    def test_nested_configuration(self) -> None:
        """Test nested configuration."""
        settings = AppSettings(
            cache={"max_size_gb": 20.0},
            download={"quality": "720p"},
            extraction={"fps": 2.0, "frame_interval": None},
            logging={"level": "DEBUG"},
        )
        assert settings.cache.max_size_gb == 20.0
        assert settings.download.quality == "720p"
        assert settings.extraction.fps == 2.0
        assert settings.logging.level == "DEBUG"

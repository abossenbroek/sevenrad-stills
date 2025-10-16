"""
Pydantic configuration models for sevenrad-stills.

These models define and validate all application settings loaded from YAML
configuration files.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClearStrategy(str, Enum):
    """Cache clearing strategy."""

    ON_START = "on_start"
    ON_EXIT = "on_exit"
    MANUAL = "manual"
    NEVER = "never"


class CacheSettings(BaseModel):
    """Cache management settings."""

    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "sevenrad_stills",
        description="Directory for caching downloaded videos and temporary files",
    )
    clear_strategy: ClearStrategy = Field(
        default=ClearStrategy.ON_EXIT,
        description="When to clear the cache",
    )
    max_size_gb: float = Field(
        default=10.0,
        gt=0,
        description="Maximum cache size in gigabytes",
    )
    retention_days: int = Field(
        default=7,
        gt=0,
        description="Number of days to retain cached files",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def expand_cache_dir(cls, v: str | Path) -> Path:
        """Expand user home directory in cache path."""
        path = Path(v)
        return path.expanduser().resolve()


class DownloadSettings(BaseModel):
    """Video download settings."""

    format: Literal["mp4", "webm", "mkv"] = Field(
        default="mp4",
        description="Video format",
    )
    quality: str = Field(
        default="best",
        description="Video quality (best, worst, or specific resolution)",
    )
    output_dir: Path = Field(
        default=Path("./videos"),
        description="Directory for downloaded videos",
    )
    temp_dir: Path | None = Field(
        default=None,
        description="Temporary download directory",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def expand_output_dir(cls, v: str | Path) -> Path:
        """Expand and resolve output directory path."""
        return Path(v).expanduser().resolve()

    @field_validator("temp_dir", mode="before")
    @classmethod
    def expand_temp_dir(cls, v: str | Path | None) -> Path | None:
        """Expand and resolve temp directory path."""
        if v is None:
            return None
        return Path(v).expanduser().resolve()


class ExtractionSettings(BaseModel):
    """Frame extraction settings."""

    fps: float | None = Field(
        default=1.0,
        gt=0,
        description="Frames per second to extract",
    )
    frame_interval: int | None = Field(
        default=None,
        gt=0,
        description="Extract every Nth frame",
    )
    output_format: Literal["jpg", "png"] = Field(
        default="jpg",
        description="Output format for extracted frames",
    )
    output_dir: Path = Field(
        default=Path("./frames"),
        description="Directory for extracted frames",
    )
    naming_pattern: str = Field(
        default="{video_id}_frame_{index:06d}.{ext}",
        description="Naming pattern for extracted frames",
    )
    jpeg_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG quality when output_format is jpg",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def expand_output_dir(cls, v: str | Path) -> Path:
        """Expand and resolve output directory path."""
        return Path(v).expanduser().resolve()

    @field_validator("fps")
    @classmethod
    def validate_extraction_mode(cls, v: float | None) -> float | None:
        """Validate that either fps or frame_interval is set, but not both."""
        # This validator runs before frame_interval is set, so we can't check here
        # The check will be done in the root model validator
        return v


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Path | None = Field(
        default=None,
        description="Log file path",
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress bars",
    )

    @field_validator("log_file", mode="before")
    @classmethod
    def expand_log_file(cls, v: str | Path | None) -> Path | None:
        """Expand and resolve log file path."""
        if v is None:
            return None
        return Path(v).expanduser().resolve()


class AppSettings(BaseModel):
    """Root application settings."""

    cache: CacheSettings = Field(default_factory=CacheSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @field_validator("extraction")
    @classmethod
    def validate_extraction_settings(cls, v: ExtractionSettings) -> ExtractionSettings:
        """Validate that either fps or frame_interval is set, but not both."""
        if v.fps is not None and v.frame_interval is not None:
            msg = "Cannot specify both fps and frame_interval"
            raise ValueError(msg)
        if v.fps is None and v.frame_interval is None:
            msg = "Must specify either fps or frame_interval"
            raise ValueError(msg)
        return v

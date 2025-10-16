"""
Video metadata models for sevenrad-stills.

Pydantic models for video information extracted from yt-dlp.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Information about a downloaded video."""

    video_id: str = Field(description="Unique video identifier")
    title: str = Field(description="Video title")
    duration: float = Field(description="Duration in seconds")
    width: int | None = Field(default=None, description="Video width in pixels")
    height: int | None = Field(default=None, description="Video height in pixels")
    fps: float | None = Field(default=None, description="Frames per second")
    format: str = Field(description="Video format")
    file_size: int | None = Field(default=None, description="File size in bytes")
    file_path: Path = Field(description="Path to downloaded video file")
    thumbnail_url: str | None = Field(default=None, description="Thumbnail URL")
    uploader: str | None = Field(default=None, description="Video uploader")
    upload_date: str | None = Field(default=None, description="Upload date (YYYYMMDD)")

    @property
    def resolution(self) -> str | None:
        """Get video resolution as string."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None

    @property
    def duration_str(self) -> str:
        """Get duration as formatted string (HH:MM:SS)."""
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

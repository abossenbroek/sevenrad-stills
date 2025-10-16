"""
YouTube video downloader using yt-dlp.

Downloads videos from YouTube and other supported platforms.
"""

from pathlib import Path
from typing import Any, Callable

import yt_dlp

from sevenrad_stills.download.metadata import VideoInfo
from sevenrad_stills.settings.models import DownloadSettings
from sevenrad_stills.storage.paths import ensure_directory, sanitize_filename
from sevenrad_stills.utils.exceptions import (
    DownloadError,
    NetworkError,
    VideoNotFoundError,
)


class VideoDownloader:
    """Downloads videos using yt-dlp."""

    def __init__(
        self,
        settings: DownloadSettings,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Initialize video downloader.

        Args:
            settings: Download configuration settings
            progress_callback: Optional callback for download progress updates

        """
        self.settings = settings
        self.progress_callback = progress_callback

    def _progress_hook(self, d: dict[str, Any]) -> None:
        """
        Handle progress updates from yt-dlp.

        Args:
            d: Progress dictionary from yt-dlp

        """
        if self.progress_callback:
            self.progress_callback(d)

    def download(self, url: str, output_path: Path | None = None) -> VideoInfo:
        """
        Download video from URL.

        Args:
            url: Video URL (YouTube or other supported platform)
            output_path: Optional custom output path

        Returns:
            VideoInfo with metadata and file path

        Raises:
            VideoNotFoundError: If video is not found or unavailable
            NetworkError: If network error occurs
            DownloadError: For other download errors

        """
        # Determine output directory
        if output_path is None:
            output_dir = self.settings.output_dir
        else:
            output_dir = output_path.parent if output_path.is_file() else output_path

        ensure_directory(output_dir)

        # Configure yt-dlp options
        ydl_opts: dict[str, Any] = {
            "format": self._get_format_string(),
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "progress_hooks": [self._progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info without downloading first
                info = ydl.extract_info(url, download=False)
                if info is None:
                    msg = f"Could not extract video information from {url}"
                    raise VideoNotFoundError(msg)

                # Download the video
                info = ydl.extract_info(url, download=True)
                if info is None:
                    msg = f"Failed to download video from {url}"
                    raise DownloadError(msg)

                # Get the downloaded file path
                video_id = info.get("id", "unknown")
                ext = info.get("ext", self.settings.format)
                file_path = output_dir / f"{video_id}.{ext}"

                # Create VideoInfo
                return self._create_video_info(info, file_path)

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg or "not available" in error_msg:
                msg = f"Video not found or unavailable: {url}"
                raise VideoNotFoundError(msg) from e
            if "network" in error_msg.lower():
                msg = f"Network error downloading video: {error_msg}"
                raise NetworkError(msg) from e
            msg = f"Failed to download video from {url}: {error_msg}"
            raise DownloadError(msg) from e
        except Exception as e:
            msg = f"Unexpected error downloading video from {url}: {e}"
            raise DownloadError(msg) from e

    def _get_format_string(self) -> str:
        """
        Get yt-dlp format string based on settings.

        Returns:
            Format string for yt-dlp

        """
        if self.settings.quality == "best":
            fmt = self.settings.format
            return f"bestvideo[ext={fmt}]+bestaudio/best[ext={fmt}]/best"
        if self.settings.quality == "worst":
            fmt = self.settings.format
            return f"worstvideo[ext={fmt}]+worstaudio/worst[ext={fmt}]/worst"

        # Specific resolution like "720p"
        if self.settings.quality.endswith("p"):
            height = self.settings.quality[:-1]
            fmt = self.settings.format
            return (
                f"bestvideo[height<={height}][ext={fmt}]+bestaudio/"
                f"best[height<={height}]/best"
            )

        # Default to best
        fmt = self.settings.format
        return f"bestvideo[ext={fmt}]+bestaudio/best[ext={fmt}]/best"

    def _create_video_info(self, info: dict[str, Any], file_path: Path) -> VideoInfo:
        """
        Create VideoInfo from yt-dlp info dict.

        Args:
            info: yt-dlp info dictionary
            file_path: Path to downloaded video file

        Returns:
            VideoInfo instance

        """
        return VideoInfo(
            video_id=info.get("id", "unknown"),
            title=info.get("title", "Unknown"),
            duration=float(info.get("duration", 0)),
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            format=info.get("ext", self.settings.format),
            file_size=info.get("filesize") or info.get("filesize_approx"),
            file_path=file_path,
            thumbnail_url=info.get("thumbnail"),
            uploader=info.get("uploader"),
            upload_date=info.get("upload_date"),
        )

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is supported by yt-dlp.

        Args:
            url: URL to check

        Returns:
            True if URL is supported

        """
        try:
            # Quick check using yt-dlp's extractors
            extractors = yt_dlp.extractor.gen_extractors()
            return any(extractor.suitable(url) for extractor in extractors)
        except Exception:
            return False

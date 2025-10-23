"""
Frame extractor using FFmpeg.

Extracts frames from video files using ffmpeg-python.
"""

from pathlib import Path
from typing import Any

import ffmpeg

from sevenrad_stills.download.metadata import VideoInfo
from sevenrad_stills.extraction.strategies import ExtractionStrategy
from sevenrad_stills.settings.models import ExtractionSettings
from sevenrad_stills.storage.paths import ensure_directory, format_frame_filename
from sevenrad_stills.utils.exceptions import FFmpegError


class FrameExtractor:
    """Extracts frames from video files using FFmpeg."""

    def __init__(self, settings: ExtractionSettings) -> None:
        """
        Initialize frame extractor.

        Args:
            settings: Extraction configuration settings

        """
        self.settings = settings

    def extract_frames(
        self,
        video_info: VideoInfo,
        strategy: ExtractionStrategy,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[Path]:
        """
        Extract frames from video using the given strategy.

        Args:
            video_info: Video metadata
            strategy: Extraction strategy to use
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds

        Returns:
            List of paths to extracted frame files

        Raises:
            FFmpegError: If frame extraction fails

        """
        # Ensure output directory exists
        output_dir = self.settings.output_dir
        ensure_directory(output_dir)

        # Create output pattern for ffmpeg
        output_pattern = self._get_output_pattern(video_info.video_id)

        try:
            # Build ffmpeg command with optional time range
            input_kwargs: dict[str, Any] = {}
            if start_time is not None:
                input_kwargs["ss"] = start_time
            if end_time is not None and start_time is not None:
                input_kwargs["t"] = end_time - start_time

            # Add strict compliance flag to handle non-standard YUV formats
            input_kwargs["strict"] = "unofficial"

            stream = ffmpeg.input(str(video_info.file_path), **input_kwargs)

            # Apply extraction filter
            filter_str = strategy.get_ffmpeg_filter()

            # Check if this is an fps filter or select filter
            if filter_str.startswith("fps="):
                # FPS filter: use fps filter directly
                fps_value = filter_str.split("=")[1]
                stream = ffmpeg.filter(stream, "fps", fps=fps_value)
            else:
                # Select filter: use select with expression
                stream = ffmpeg.filter(stream, "select", filter_str)

            # Add color space conversion for robust video format handling
            # Convert limited-range YUV to full-range RGB for compatibility
            stream = ffmpeg.filter(
                stream, "scale", in_range="limited", out_range="full"
            )
            stream = ffmpeg.filter(stream, "format", pix_fmts="rgb24")

            # Set output options
            output_args: dict[str, Any] = {
                "fps_mode": "vfr",  # Variable frame rate (replaces deprecated vsync)
                "strict": "unofficial",  # Handle non-standard YUV formats in encoder
            }

            # Configure output format-specific options
            if self.settings.output_format == "jpg":
                output_args["q:v"] = int((100 - self.settings.jpeg_quality) / 10)
            else:
                # For PNG or other formats, use high quality
                output_args["q:v"] = 2

            # Run ffmpeg
            stream = ffmpeg.output(
                stream,
                str(output_pattern),
                **output_args,
            )

            ffmpeg.run(stream, overwrite_output=True, quiet=True)

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            msg = f"FFmpeg error during frame extraction: {error_msg}"
            raise FFmpegError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during frame extraction: {e}"
            raise FFmpegError(msg) from e

        # Collect extracted frame paths
        return self._collect_extracted_frames(video_info.video_id)

    def _get_output_pattern(self, video_id: str) -> Path:
        """
        Get output file pattern for ffmpeg.

        Args:
            video_id: Video identifier

        Returns:
            Path pattern with ffmpeg frame number placeholder

        """
        # Use ffmpeg's %06d for frame numbering
        filename = format_frame_filename(
            pattern=self.settings.naming_pattern,
            video_id=video_id,
            index=0,  # Will be replaced by ffmpeg
            ext=self.settings.output_format,
        )

        # Replace the index with ffmpeg's pattern
        # The format_frame_filename uses {index:06d} which becomes 000000
        filename = filename.replace("000000", "%06d")

        return self.settings.output_dir / filename

    def _collect_extracted_frames(self, video_id: str) -> list[Path]:
        """
        Collect paths of extracted frames.

        Args:
            video_id: Video identifier

        Returns:
            Sorted list of frame file paths

        """
        # Pattern to match extracted frames - use wildcard to match any naming pattern
        pattern = f"{video_id}_*.{self.settings.output_format}"
        frames = sorted(self.settings.output_dir.glob(pattern))
        return frames

    def get_frame_count_estimate(
        self,
        video_info: VideoInfo,
        strategy: ExtractionStrategy,
    ) -> int:
        """
        Estimate number of frames that will be extracted.

        Args:
            video_info: Video metadata
            strategy: Extraction strategy

        Returns:
            Estimated frame count

        """
        if hasattr(strategy, "fps"):
            # FPS strategy
            fps_val: float = strategy.fps
            return int(video_info.duration * fps_val)
        if hasattr(strategy, "interval"):
            # Interval strategy
            interval_val: int = strategy.interval
            if video_info.fps:
                total_frames = int(video_info.duration * video_info.fps)
                return total_frames // interval_val
            # Fallback estimate (assume 30 fps if not known)
            total_frames = int(video_info.duration * 30)
            return total_frames // interval_val

        return 0

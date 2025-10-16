"""
Video processing pipeline for sevenrad-stills.

Orchestrates the complete workflow: download → extract → cleanup.
"""

from pathlib import Path

from sevenrad_stills.download.downloader import VideoDownloader
from sevenrad_stills.download.metadata import VideoInfo
from sevenrad_stills.extraction.extractor import FrameExtractor
from sevenrad_stills.extraction.strategies import (
    ExtractionStrategy,
    create_extraction_strategy,
)
from sevenrad_stills.settings.models import AppSettings
from sevenrad_stills.storage.cache import CacheManager
from sevenrad_stills.utils.exceptions import PipelineError
from sevenrad_stills.utils.logging import get_logger


class VideoProcessor:
    """Main pipeline for processing videos."""

    def __init__(self, settings: AppSettings) -> None:
        """
        Initialize video processor.

        Args:
            settings: Application settings

        """
        self.settings = settings
        self.logger = get_logger()

        # Initialize components
        self.cache_manager = CacheManager(settings.cache)
        self.downloader = VideoDownloader(settings.download)
        self.extractor = FrameExtractor(settings.extraction)

    def process(self, url: str) -> tuple[VideoInfo, list[Path]]:
        """
        Process video: download and extract frames.

        Args:
            url: Video URL

        Returns:
            Tuple of (VideoInfo, list of extracted frame paths)

        Raises:
            PipelineError: If processing fails

        """
        self.logger.info(f"Starting video processing for: {url}")

        try:
            # Setup cache
            self.cache_manager.setup()

            # Download video
            self.logger.info("Downloading video...")
            video_info = self.downloader.download(url)
            self.logger.info(
                f"Downloaded: {video_info.title} ({video_info.duration_str})"
            )

            # Create extraction strategy
            strategy = create_extraction_strategy(
                fps=self.settings.extraction.fps,
                frame_interval=self.settings.extraction.frame_interval,
            )
            self.logger.info(f"Extraction strategy: {strategy.get_description()}")

            # Estimate frame count
            estimated_frames = self.extractor.get_frame_count_estimate(
                video_info, strategy
            )
            self.logger.info(f"Estimated frames to extract: {estimated_frames}")

            # Extract frames
            self.logger.info("Extracting frames...")
            frame_paths = self.extractor.extract_frames(video_info, strategy)
            self.logger.info(f"Extracted {len(frame_paths)} frames")

            # Cleanup
            self.cache_manager.cleanup()

            self.logger.info("Processing completed successfully")
            return video_info, frame_paths

        except Exception as e:
            msg = f"Pipeline error: {e}"
            self.logger.error(msg)
            raise PipelineError(msg) from e

    def process_with_custom_settings(
        self,
        url: str,
        fps: float | None = None,
        frame_interval: int | None = None,
        output_dir: Path | None = None,
    ) -> tuple[VideoInfo, list[Path]]:
        """
        Process video with custom extraction settings.

        Args:
            url: Video URL
            fps: Optional FPS override
            frame_interval: Optional frame interval override
            output_dir: Optional output directory override

        Returns:
            Tuple of (VideoInfo, list of extracted frame paths)

        Raises:
            PipelineError: If processing fails

        """
        # Temporarily override settings
        original_fps = self.settings.extraction.fps
        original_interval = self.settings.extraction.frame_interval
        original_output = self.settings.extraction.output_dir

        try:
            if fps is not None:
                self.settings.extraction.fps = fps
                self.settings.extraction.frame_interval = None
            if frame_interval is not None:
                self.settings.extraction.frame_interval = frame_interval
                self.settings.extraction.fps = None
            if output_dir is not None:
                self.settings.extraction.output_dir = output_dir

            return self.process(url)

        finally:
            # Restore original settings
            self.settings.extraction.fps = original_fps
            self.settings.extraction.frame_interval = original_interval
            self.settings.extraction.output_dir = original_output

    def __enter__(self) -> "VideoProcessor":
        """Context manager entry."""
        self.cache_manager.setup()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.cache_manager.cleanup()

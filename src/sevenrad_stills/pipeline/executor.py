"""
Pipeline executor for YAML-based image processing workflows.

Orchestrates video download, segmentation, and image operations.
"""

from pathlib import Path

from PIL import Image

from sevenrad_stills.download.downloader import VideoDownloader
from sevenrad_stills.download.metadata import VideoInfo
from sevenrad_stills.extraction.extractor import FrameExtractor
from sevenrad_stills.extraction.strategies import create_extraction_strategy
from sevenrad_stills.operations import get_operation
from sevenrad_stills.pipeline.models import PipelineConfig
from sevenrad_stills.settings.models import (
    AppSettings,
    DownloadSettings,
    ExtractionSettings,
)
from sevenrad_stills.storage.cache import CacheManager
from sevenrad_stills.storage.paths import ensure_directory
from sevenrad_stills.utils.exceptions import PipelineError
from sevenrad_stills.utils.logging import get_logger


class PipelineExecutor:
    """
    Execute YAML-defined image processing pipelines.

    Handles the full workflow from video download to final processed images.
    """

    def __init__(
        self, pipeline_config: PipelineConfig, app_settings: AppSettings | None = None
    ) -> None:
        """
        Initialize pipeline executor.

        Args:
            pipeline_config: Pipeline configuration
            app_settings: Optional app settings (creates defaults if not provided)

        """
        self.config = pipeline_config
        self.logger = get_logger()

        # Use provided settings or create minimal defaults
        if app_settings is None:
            app_settings = AppSettings()

        self.app_settings = app_settings
        self.cache_manager = CacheManager(app_settings.cache)
        self.downloader = VideoDownloader(app_settings.download)

    def execute(self) -> dict[str, list[Path]]:
        """
        Execute the complete pipeline.

        Returns:
            Dictionary mapping step names to lists of output file paths

        Raises:
            PipelineError: If execution fails

        """
        self.logger.info("Starting pipeline execution")

        try:
            # Setup
            self._setup_output_directories()
            self.cache_manager.setup()

            # Download video
            self.logger.info("Downloading video: %s", self.config.source.youtube_url)
            video_info = self.downloader.download(self.config.source.youtube_url)
            self.logger.info("Downloaded: %s", video_info.title)

            # Extract segment frames
            self.logger.info("Extracting frames from segment")
            frame_paths = self._extract_segment_frames(video_info)
            self.logger.info("Extracted %d frames", len(frame_paths))

            # Process frames through pipeline
            results = self._process_frames(frame_paths)

            # Cleanup
            self.cache_manager.cleanup()

            self.logger.info("Pipeline execution completed successfully")
            return results

        except Exception as e:
            msg = f"Pipeline execution failed: {e}"
            self.logger.error(msg)
            raise PipelineError(msg) from e

    def _setup_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        ensure_directory(self.config.output.base_dir)
        ensure_directory(self.config.output.intermediate_dir)
        ensure_directory(self.config.output.final_dir)

    def _extract_segment_frames(self, video_info: VideoInfo) -> list[Path]:
        """
        Extract frames from the specified video segment.

        Args:
            video_info: Video metadata

        Returns:
            List of extracted frame paths

        """
        segment = self.config.segment

        # Calculate FPS based on interval
        # If interval is 1 second, fps = 1
        # If interval is 0.5 seconds, fps = 2
        fps = 1.0 / segment.interval

        # Create extraction settings for the segment
        extraction_settings = ExtractionSettings(
            fps=fps,
            frame_interval=None,
            output_format="jpg",
            output_dir=self.config.output.intermediate_dir / "extracted",
            naming_pattern="{video_id}_segment_{index:06d}.{ext}",
            jpeg_quality=95,
        )

        # Create extractor with segment-specific settings
        extractor = FrameExtractor(extraction_settings)
        strategy = create_extraction_strategy(fps=fps)

        # Extract frames
        all_frames = extractor.extract_frames(video_info, strategy)

        # Filter frames based on segment time range
        # Calculate which frame indices fall within the segment
        start_idx = int(segment.start * fps)
        end_idx = int(segment.end * fps)

        # Return only frames within the segment
        segment_frames = all_frames[start_idx:end_idx]
        self.logger.info(
            "Selected frames %d-%d from segment %.1fs-%.1fs",
            start_idx,
            end_idx,
            segment.start,
            segment.end,
        )

        return segment_frames

    def _process_frames(self, frame_paths: list[Path]) -> dict[str, list[Path]]:
        """
        Process frames through all pipeline steps.

        Args:
            frame_paths: Input frame paths

        Returns:
            Dictionary mapping step names to output paths

        """
        results: dict[str, list[Path]] = {}
        current_frames = frame_paths

        for step_idx, step in enumerate(self.config.steps):
            self.logger.info(
                "Executing step %d/%d: %s (%s)",
                step_idx + 1,
                len(self.config.steps),
                step.name,
                step.operation,
            )

            # Get operation
            operation = get_operation(step.operation)

            # Determine output directory
            if step_idx == len(self.config.steps) - 1:
                # Last step goes to final directory
                output_dir = self.config.output.final_dir
            else:
                # Intermediate steps
                output_dir = self.config.output.intermediate_dir / step.name

            ensure_directory(output_dir)

            # Process each frame
            output_paths: list[Path] = []
            for frame_path in current_frames:
                # Load image
                image = Image.open(frame_path)

                # Apply operation
                processed_image = operation.apply(image, step.params)

                # Generate output filename
                output_filename = (
                    f"{step.name}_{frame_path.stem}_step{step_idx:02d}.jpg"
                )
                output_path = output_dir / output_filename

                # Save processed image
                operation.save_image(processed_image, output_path)
                output_paths.append(output_path)

            self.logger.info(
                "Step '%s' processed %d frames", step.name, len(output_paths)
            )
            results[step.name] = output_paths

            # Next step uses output of current step
            current_frames = output_paths

        return results

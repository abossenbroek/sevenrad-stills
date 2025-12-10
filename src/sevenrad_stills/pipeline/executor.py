"""
Pipeline executor for YAML-based image processing workflows.

Orchestrates video download, segmentation, and image operations.
Supports both legacy PIL-based execution and Taichi GPU pipeline execution.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from sevenrad_stills.download.downloader import VideoDownloader
from sevenrad_stills.download.metadata import VideoInfo
from sevenrad_stills.extraction.extractor import FrameExtractor
from sevenrad_stills.extraction.strategies import create_extraction_strategy
from sevenrad_stills.operations import (
    BackendNotAvailableError,
    get_backend_implementation,
    get_operation,
    has_taichi_operation,
)
from sevenrad_stills.operations.backend import BackendType
from sevenrad_stills.pipeline.models import ImageOperationStep, PipelineConfig
from sevenrad_stills.pipeline.taichi_executor import TaichiArch, TaichiPipelineExecutor
from sevenrad_stills.settings.models import (
    AppSettings,
    DownloadSettings,
    ExtractionSettings,
)
from sevenrad_stills.storage.cache import CacheManager
from sevenrad_stills.storage.paths import ensure_directory
from sevenrad_stills.utils.exceptions import PipelineError
from sevenrad_stills.utils.logging import get_logger


def _process_single_frame(
    frame_path: Path,
    operation_name: str,
    params: dict[str, object],
    output_path: Path,
    repeat: int = 1,
    backend: str = "cpu",
) -> Path:
    """
    Process a single frame with an operation.

    Must be module-level function for multiprocessing pickling.

    Args:
        frame_path: Input frame path
        operation_name: Name of operation to apply
        params: Operation parameters
        output_path: Output file path
        repeat: Number of times to repeat the operation
        backend: Compute backend to use (cpu, gpu, metal)

    Returns:
        Path to processed image

    Raises:
        BackendNotAvailableError: If requested backend not available

    """
    # Import here to avoid circular dependencies in multiprocessing
    from PIL import Image

    from sevenrad_stills.operations import get_backend_implementation

    # Get operation for specified backend
    operation = get_backend_implementation(operation_name, backend)  # type: ignore[arg-type]
    image: Image.Image = Image.open(frame_path)

    # Apply operation with repeat support
    processed_image: Image.Image = image
    for _ in range(repeat):
        processed_image = operation.apply(processed_image, params)

    operation.save_image(processed_image, output_path)

    return output_path


class PipelineExecutor:
    """
    Execute YAML-defined image processing pipelines.

    Handles the full workflow from video download to final processed images.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        app_settings: AppSettings | None = None,
        max_workers: int | None = None,
        parallel: bool = True,
    ) -> None:
        """
        Initialize pipeline executor.

        Args:
            pipeline_config: Pipeline configuration
            app_settings: Optional app settings (creates defaults if not provided)
            max_workers: Maximum parallel workers (defaults to CPU count)
            parallel: Enable parallel processing (default True)

        """
        self.config = pipeline_config
        self.logger = get_logger()

        # Use provided settings or create minimal defaults
        if app_settings is None:
            app_settings = AppSettings()

        self.app_settings = app_settings
        self.cache_manager = CacheManager(app_settings.cache)
        self.downloader = VideoDownloader(app_settings.download)

        # Parallel processing configuration
        self.parallel = parallel
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.logger.debug(
            "Parallel processing: %s (workers: %d)", self.parallel, self.max_workers
        )

        # Taichi executor (lazily initialized)
        self._taichi_executor: TaichiPipelineExecutor | None = None
        self._use_taichi = self._should_use_taichi()

    def _should_use_taichi(self) -> bool:
        """
        Determine whether to use Taichi pipeline execution.

        Returns:
            True if Taichi execution should be used, False for legacy execution.

        """
        mode = self.config.execution_mode

        if mode == "legacy":
            self.logger.info("Using legacy PIL-based execution (explicit)")
            return False

        if mode == "taichi":
            # Check all operations have Taichi implementations
            missing_ops = [
                step.operation
                for step in self.config.steps
                if not has_taichi_operation(step.operation)
            ]
            if missing_ops:
                msg = (
                    f"Taichi mode requested but these operations lack "
                    f"Taichi implementations: {', '.join(set(missing_ops))}"
                )
                raise PipelineError(msg)
            self.logger.info("Using Taichi GPU pipeline (explicit)")
            return True

        # mode == "auto"
        # Check if all operations support Taichi
        all_taichi = all(
            has_taichi_operation(step.operation) for step in self.config.steps
        )

        if all_taichi:
            self.logger.info(
                "Using Taichi GPU pipeline (auto-detected: all ops support Taichi)"
            )
            return True

        # Some operations don't have Taichi implementations
        legacy_ops = [
            step.operation
            for step in self.config.steps
            if not has_taichi_operation(step.operation)
        ]
        self.logger.info(
            "Using legacy PIL-based execution (auto-detected: %s lack Taichi support)",
            ", ".join(set(legacy_ops)),
        )
        return False

    def _get_taichi_executor(self) -> TaichiPipelineExecutor:
        """
        Get or create the Taichi pipeline executor.

        Returns:
            Initialized TaichiPipelineExecutor instance.

        """
        if self._taichi_executor is None:
            # Map backend to Taichi architecture
            arch_map = {
                "cpu": TaichiArch.CPU,
                "gpu": TaichiArch.CUDA,
                "metal": TaichiArch.METAL,
            }
            arch = arch_map.get(self.config.backend, TaichiArch.METAL)

            self._taichi_executor = TaichiPipelineExecutor(
                arch=arch,
                debug=self.config.debug,
            )
        return self._taichi_executor

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

        # Extract frames from the segment time range
        # FFmpeg will handle the time-based extraction
        segment_frames = extractor.extract_frames(
            video_info,
            strategy,
            start_time=segment.start,
            end_time=segment.end,
        )

        self.logger.info(
            "Extracted %d frames from segment %.1fs-%.1fs at %.1f fps",
            len(segment_frames),
            segment.start,
            segment.end,
            fps,
        )

        return segment_frames

    def _process_frames(self, frame_paths: list[Path]) -> dict[str, list[Path]]:
        """
        Process frames through all pipeline steps.

        Routes to Taichi or legacy execution based on configuration.

        Args:
            frame_paths: Input frame paths

        Returns:
            Dictionary mapping step names to output paths

        """
        if self._use_taichi:
            return self._process_frames_taichi(frame_paths)
        return self._process_frames_legacy(frame_paths)

    def _process_frames_taichi(self, frame_paths: list[Path]) -> dict[str, list[Path]]:
        """
        Process frames using Taichi end-to-end GPU pipeline.

        Processes each frame through all steps on GPU with minimal transfers.

        Args:
            frame_paths: Input frame paths

        Returns:
            Dictionary mapping step names to output paths

        """
        results: dict[str, list[Path]] = {}
        executor = self._get_taichi_executor()

        # Warmup all operations
        executor.warmup(self.config.steps)

        # Prepare output directory (final step output)
        output_dir = self.config.output.final_dir
        ensure_directory(output_dir)

        output_paths: list[Path] = []

        for frame_idx, frame_path in enumerate(frame_paths):
            self.logger.debug(
                "Processing frame %d/%d via Taichi: %s",
                frame_idx + 1,
                len(frame_paths),
                frame_path.name,
            )

            # Load frame
            image = Image.open(frame_path)

            # Process through entire pipeline on GPU
            result = executor.process_image(image, self.config.steps)

            # Generate output filename
            output_filename = f"final_{frame_path.stem}_taichi.jpg"
            output_path = output_dir / output_filename

            # Save result
            result.save(output_path, "JPEG", quality=95)
            output_paths.append(output_path)

        # For Taichi mode, all steps are executed together
        # Return final results under "final" key
        results["final"] = output_paths

        self.logger.info(
            "Taichi pipeline processed %d frames through %d steps",
            len(frame_paths),
            len(self.config.steps),
        )

        return results

    def _process_frames_legacy(self, frame_paths: list[Path]) -> dict[str, list[Path]]:
        """
        Process frames using legacy PIL-based execution.

        Processes frames step by step with CPU/GPU operations.

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

            # Determine output directory
            if step_idx == len(self.config.steps) - 1:
                # Last step goes to final directory
                output_dir = self.config.output.final_dir
            else:
                # Intermediate steps
                output_dir = self.config.output.intermediate_dir / step.name

            ensure_directory(output_dir)

            # Process frames (parallel or sequential)
            if self.parallel and len(current_frames) > 1:
                output_paths = self._process_frames_parallel(
                    current_frames, step, output_dir, step_idx
                )
            else:
                output_paths = self._process_frames_sequential(
                    current_frames, step, output_dir, step_idx
                )

            self.logger.info(
                "Step '%s' processed %d frames", step.name, len(output_paths)
            )
            results[step.name] = output_paths

            # Next step uses output of current step
            current_frames = output_paths

        return results

    def _process_frames_sequential(
        self,
        frame_paths: list[Path],
        step: ImageOperationStep,
        output_dir: Path,
        step_idx: int,
    ) -> list[Path]:
        """
        Process frames sequentially (original implementation).

        Args:
            frame_paths: Input frame paths
            step: Pipeline step configuration
            output_dir: Output directory
            step_idx: Step index

        Returns:
            List of output file paths

        Raises:
            BackendNotAvailableError: If requested backend not available
            PipelineError: If operation fails

        """
        output_paths: list[Path] = []

        # Get operation for configured backend
        try:
            operation = get_backend_implementation(step.operation, self.config.backend)
        except BackendNotAvailableError as e:
            msg = (
                f"Cannot execute step '{step.name}': {e}. "
                f"Please use a different backend or implement the missing variant."
            )
            raise PipelineError(msg) from e

        for frame_path in frame_paths:
            # Load image
            image: Image.Image = Image.open(frame_path)

            # Apply operation (with repeat support)
            processed_image: Image.Image = image
            for repeat_idx in range(step.repeat):
                processed_image = operation.apply(processed_image, step.params)

                # Log repeat iterations if > 1
                if step.repeat > 1 and repeat_idx > 0:
                    self.logger.debug(
                        "Step '%s' repeat %d/%d for %s",
                        step.name,
                        repeat_idx + 1,
                        step.repeat,
                        frame_path.name,
                    )

            # Generate output filename
            output_filename = f"{step.name}_{frame_path.stem}_step{step_idx:02d}.jpg"
            output_path = output_dir / output_filename

            # Save processed image
            operation.save_image(processed_image, output_path)
            output_paths.append(output_path)

        return output_paths

    def _process_frames_parallel(
        self,
        frame_paths: list[Path],
        step: ImageOperationStep,
        output_dir: Path,
        step_idx: int,
    ) -> list[Path]:
        """
        Process frames in parallel using ProcessPoolExecutor.

        Args:
            frame_paths: Input frame paths
            step: Pipeline step configuration
            output_dir: Output directory
            step_idx: Step index

        Returns:
            List of output file paths (in original order)

        """
        # Validate backend availability before parallel processing
        try:
            # Test that operation exists for this backend
            _ = get_backend_implementation(step.operation, self.config.backend)
        except BackendNotAvailableError as e:
            msg = (
                f"Cannot execute step '{step.name}': {e}. "
                f"Please use a different backend or implement the missing variant."
            )
            raise PipelineError(msg) from e

        self.logger.info(
            "Processing %d frames in parallel (workers: %d, backend: %s)",
            len(frame_paths),
            self.max_workers,
            self.config.backend,
        )

        # Prepare tasks for parallel processing
        tasks = []
        for frame_path in frame_paths:
            output_filename = f"{step.name}_{frame_path.stem}_step{step_idx:02d}.jpg"
            output_path = output_dir / output_filename
            tasks.append(
                (
                    frame_path,
                    step.operation,
                    step.params,
                    output_path,
                    step.repeat,
                    self.config.backend,
                )
            )

        # Process in parallel
        output_paths: list[Path] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single_frame, *task): idx
                for idx, task in enumerate(tasks)
            }

            # Collect results in order
            results_dict: dict[int, Path] = {}
            for future in as_completed(futures):
                idx = futures[future]
                results_dict[idx] = future.result()

            # Return in original order
            output_paths = [results_dict[i] for i in sorted(results_dict.keys())]

        return output_paths

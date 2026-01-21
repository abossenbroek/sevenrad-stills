"""
Taichi-based pipeline executor for end-to-end GPU acceleration.

Executes image processing pipelines with minimal CPU↔GPU data transfers by
keeping image data on the GPU throughout the entire pipeline execution.
"""

from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from sevenrad_stills.operations.backend import (
    get_taichi_operation,
    has_taichi_operation,
)
from sevenrad_stills.pipeline.buffer_pool import PipelineBufferPool
from sevenrad_stills.pipeline.models import ImageOperationStep
from sevenrad_stills.utils.exceptions import PipelineError
from sevenrad_stills.utils.logging import get_logger

if TYPE_CHECKING:
    from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False


class TaichiArch(Enum):
    """Supported Taichi backend architectures."""

    CPU = "cpu"
    CUDA = "cuda"
    VULKAN = "vulkan"
    METAL = "metal"
    OPENGL = "opengl"


class GPUOperationError(PipelineError):
    """Error during GPU operation execution."""


class TaichiContext:
    """
    Singleton managing Taichi runtime and resources.

    Ensures Taichi is initialized once and provides access to shared resources
    like the buffer pool.
    """

    _instance: "TaichiContext | None" = None
    _initialized: bool = False

    def __new__(cls) -> "TaichiContext":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize context (only once)."""
        if not hasattr(self, "_buffer_pool"):
            self._buffer_pool = PipelineBufferPool()
            self._logger = get_logger()

    def startup(self, arch: TaichiArch = TaichiArch.METAL) -> None:
        """
        Initialize Taichi runtime.

        Should be called once at application start before any GPU operations.

        Args:
            arch: Backend architecture to use (default: Metal for macOS)

        Raises:
            RuntimeError: If Taichi is not available or initialization fails

        """
        if self._initialized:
            self._logger.debug("Taichi already initialized")
            return

        if ti is None:
            msg = "Taichi is not available. Install with: pip install taichi"
            raise RuntimeError(msg)

        try:
            # Map enum to Taichi architecture
            arch_map = {
                TaichiArch.CPU: ti.cpu,
                TaichiArch.CUDA: ti.cuda,
                TaichiArch.VULKAN: ti.vulkan,
                TaichiArch.METAL: ti.metal,
                TaichiArch.OPENGL: ti.opengl,
            }
            ti_arch = arch_map[arch]

            # Initialize Taichi with specified backend
            ti.init(arch=ti_arch, default_fp=ti.f32)
            self._initialized = True
            self._logger.info("Taichi initialized with %s backend", arch.value)

        except Exception as e:
            msg = f"Failed to initialize Taichi with {arch.value} backend: {e}"
            raise RuntimeError(msg) from e

    def shutdown(self) -> None:
        """
        Release all GPU resources and reset Taichi runtime.

        Should be called before application exit or when switching configurations.
        """
        if not self._initialized:
            return

        try:
            self._buffer_pool.release()
            if ti is not None:
                ti.reset()
            self._initialized = False
            self._logger.info("Taichi context shut down")
        except Exception as e:
            self._logger.error("Error during Taichi shutdown: %s", e)

    @property
    def buffer_pool(self) -> PipelineBufferPool:
        """Access the shared buffer pool."""
        return self._buffer_pool

    @property
    def is_initialized(self) -> bool:
        """Check if Taichi runtime is initialized."""
        return self._initialized


class TaichiPipelineExecutor:
    """
    Execute image pipelines with end-to-end GPU acceleration.

    Manages buffer allocation, operation execution, and graph optimization to
    minimize CPU↔GPU data transfers. Supports both single-image and batch
    processing.

    Example:
        >>> executor = TaichiPipelineExecutor(arch=TaichiArch.METAL, debug=True)
        >>> executor.prepare(steps, height=1080, width=1920)
        >>> executor.warmup(steps)
        >>> result = executor.process_image(image, steps)
        >>> executor.cleanup()

    """

    def __init__(
        self,
        arch: TaichiArch = TaichiArch.METAL,
        debug: bool = False,
    ) -> None:
        """
        Initialize pipeline executor.

        Args:
            arch: Taichi backend architecture (default: Metal)
            debug: Enable debug mode with synchronization after each kernel

        """
        self._arch = arch
        self._debug = debug
        self._context = TaichiContext()
        self._prepared = False
        self._logger = get_logger()

        # Ensure Taichi is initialized
        if not self._context.is_initialized:
            self._context.startup(arch)

    @property
    def buffer_pool(self) -> PipelineBufferPool:
        """Access the buffer pool for this executor."""
        return self._context.buffer_pool

    def prepare(
        self,
        steps: list[ImageOperationStep],
        height: int,
        width: int,
        batch_size: int = 1,
    ) -> None:
        """
        Analyze pipeline and pre-allocate all required buffers.

        Examines all operations to determine required buffer shapes (accounting
        for dimension-changing operations like downscale) and pre-allocates
        buffer pairs for each unique shape.

        Args:
            steps: Pipeline operation steps
            height: Input image height
            width: Input image width
            batch_size: Number of images to process in parallel (default: 1)

        Raises:
            PipelineError: If pipeline analysis fails

        """
        if self._prepared:
            self._logger.debug("Pipeline already prepared")
            return

        try:
            self._logger.info(
                "Preparing pipeline with %d steps for %dx%dx%d",
                len(steps),
                batch_size,
                height,
                width,
            )

            # Analyze pipeline to determine all required buffer shapes
            required_shapes = self._analyze_shapes(steps, batch_size, height, width)

            # Pre-allocate buffer pairs for each unique shape
            for shape in required_shapes:
                batch, h, w = shape
                self._logger.debug(
                    "Pre-allocating buffers for shape: %dx%dx%d", batch, h, w
                )
                self.buffer_pool.ensure_shape(batch, h, w)

            self._prepared = True
            self._logger.info(
                "Pipeline prepared with %d buffer shapes", len(required_shapes)
            )

        except Exception as e:
            msg = f"Failed to prepare pipeline: {e}"
            raise PipelineError(msg) from e

    def _analyze_shapes(
        self,
        steps: list[ImageOperationStep],
        batch: int,
        height: int,
        width: int,
    ) -> list[tuple[int, int, int]]:
        """
        Analyze pipeline to determine all required buffer shapes.

        Tracks dimension changes through the pipeline to identify all unique
        (batch, height, width) combinations needed for buffer allocation.

        Args:
            steps: Pipeline operation steps
            batch: Initial batch size
            height: Initial height
            width: Initial width

        Returns:
            List of unique (batch, height, width) tuples needed

        """
        shapes: set[tuple[int, int, int]] = set()
        current_h, current_w = height, width

        # Add initial shape
        shapes.add((batch, current_h, current_w))

        # TODO: Implement shape tracking through operations
        # For now, assume all operations maintain dimensions
        # Future: Query each operation for output_shape_factor

        for _step in steps:
            # Placeholder: would query operation.output_shape_factor
            # For dimension-changing ops like downscale
            shapes.add((batch, current_h, current_w))

        return list(shapes)

    def warmup(self, steps: list[ImageOperationStep]) -> None:
        """
        Force JIT compilation with tiny dummy data.

        Runs each operation with minimal 2x2 dummy fields to trigger Taichi's
        JIT compilation before actual processing. This isolates compilation
        overhead from processing time.

        Args:
            steps: Pipeline operation steps to warm up

        Raises:
            GPUOperationError: If warmup fails

        """
        if ti is None:
            return

        try:
            self._logger.info("Warming up pipeline with %d operations", len(steps))

            # Warmup each Taichi operation
            for step in steps:
                if has_taichi_operation(step.operation):
                    op = get_taichi_operation(step.operation)
                    self._logger.debug("Warming up operation: %s", step.operation)
                    op.warmup()

            self._logger.info("Pipeline warmup completed")

        except Exception as e:
            msg = f"Pipeline warmup failed: {e}"
            raise GPUOperationError(msg) from e

    def process_image(
        self,
        image: Image.Image,
        steps: list[ImageOperationStep],
    ) -> Image.Image:
        """
        Process single image through pipeline.

        Uploads image to GPU once, executes all operations on GPU, then downloads
        final result. This minimizes CPU↔GPU transfers.

        Args:
            image: Input PIL Image
            steps: Pipeline operation steps

        Returns:
            Processed PIL Image

        Raises:
            PipelineError: If pipeline is not prepared
            GPUOperationError: If processing fails

        """
        if not self._prepared:
            # Auto-prepare with image dimensions
            self.prepare(steps, image.height, image.width, batch_size=1)

        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)

            # Get buffer pair for image dimensions
            height, width = image.height, image.width
            buffer_pair = self.buffer_pool.get_pair(1, height, width)

            # Upload to GPU (once)
            self.buffer_pool.load_image(img_array, buffer_pair.a, batch_idx=0)

            # Execute pipeline operations
            current_buffer = buffer_pair.a
            next_buffer = buffer_pair.b

            for step_idx, step in enumerate(steps):
                self._logger.debug(
                    "Executing step %d/%d: %s (%s)",
                    step_idx + 1,
                    len(steps),
                    step.name,
                    step.operation,
                )

                # Execute operation on GPU
                self._execute_step(
                    step=step,
                    source=current_buffer,
                    dest=next_buffer,
                    height=height,
                    width=width,
                )

                if self._debug and ti is not None:
                    ti.sync()  # Force synchronization for error detection

                # Swap buffers for ping-pong
                current_buffer, next_buffer = next_buffer, current_buffer

            # Download from GPU (once)
            result_array = self.buffer_pool.extract_result(current_buffer, batch_idx=0)

            # Convert back to PIL Image
            return Image.fromarray(result_array)

        except Exception as e:
            msg = f"Image processing failed: {e}"
            raise GPUOperationError(msg) from e

    def _execute_step(
        self,
        step: ImageOperationStep,
        source: object,  # ti.Vector.field
        dest: object,  # ti.Vector.field
        height: int,
        width: int,
    ) -> None:
        """
        Execute a single pipeline step on GPU.

        Args:
            step: Pipeline step to execute
            source: Source Taichi Vector.field
            dest: Destination Taichi Vector.field
            height: Image height
            width: Image width

        Raises:
            GPUOperationError: If operation not found or execution fails

        """
        operation_name = step.operation

        # Check if Taichi implementation exists
        if not has_taichi_operation(operation_name):
            msg = (
                f"Operation '{operation_name}' has no Taichi implementation. "
                f"Cannot execute in end-to-end GPU mode."
            )
            raise GPUOperationError(msg)

        # Get the Taichi operation
        op = get_taichi_operation(operation_name)

        # Validate parameters
        op.validate_params(step.params)

        # Execute operation (with repeat support)
        for repeat_idx in range(step.repeat):
            if step.repeat > 1:
                self._logger.debug(
                    "Step '%s' repeat %d/%d", step.name, repeat_idx + 1, step.repeat
                )

            # Execute on GPU fields
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},  # TODO: Support temp fields
                params=step.params,
                height=height,
                width=width,
            )

            # For repeated operations, swap source/dest
            if repeat_idx < step.repeat - 1:
                source, dest = dest, source

    def process_batch(
        self,
        images: list[Image.Image],
        steps: list[ImageOperationStep],
    ) -> list[Image.Image]:
        """
        Process multiple images with shared buffers.

        Processes a batch of images through the pipeline, amortizing GPU setup
        overhead across all images.

        Args:
            images: List of input PIL Images (must all have same dimensions)
            steps: Pipeline operation steps

        Returns:
            List of processed PIL Images

        Raises:
            ValueError: If images have different dimensions
            PipelineError: If pipeline is not prepared
            GPUOperationError: If processing fails

        """
        if not images:
            return []

        # Validate all images have same dimensions
        first_size = (images[0].height, images[0].width)
        if not all((img.height, img.width) == first_size for img in images):
            msg = "All images in batch must have same dimensions"
            raise ValueError(msg)

        # Prepare for batch processing
        batch_size = len(images)
        height, width = first_size

        if not self._prepared:
            self.prepare(steps, height, width, batch_size)

        try:
            self._logger.info("Processing batch of %d images", batch_size)

            # Process each image individually for now
            # TODO: Implement true batch processing with batch dimension
            results = []
            for idx, image in enumerate(images):
                self._logger.debug("Processing image %d/%d", idx + 1, batch_size)
                result = self.process_image(image, steps)
                results.append(result)

            return results

        except Exception as e:
            msg = f"Batch processing failed: {e}"
            raise GPUOperationError(msg) from e

    def _coalesce_operations(
        self, steps: list[ImageOperationStep]
    ) -> list[list[ImageOperationStep]]:
        """
        Group consecutive native/legacy operations to minimize transfers.

        Analyzes the pipeline to identify sequences of operations that can be
        executed together on the same device (GPU or CPU), reducing the number
        of data transfers between devices.

        Args:
            steps: Pipeline operation steps

        Returns:
            List of grouped operation sequences
            [[native_ops], [legacy_ops], [native_ops], ...]

        """
        # TODO: Implement operation grouping
        # Requires knowing which operations are GPU-native vs CPU-only
        # For now, assume all operations are GPU-capable

        # Placeholder: return all steps as single GPU group
        return [steps]

    def cleanup(self) -> None:
        """
        Release resources and clean up executor state.

        Clears prepared state but does not shut down the shared Taichi context.
        Call TaichiContext.shutdown() separately for full cleanup.
        """
        self._prepared = False
        self._logger.debug("Executor cleanup completed")

    def __enter__(self) -> "TaichiPipelineExecutor":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.cleanup()

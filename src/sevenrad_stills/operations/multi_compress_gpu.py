"""
GPU-accelerated multi-generation compression for compound quality degradation.

NOTE: This operation is fundamentally CPU-bound due to the inherently serial
nature of Huffman encoding in JPEG compression. Following industry standards
(NVIDIA nvJPEG, AMD GPUOpen), we use a hybrid approach:
- CPU (PIL/libjpeg-turbo): JPEG encoding/decoding with Huffman coding
- GPU (Taichi): Reserved for future parallelizable preprocessing operations

This implementation exists for API consistency within GPU-based image processing
pipelines. The iterative nature of multi-compression (iteration N depends on N-1)
combined with serial Huffman encoding makes GPU acceleration impractical, as
GPU↔CPU memory transfer overhead would exceed any potential gains.

Performance Characteristics:
- CPU version: Optimal (uses libjpeg-turbo)
- GPU version: Similar to CPU (CPU-bound by Huffman encoding)
- Metal version: Similar to CPU (CPU-bound by Huffman encoding)

For more details on why JPEG encoding remains CPU-bound, see:
https://developer.nvidia.com/blog/leveraging-hardware-jpeg-decoder-and-nvjpeg-on-a100/
"""

import io
import math
from typing import Any, Literal

import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
MIN_ITERATIONS = 1
MAX_ITERATIONS = 50
MIN_QUALITY = 1
MAX_QUALITY = 100

# Decay types
DecayType = Literal["linear", "exponential", "fixed"]


class MultiCompressGPUOperation(BaseImageOperation):
    """
    Apply JPEG compression multiple times to compound artifacts (GPU version).

    **Performance Note**: This operation is CPU-bound due to serial Huffman encoding.
    The GPU version exists for API consistency but delegates all work to PIL's
    highly optimized CPU implementation (libjpeg-turbo).

    Simulates multiple save/load cycles where each iteration compounds
    quality loss. Supports various quality decay curves.

    Iterations:
    - 2-3: Subtle multi-generation artifacts
    - 4-6: Noticeable compound degradation
    - 7-10: Heavy compound artifacts
    - 10+: Extreme degradation

    Decay types:
    - fixed: Use same quality each iteration
    - linear: Quality decreases linearly from start to end
    - exponential: Quality decreases rapidly then levels off
    """

    def __init__(self) -> None:
        """Initialize GPU multi-compress operation."""
        super().__init__("multi_compress_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901, PLR0912
        """
        Validate multi-compress operation parameters.

        Expected params:
        - iterations: int (1-50) - Number of compression cycles
        - quality_start: int (1-100) - Starting quality level
        - quality_end: int (1-100) - Ending quality level (for decay modes)
        - decay: str - Quality decay type: "fixed", "linear", "exponential"
        - subsampling: int (0, 1, or 2) - Chroma subsampling mode (optional)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        # Validate iterations
        if "iterations" not in params:
            msg = "Multi-compress operation requires 'iterations' parameter"
            raise ValueError(msg)

        iterations = params["iterations"]
        if not isinstance(iterations, int):
            msg = f"Iterations must be an integer, got {type(iterations)}"
            raise ValueError(msg)
        if not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS:
            msg = (
                f"Iterations must be between {MIN_ITERATIONS} and "
                f"{MAX_ITERATIONS}, got {iterations}"
            )
            raise ValueError(msg)

        # Validate quality_start
        if "quality_start" not in params:
            msg = "Multi-compress operation requires 'quality_start' parameter"
            raise ValueError(msg)

        quality_start = params["quality_start"]
        if not isinstance(quality_start, int):
            msg = f"Quality start must be an integer, got {type(quality_start)}"
            raise ValueError(msg)
        if not MIN_QUALITY <= quality_start <= MAX_QUALITY:
            msg = (
                f"Quality start must be between {MIN_QUALITY} and "
                f"{MAX_QUALITY}, got {quality_start}"
            )
            raise ValueError(msg)

        # Validate decay type
        decay: str = params.get("decay", "fixed")
        if decay not in ("fixed", "linear", "exponential"):
            msg = f"Decay must be 'fixed', 'linear', or 'exponential', got {decay}"
            raise ValueError(msg)

        # Validate quality_end if not using fixed decay
        if decay != "fixed":
            if "quality_end" not in params:
                msg = f"Decay mode '{decay}' requires 'quality_end' parameter"
                raise ValueError(msg)

            quality_end = params["quality_end"]
            if not isinstance(quality_end, int):
                msg = f"Quality end must be an integer, got {type(quality_end)}"
                raise ValueError(msg)
            if not MIN_QUALITY <= quality_end <= MAX_QUALITY:
                msg = (
                    f"Quality end must be between {MIN_QUALITY} and "
                    f"{MAX_QUALITY}, got {quality_end}"
                )
                raise ValueError(msg)
            if quality_end >= quality_start:
                msg = (
                    f"Quality end ({quality_end}) must be less than "
                    f"quality start ({quality_start})"
                )
                raise ValueError(msg)

        # Validate subsampling if provided
        if "subsampling" in params:
            subsampling = params["subsampling"]
            if not isinstance(subsampling, int):
                msg = f"Subsampling must be an integer, got {type(subsampling)}"
                raise ValueError(msg)
            if subsampling not in (0, 1, 2):
                msg = f"Subsampling must be 0, 1, or 2, got {subsampling}"
                raise ValueError(msg)

    def _calculate_quality(
        self,
        iteration: int,
        total_iterations: int,
        quality_start: int,
        quality_end: int,
        decay: DecayType,
    ) -> int:
        """
        Calculate quality for a given iteration based on decay type.

        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total number of iterations
            quality_start: Starting quality
            quality_end: Ending quality
            decay: Decay type

        Returns:
            Quality value for this iteration

        """
        if decay == "fixed":
            return quality_start

        if total_iterations == 1:
            return quality_start

        # Calculate progress (0.0 to 1.0)
        progress = iteration / (total_iterations - 1)

        if decay == "linear":
            # Linear interpolation
            quality = quality_start + (quality_end - quality_start) * progress
            return int(quality)

        # Exponential decay
        # Use exponential function: Q = Q_start * e^(-k*progress)
        # Solve for k such that Q(1) = Q_end
        # k = -ln(Q_end / Q_start)
        if quality_end == 0:
            quality_end = 1  # Avoid log(0)

        k = -math.log(quality_end / quality_start)
        quality = quality_start * math.exp(-k * progress)
        return max(MIN_QUALITY, int(quality))

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply multi-generation compression to image (CPU-based).

        **Implementation Note**: Despite being a GPU operation class, this
        implementation uses CPU-based JPEG encoding via PIL. This is intentional
        and follows industry best practices (NVIDIA nvJPEG, AMD GPUOpen).

        The iterative compression loop has strict data dependencies (iteration N
        requires result from N-1) and relies on Huffman encoding, which is
        inherently serial. GPU acceleration would introduce memory transfer
        overhead without providing performance benefits.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Multi-compressed PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        iterations: int = params["iterations"]
        quality_start: int = params["quality_start"]
        quality_end: int = params.get("quality_end", quality_start)
        decay: DecayType = params.get("decay", "fixed")
        subsampling: int = params.get("subsampling", 2)

        # Convert to RGB if necessary
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        current_image = image

        # Apply compression iteratively (CPU-based via PIL)
        for i in range(iterations):
            # Calculate quality for this iteration
            quality = self._calculate_quality(
                i, iterations, quality_start, quality_end, decay
            )

            # Apply compression via in-memory save/load
            buffer = io.BytesIO()
            current_image.save(
                buffer,
                format="JPEG",
                quality=quality,
                subsampling=subsampling,
                optimize=True,
            )
            buffer.seek(0)
            current_image = Image.open(buffer)
            current_image.load()
            buffer.close()

        return current_image

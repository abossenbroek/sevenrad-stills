"""
GPU-accelerated JPEG compression with Taichi gamma correction.

Provides GPU-accelerated gamma correction for JPEG compression operations.
Uses Taichi for GPU acceleration of the gamma correction step, while leveraging
PIL for the JPEG encoding itself (hybrid CPU/GPU approach).
"""

import io
from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants (same as CPU version)
MIN_QUALITY = 1
MAX_QUALITY = 100
DEFAULT_QUALITY = 85


@ti.kernel  # type: ignore[misc]
def apply_gamma_correction(  # type: ignore[no-untyped-def]
    input_img: ti.template(),  # type: ignore[valid-type]
    output_img: ti.template(),  # type: ignore[valid-type]
    gamma: ti.f32,
    height: ti.i32,
    width: ti.i32,
    channels: ti.i32,
):
    """
    GPU kernel to apply gamma correction element-wise.

    Args:
        input_img: Input image field (normalized to 0.0-1.0).
        output_img: Output image field.
        gamma: Gamma correction factor.
        height: Image height.
        width: Image width.
        channels: Number of channels.

    """
    for i, j, c in ti.ndrange(height, width, channels):
        # Apply power function for gamma correction
        output_img[i, j, c] = ti.pow(input_img[i, j, c], gamma)


class CompressionGPUOperation(BaseImageOperation):
    """
    GPU-accelerated JPEG compression with configurable quality and gamma.

    This hybrid implementation uses Taichi for GPU-accelerated gamma correction
    while leveraging PIL's optimized JPEG encoder for the compression step.

    Subsampling modes:
    - 0 (4:4:4): No subsampling - highest quality, minimal artifacts
    - 1 (4:2:2): Moderate subsampling - visible artifacts in color transitions
    - 2 (4:2:0): Heavy subsampling - severe 8x8 blocking, default JPEG behavior

    Quality range 1-100:
    - 1-15: Severe compression, heavy blocking and banding
    - 16-50: Moderate compression, visible artifacts
    - 51-85: Standard compression, balanced quality/size
    - 86-100: High quality, minimal artifacts

    Performance: GPU acceleration provides significant speedup for gamma correction
    on large images, especially when gamma is specified (non-None).
    """

    def __init__(self) -> None:
        """Initialize GPU-accelerated compression operation."""
        super().__init__("compression_gpu")

    def _validate_gamma(self, gamma: object) -> None:
        """
        Validate gamma correction parameter.

        Args:
            gamma: Gamma value to validate

        Raises:
            ValueError: If gamma is invalid

        """
        if not isinstance(gamma, (int, float)):
            msg = f"Gamma must be a number, got {type(gamma)}"
            raise ValueError(msg)
        if gamma <= 0:
            msg = f"Gamma must be positive, got {gamma}"
            raise ValueError(msg)

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate compression operation parameters.

        Expected params:
        - quality: int (1-100) - JPEG quality level
        - subsampling: int (0, 1, or 2) - Chroma subsampling mode (optional)
        - optimize: bool - Apply JPEG optimization (optional, default True)
        - gamma: float | None - Gamma correction factor (optional, default None)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "quality" not in params:
            msg = "Compression operation requires 'quality' parameter"
            raise ValueError(msg)

        quality = params["quality"]
        if not isinstance(quality, int):
            msg = f"Quality must be an integer, got {type(quality)}"
            raise ValueError(msg)
        if not MIN_QUALITY <= quality <= MAX_QUALITY:
            msg = (
                f"Quality must be between {MIN_QUALITY} and {MAX_QUALITY}, "
                f"got {quality}"
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

        # Validate optimize if provided
        if "optimize" in params:
            optimize = params["optimize"]
            if not isinstance(optimize, bool):
                msg = f"Optimize must be a boolean, got {type(optimize)}"
                raise ValueError(msg)

        # Validate gamma if provided
        if "gamma" in params and params["gamma"] is not None:
            self._validate_gamma(params["gamma"])

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated JPEG compression with optional gamma correction.

        The gamma correction (if specified) is GPU-accelerated using Taichi,
        while JPEG compression uses PIL's optimized encoder.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Compressed PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        quality: int = params["quality"]
        subsampling: int = params.get("subsampling", 2)  # Default to 4:2:0
        optimize: bool = params.get("optimize", True)
        gamma: float | None = params.get("gamma")

        # Convert to RGB if necessary (JPEG doesn't support transparency)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Apply GPU-accelerated gamma correction if specified
        if gamma is not None:
            img_array = np.array(image, dtype=np.float32) / 255.0
            height, width = img_array.shape[:2]

            # Handle both RGB and grayscale
            if img_array.ndim == 3:  # noqa: PLR2004
                channels = img_array.shape[2]
            else:
                channels = 1
                img_array = img_array.reshape(height, width, 1)

            # Create Taichi fields
            input_field = ti.field(dtype=ti.f32, shape=(height, width, channels))
            output_field = ti.field(dtype=ti.f32, shape=(height, width, channels))

            # Copy data to GPU
            input_field.from_numpy(img_array)

            # Apply gamma correction on GPU
            apply_gamma_correction(
                input_field, output_field, gamma, height, width, channels
            )

            # Copy result back to CPU
            gamma_corrected = output_field.to_numpy()

            # Convert back to uint8
            gamma_corrected_uint8 = (np.clip(gamma_corrected, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )

            # Handle channel dimension
            if channels == 1:
                gamma_corrected_uint8 = gamma_corrected_uint8.reshape(height, width)

            image = Image.fromarray(gamma_corrected_uint8)

        # Apply compression via in-memory save/load cycle
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=optimize,
        )
        buffer.seek(0)
        compressed_image = Image.open(buffer)

        # Load the image data to ensure buffer can be closed
        compressed_image.load()
        buffer.close()

        return compressed_image

"""
GPU-accelerated resolution downscaling operation using Taichi.

Provides control over scale factors and resampling methods to achieve
various levels of pixelation and loss of detail. Uses Taichi for GPU
acceleration of the resizing operations.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants (same as CPU version)
MIN_SCALE = 0.01
MAX_SCALE = 1.0

# Resampling method mapping for GPU
RESAMPLING_METHODS = {
    "nearest": 0,  # Harsh pixelation
    "bilinear": 1,  # Smooth downscaling
}


@ti.kernel  # type: ignore[misc]
def resize_nearest(  # type: ignore[no-untyped-def]
    src: ti.types.ndarray(),  # type: ignore[valid-type]
    dst: ti.types.ndarray(),  # type: ignore[valid-type]
    src_h: ti.i32,
    src_w: ti.i32,
    dst_h: ti.i32,
    dst_w: ti.i32,
    channels: ti.i32,
):
    """
    GPU kernel for nearest-neighbor resampling.

    Args:
        src: Source image array
        dst: Destination image array
        src_h: Source height
        src_w: Source width
        dst_h: Destination height
        dst_w: Destination width
        channels: Number of color channels

    """
    for y, x, c in ti.ndrange(dst_h, dst_w, channels):
        # Calculate source coordinates (nearest neighbor)
        src_y = ti.cast(y * src_h / dst_h, ti.i32)
        src_x = ti.cast(x * src_w / dst_w, ti.i32)

        # Clamp to valid range
        src_y = ti.max(0, ti.min(src_h - 1, src_y))
        src_x = ti.max(0, ti.min(src_w - 1, src_x))

        # Copy pixel value
        dst[y, x, c] = src[src_y, src_x, c]


@ti.kernel  # type: ignore[misc]
def resize_bilinear(  # type: ignore[no-untyped-def]
    src: ti.types.ndarray(),  # type: ignore[valid-type]
    dst: ti.types.ndarray(),  # type: ignore[valid-type]
    src_h: ti.i32,
    src_w: ti.i32,
    dst_h: ti.i32,
    dst_w: ti.i32,
    channels: ti.i32,
):
    """
    GPU kernel for bilinear resampling.

    Args:
        src: Source image array
        dst: Destination image array
        src_h: Source height
        src_w: Source width
        dst_h: Destination height
        dst_w: Destination width
        channels: Number of color channels

    """
    for y, x, c in ti.ndrange(dst_h, dst_w, channels):
        # Calculate source coordinates (floating point)
        src_y_f = (y + 0.5) * src_h / dst_h - 0.5
        src_x_f = (x + 0.5) * src_w / dst_w - 0.5

        # Get integer parts (floor)
        src_y0 = ti.cast(ti.floor(src_y_f), ti.i32)
        src_x0 = ti.cast(ti.floor(src_x_f), ti.i32)
        src_y1 = src_y0 + 1
        src_x1 = src_x0 + 1

        # Clamp to valid range
        src_y0 = ti.max(0, ti.min(src_h - 1, src_y0))
        src_y1 = ti.max(0, ti.min(src_h - 1, src_y1))
        src_x0 = ti.max(0, ti.min(src_w - 1, src_x0))
        src_x1 = ti.max(0, ti.min(src_w - 1, src_x1))

        # Get fractional parts
        fy = src_y_f - ti.floor(src_y_f)
        fx = src_x_f - ti.floor(src_x_f)

        # Bilinear interpolation
        v00 = ti.cast(src[src_y0, src_x0, c], ti.f32)
        v01 = ti.cast(src[src_y0, src_x1, c], ti.f32)
        v10 = ti.cast(src[src_y1, src_x0, c], ti.f32)
        v11 = ti.cast(src[src_y1, src_x1, c], ti.f32)

        v0 = v00 * (1.0 - fx) + v01 * fx
        v1 = v10 * (1.0 - fx) + v11 * fx
        result = v0 * (1.0 - fy) + v1 * fy

        # Clamp and store
        dst[y, x, c] = ti.cast(ti.max(0.0, ti.min(255.0, result)), ti.u8)


class DownscaleGPUOperation(BaseImageOperation):
    """
    GPU-accelerated downscale operation using Taichi.

    Downscale image resolution to create pixelation effects with GPU acceleration.
    Supports downscaling to a fraction of original size, with optional
    upscaling back to original dimensions to create visible pixelation.

    Scale factor:
    - 0.01-0.10: Extreme pixelation, heavily degraded
    - 0.10-0.25: Heavy pixelation, architectural details lost
    - 0.25-0.50: Moderate pixelation, visible block structures
    - 0.50-1.00: Subtle quality reduction

    Resampling methods:
    - nearest: Maximum pixelation, harsh block edges
    - bilinear: Softer pixelation with blended edges

    Note: GPU version supports only nearest and bilinear methods for optimal
    performance. For bicubic/lanczos, use the CPU version.

    Performance: GPU acceleration provides significant speedup for large images,
    as each pixel's interpolation can be parallelized across thousands of GPU threads.
    """

    def __init__(self) -> None:
        """Initialize GPU-accelerated downscale operation."""
        super().__init__("downscale_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
        """
        Validate downscale operation parameters.

        Expected params:
        - scale: float (0.01-1.0) - Scale factor for downscaling
        - upscale: bool - Whether to upscale back to original size (default: True)
        - downscale_method: str - Resampling method for downscaling
          (default: "bilinear")
        - upscale_method: str - Resampling method for upscaling
          (default: "nearest")

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "scale" not in params:
            msg = "Downscale operation requires 'scale' parameter"
            raise ValueError(msg)

        scale = params["scale"]
        if not isinstance(scale, (int, float)):
            msg = f"Scale must be a number, got {type(scale)}"
            raise ValueError(msg)
        if not MIN_SCALE <= scale <= MAX_SCALE:
            msg = f"Scale must be between {MIN_SCALE} and {MAX_SCALE}, got {scale}"
            raise ValueError(msg)

        # Validate upscale if provided
        if "upscale" in params:
            upscale = params["upscale"]
            if not isinstance(upscale, bool):
                msg = f"Upscale must be a boolean, got {type(upscale)}"
                raise ValueError(msg)

        # Validate downscale_method if provided
        if "downscale_method" in params:
            method = params["downscale_method"]
            if not isinstance(method, str):
                msg = f"Downscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = (
                    f"Invalid downscale method '{method}'. "
                    f"GPU version supports: {available}"
                )
                raise ValueError(msg)

        # Validate upscale_method if provided
        if "upscale_method" in params:
            method = params["upscale_method"]
            if not isinstance(method, str):
                msg = f"Upscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = (
                    f"Invalid upscale method '{method}'. "
                    f"GPU version supports: {available}"
                )
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated downscaling to image.

        Args:
            image: Input PIL Image (RGB or RGBA)
            params: Operation parameters (validated)

        Returns:
            Downscaled (and optionally upscaled) PIL Image

        Raises:
            ValueError: If image is not RGB or RGBA mode

        """
        # Validate parameters first
        self.validate_params(params)

        # Only support RGB/RGBA for now
        if image.mode not in ("RGB", "RGBA"):
            msg = f"GPU downscale requires RGB or RGBA image, got {image.mode}"
            raise ValueError(msg)

        scale: float = params["scale"]
        upscale: bool = params.get("upscale", True)
        downscale_method_name: str = params.get("downscale_method", "bilinear")
        upscale_method_name: str = params.get("upscale_method", "nearest")

        # Convert to numpy array
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
        else:
            rgb = img_array.copy()
            alpha = None

        h, w = rgb.shape[:2]

        # Calculate new size
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))

        # Downscale using GPU
        downscaled = self._resize_gpu(
            rgb, new_height, new_width, RESAMPLING_METHODS[downscale_method_name]
        )

        # Upscale back if requested
        if upscale:
            result = self._resize_gpu(
                downscaled, h, w, RESAMPLING_METHODS[upscale_method_name]
            )
        else:
            result = downscaled

        # Recombine with alpha if needed
        if alpha is not None:
            # Resize alpha channel to match result
            if upscale:
                alpha_resized = alpha
            else:
                # Downscale alpha to match
                alpha_2d = alpha.reshape(h, w, 1)
                alpha_down = self._resize_gpu(
                    alpha_2d, new_height, new_width, RESAMPLING_METHODS["nearest"]
                )
                alpha_resized = alpha_down.reshape(new_height, new_width)

            output_array = np.dstack([result, alpha_resized])
        else:
            output_array = result

        return Image.fromarray(output_array)

    def _resize_gpu(
        self, img: np.ndarray, dst_h: int, dst_w: int, method: int
    ) -> np.ndarray:
        """
        Resize image using GPU kernels.

        Args:
            img: Input image array (H, W, C)
            dst_h: Destination height
            dst_w: Destination width
            method: Resampling method (0=nearest, 1=bilinear)

        Returns:
            Resized image array

        """
        src_h, src_w, channels = img.shape

        # Allocate output array
        output = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)

        # Call appropriate GPU kernel
        if method == 0:  # nearest
            resize_nearest(img, output, src_h, src_w, dst_h, dst_w, channels)
        else:  # bilinear
            resize_bilinear(img, output, src_h, src_w, dst_h, dst_w, channels)

        return output

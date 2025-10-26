"""
GPU-accelerated chromatic aberration effect using Taichi.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses Taichi for GPU acceleration on macOS Metal,
providing significant performance improvements over CPU-based scipy.ndimage.shift.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)


@ti.kernel  # type: ignore[misc]
def apply_chromatic_aberration_fused(  # type: ignore[no-untyped-def]
    input_img: ti.types.ndarray(),  # type: ignore[valid-type]
    output_img: ti.types.ndarray(),  # type: ignore[valid-type]
    shift_y: ti.i32,
    shift_x: ti.i32,
    height: ti.i32,
    width: ti.i32,
):
    """
    Optimized fused GPU kernel processing all RGB channels in a single pass.

    This kernel eliminates the need for multiple kernel launches and array copies
    by processing all three channels together. It provides better memory coalescing
    and reduces GPU overhead significantly.

    Performance optimizations:
    - Single kernel launch for all channels (vs 2 separate launches)
    - Direct RGB array processing (no channel extraction/copying)
    - Better memory coalescing with contiguous RGB reads/writes
    - Reduced CPU-GPU synchronization overhead

    Args:
        input_img: Input RGB image array (H, W, 3)
        output_img: Output RGB image array (H, W, 3)
        shift_y: Vertical shift in pixels (positive = down)
        shift_x: Horizontal shift in pixels (positive = right)
        height: Image height
        width: Image width

    """
    for i, j in ti.ndrange(height, width):
        # Red channel: shift in positive direction
        src_r_i = i - shift_y
        src_r_j = j - shift_x

        if 0 <= src_r_i < height and 0 <= src_r_j < width:
            output_img[i, j, 0] = input_img[src_r_i, src_r_j, 0]
        else:
            output_img[i, j, 0] = 0

        # Green channel: no shift (reference channel)
        output_img[i, j, 1] = input_img[i, j, 1]

        # Blue channel: shift in negative direction (opposite of red)
        src_b_i = i + shift_y
        src_b_j = j + shift_x

        if 0 <= src_b_i < height and 0 <= src_b_j < width:
            output_img[i, j, 2] = input_img[src_b_i, src_b_j, 2]
        else:
            output_img[i, j, 2] = 0


class ChromaticAberrationGPUOperation(BaseImageOperation):
    """
    GPU-accelerated chromatic aberration simulation using Taichi.

    Chromatic aberration is a common optical phenomenon where a lens
    fails to focus all colors to the same point, causing color fringing
    at edges. This operation simulates the effect by shifting the red
    and blue channels in opposite directions.

    The GPU implementation provides significant performance improvements,
    especially for large images, by parallelizing the shift operation
    across thousands of GPU threads.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated operation."""
        super().__init__("chromatic_aberration_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters.

        Args:
            params: A dictionary containing:
                - shift_x (int): Horizontal shift in pixels.
                - shift_y (int): Vertical shift in pixels.

        Raises:
            ValueError: If parameters are invalid.

        """
        for key in ("shift_x", "shift_y"):
            if key not in params:
                msg = f"Parameter '{key}' is required."
                raise ValueError(msg)
            if not isinstance(params[key], int):
                msg = f"Parameter '{key}' must be an integer."
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated chromatic aberration with optimized fused kernel.

        The green channel is kept as reference, while red and blue channels
        are shifted in opposite directions to create color fringing. All three
        channels are processed in a single fused GPU kernel for maximum performance.

        Performance: Uses a single kernel launch processing all RGB channels together,
        eliminating overhead from multiple kernel launches and array copies.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'shift_x' and 'shift_y'.

        Returns:
            The transformed PIL Image.

        """
        self.validate_params(params)
        shift_x: int = params["shift_x"]
        shift_y: int = params["shift_y"]

        if image.mode not in ("RGB", "RGBA"):
            return image.copy()  # Effect only applies to color images

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Separate RGB from alpha if needed
        if image.mode == "RGBA":
            rgb = np.ascontiguousarray(img_array[..., :3])
            alpha = img_array[..., 3]
        else:
            rgb = np.ascontiguousarray(img_array)
            alpha = None

        # Create output array (contiguous for GPU)
        output_rgb = np.zeros_like(rgb)

        # Single fused kernel launch for all RGB channels
        apply_chromatic_aberration_fused(
            rgb,
            output_rgb,
            shift_y,
            shift_x,
            height,
            width,
        )

        # Recombine with alpha if needed
        if alpha is not None:
            output_array = np.dstack([output_rgb, alpha])
        else:
            output_array = output_rgb

        return Image.fromarray(output_array)

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
def apply_channel_shift(  # type: ignore[no-untyped-def]
    input_channel: ti.types.ndarray(),  # type: ignore[valid-type]
    output_channel: ti.types.ndarray(),  # type: ignore[valid-type]
    shift_y: ti.i32,
    shift_x: ti.i32,
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to shift a single channel with constant boundary mode.

    This kernel reads from input_channel and writes shifted values to output_channel.
    Pixels that would read from outside the image boundaries are set to 0 (black),
    matching scipy's mode='constant', cval=0 behavior.

    Args:
        input_channel: Input channel array (H, W)
        output_channel: Output channel array (H, W)
        shift_y: Vertical shift in pixels (positive = down)
        shift_x: Horizontal shift in pixels (positive = right)
        height: Image height
        width: Image width

    """
    for i, j in ti.ndrange(height, width):
        # Calculate source position (reverse shift)
        src_i = i - shift_y
        src_j = j - shift_x

        # Check if source is within bounds (constant boundary mode)
        if 0 <= src_i < height and 0 <= src_j < width:
            output_channel[i, j] = input_channel[src_i, src_j]
        else:
            output_channel[i, j] = 0  # Constant value (black)


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
        Apply GPU-accelerated chromatic aberration.

        The green channel is kept as reference, while red and blue channels
        are shifted in opposite directions to create color fringing. The shift
        operation is parallelized on GPU for improved performance.

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

        # Create output array
        output_array = np.zeros_like(img_array)

        # Process RGB channels with GPU acceleration
        # Define shifts for R, G, B channels. G channel is kept as reference.
        # R is shifted one way, B is shifted the other.
        shifts = {
            0: (shift_y, shift_x),  # Red channel shift
            1: (0, 0),  # Green channel (no shift)
            2: (-shift_y, -shift_x),  # Blue channel shift
        }

        for i in range(3):  # Iterate through R, G, B
            if shifts[i] == (0, 0):
                # Green channel - direct copy (no shift)
                output_array[..., i] = img_array[..., i]
            else:
                # Red or Blue channel - apply GPU shift
                # Taichi requires contiguous arrays
                input_channel = np.ascontiguousarray(img_array[..., i])
                output_channel = np.ascontiguousarray(output_array[..., i])

                sy, sx = shifts[i]
                apply_channel_shift(
                    input_channel,
                    output_channel,
                    sy,
                    sx,
                    height,
                    width,
                )

                # Copy result back
                output_array[..., i] = output_channel

        # Preserve alpha channel if it exists
        if image.mode == "RGBA":
            output_array[..., 3] = img_array[..., 3]

        return Image.fromarray(output_array)

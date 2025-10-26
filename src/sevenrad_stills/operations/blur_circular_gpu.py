"""
GPU-accelerated circular blur operation for bokeh effects using Taichi.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses Taichi for GPU acceleration, providing significant
performance improvements over CPU-based convolution.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi (will use GPU if available, otherwise CPU)
ti.init(arch=ti.gpu)

# Constants
RGB_CHANNELS = 3  # Number of color channels in RGB image


@ti.kernel  # type: ignore[misc]
def create_circular_kernel(kernel: ti.template(), radius: ti.i32):  # type: ignore[valid-type]
    """
    Create a circular (disc-shaped) convolution kernel on GPU.

    Args:
        kernel: Taichi field to store the kernel.
        radius: The radius of the circular kernel.

    """
    diameter = 2 * radius + 1
    total = 0.0

    # First pass: create mask and count pixels
    for i, j in ti.ndrange(diameter, diameter):
        x = i - radius
        y = j - radius
        if x * x + y * y <= radius * radius:
            kernel[i, j] = 1.0
            total += 1.0
        else:
            kernel[i, j] = 0.0

    # Second pass: normalize
    for i, j in kernel:
        if total > 0:
            kernel[i, j] /= total


@ti.kernel  # type: ignore[misc]
def convolve_2d(  # type: ignore[valid-type]
    input_img: ti.template(),
    output_img: ti.template(),
    kernel: ti.template(),
    radius: ti.i32,
    height: ti.i32,
    width: ti.i32,
):
    """
    Perform 2D convolution with reflect boundary mode on GPU.

    Args:
        input_img: Input image field.
        output_img: Output image field.
        kernel: Convolution kernel field.
        radius: Kernel radius.
        height: Image height.
        width: Image width.

    """
    diameter = 2 * radius + 1

    for i, j in ti.ndrange(height, width):
        sum_val = 0.0

        for ki, kj in ti.ndrange(diameter, diameter):
            # Calculate source pixel coordinates
            src_i = i + ki - radius
            src_j = j + kj - radius

            # Reflect boundary handling
            if src_i < 0:
                src_i = -src_i
            elif src_i >= height:
                src_i = 2 * height - src_i - 2

            if src_j < 0:
                src_j = -src_j
            elif src_j >= width:
                src_j = 2 * width - src_j - 2

            # Clamp to valid range (for edge cases in reflection)
            src_i = ti.max(0, ti.min(height - 1, src_i))
            src_j = ti.max(0, ti.min(width - 1, src_j))

            sum_val += input_img[src_i, src_j] * kernel[ki, kj]

        output_img[i, j] = sum_val


class CircularBlurGPUOperation(BaseImageOperation):
    """
    Apply a circular blur to an image for bokeh effects using GPU acceleration.

    Uses a circular (disc-shaped) kernel to create a bokeh-like blur effect,
    simulating the behavior of a camera lens with a circular aperture. This
    implementation leverages Taichi for GPU acceleration, providing significant
    performance improvements for large images or large kernel radii.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated circular blur operation."""
        super().__init__("blur_circular_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the circular blur operation.

        Args:
            params: A dictionary containing:
                - radius (int): The radius of the circular kernel.
                  Must be a positive integer. A radius of 0 returns the original image.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "radius" not in params:
            msg = "Circular blur requires a 'radius' parameter."
            raise ValueError(msg)

        radius = params["radius"]
        if not isinstance(radius, int):
            msg = f"Radius must be an integer, got {type(radius)}."
            raise ValueError(msg)
        if radius < 0:
            msg = f"Radius must be non-negative, got {radius}."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply circular blur to the image using GPU acceleration.

        Args:
            image: The input PIL Image.
            params: A dictionary with a 'radius' key.

        Returns:
            The blurred PIL Image.

        """
        self.validate_params(params)
        radius: int = params["radius"]

        # If radius is zero, no blur is applied. Return original image.
        if radius == 0:
            return image.copy()

        # Convert PIL image to NumPy array
        img_array = np.array(image)
        original_dtype = img_array.dtype

        # Handle RGBA images separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            rgb_array = np.array(rgb)
            blurred_rgb_array = self._apply_blur_to_array(rgb_array, radius)

            # Restore original data type and create new image
            blurred_rgb = Image.fromarray(
                np.clip(blurred_rgb_array, 0, 255).astype(original_dtype)
            )
            # Create RGBA image with blurred RGB and original alpha
            result = Image.new("RGBA", image.size)
            result.paste(blurred_rgb, (0, 0))
            result.putalpha(alpha)
            return result

        # For RGB or L images, apply blur directly
        blurred_array = self._apply_blur_to_array(img_array, radius)

        # Convert back to PIL Image
        return Image.fromarray(np.clip(blurred_array, 0, 255).astype(original_dtype))

    def _apply_blur_to_array(self, img_array: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply circular blur to a numpy array using GPU.

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB).
            radius: Blur radius.

        Returns:
            Blurred image array.

        """
        # Create kernel
        diameter = 2 * radius + 1
        kernel_field = ti.field(dtype=ti.f32, shape=(diameter, diameter))
        create_circular_kernel(kernel_field, radius)

        # Handle RGB and grayscale images
        if img_array.ndim == RGB_CHANNELS:  # RGB image
            height, width, channels = img_array.shape
            blurred_array = np.zeros_like(img_array, dtype=np.float32)

            # Process each channel separately
            for c in range(channels):
                channel_data: np.ndarray = img_array[..., c].astype(np.float32)

                # Create Taichi fields for this channel
                input_field = ti.field(dtype=ti.f32, shape=(height, width))
                output_field = ti.field(dtype=ti.f32, shape=(height, width))

                # Copy data to GPU
                input_field.from_numpy(channel_data)

                # Perform convolution
                convolve_2d(
                    input_field, output_field, kernel_field, radius, height, width
                )

                # Copy result back to CPU
                blurred_array[..., c] = output_field.to_numpy()

        else:  # Grayscale image
            height, width = img_array.shape
            img_float: np.ndarray = img_array.astype(np.float32)

            # Create Taichi fields
            input_field = ti.field(dtype=ti.f32, shape=(height, width))
            output_field = ti.field(dtype=ti.f32, shape=(height, width))

            # Copy data to GPU
            input_field.from_numpy(img_float)

            # Perform convolution
            convolve_2d(input_field, output_field, kernel_field, radius, height, width)

            # Copy result back to CPU
            blurred_array = output_field.to_numpy()

        return blurred_array  # type: ignore[no-any-return]

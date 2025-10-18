"""
Circular blur operation for bokeh effects.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses SciPy instead of PyTorch for better integration
with PIL-based image processing pipeline.
"""

from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
RGB_CHANNELS = 3  # Number of color channels in RGB image


class CircularBlurOperation(BaseImageOperation):
    """
    Apply a circular blur to an image for bokeh effects.

    Uses a circular (disc-shaped) kernel to create a bokeh-like blur effect,
    simulating the behavior of a camera lens with a circular aperture. The
    blur intensity is controlled by the radius parameter.
    """

    def __init__(self) -> None:
        """Initialize the circular blur operation."""
        super().__init__("blur_circular")

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
        Apply circular blur to the image.

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

        # Create circular kernel
        kernel = self._create_circular_kernel(radius)

        # Convert PIL image to NumPy array. Preserve original dtype.
        img_array = np.array(image)
        original_dtype = img_array.dtype

        # Handle RGBA images separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            rgb_array = np.array(rgb)
            # Apply blur to each channel separately
            blurred_rgb_array = np.zeros_like(rgb_array)
            for c in range(RGB_CHANNELS):
                blurred_rgb_array[..., c] = convolve(
                    rgb_array[..., c].astype(float), kernel, mode="reflect"
                )

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
        if img_array.ndim == RGB_CHANNELS:  # RGB image
            blurred_array = np.zeros_like(img_array)
            for c in range(RGB_CHANNELS):
                blurred_array[..., c] = convolve(
                    img_array[..., c].astype(float), kernel, mode="reflect"
                )
        else:  # Grayscale image
            blurred_array = convolve(img_array.astype(float), kernel, mode="reflect")

        # Convert back to PIL Image, ensuring original dtype is respected.
        return Image.fromarray(np.clip(blurred_array, 0, 255).astype(original_dtype))

    def _create_circular_kernel(self, radius: int) -> np.ndarray:
        """
        Create a circular (disc-shaped) convolution kernel.

        Args:
            radius: The radius of the circular kernel.

        Returns:
            A 2D numpy array representing the normalized circular kernel.

        """
        # Create a grid with diameter = 2*radius + 1
        diameter = 2 * radius + 1
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

        # Create circular mask where distance from center <= radius
        mask = x * x + y * y <= radius * radius

        # Create kernel with ones inside circle, zeros outside
        kernel: np.ndarray = np.zeros((diameter, diameter), dtype=float)
        kernel[mask] = 1.0

        # Normalize so sum equals 1
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        return kernel

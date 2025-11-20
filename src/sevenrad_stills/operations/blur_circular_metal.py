"""
Pure Metal-accelerated circular blur operation for bokeh effects.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses Apple's MLX framework for Metal acceleration, providing
the best performance on Apple Silicon for circular blur effects.
"""

from typing import Any

import numpy as np
from PIL import Image

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError(
        "MLX is required for Metal acceleration. Install with: pip install mlx"
    ) from e

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
RGB_CHANNELS = 3  # Number of channels in RGB images
RGBA_CHANNELS = 4  # Number of channels in RGBA images


class CircularBlurMetalOperation(BaseImageOperation):
    """
    Apply a circular blur to an image for bokeh effects using Metal acceleration.

    Uses a circular (disc-shaped) kernel to create a bokeh-like blur effect,
    simulating the behavior of a camera lens with a circular aperture. This
    implementation leverages Apple's MLX framework for Metal acceleration,
    providing the best performance on Apple Silicon.

    The blur intensity is controlled by the radius parameter:
    - 0: No blur (returns original image)
    - 1-3: Minimal blur, subtle bokeh
    - 4-8: Moderate blur, noticeable bokeh
    - 8-15: Heavy blur, strong bokeh
    - 15+: Extreme blur, dramatic bokeh
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated circular blur operation."""
        super().__init__("blur_circular_metal")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the circular blur operation.

        Args:
            params: A dictionary containing:
                - radius (int): The radius of the circular kernel.
                  Must be a non-negative integer. A radius of 0 returns the
                  original image.

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
        kernel: np.ndarray = np.zeros((diameter, diameter), dtype=np.float32)
        kernel[mask] = 1.0

        # Normalize so sum equals 1
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum

        return kernel

    def _convolve_2d_metal(
        self, image: mx.array, kernel: mx.array, height: int, width: int
    ) -> mx.array:
        """
        Perform 2D convolution using MLX with reflect padding.

        Args:
            image: Input image as MLX array [H, W].
            kernel: Convolution kernel as MLX array [K, K].
            height: Image height.
            width: Image width.

        Returns:
            Convolved image as MLX array.

        """
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2

        # Pad image with reflect mode
        padded = self._reflect_pad_2d(image, pad_size)

        # Perform correlation (not convolution) - don't flip the kernel
        # The sliding window approach implements correlation semantics
        # Since circular kernel is symmetric, correlation == convolution anyway
        result = mx.zeros((height, width), dtype=mx.float32)

        # Use MLX's efficient operations
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Extract the shifted window
                # Window starts at (i, j) in padded image, size (height, width)
                window = padded[i : i + height, j : j + width]
                # Multiply by kernel weight and accumulate
                result = result + window * kernel[i, j]

        # Evaluate the computation graph
        mx.eval(result)
        return result

    def _reflect_pad_2d(self, image: mx.array, pad_size: int) -> mx.array:
        """
        Apply reflect padding to a 2D image.

        Args:
            image: Input image [H, W].
            pad_size: Padding size.

        Returns:
            Padded image.

        """
        h, w = image.shape

        # Pad horizontally first
        left_pad = image[:, 1 : pad_size + 1][:, ::-1]
        right_pad = image[:, w - pad_size - 1 : w - 1][:, ::-1]
        h_padded = mx.concatenate([left_pad, image, right_pad], axis=1)

        # Pad vertically
        top_pad = h_padded[1 : pad_size + 1, :][::-1, :]
        bottom_pad = h_padded[h - pad_size - 1 : h - 1, :][::-1, :]
        v_padded = mx.concatenate([top_pad, h_padded, bottom_pad], axis=0)

        return v_padded

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply circular blur to image using Metal acceleration.

        Args:
            image: Input PIL Image.
            params: Operation parameters (validated).

        Returns:
            Circular-blurred PIL Image.

        """
        # Validate parameters first
        self.validate_params(params)

        radius: int = params["radius"]

        # If radius is zero, no blur is applied. Return original image.
        if radius == 0:
            return image.copy()

        # Convert image to numpy array
        img_array = np.array(image)
        original_mode = image.mode
        original_dtype = img_array.dtype

        # Create circular blur kernel
        kernel_np = self._create_circular_kernel(radius)
        kernel_mlx = mx.array(kernel_np)

        # Apply blur using Metal
        blurred_array = self._apply_blur_to_array(img_array, kernel_mlx)

        # Clip values and convert back to original dtype
        blurred_array = np.clip(blurred_array, 0, 255).astype(original_dtype)

        # Convert back to PIL Image
        return Image.fromarray(blurred_array, mode=original_mode)

    def _apply_blur_to_array(
        self, img_array: np.ndarray, kernel: mx.array
    ) -> np.ndarray:
        """
        Apply circular blur to a numpy array using Metal.

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB/RGBA).
            kernel: Circular blur kernel as MLX array.

        Returns:
            Blurred image array.

        """
        # Handle RGBA images separately to preserve alpha channel
        if img_array.ndim == RGB_CHANNELS and img_array.shape[2] == RGBA_CHANNELS:
            height, width, channels = img_array.shape
            blurred_array = np.zeros_like(img_array, dtype=np.float32)

            # Process RGB channels only
            for c in range(3):
                rgba_channel_data: np.ndarray = img_array[..., c].astype(np.float32)

                # Convert to MLX array
                channel_mlx = mx.array(rgba_channel_data)

                # Perform convolution
                result_mlx = self._convolve_2d_metal(channel_mlx, kernel, height, width)

                # Convert back to numpy
                blurred_array[..., c] = np.array(result_mlx)

            # Preserve alpha channel unchanged
            blurred_array[..., 3] = img_array[..., 3].astype(np.float32)

        elif img_array.ndim == RGB_CHANNELS:  # RGB
            height, width, channels = img_array.shape
            blurred_array = np.zeros_like(img_array, dtype=np.float32)

            # Process each channel separately
            for c in range(channels):
                rgb_channel_data: np.ndarray = img_array[..., c].astype(np.float32)

                # Convert to MLX array
                channel_mlx = mx.array(rgb_channel_data)

                # Perform convolution
                result_mlx = self._convolve_2d_metal(channel_mlx, kernel, height, width)

                # Convert back to numpy
                blurred_array[..., c] = np.array(result_mlx)

        else:  # Grayscale
            height, width = img_array.shape
            img_float: np.ndarray = img_array.astype(np.float32)

            # Convert to MLX array
            img_mlx = mx.array(img_float)

            # Perform convolution
            result_mlx = self._convolve_2d_metal(img_mlx, kernel, height, width)

            # Convert back to numpy
            blurred_array = np.array(result_mlx)

        return blurred_array  # type: ignore[no-any-return]

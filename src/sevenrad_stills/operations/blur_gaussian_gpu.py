"""
GPU-accelerated Gaussian blur operation using Taichi.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses Taichi for GPU acceleration with separable 1D filters,
providing significant performance improvements over CPU-based implementations
for large images and/or large sigma values.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi (will use GPU if available, otherwise CPU)
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
RGB_CHANNELS = 3  # Number of color channels in RGB image


def compute_gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Compute 1D Gaussian kernel weights on CPU.

    The kernel size is determined by the kernel array size.
    Uses the standard Gaussian formula: exp(-x^2 / (2 * sigma^2))

    Args:
        kernel_size: Size of the kernel (odd number).
        sigma: Standard deviation of the Gaussian.

    Returns:
        1D array of normalized Gaussian weights.

    """
    radius = kernel_size // 2
    x = np.arange(kernel_size) - radius
    weights = np.exp(-(x**2) / (2.0 * sigma**2))
    return (weights / weights.sum()).astype(np.float32)


@ti.kernel  # type: ignore[misc]
def convolve_1d_horizontal(  # type: ignore[no-untyped-def]
    input_img: ti.template(),
    output_img: ti.template(),
    kernel: ti.template(),
    height: ti.i32,
    width: ti.i32,
):
    """
    Apply 1D horizontal Gaussian convolution with reflect boundary mode.

    Args:
        input_img: Input image field (H, W).
        output_img: Output image field (H, W).
        kernel: 1D Gaussian kernel weights.
        height: Image height.
        width: Image width.

    """
    radius = kernel.shape[0] // 2

    for i, j in ti.ndrange(height, width):
        sum_val = 0.0

        for k in range(kernel.shape[0]):
            # Calculate source column
            src_j = j + k - radius

            # Reflect boundary handling
            if src_j < 0:
                src_j = -src_j
            elif src_j >= width:
                src_j = 2 * width - src_j - 2

            # Clamp to valid range
            src_j = ti.max(0, ti.min(width - 1, src_j))

            sum_val += input_img[i, src_j] * kernel[k]

        output_img[i, j] = sum_val


@ti.kernel  # type: ignore[misc]
def convolve_1d_vertical(  # type: ignore[no-untyped-def]
    input_img: ti.template(),
    output_img: ti.template(),
    kernel: ti.template(),
    height: ti.i32,
    width: ti.i32,
):
    """
    Apply 1D vertical Gaussian convolution with reflect boundary mode.

    Args:
        input_img: Input image field (H, W).
        output_img: Output image field (H, W).
        kernel: 1D Gaussian kernel weights.
        height: Image height.
        width: Image width.

    """
    radius = kernel.shape[0] // 2

    for i, j in ti.ndrange(height, width):
        sum_val = 0.0

        for k in range(kernel.shape[0]):
            # Calculate source row
            src_i = i + k - radius

            # Reflect boundary handling
            if src_i < 0:
                src_i = -src_i
            elif src_i >= height:
                src_i = 2 * height - src_i - 2

            # Clamp to valid range
            src_i = ti.max(0, ti.min(height - 1, src_i))

            sum_val += input_img[src_i, j] * kernel[k]

        output_img[i, j] = sum_val


class GaussianBlurGPUOperation(BaseImageOperation):
    """
    Apply GPU-accelerated Gaussian blur using separable 1D filters.

    This implementation uses Taichi to accelerate Gaussian blur through:
    1. Separable filter decomposition (2D → horizontal + vertical 1D passes)
    2. Parallel GPU execution for each pixel
    3. Efficient memory access patterns

    The separable approach provides ~10x reduction in operations compared to
    2D convolution, and GPU parallelization provides additional speedup for
    large images and sigma values.

    Performance: Significant speedup over scipy.ndimage.gaussian_filter for:
    - Large images (>1024x1024)
    - Large sigma values (>5.0)
    - Batch processing of multiple images
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated Gaussian blur operation."""
        super().__init__("blur_gaussian_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the Gaussian blur operation.

        Args:
            params: A dictionary containing:
                - sigma (float): The standard deviation for the Gaussian kernel.
                  Must be non-negative. A sigma of 0 returns the original image.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "sigma" not in params:
            msg = "Gaussian blur requires a 'sigma' parameter."
            raise ValueError(msg)

        sigma = params["sigma"]
        if not isinstance(sigma, (int, float)):
            msg = f"Sigma must be a number, got {type(sigma)}."
            raise ValueError(msg)
        if sigma < 0:
            msg = f"Sigma must be non-negative, got {sigma}."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated Gaussian blur to the image.

        Uses separable 1D filters for efficiency:
        1. Apply horizontal Gaussian blur
        2. Apply vertical Gaussian blur to the result

        Args:
            image: The input PIL Image.
            params: A dictionary with a 'sigma' key.

        Returns:
            The blurred PIL Image.

        """
        self.validate_params(params)
        sigma: float = params["sigma"]

        # If sigma is zero, no blur is applied. Return original image.
        if sigma == 0:
            return image.copy()

        # Convert PIL image to NumPy array. Preserve original dtype.
        img_array = np.array(image)
        original_dtype = img_array.dtype

        # Handle RGBA images separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            rgb_array = np.array(rgb)
            blurred_rgb_array = self._apply_blur_to_array(rgb_array, sigma)

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
        blurred_array = self._apply_blur_to_array(img_array, sigma)

        # Convert back to PIL Image, ensuring original dtype is respected.
        return Image.fromarray(np.clip(blurred_array, 0, 255).astype(original_dtype))

    def _apply_blur_to_array(self, img_array: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply separable Gaussian blur to a numpy array using GPU.

        Optimized to allocate GPU fields once and reuse them for all channels,
        eliminating expensive allocation/deallocation overhead.

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB).
            sigma: Blur sigma value.

        Returns:
            Blurred image array with same shape as input.

        """
        # Calculate kernel size based on sigma (rule of thumb: 6*sigma covers 99.7%)
        kernel_size = int(np.ceil(sigma * 6))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size for symmetric kernel
        kernel_size = max(3, kernel_size)  # Minimum size of 3

        h, w = img_array.shape[:2]
        is_color = img_array.ndim == RGB_CHANNELS

        # Compute Gaussian kernel weights on CPU once
        kernel_weights = compute_gaussian_kernel_1d(kernel_size, sigma)

        # Create Taichi fields ONCE - major performance optimization
        kernel_field = ti.field(dtype=ti.f32, shape=kernel_size)
        kernel_field.from_numpy(kernel_weights)

        # Allocate GPU fields once outside loop
        input_field = ti.field(dtype=ti.f32, shape=(h, w))
        temp_field = ti.field(dtype=ti.f32, shape=(h, w))
        output_field = ti.field(dtype=ti.f32, shape=(h, w))

        if is_color:
            # Process each channel, reusing GPU fields
            result = np.zeros_like(img_array, dtype=np.float32)

            for c in range(RGB_CHANNELS):
                # Reuse existing fields - just copy new channel data
                input_field.from_numpy(img_array[:, :, c].astype(np.float32))

                # Apply separable filter: horizontal then vertical
                convolve_1d_horizontal(input_field, temp_field, kernel_field, h, w)
                convolve_1d_vertical(temp_field, output_field, kernel_field, h, w)

                # Copy result back to CPU
                result[:, :, c] = output_field.to_numpy()

            return result

        # Grayscale image
        input_field.from_numpy(img_array.astype(np.float32))

        # Apply separable filter: horizontal then vertical
        convolve_1d_horizontal(input_field, temp_field, kernel_field, h, w)
        convolve_1d_vertical(temp_field, output_field, kernel_field, h, w)

        # Copy result back to CPU
        return output_field.to_numpy()

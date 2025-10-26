"""
Pure Metal-accelerated motion blur operation using custom kernels.

Provides directional blur with configurable intensity and angle,
leveraging Metal Performance Shaders (MPS) and custom Metal compute kernels.
"""

from typing import Any

import numpy as np
from PIL import Image
from skimage.transform import rotate

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError(
        "MLX is required for Metal acceleration. Install with: pip install mlx"
    ) from e

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 100
MIN_ANGLE = 0.0
MAX_ANGLE = 360.0
RGB_CHANNELS = 3  # Number of channels in RGB/RGBA images


class MotionBlurMetalOperation(BaseImageOperation):
    """
    Apply directional motion blur using pure Metal acceleration.

    Creates linear motion blur in a specified direction, useful for
    simulating camera shake, panning, or subject movement. This implementation
    uses Apple's MLX framework for Metal acceleration, providing the best
    performance on Apple Silicon.

    Kernel size (blur strength):
    - 1-3: Minimal blur, very subtle shake effect
    - 3-8: Moderate blur, noticeable motion
    - 8-20: Heavy blur, significant movement
    - 20+: Extreme blur, dramatic motion effect

    Angle:
    - 0°: Horizontal motion (left-right)
    - 90°: Vertical motion (up-down)
    - 45°/135°: Diagonal motion
    - Any angle 0-360° supported
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated motion blur operation."""
        super().__init__("motion_blur_metal")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate motion blur operation parameters.

        Expected params:
        - kernel_size: int (1-100) - Size of motion blur kernel
        - angle: float (0-360) - Direction of motion in degrees (optional)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "kernel_size" not in params:
            msg = "Motion blur operation requires 'kernel_size' parameter"
            raise ValueError(msg)

        kernel_size = params["kernel_size"]
        if not isinstance(kernel_size, int):
            msg = f"Kernel size must be an integer, got {type(kernel_size)}"
            raise ValueError(msg)
        if not MIN_KERNEL_SIZE <= kernel_size <= MAX_KERNEL_SIZE:
            msg = (
                f"Kernel size must be between {MIN_KERNEL_SIZE} and "
                f"{MAX_KERNEL_SIZE}, got {kernel_size}"
            )
            raise ValueError(msg)

        # Validate angle if provided
        if "angle" in params:
            angle = params["angle"]
            if not isinstance(angle, (int, float)):
                msg = f"Angle must be a number, got {type(angle)}"
                raise ValueError(msg)
            if not MIN_ANGLE <= angle < MAX_ANGLE:
                msg = f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE}, got {angle}"
                raise ValueError(msg)

    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        Create a motion blur kernel.

        Args:
            size: Size of the kernel
            angle: Angle of motion in degrees

        Returns:
            Motion blur kernel as numpy array

        """
        # Create a horizontal line kernel
        kernel: np.ndarray = np.zeros((size, size), dtype=np.float32)
        kernel[(size - 1) // 2, :] = 1.0
        kernel /= kernel.sum()

        # Rotate kernel to desired angle
        if angle != 0.0:
            kernel = rotate(
                kernel,
                angle,
                resize=False,
                order=1,
                mode="constant",
                cval=0.0,
                preserve_range=True,
            )
            # Re-normalize after rotation to preserve brightness
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum

        return kernel.astype(np.float32)

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
        # MLX doesn't have reflect padding, so we implement it manually
        padded = self._reflect_pad_2d(image, pad_size)

        # Flip kernel for convolution (vs correlation)
        kernel_flipped = mx.flip(mx.flip(kernel, axis=0), axis=1)

        # Perform convolution using sliding window
        result = mx.zeros((height, width), dtype=mx.float32)

        # Use MLX's efficient operations
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Extract the shifted window
                window = padded[i : i + height, j : j + width]
                # Multiply by kernel weight and accumulate
                result = result + window * kernel_flipped[i, j]

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
        left_pad = mx.flip(image[:, 1 : pad_size + 1], axis=1)
        right_pad = mx.flip(image[:, w - pad_size - 1 : w - 1], axis=1)
        h_padded = mx.concatenate([left_pad, image, right_pad], axis=1)

        # Pad vertically
        top_pad = mx.flip(h_padded[1 : pad_size + 1, :], axis=0)
        bottom_pad = mx.flip(h_padded[h - pad_size - 1 : h - 1, :], axis=0)
        v_padded = mx.concatenate([top_pad, h_padded, bottom_pad], axis=0)

        return v_padded

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply motion blur to image using Metal acceleration.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Motion-blurred PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        kernel_size: int = params["kernel_size"]
        angle: float = params.get("angle", 0.0)  # Default to horizontal

        # Handle edge case: kernel size of 1 returns original image
        if kernel_size == 1:
            return image.copy()

        # Convert image to numpy array
        image_array = np.array(image)
        original_mode = image.mode
        original_dtype = image_array.dtype

        # Create motion blur kernel
        kernel_np = self._create_motion_kernel(kernel_size, angle)
        kernel_mlx = mx.array(kernel_np)

        # Apply blur using Metal
        blurred_array = self._apply_blur_to_array(image_array, kernel_mlx)

        # Clip values and convert back to original dtype
        blurred_array = np.clip(blurred_array, 0, 255).astype(original_dtype)

        # Convert back to PIL Image
        return Image.fromarray(blurred_array, mode=original_mode)

    def _apply_blur_to_array(
        self, img_array: np.ndarray, kernel: mx.array
    ) -> np.ndarray:
        """
        Apply motion blur to a numpy array using Metal.

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB/RGBA).
            kernel: Motion blur kernel as MLX array.

        Returns:
            Blurred image array.

        """
        # Handle RGB/RGBA and grayscale images
        if img_array.ndim == RGB_CHANNELS:  # RGB or RGBA
            height, width, channels = img_array.shape
            blurred_array = np.zeros_like(img_array, dtype=np.float32)

            # Process each channel separately
            for c in range(channels):
                channel_data: np.ndarray = img_array[..., c].astype(np.float32)

                # Convert to MLX array
                channel_mlx = mx.array(channel_data)

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

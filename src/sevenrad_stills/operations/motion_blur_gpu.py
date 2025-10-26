"""
GPU-accelerated motion blur operation using Taichi.

Provides directional blur with configurable intensity and angle,
supporting minimal blur amounts for subtle effects.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image
from skimage.transform import rotate

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi (will use GPU if available, otherwise CPU)
ti.init(arch=ti.gpu)

# Constants
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 100
MIN_ANGLE = 0.0
MAX_ANGLE = 360.0
RGB_CHANNELS = 3  # Number of channels in RGB/RGBA images


@ti.kernel  # type: ignore[misc]
def convolve_2d_motion(  # noqa: PLR0913
    input_img: ti.template(),  # type: ignore[valid-type]
    output_img: ti.template(),  # type: ignore[valid-type]
    kernel: ti.template(),  # type: ignore[valid-type]
    kernel_size: ti.i32,
    height: ti.i32,
    width: ti.i32,
) -> None:
    """
    Perform 2D convolution with reflect boundary mode on GPU.

    Args:
        input_img: Input image field.
        output_img: Output image field.
        kernel: Convolution kernel field.
        kernel_size: Size of the kernel (assumed square).
        height: Image height.
        width: Image width.

    """
    radius = kernel_size // 2

    for i, j in ti.ndrange(height, width):
        sum_val = 0.0

        for ki, kj in ti.ndrange(kernel_size, kernel_size):
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


class MotionBlurGPUOperation(BaseImageOperation):
    """
    Apply directional motion blur to simulate camera movement using GPU acceleration.

    Creates linear motion blur in a specified direction, useful for
    simulating camera shake, panning, or subject movement. This implementation
    leverages Taichi for GPU acceleration, providing significant performance
    improvements for large images or large kernel sizes.

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
        """Initialize GPU-accelerated motion blur operation."""
        super().__init__("motion_blur_gpu")

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

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply motion blur to image using GPU acceleration.

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
        kernel = self._create_motion_kernel(kernel_size, angle)

        # Apply blur using GPU
        blurred_array = self._apply_blur_to_array(image_array, kernel, kernel_size)

        # Clip values and convert back to original dtype
        blurred_array = np.clip(blurred_array, 0, 255).astype(original_dtype)

        # Convert back to PIL Image
        return Image.fromarray(blurred_array, mode=original_mode)

    def _apply_blur_to_array(
        self, img_array: np.ndarray, kernel: np.ndarray, kernel_size: int
    ) -> np.ndarray:
        """
        Apply motion blur to a numpy array using GPU.

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB/RGBA).
            kernel: Motion blur kernel.
            kernel_size: Size of the kernel.

        Returns:
            Blurred image array.

        """
        # Create Taichi field for kernel
        kernel_field = ti.field(dtype=ti.f32, shape=(kernel_size, kernel_size))
        kernel_field.from_numpy(kernel)

        # Handle RGB/RGBA and grayscale images
        if img_array.ndim == RGB_CHANNELS:  # RGB or RGBA
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
                convolve_2d_motion(
                    input_field, output_field, kernel_field, kernel_size, height, width
                )

                # Copy result back to CPU
                blurred_array[..., c] = output_field.to_numpy()

        else:  # Grayscale
            height, width = img_array.shape
            img_float: np.ndarray = img_array.astype(np.float32)

            # Create Taichi fields
            input_field = ti.field(dtype=ti.f32, shape=(height, width))
            output_field = ti.field(dtype=ti.f32, shape=(height, width))

            # Copy data to GPU
            input_field.from_numpy(img_float)

            # Perform convolution
            convolve_2d_motion(
                input_field, output_field, kernel_field, kernel_size, height, width
            )

            # Copy result back to CPU
            blurred_array = output_field.to_numpy()

        return blurred_array  # type: ignore[no-any-return]

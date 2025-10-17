"""
Motion blur operation for simulating camera movement and shake.

Provides directional blur with configurable intensity and angle,
supporting minimal blur amounts for subtle effects.
"""

from typing import Any

import numpy as np  # type: ignore[import-not-found]
from PIL import Image
from scipy.ndimage import convolve  # type: ignore[import-not-found]
from skimage.transform import rotate  # type: ignore[import-not-found]

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 100
MIN_ANGLE = 0.0
MAX_ANGLE = 360.0
RGB_CHANNELS = 3  # Number of channels in RGB/RGBA images


class MotionBlurOperation(BaseImageOperation):
    """
    Apply directional motion blur to simulate camera movement.

    Creates linear motion blur in a specified direction, useful for
    simulating camera shake, panning, or subject movement.

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
        """Initialize motion blur operation."""
        super().__init__("motion_blur")

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
                msg = (
                    f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE}, "
                    f"got {angle}"
                )
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
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size

        # Rotate kernel to desired angle
        if angle != 0:
            kernel = rotate(kernel, angle, resize=False)

        return kernel

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply motion blur to image.

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

        # Create motion blur kernel
        kernel = self._create_motion_kernel(kernel_size, angle)

        # Apply convolution to each channel
        if len(image_array.shape) == RGB_CHANNELS:  # RGB or RGBA
            blurred_channels = []
            for channel in range(image_array.shape[2]):
                blurred = convolve(
                    image_array[:, :, channel].astype(float),
                    kernel,
                    mode="reflect",
                )
                blurred_channels.append(blurred)
            blurred_array = np.stack(blurred_channels, axis=2)
        else:  # Grayscale
            blurred_array = convolve(image_array.astype(float), kernel, mode="reflect")

        # Clip values and convert back to uint8
        blurred_array = np.clip(blurred_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(blurred_array, mode=original_mode)

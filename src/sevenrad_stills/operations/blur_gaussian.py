"""
Gaussian blur operation.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses SciPy instead of PyTorch for better integration
with PIL-based image processing pipeline.
"""

from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
RGB_CHANNELS = 3  # Number of color channels in RGB image


class GaussianBlurOperation(BaseImageOperation):
    """
    Apply a Gaussian blur to an image.

    Uses scipy.ndimage.gaussian_filter to apply a Gaussian blur kernel
    to an image. The blur intensity is controlled by the sigma parameter,
    with higher values producing stronger blur effects.
    """

    def __init__(self) -> None:
        """Initialize the Gaussian blur operation."""
        super().__init__("blur_gaussian")

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
        Apply Gaussian blur to the image.

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
            # Apply blur only to spatial dimensions (height, width), not channels
            blurred_rgb_array = gaussian_filter(
                rgb_array, sigma=[sigma, sigma, 0], mode="reflect"
            )

            # Restore original data type and create new image
            blurred_rgb = Image.fromarray(blurred_rgb_array.astype(original_dtype))
            # Create RGBA image with blurred RGB and original alpha
            result = Image.new("RGBA", image.size)
            result.paste(blurred_rgb, (0, 0))
            result.putalpha(alpha)
            return result

        # For RGB or L images, apply blur directly
        # The sigma sequence is (height, width, channels).
        # We don't blur across channels, so the channel sigma is 0.
        sigma_vector = (
            [sigma, sigma, 0] if img_array.ndim == RGB_CHANNELS else [sigma, sigma]
        )
        blurred_array = gaussian_filter(
            img_array, sigma=sigma_vector[: img_array.ndim], mode="reflect"
        )

        # Convert back to PIL Image, ensuring original dtype is respected.
        return Image.fromarray(blurred_array.astype(original_dtype))

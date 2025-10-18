"""
Chromatic aberration effect.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses SciPy instead of PyTorch for better integration
with PIL-based image processing pipeline.
"""

from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import shift

from sevenrad_stills.operations.base import BaseImageOperation


class ChromaticAberrationOperation(BaseImageOperation):
    """
    Simulate chromatic aberration by shifting color channels.

    Chromatic aberration is a common optical phenomenon where a lens
    fails to focus all colors to the same point, causing color fringing
    at edges. This operation simulates the effect by shifting the red
    and blue channels in opposite directions.
    """

    def __init__(self) -> None:
        """Initialize the operation."""
        super().__init__("chromatic_aberration")

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
        Apply chromatic aberration.

        The green channel is kept as reference, while red and blue channels
        are shifted in opposite directions to create color fringing.

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

        # Create an empty array for the output
        output_array = np.zeros_like(img_array)

        # Define shifts for R, G, B channels. G channel is kept as reference.
        # R is shifted one way, B is shifted the other.
        # Shift expects (rows, cols) for 2D arrays
        shifts = {
            0: (shift_y, shift_x),  # Red channel shift
            1: (0, 0),  # Green channel (no shift)
            2: (-shift_y, -shift_x),  # Blue channel shift
        }

        for i in range(3):  # Iterate through R, G, B
            output_array[..., i] = shift(
                img_array[..., i], shifts[i], mode="constant", cval=0
            )

        # Preserve alpha channel if it exists
        if image.mode == "RGBA":
            output_array[..., 3] = img_array[..., 3]

        return Image.fromarray(output_array)

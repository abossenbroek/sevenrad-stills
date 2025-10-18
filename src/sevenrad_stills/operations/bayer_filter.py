"""
Bayer filter operation to simulate digital camera sensor artifacts.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses colour-demosaicing for demosaicing.
"""

from typing import Any, Literal

import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from PIL import Image
from skimage.util import img_as_float, img_as_ubyte

from sevenrad_stills.operations.base import BaseImageOperation

# Type alias for Bayer patterns
BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
VALID_PATTERNS: set[BayerPattern] = {"RGGB", "BGGR", "GRBG", "GBRG"}


class BayerFilterOperation(BaseImageOperation):
    """
    Simulates digital sensor artifacts using Bayer filter mosaicing and demosaicing.

    This operation first creates a Bayer mosaic from an RGB image, simulating
    the raw data from a digital camera sensor. It then applies a demosaicing
    algorithm to reconstruct a full-color image. This process can introduce
    artifacts like color fringing and aliasing, mimicking the behavior of
    a real digital camera sensor.
    """

    def __init__(self) -> None:
        """Initialize the Bayer filter operation."""
        super().__init__("bayer_filter")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the Bayer filter operation.

        Args:
            params: A dictionary that can contain:
                - pattern (str): The Bayer filter pattern. Must be one of
                  "RGGB", "BGGR", "GRBG", or "GBRG". Optional, defaults to "RGGB".

        Raises:
            ValueError: If the pattern parameter is invalid.

        """
        if "pattern" in params:
            pattern = params["pattern"]
            if not isinstance(pattern, str):
                msg = f"Pattern must be a string, got {type(pattern)}."
                raise ValueError(msg)
            if pattern not in VALID_PATTERNS:
                valid = ", ".join(sorted(VALID_PATTERNS))
                msg = f"Invalid pattern '{pattern}'. Must be one of {valid}."
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply the Bayer filter and demosaicing effect.

        Args:
            image: The input PIL Image.
            params: A dictionary with an optional 'pattern' key.

        Returns:
            The transformed PIL Image with simulated sensor artifacts.

        """
        self.validate_params(params)
        pattern: BayerPattern = params.get("pattern", "RGGB")

        if image.mode not in ("RGB", "RGBA"):
            return image.copy()

        # Handle RGBA images by separating alpha
        alpha = None
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            image = image.convert("RGB")

        # Convert to float for processing
        img_array_float = img_as_float(image)

        # 1. Mosaicing: Create the Bayer pattern Color Filter Array (CFA)
        mosaic = np.zeros(img_array_float.shape[:2], dtype=np.float64)

        # Pattern is defined by the top-left 2x2 pixel block
        if pattern == "RGGB":
            mosaic[0::2, 0::2] = img_array_float[0::2, 0::2, 0]  # R
            mosaic[0::2, 1::2] = img_array_float[0::2, 1::2, 1]  # G
            mosaic[1::2, 0::2] = img_array_float[1::2, 0::2, 1]  # G
            mosaic[1::2, 1::2] = img_array_float[1::2, 1::2, 2]  # B
        elif pattern == "BGGR":
            mosaic[0::2, 0::2] = img_array_float[0::2, 0::2, 2]  # B
            mosaic[0::2, 1::2] = img_array_float[0::2, 1::2, 1]  # G
            mosaic[1::2, 0::2] = img_array_float[1::2, 0::2, 1]  # G
            mosaic[1::2, 1::2] = img_array_float[1::2, 1::2, 0]  # R
        elif pattern == "GRBG":
            mosaic[0::2, 0::2] = img_array_float[0::2, 0::2, 1]  # G
            mosaic[0::2, 1::2] = img_array_float[0::2, 1::2, 0]  # R
            mosaic[1::2, 0::2] = img_array_float[1::2, 0::2, 2]  # B
            mosaic[1::2, 1::2] = img_array_float[1::2, 1::2, 1]  # G
        elif pattern == "GBRG":
            mosaic[0::2, 0::2] = img_array_float[0::2, 0::2, 1]  # G
            mosaic[0::2, 1::2] = img_array_float[0::2, 1::2, 2]  # B
            mosaic[1::2, 0::2] = img_array_float[1::2, 0::2, 0]  # R
            mosaic[1::2, 1::2] = img_array_float[1::2, 1::2, 1]  # G

        # 2. Demosaicing
        demosaiced_array = demosaicing_CFA_Bayer_Malvar2004(mosaic, pattern)

        # Convert back to uint8 for PIL
        # np.clip is used to ensure values are in [0, 1] range before conversion
        demosaiced_array_clipped = np.clip(demosaiced_array, 0, 1)
        output_array_uint8 = img_as_ubyte(demosaiced_array_clipped)

        result_img = Image.fromarray(output_array_uint8)

        # Restore alpha channel if necessary
        if alpha:
            result_img.putalpha(alpha)

        return result_img

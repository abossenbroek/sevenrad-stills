"""
Saturation adjustment operation.

Adjusts image saturation using HSV color space conversion.
Supports either fixed percentage or random variation.
"""

import random
from typing import Any, Literal

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
RANGE_SIZE = 2
MIN_RANGE_VALUE = -1.0

# HSV conversion constants
EPSILON = 1e-6
HUE_SECTOR_0 = 60.0
HUE_SECTOR_1 = 120.0
HUE_SECTOR_2 = 180.0
HUE_SECTOR_3 = 240.0
HUE_SECTOR_4 = 300.0
HUE_MODULO = 360.0
HUE_SECTORS = 6.0


class SaturationOperation(BaseImageOperation):
    """
    Adjust image saturation using HSV color space conversion.

    Converts RGB to HSV, multiplies the S (saturation) component by the factor,
    then converts back to RGB. This provides precise control over color intensity
    while preserving hue and value.

    Supports two modes:
    - fixed: Apply a fixed saturation multiplier
    - random: Apply a random saturation multiplier within a range
    """

    def __init__(self) -> None:
        """Initialize saturation operation."""
        super().__init__("saturation")

    def _validate_mode(self, params: dict[str, Any]) -> str:
        """Validate and return mode parameter."""
        if "mode" not in params:
            msg = "Saturation operation requires 'mode' parameter"
            raise ValueError(msg)

        mode: str = params["mode"]
        if mode not in ("fixed", "random"):
            msg = f"Invalid mode '{mode}'. Must be 'fixed' or 'random'"
            raise ValueError(msg)
        return mode

    def _validate_fixed_params(self, params: dict[str, Any]) -> None:
        """Validate fixed mode parameters."""
        if "value" not in params:
            msg = "Fixed mode requires 'value' parameter"
            raise ValueError(msg)
        value = params["value"]
        if not isinstance(value, (int, float)):
            msg = f"Value must be a number, got {type(value)}"
            raise ValueError(msg)
        if value < -1.0:
            msg = f"Value must be >= -1.0 (for complete grayscale), got {value}"
            raise ValueError(msg)

    def _validate_random_params(self, params: dict[str, Any]) -> None:
        """Validate random mode parameters."""
        if "range" not in params:
            msg = "Random mode requires 'range' parameter"
            raise ValueError(msg)
        range_val = params["range"]
        if not isinstance(range_val, (list, tuple)) or len(range_val) != RANGE_SIZE:
            msg = "Range must be a list/tuple of two numbers"
            raise ValueError(msg)
        min_val, max_val = range_val
        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            msg = "Range values must be numbers"
            raise ValueError(msg)
        if min_val >= max_val:
            msg = f"Range min ({min_val}) must be less than max ({max_val})"
            raise ValueError(msg)
        if min_val < MIN_RANGE_VALUE:
            msg = f"Range min must be >= {MIN_RANGE_VALUE}, got {min_val}"
            raise ValueError(msg)

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate saturation operation parameters.

        Expected params:
        - mode: "fixed" or "random"
        - value: float (for fixed mode) - multiplier (1.0 = no change)
        - range: [float, float] (for random mode) - min/max multipliers

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        mode = self._validate_mode(params)

        if mode == "fixed":
            self._validate_fixed_params(params)
        else:  # random
            self._validate_random_params(params)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply saturation adjustment using HSV color space conversion.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Saturation-adjusted PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        mode: Literal["fixed", "random"] = params["mode"]

        # Calculate saturation factor
        if mode == "fixed":
            factor = float(params["value"])
        else:  # random
            min_val, max_val = params["range"]
            factor = random.uniform(min_val, max_val)  # noqa: S311

        # Ensure factor is non-negative (saturation can't be negative)
        # Factor of 0 = grayscale, 1 = original, > 1 = more saturated
        factor = max(0.0, 1.0 + factor)

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        height, width = img_array.shape[:2]

        # Apply HSV-based saturation adjustment
        result = self._apply_hsv_saturation(img_array, factor, height, width)

        # Convert back to uint8
        result_uint8 = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)

        return Image.fromarray(result_uint8, mode="RGB")

    def _apply_hsv_saturation(
        self,
        img_array: np.ndarray,
        factor: float,
        height: int,  # noqa: ARG002
        width: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Apply HSV-based saturation adjustment using vectorized numpy operations.

        Args:
            img_array: Normalized RGB image array (0.0-1.0)
            factor: Saturation factor (0.0 = grayscale, 1.0 = original)
            height: Image height (unused, for API compatibility)
            width: Image width (unused, for API compatibility)

        Returns:
            Adjusted image array

        """
        # Extract RGB channels
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        # Convert RGB to HSV (vectorized)
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Hue calculation
        h = np.zeros_like(r)
        mask = delta > EPSILON

        # Where max is R
        mask_r = mask & (np.abs(cmax - r) < EPSILON)
        h[mask_r] = HUE_SECTOR_0 * (
            ((g[mask_r] - b[mask_r]) / delta[mask_r]) % HUE_SECTORS
        )

        # Where max is G
        mask_g = mask & (np.abs(cmax - g) < EPSILON)
        h[mask_g] = HUE_SECTOR_0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)

        # Where max is B
        mask_b = mask & ~mask_r & ~mask_g
        h[mask_b] = HUE_SECTOR_0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        # Saturation
        s = np.where(cmax > EPSILON, delta / cmax, 0.0)

        # Value
        v = cmax

        # Adjust saturation
        s = np.clip(s * factor, 0.0, 1.0)

        # Convert HSV back to RGB
        c = v * s
        x = c * (1.0 - np.abs(((h / HUE_SECTOR_0) % 2.0) - 1.0))
        m = v - c

        # Initialize output
        result = np.zeros_like(img_array)

        # Map h ranges to RGB
        h_mod = h % HUE_MODULO
        mask_0 = h_mod < HUE_SECTOR_0
        mask_1 = (h_mod >= HUE_SECTOR_0) & (h_mod < HUE_SECTOR_1)
        mask_2 = (h_mod >= HUE_SECTOR_1) & (h_mod < HUE_SECTOR_2)
        mask_3 = (h_mod >= HUE_SECTOR_2) & (h_mod < HUE_SECTOR_3)
        mask_4 = (h_mod >= HUE_SECTOR_3) & (h_mod < HUE_SECTOR_4)
        mask_5 = h_mod >= HUE_SECTOR_4

        result[mask_0] = np.stack(
            [c[mask_0], x[mask_0], np.zeros_like(c[mask_0])], axis=-1
        )
        result[mask_1] = np.stack(
            [x[mask_1], c[mask_1], np.zeros_like(c[mask_1])], axis=-1
        )
        result[mask_2] = np.stack(
            [np.zeros_like(c[mask_2]), c[mask_2], x[mask_2]], axis=-1
        )
        result[mask_3] = np.stack(
            [np.zeros_like(c[mask_3]), x[mask_3], c[mask_3]], axis=-1
        )
        result[mask_4] = np.stack(
            [x[mask_4], np.zeros_like(c[mask_4]), c[mask_4]], axis=-1
        )
        result[mask_5] = np.stack(
            [c[mask_5], np.zeros_like(c[mask_5]), x[mask_5]], axis=-1
        )

        # Add m to all channels
        result += m[:, :, np.newaxis]

        return result  # type: ignore[no-any-return]

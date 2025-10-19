"""
Saturation adjustment operation.

Adjusts image saturation using either fixed percentage or random variation.
"""

import random
from typing import Any, Literal

from PIL import Image, ImageEnhance

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
RANGE_SIZE = 2
MIN_RANGE_VALUE = -1.0


class SaturationOperation(BaseImageOperation):
    """
    Adjust image saturation.

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
        Apply saturation adjustment to image.

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

        # Apply saturation enhancement
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

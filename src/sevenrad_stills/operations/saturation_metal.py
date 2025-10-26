"""
Metal-accelerated saturation adjustment with explicit Metal backend.

Provides Metal-accelerated saturation adjustment using Taichi with explicit
Metal backend and HSV color space conversion. Uses Taichi with explicit Metal
backend for GPU acceleration while leveraging PIL for image I/O (hybrid approach).

Performance: Explicit Metal backend ensures maximum performance on Apple Silicon
for parallel HSV conversion and saturation adjustment operations.
"""

import random
import sys
from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Ensure we are on macOS
if sys.platform != "darwin":
    msg = "saturation_metal operation is only available on macOS"
    raise ImportError(msg)

# Initialize Taichi with explicit Metal backend for Apple Silicon
ti.init(arch=ti.metal, default_fp=ti.f32)

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
HUE_SECTORS = 6.0


@ti.kernel  # type: ignore[misc]
def apply_saturation_adjustment(  # type: ignore[no-untyped-def]  # noqa: C901, PLR0915, ANN201
    input_img: ti.template(),  # type: ignore[valid-type]
    output_img: ti.template(),  # type: ignore[valid-type]
    factor: ti.f32,
    height: ti.i32,
    width: ti.i32,
):
    """
    Metal kernel to apply saturation adjustment using HSV color space.

    Converts RGB to HSV, adjusts saturation, and converts back to RGB.

    Args:
        input_img: Input image field (normalized to 0.0-1.0).
        output_img: Output image field.
        factor: Saturation adjustment factor (0.0 = grayscale, 1.0 = original).
        height: Image height.
        width: Image width.

    """
    for i, j in ti.ndrange(height, width):
        r = input_img[i, j, 0]
        g = input_img[i, j, 1]
        b = input_img[i, j, 2]

        # Find min and max RGB values
        cmax = ti.max(ti.max(r, g), b)
        cmin = ti.min(ti.min(r, g), b)
        delta = cmax - cmin

        # Calculate HSV
        # Hue
        h = 0.0
        if delta > EPSILON:
            if ti.abs(cmax - r) < EPSILON:
                h = HUE_SECTOR_0 * (((g - b) / delta) % HUE_SECTORS)
            elif ti.abs(cmax - g) < EPSILON:
                h = HUE_SECTOR_0 * (((b - r) / delta) + 2.0)
            else:
                h = HUE_SECTOR_0 * (((r - g) / delta) + 4.0)

        # Saturation
        s = 0.0
        if cmax > EPSILON:
            s = delta / cmax

        # Value
        v = cmax

        # Adjust saturation
        s = s * factor

        # Clamp saturation
        s = ti.max(0.0, ti.min(1.0, s))

        # Convert back to RGB
        c = v * s
        x = c * (1.0 - ti.abs(((h / HUE_SECTOR_0) % 2.0) - 1.0))
        m = v - c

        r_new = 0.0
        g_new = 0.0
        b_new = 0.0

        if h < HUE_SECTOR_0:
            r_new = c
            g_new = x
            b_new = 0.0
        elif h < HUE_SECTOR_1:
            r_new = x
            g_new = c
            b_new = 0.0
        elif h < HUE_SECTOR_2:
            r_new = 0.0
            g_new = c
            b_new = x
        elif h < HUE_SECTOR_3:
            r_new = 0.0
            g_new = x
            b_new = c
        elif h < HUE_SECTOR_4:
            r_new = x
            g_new = 0.0
            b_new = c
        else:
            r_new = c
            g_new = 0.0
            b_new = x

        output_img[i, j, 0] = r_new + m
        output_img[i, j, 1] = g_new + m
        output_img[i, j, 2] = b_new + m


class SaturationMetalOperation(BaseImageOperation):
    """
    Metal-accelerated saturation adjustment using HSV color space.

    This implementation uses Taichi with explicit Metal backend for
    GPU-accelerated HSV color space conversion and saturation manipulation,
    ensuring machine-epsilon numerical accuracy with the CPU version while
    providing maximum performance on Apple Silicon.

    **Implementation Philosophy:**
    Uses Metal GPU for parallel HSV conversion and saturation adjustment,
    delivering real performance gains for large images while maintaining
    exact numerical accuracy with CPU and GPU implementations.

    Supports two modes:
    - fixed: Apply a fixed saturation multiplier
    - random: Apply a random saturation multiplier within a range

    Performance: Metal backend provides maximum performance on Apple Silicon
    for parallel HSV color space conversion and saturation adjustment.
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated saturation operation."""
        super().__init__("saturation_metal")

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
        Apply Metal-accelerated saturation adjustment to image.

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
        channels = 3

        # Create Taichi fields
        input_field = ti.field(dtype=ti.f32, shape=(height, width, channels))
        output_field = ti.field(dtype=ti.f32, shape=(height, width, channels))

        # Copy data to GPU
        input_field.from_numpy(img_array)

        # Apply saturation adjustment on Metal GPU
        apply_saturation_adjustment(input_field, output_field, factor, height, width)

        # Copy result back to CPU
        result_array = output_field.to_numpy()

        # Convert back to uint8
        result_uint8 = (np.clip(result_array, 0.0, 1.0) * 255.0).astype(np.uint8)

        return Image.fromarray(result_uint8, mode="RGB")

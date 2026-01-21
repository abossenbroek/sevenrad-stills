"""
GPU-accelerated corduroy striping operation using Taichi.

Simulates "corduroy" or "banding" artifacts from push-broom and whisk-broom
scanners where individual detector elements have slightly different sensitivity
due to calibration drift or manufacturing variations. Uses Taichi for GPU
acceleration of the line-wise multiplier application.
"""

from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image
from skimage.util import img_as_float32, img_as_ubyte

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0


@ti.kernel  # type: ignore[misc]
def apply_vertical_stripes_gray(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    multipliers: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply vertical striping to grayscale images.

    Args:
        img: Image array (H, W) to modify in-place
        multipliers: Array of multipliers per column (W,)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        mult = multipliers[x]
        img[y, x] = ti.math.clamp(img[y, x] * mult, 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_vertical_stripes_rgb(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    multipliers: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply vertical striping to RGB images.

    Args:
        img: Image array (H, W, 3) to modify in-place
        multipliers: Array of multipliers per column (W,)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        mult = multipliers[x]
        for c in ti.static(range(3)):
            img[y, x, c] = ti.math.clamp(img[y, x, c] * mult, 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_horizontal_stripes_gray(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    multipliers: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply horizontal striping to grayscale images.

    Args:
        img: Image array (H, W) to modify in-place
        multipliers: Array of multipliers per row (H,)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        mult = multipliers[y]
        img[y, x] = ti.math.clamp(img[y, x] * mult, 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_horizontal_stripes_rgb(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    multipliers: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply horizontal striping to RGB images.

    Args:
        img: Image array (H, W, 3) to modify in-place
        multipliers: Array of multipliers per row (H,)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        mult = multipliers[y]
        for c in ti.static(range(3)):
            img[y, x, c] = ti.math.clamp(img[y, x, c] * mult, 0.0, 1.0)


class CorduroyGPUOperation(BaseImageOperation):
    """
    Apply GPU-accelerated corduroy striping to simulate detector calibration errors.

    Creates subtle vertical or horizontal banding by simulating "hot" (overly
    sensitive) and "cold" (less sensitive) detector elements in a push-broom
    or whisk-broom scanner array.

    In real satellite sensors, each detector in a linear array may have slightly
    different gain due to:
    - Manufacturing variation in sensitivity
    - Calibration drift over time
    - Temperature effects on individual detectors
    - Radiation damage accumulation

    This creates characteristic "corduroy" patterns - subtle repeating lines
    of slightly brighter or darker pixels running perpendicular to the scan
    direction.

    Performance: GPU acceleration provides significant speedup for large images
    by parallelizing the multiplier application across all pixels.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated corduroy striping operation."""
        super().__init__("corduroy_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for corduroy striping operation.

        Args:
            params: A dictionary containing:
                - strength (float): Striping intensity (0.0 to 1.0), maps to
                  multiplier of 1.0 ± strength x 0.2
                - orientation (str): 'vertical' or 'horizontal' line direction
                - density (float): Proportion of lines affected (0.0 to 1.0)
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "strength" not in params:
            msg = "Corduroy operation requires 'strength' parameter."
            raise ValueError(msg)
        strength = params["strength"]
        if not isinstance(strength, (int, float)) or not (
            MIN_STRENGTH <= strength <= MAX_STRENGTH
        ):
            msg = f"Strength must be a float between {MIN_STRENGTH} and {MAX_STRENGTH}."
            raise ValueError(msg)

        if "orientation" not in params:
            msg = "Corduroy operation requires 'orientation' parameter."
            raise ValueError(msg)
        orientation = params["orientation"]
        if orientation not in ("vertical", "horizontal"):
            msg = "Orientation must be 'vertical' or 'horizontal'."
            raise ValueError(msg)

        if "density" not in params:
            msg = "Corduroy operation requires 'density' parameter."
            raise ValueError(msg)
        density = params["density"]
        if not isinstance(density, (int, float)) or not (
            MIN_DENSITY <= density <= MAX_DENSITY
        ):
            msg = f"Density must be a float between {MIN_DENSITY} and {MAX_DENSITY}."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated corduroy striping to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'strength', 'orientation', 'density',
                    and optional 'seed'.

        Returns:
            The PIL Image with corduroy striping applied.

        """
        self.validate_params(params)
        strength: float = params["strength"]
        orientation: Literal["vertical", "horizontal"] = params["orientation"]
        density: float = params["density"]
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert to float array (0.0 to 1.0) using skimage utility
        img_float = img_as_float32(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_float[..., :3].copy()
            alpha = img_float[..., 3:4]
            h, w = rgb.shape[:2]
            num_channels = 3
        else:
            rgb = img_float.copy()
            alpha = None
            h, w = rgb.shape[:2]
            # Determine if grayscale or RGB
            num_channels = 1 if rgb.ndim == 2 else 3  # noqa: PLR2004

        # Determine number of lines to affect
        if orientation == "vertical":
            num_lines = int(density * w)
            total_lines = w
        else:  # horizontal
            num_lines = int(density * h)
            total_lines = h

        if num_lines > 0:
            # Select random lines
            affected_lines = rng.choice(total_lines, size=num_lines, replace=False)

            # Generate random multipliers for each line
            # strength maps to range [1.0 - strength*0.2, 1.0 + strength*0.2]
            multipliers_affected = rng.uniform(
                1.0 - strength * 0.2,
                1.0 + strength * 0.2,
                size=num_lines,
            ).astype(np.float32)

            # Create an array of multipliers, with 1.0 for unaffected lines
            multipliers_array = np.ones(total_lines, dtype=np.float32)
            multipliers_array[affected_lines] = multipliers_affected

            # Apply GPU kernel based on orientation and image type
            if orientation == "vertical":
                if num_channels == 1:
                    apply_vertical_stripes_gray(rgb, multipliers_array, h, w)
                else:
                    apply_vertical_stripes_rgb(rgb, multipliers_array, h, w)
            elif num_channels == 1:
                apply_horizontal_stripes_gray(rgb, multipliers_array, h, w)
            else:
                apply_horizontal_stripes_rgb(rgb, multipliers_array, h, w)

        # Recombine with alpha if needed
        if alpha is not None:
            output_float = np.concatenate([rgb, alpha], axis=2)
        else:
            output_float = rgb

        # Convert back to uint8 using skimage utility
        output_array = img_as_ubyte(output_float)
        return Image.fromarray(output_array)

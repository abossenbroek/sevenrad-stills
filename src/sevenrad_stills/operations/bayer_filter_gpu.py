"""
GPU-accelerated Bayer filter operation using Taichi.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses Taichi for GPU acceleration with Metal backend on macOS,
providing significant performance improvements over CPU-based processing through
optimized single-pass kernels.
"""
# mypy: ignore-errors

from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image
from skimage.util import img_as_float, img_as_ubyte

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi with Metal backend for macOS
ti.init(arch=ti.metal)

# Type alias for Bayer patterns
BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
VALID_PATTERNS: set[BayerPattern] = {"RGGB", "BGGR", "GRBG", "GBRG"}

# Pattern encoding for GPU kernel
PATTERN_RGGB = 0
PATTERN_BGGR = 1
PATTERN_GRBG = 2
PATTERN_GBRG = 3


@ti.func  # type: ignore[misc]
def safe_get_channel(  # type: ignore[valid-type]  # noqa: ANN201, PLR0913
    field: ti.template(), i: ti.i32, j: ti.i32, c: ti.i32, h: ti.i32, w: ti.i32
):
    """Safely get field channel value with bounds checking."""
    ii = ti.max(0, ti.min(h - 1, i))
    jj = ti.max(0, ti.min(w - 1, j))
    return field[ii, jj, c]


@ti.kernel  # type: ignore[misc]
def bayer_filter_fast(  # type: ignore[valid-type]  # noqa: ANN201, C901, PLR0912, PLR0915
    rgb_in: ti.template(),
    rgb_out: ti.template(),
    pattern_id: ti.i32,
    h: ti.i32,
    w: ti.i32,
):
    """
    Optimized single-pass Bayer filter with edge-directed demosaicing.

    This kernel performs mosaicing and demosaicing in minimal passes for
    maximum GPU performance. Uses simplified Malvar-style interpolation.

    Args:
        rgb_in: Input RGB image (height, width, 3).
        rgb_out: Output RGB image (height, width, 3).
        pattern_id: Bayer pattern (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG).
        h: Image height.
        w: Image width.

    """
    # Process all pixels in parallel
    for i, j in ti.ndrange(h, w):
        row_even = (i % 2) == 0
        col_even = (j % 2) == 0

        # Determine pixel type based on pattern
        is_red = ti.cast(0, ti.i32)
        is_green = ti.cast(0, ti.i32)
        is_blue = ti.cast(0, ti.i32)

        if pattern_id == PATTERN_RGGB:
            is_red = row_even and col_even
            is_green = (row_even and not col_even) or (not row_even and col_even)
            is_blue = not row_even and not col_even
        elif pattern_id == PATTERN_BGGR:
            is_blue = row_even and col_even
            is_green = (row_even and not col_even) or (not row_even and col_even)
            is_red = not row_even and not col_even
        elif pattern_id == PATTERN_GRBG:
            is_green = (row_even and col_even) or (not row_even and not col_even)
            is_red = row_even and not col_even
            is_blue = not row_even and col_even
        else:  # GBRG
            is_green = (row_even and col_even) or (not row_even and not col_even)
            is_blue = row_even and not col_even
            is_red = not row_even and col_even

        # Get mosaic value (sample appropriate channel)
        mosaic_val = 0.0
        if is_red:
            mosaic_val = rgb_in[i, j, 0]
        elif is_green:
            mosaic_val = rgb_in[i, j, 1]
        else:  # is_blue
            mosaic_val = rgb_in[i, j, 2]

        # Interpolate all three channels
        r_val = 0.0
        g_val = 0.0
        b_val = 0.0

        # Green channel (most important, interpolate first)
        if is_green:
            g_val = mosaic_val
        # Edge-directed green interpolation
        elif i > 0 and i < h - 1 and j > 0 and j < w - 1:
            g_n = safe_get_channel(rgb_in, i - 1, j, 1, h, w)
            g_s = safe_get_channel(rgb_in, i + 1, j, 1, h, w)
            g_w = safe_get_channel(rgb_in, i, j - 1, 1, h, w)
            g_e = safe_get_channel(rgb_in, i, j + 1, 1, h, w)

            # Simple gradient
            dh = ti.abs(g_w - g_e)
            dv = ti.abs(g_n - g_s)

            if dh < dv:
                g_val = (g_w + g_e) * 0.5
            elif dv < dh:
                g_val = (g_n + g_s) * 0.5
            else:
                g_val = (g_n + g_s + g_w + g_e) * 0.25
        else:
            # Edge pixels: simple average
            count = 0.0
            if i > 0:
                g_val += safe_get_channel(rgb_in, i - 1, j, 1, h, w)
                count += 1.0
            if i < h - 1:
                g_val += safe_get_channel(rgb_in, i + 1, j, 1, h, w)
                count += 1.0
            if j > 0:
                g_val += safe_get_channel(rgb_in, i, j - 1, 1, h, w)
                count += 1.0
            if j < w - 1:
                g_val += safe_get_channel(rgb_in, i, j + 1, 1, h, w)
                count += 1.0
            if count > 0:
                g_val /= count

        # Red channel
        if is_red:
            r_val = mosaic_val
        elif i > 0 and i < h - 1 and j > 0 and j < w - 1:
            if is_green:
                # At green, interpolate from neighbors
                if (pattern_id == PATTERN_RGGB and row_even) or (
                    pattern_id == PATTERN_BGGR and not row_even
                ):
                    r_val = (
                        safe_get_channel(rgb_in, i, j - 1, 0, h, w)
                        + safe_get_channel(rgb_in, i, j + 1, 0, h, w)
                    ) * 0.5
                elif (pattern_id == PATTERN_GRBG and col_even) or (
                    pattern_id == PATTERN_GBRG and not col_even
                ):
                    r_val = (
                        safe_get_channel(rgb_in, i - 1, j, 0, h, w)
                        + safe_get_channel(rgb_in, i + 1, j, 0, h, w)
                    ) * 0.5
                elif (pattern_id == PATTERN_GRBG and not col_even) or (
                    pattern_id == PATTERN_GBRG and col_even
                ):
                    r_val = (
                        safe_get_channel(rgb_in, i, j - 1, 0, h, w)
                        + safe_get_channel(rgb_in, i, j + 1, 0, h, w)
                    ) * 0.5
                else:
                    r_val = (
                        safe_get_channel(rgb_in, i - 1, j, 0, h, w)
                        + safe_get_channel(rgb_in, i + 1, j, 0, h, w)
                    ) * 0.5
            else:
                # At blue, interpolate from diagonals
                r_val = (
                    safe_get_channel(rgb_in, i - 1, j - 1, 0, h, w)
                    + safe_get_channel(rgb_in, i - 1, j + 1, 0, h, w)
                    + safe_get_channel(rgb_in, i + 1, j - 1, 0, h, w)
                    + safe_get_channel(rgb_in, i + 1, j + 1, 0, h, w)
                ) * 0.25
        else:
            # Edge case
            r_val = rgb_in[i, j, 0]

        # Blue channel (symmetric to red)
        if is_blue:
            b_val = mosaic_val
        elif i > 0 and i < h - 1 and j > 0 and j < w - 1:
            if is_green:
                # At green, interpolate from neighbors
                if (pattern_id == PATTERN_RGGB and not row_even) or (
                    pattern_id == PATTERN_BGGR and row_even
                ):
                    b_val = (
                        safe_get_channel(rgb_in, i, j - 1, 2, h, w)
                        + safe_get_channel(rgb_in, i, j + 1, 2, h, w)
                    ) * 0.5
                elif (pattern_id == PATTERN_GRBG and not col_even) or (
                    pattern_id == PATTERN_GBRG and col_even
                ):
                    b_val = (
                        safe_get_channel(rgb_in, i - 1, j, 2, h, w)
                        + safe_get_channel(rgb_in, i + 1, j, 2, h, w)
                    ) * 0.5
                elif (pattern_id == PATTERN_GRBG and col_even) or (
                    pattern_id == PATTERN_GBRG and not col_even
                ):
                    b_val = (
                        safe_get_channel(rgb_in, i, j - 1, 2, h, w)
                        + safe_get_channel(rgb_in, i, j + 1, 2, h, w)
                    ) * 0.5
                else:
                    b_val = (
                        safe_get_channel(rgb_in, i - 1, j, 2, h, w)
                        + safe_get_channel(rgb_in, i + 1, j, 2, h, w)
                    ) * 0.5
            else:
                # At red, interpolate from diagonals
                b_val = (
                    safe_get_channel(rgb_in, i - 1, j - 1, 2, h, w)
                    + safe_get_channel(rgb_in, i - 1, j + 1, 2, h, w)
                    + safe_get_channel(rgb_in, i + 1, j - 1, 2, h, w)
                    + safe_get_channel(rgb_in, i + 1, j + 1, 2, h, w)
                ) * 0.25
        else:
            # Edge case
            b_val = rgb_in[i, j, 2]

        # Write output with clamping
        rgb_out[i, j, 0] = ti.max(0.0, ti.min(1.0, r_val))
        rgb_out[i, j, 1] = ti.max(0.0, ti.min(1.0, g_val))
        rgb_out[i, j, 2] = ti.max(0.0, ti.min(1.0, b_val))


class BayerFilterGPUOperation(BaseImageOperation):
    """
    GPU-accelerated Bayer filter operation simulating digital sensor artifacts.

    This operation first creates a Bayer mosaic from an RGB image, simulating
    the raw data from a digital camera sensor. It then applies an edge-directed
    demosaicing algorithm to reconstruct a full-color image. This implementation
    uses Taichi for GPU acceleration with Metal backend on macOS, providing
    significant performance improvements over CPU-based processing.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated Bayer filter operation."""
        super().__init__("bayer_filter_gpu")

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
        Apply the Bayer filter and demosaicing effect using GPU acceleration.

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

        # Convert to float32 array
        img_array = np.array(image, dtype=np.float32) / 255.0
        height, width = img_array.shape[:2]

        # Convert pattern to ID
        pattern_map = {
            "RGGB": PATTERN_RGGB,
            "BGGR": PATTERN_BGGR,
            "GRBG": PATTERN_GRBG,
            "GBRG": PATTERN_GBRG,
        }
        pattern_id = pattern_map[pattern]

        # Create Taichi fields sized to the image
        rgb_in = ti.field(dtype=ti.f32, shape=(height, width, 3))
        rgb_out = ti.field(dtype=ti.f32, shape=(height, width, 3))

        # Transfer data to GPU
        rgb_in.from_numpy(img_array)

        # Execute single-pass kernel
        bayer_filter_fast(rgb_in, rgb_out, pattern_id, height, width)

        # Transfer result back from GPU
        result_array = rgb_out.to_numpy()

        # Convert to uint8
        result_array_uint8 = np.clip(result_array * 255.0, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result_array_uint8)

        # Restore alpha if needed
        if alpha:
            result_img.putalpha(alpha)

        return result_img

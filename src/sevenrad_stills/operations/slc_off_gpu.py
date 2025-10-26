"""
GPU-accelerated SLC-Off operation using Taichi.

Simulates the 2003 Landsat 7 ETM+ Scan Line Corrector failure using GPU
acceleration for the gap mask generation and fill operations.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image
from skimage.util import img_as_float32, img_as_ubyte

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
MIN_GAP_WIDTH = 0.0
MAX_GAP_WIDTH = 0.5
MIN_SCAN_PERIOD = 2
MAX_SCAN_PERIOD = 100


@ti.kernel  # type: ignore[misc]
def create_gap_mask(  # type: ignore[no-untyped-def]  # noqa: PLR0913, ANN201
    gap_mask: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
    center_y: ti.i32,
    gap_width: ti.f32,
    scan_period: ti.i32,
    diagonal_offset_per_row: ti.f32,
) -> None:
    """
    GPU kernel to create SLC-Off gap mask with diagonal wedge-shaped gaps.

    Args:
        gap_mask: Output boolean mask array (H, W)
        height: Image height
        width: Image width
        center_y: Center row index
        gap_width: Maximum gap width at edges (fraction of image width)
        scan_period: Number of rows per scan cycle
        diagonal_offset_per_row: Diagonal shift per row (pixels)

    """
    for y, x in ti.ndrange(height, width):
        # Initialize to no gap
        gap_mask[y, x] = 0

        # Find the scan line that this pixel belongs to
        # Scan lines start at y % scan_period == 0
        scan_cycle_start = (y // scan_period) * scan_period
        offset_in_cycle = y - scan_cycle_start

        # Distance from center for the scan line start
        distance_from_center = ti.abs(scan_cycle_start - center_y) / (height / 2.0)
        current_gap_width = ti.cast(distance_from_center * gap_width * width, ti.i32)

        if current_gap_width > 0:
            # Determine scan direction (zig-zag pattern)
            scan_line_number = scan_cycle_start // scan_period
            scan_direction = 1 if (scan_line_number % 2 == 0) else -1

            # Calculate diagonal offset for this row within the cycle
            diagonal_shift = ti.cast(
                diagonal_offset_per_row * offset_in_cycle * scan_direction, ti.i32
            )

            # Gap width at current row's distance from center
            row_distance = ti.abs(y - center_y) / (height / 2.0)
            row_gap_width = ti.cast(row_distance * gap_width * width, ti.i32)

            if row_gap_width > 0:
                # Center gap position with diagonal shift
                gap_center = width // 2 + diagonal_shift
                gap_start = ti.max(0, gap_center - row_gap_width // 2)
                gap_end = ti.min(width, gap_center + row_gap_width // 2)

                # Check if current pixel is within the gap
                if x >= gap_start and x < gap_end:
                    gap_mask[y, x] = 1


@ti.kernel  # type: ignore[misc]
def apply_constant_fill_rgb(  # type: ignore[no-untyped-def]  # noqa: PLR0913, ANN201
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    gap_mask: ti.types.ndarray(),  # type: ignore[valid-type]
    fill_r: ti.f32,
    fill_g: ti.f32,
    fill_b: ti.f32,
    height: ti.i32,
    width: ti.i32,
) -> None:
    """
    GPU kernel to fill gaps with constant RGB value.

    Args:
        img: Image array (H, W, 3) to modify in-place (float32)
        gap_mask: Boolean mask (H, W) indicating gaps
        fill_r: Red channel fill value (0.0-1.0)
        fill_g: Green channel fill value (0.0-1.0)
        fill_b: Blue channel fill value (0.0-1.0)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        if gap_mask[y, x] == 1:
            img[y, x, 0] = fill_r
            img[y, x, 1] = fill_g
            img[y, x, 2] = fill_b


@ti.kernel  # type: ignore[misc]
def apply_constant_fill_gray(  # type: ignore[no-untyped-def]  # noqa: ANN201
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    gap_mask: ti.types.ndarray(),  # type: ignore[valid-type]
    fill_value: ti.f32,
    height: ti.i32,
    width: ti.i32,
) -> None:
    """
    GPU kernel to fill gaps with constant grayscale value.

    Args:
        img: Image array (H, W) to modify in-place (float32)
        gap_mask: Boolean mask (H, W) indicating gaps
        fill_value: Grayscale fill value (0.0-1.0)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        if gap_mask[y, x] == 1:
            img[y, x] = fill_value


class SlcOffGPUOperation(BaseImageOperation):
    """
    GPU-accelerated SLC-Off artifacts simulating Landsat 7 scan line corrector failure.

    Uses Taichi for GPU-accelerated gap mask generation and filling operations.
    See SlcOffOperation for detailed documentation of the algorithm.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated SLC-Off operation."""
        super().__init__("slc_off_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for SLC-Off operation.

        Args:
            params: A dictionary containing:
                - gap_width (float): Maximum gap width at edges as fraction
                  of image width (0.0 to 0.5, represents ~0-50% loss)
                - scan_period (int): Number of rows per scan cycle (2 to 100)
                - fill_mode (str): Gap fill strategy - 'black', 'white', or 'mean'
                - seed (int, optional): Random seed for reproducibility
                  (used for mean fill variation)

        Raises:
            ValueError: If parameters are invalid.

        """
        if "gap_width" not in params:
            msg = "SLC-Off operation requires 'gap_width' parameter."
            raise ValueError(msg)
        gap_width = params["gap_width"]
        if not isinstance(gap_width, (int, float)) or not (
            MIN_GAP_WIDTH <= gap_width <= MAX_GAP_WIDTH
        ):
            msg = (
                f"gap_width must be a number between "
                f"{MIN_GAP_WIDTH} and {MAX_GAP_WIDTH}."
            )
            raise ValueError(msg)

        if "scan_period" not in params:
            msg = "SLC-Off operation requires 'scan_period' parameter."
            raise ValueError(msg)
        scan_period = params["scan_period"]
        if not isinstance(scan_period, int) or not (
            MIN_SCAN_PERIOD <= scan_period <= MAX_SCAN_PERIOD
        ):
            msg = (
                f"scan_period must be an integer between {MIN_SCAN_PERIOD} "
                f"and {MAX_SCAN_PERIOD}."
            )
            raise ValueError(msg)

        if "fill_mode" not in params:
            msg = "SLC-Off operation requires 'fill_mode' parameter."
            raise ValueError(msg)
        fill_mode = params["fill_mode"]
        if fill_mode not in ("black", "white", "mean"):
            msg = "fill_mode must be one of: 'black', 'white', 'mean'."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(  # noqa: PLR0912, PLR0915
        self, image: Image.Image, params: dict[str, Any]
    ) -> Image.Image:
        """
        Apply SLC-Off pattern to the image using GPU acceleration.

        Creates wedge-shaped gaps that widen from center to edges, simulating
        the Landsat 7 scan line corrector failure.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'gap_width', 'scan_period', 'fill_mode',
                    and optional 'seed'.

        Returns:
            The PIL Image with SLC-Off pattern applied.

        """
        self.validate_params(params)

        gap_width: float = params["gap_width"]
        scan_period: int = params["scan_period"]
        fill_mode: str = params["fill_mode"]
        seed: int | None = params.get("seed")

        # Create random number generator (used for mean fill variation)
        rng = np.random.default_rng(seed)

        # Convert to float array (0.0 to 1.0) using skimage utility
        img_float = img_as_float32(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_float[..., :3].copy()
            alpha = img_float[..., 3:4]
            has_alpha = True
        else:
            rgb = img_float.copy()
            alpha = None
            has_alpha = False

        h, w = rgb.shape[:2]
        center_y = h // 2

        # Create gap mask using GPU
        gap_mask = np.zeros((h, w), dtype=np.uint8)
        diagonal_offset_per_row = 0.3  # pixels per row (shallow angle)

        create_gap_mask(
            gap_mask,
            h,
            w,
            center_y,
            gap_width,
            scan_period,
            diagonal_offset_per_row,
        )

        # Apply fill mode (use CPU for simplicity and correctness)
        gap_mask_bool = gap_mask.astype(bool)

        if fill_mode == "black":
            if rgb.ndim == 3:  # RGB/RGBA  # noqa: PLR2004
                rgb[gap_mask_bool] = np.array([0.0, 0.0, 0.0])
            else:  # Grayscale
                rgb[gap_mask_bool] = 0.0
        elif fill_mode == "white":
            if rgb.ndim == 3:  # RGB/RGBA  # noqa: PLR2004
                rgb[gap_mask_bool] = np.array([1.0, 1.0, 1.0])
            else:  # Grayscale
                rgb[gap_mask_bool] = 1.0
        else:  # mean fill
            # For mean fill, use CPU implementation as it requires row-wise computation
            # that's less efficient on GPU
            gap_mask_bool = gap_mask.astype(bool)
            for y in range(h):
                if np.any(gap_mask_bool[y]):
                    # Calculate mean of non-gap pixels in this row
                    if rgb.ndim == 3:  # RGB  # noqa: PLR2004
                        row_mean = np.mean(rgb[y, ~gap_mask_bool[y]], axis=0)
                        # Add small variation to avoid perfect uniformity
                        # Use same integer variation as CPU version for consistency
                        variation_int = rng.integers(-5, 6, size=3, dtype=np.int16)
                        variation = variation_int / 255.0  # Convert to float32 range
                        row_mean = np.clip(row_mean + variation, 0.0, 1.0)
                        rgb[y, gap_mask_bool[y]] = row_mean
                    else:  # Grayscale
                        row_mean = float(np.mean(rgb[y, ~gap_mask_bool[y]]))
                        # Add small variation
                        variation_int = rng.integers(-5, 6)
                        variation = variation_int / 255.0
                        row_mean = np.clip(row_mean + variation, 0.0, 1.0)
                        rgb[y, gap_mask_bool[y]] = row_mean

        # Recombine with alpha if needed
        output_float = np.dstack([rgb, alpha]) if has_alpha else rgb

        # Convert back to uint8 using skimage utility
        output_array = img_as_ubyte(output_float)
        return Image.fromarray(output_array)

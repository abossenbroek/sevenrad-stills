"""
SLC-Off operation for simulating Landsat 7 Scan Line Corrector failure.

Simulates the 2003 Landsat 7 ETM+ Scan Line Corrector failure that created
characteristic wedge-shaped data gaps widening towards image edges.
"""

from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_GAP_WIDTH = 0.0
MAX_GAP_WIDTH = 0.5
MIN_SCAN_PERIOD = 2
MAX_SCAN_PERIOD = 100


class SlcOffOperation(BaseImageOperation):
    """
    Apply SLC-Off artifacts to simulate Landsat 7 scan line corrector failure.

    Simulates the May 31, 2003 failure of the Scan Line Corrector (SLC) on
    Landsat 7's Enhanced Thematic Mapper Plus (ETM+) sensor. The SLC was a
    mirror mechanism that compensated for the forward motion of the spacecraft,
    ensuring scan lines were properly aligned.

    After the SLC failure, the satellite continued to collect data, but with
    characteristic artifacts:
    - Zig-zag scan pattern instead of straight lines
    - Wedge-shaped data gaps that widen towards image edges
    - Maximum gap width at edges (up to 22% of scene width)
    - No gaps at the center line
    - Affects about 22% of pixels across the entire scene

    The failure pattern is geometric and deterministic:
    - Gap width increases linearly from center to edges
    - Scan lines alternate direction (whisk-broom pattern)
    - Creates distinctive diagonal striping

    This operation simulates the geometric distortion by:
    1. Calculating distance from image center for each row
    2. Determining gap width proportional to that distance
    3. Creating diagonal gaps with alternating direction (zig-zag pattern)
    4. Simulating forward satellite motion via diagonal offset
    5. Filling gaps with configurable strategy (black, white, or mean)

    Historical context: This failure affected all Landsat 7 data from 2003
    onwards, but the satellite remained operational for scientific use with
    special gap-filling algorithms applied during post-processing.
    """

    def __init__(self) -> None:
        """Initialize the SLC-Off operation."""
        super().__init__("slc_off")

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

    def apply(  # noqa: C901, PLR0912, PLR0915
        self, image: Image.Image, params: dict[str, Any]
    ) -> Image.Image:
        """
        Apply SLC-Off pattern to the image.

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

        # Convert to array and handle different modes
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            has_alpha = True
        elif image.mode == "RGB":
            rgb = img_array.copy()
            alpha = None
            has_alpha = False
        else:  # Grayscale
            rgb = img_array.copy()
            alpha = None
            has_alpha = False

        h, w = rgb.shape[:2]

        # Calculate center row
        center_y = h // 2

        # Create gap mask
        # For each row, calculate distance from center (normalized to 0..1)
        y_coords = np.arange(h)
        distance_from_center = np.abs(y_coords - center_y) / (h / 2)

        # Gap width increases linearly from center (0) to edges (gap_width)
        # gap_width is a fraction of image width
        current_gap_widths = distance_from_center * gap_width * w

        # Create scan line pattern
        # Alternate scan lines (whisk-broom pattern)
        scan_line_indices: np.ndarray = y_coords % scan_period

        # Create the gap mask with diagonal wedge-shaped gaps (zig-zag pattern)
        # The real SLC-Off creates diagonal gaps due to uncompensated forward motion
        # Gaps alternate direction (left/right) creating characteristic zig-zag
        gap_mask = np.zeros((h, w), dtype=bool)

        # Diagonal offset per row (simulates satellite forward motion during scan)
        # Small angle creates shallow diagonal gaps typical of real SLC-Off
        diagonal_offset_per_row = 0.3  # pixels per row (shallow angle)

        scan_line_number = 0  # Track which scan line we're on
        for y in range(h):
            # Only create gaps on scan lines (every scan_period rows)
            if scan_line_indices[y] == 0:
                current_gap_width = int(current_gap_widths[y])
                if current_gap_width > 0:
                    # Determine scan direction (alternating for zig-zag pattern)
                    # Even scan lines go left-to-right, odd go right-to-left
                    scan_direction = 1 if (scan_line_number % 2 == 0) else -1

                    # Create diagonal gap across multiple rows
                    # Gap extends for scan_period rows with diagonal offset
                    for offset_row in range(min(scan_period, h - y)):
                        actual_y = y + offset_row
                        if actual_y >= h:
                            break

                        # Calculate diagonal offset for this row
                        diagonal_shift = int(
                            diagonal_offset_per_row * offset_row * scan_direction
                        )

                        # Gap width at this distance from center
                        row_distance = np.abs(actual_y - center_y) / (h / 2)
                        row_gap_width = int(row_distance * gap_width * w)

                        if row_gap_width > 0:
                            # Center gap position with diagonal shift
                            gap_center = w // 2 + diagonal_shift
                            gap_start = max(0, gap_center - row_gap_width // 2)
                            gap_end = min(w, gap_center + row_gap_width // 2)
                            gap_mask[actual_y, gap_start:gap_end] = True

                    scan_line_number += 1

        # Apply fill mode to gaps
        if fill_mode == "black":
            fill_value = np.array([0, 0, 0], dtype=rgb.dtype)
        elif fill_mode == "white":
            fill_value = np.array([255, 255, 255], dtype=rgb.dtype)
        else:  # mean
            # Calculate mean color of image (or use per-row mean for variation)
            # Use per-row mean for more realistic gap filling
            fill_value = None  # Will be calculated per row

        # Apply gaps
        if fill_mode in ("black", "white"):
            # Simple fill with constant value
            if rgb.ndim == 3:  # RGB/RGBA  # noqa: PLR2004
                rgb[gap_mask] = fill_value
            else:  # Grayscale
                rgb[gap_mask] = fill_value[0]
        else:  # mean fill
            # Fill with row mean (interpolation-like approach)
            for y in range(h):
                if np.any(gap_mask[y]):
                    # Calculate mean of non-gap pixels in this row
                    if rgb.ndim == 3:  # RGB  # noqa: PLR2004
                        row_mean = np.mean(rgb[y, ~gap_mask[y]], axis=0).astype(
                            rgb.dtype
                        )
                        # Add small variation to avoid perfect uniformity
                        variation = rng.integers(-5, 6, size=3, dtype=np.int16)
                        row_mean = np.clip(
                            row_mean.astype(np.int16) + variation, 0, 255
                        ).astype(rgb.dtype)
                        rgb[y, gap_mask[y]] = row_mean
                    else:  # Grayscale
                        row_mean = int(np.mean(rgb[y, ~gap_mask[y]]))
                        # Add small variation
                        variation = rng.integers(-5, 6)
                        row_mean = np.clip(row_mean + variation, 0, 255)
                        rgb[y, gap_mask[y]] = row_mean

        # Recombine with alpha if needed
        output_array = np.dstack([rgb, alpha]) if has_alpha else rgb

        return Image.fromarray(output_array)

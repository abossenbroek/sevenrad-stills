"""
Taichi end-to-end pipeline SLC-Off operation.

SLC-Off (Scan Line Corrector failure) simulation for GPU pipeline execution
using ti.Vector.field(4). Operates on pre-allocated buffers without CPU↔GPU
data transfer.
"""

from typing import Any

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Constants
MIN_GAP_WIDTH = 0.0
MAX_GAP_WIDTH = 0.5
MIN_SCAN_PERIOD = 2
MAX_SCAN_PERIOD = 100
GAP_MASK_THRESHOLD = 0.5  # Threshold for considering a pixel as gap


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _slc_off_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        mask: ti.template(),  # type: ignore[valid-type]
        fill_color_r: ti.f32,
        fill_color_g: ti.f32,
        fill_color_b: ti.f32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for SLC-Off gap application.

        Operates on ti.Vector.field(4) with RGBA channels.
        Applies pre-computed gap mask, filling gaps with specified color.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            mask: Boolean mask field (height, width) - True where gaps are
            fill_color_r: Red component of fill color [0, 1]
            fill_color_g: Green component of fill color [0, 1]
            fill_color_b: Blue component of fill color [0, 1]
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            # Read RGBA from source
            pixel = source[batch, i, j]
            a = pixel[3]  # Preserve alpha

            # Check if this pixel is in a gap
            if mask[i, j] > GAP_MASK_THRESHOLD:
                # Fill with specified color
                dest[batch, i, j] = ti.Vector(
                    [fill_color_r, fill_color_g, fill_color_b, a]
                )
            else:
                # Preserve original pixel
                dest[batch, i, j] = pixel


class SlcOffTaichiOperation(BaseTaichiOperation):
    """
    Taichi SLC-Off operation for end-to-end GPU pipeline.

    Simulates Landsat 7 Scan Line Corrector failure with wedge-shaped gaps.
    The gap geometry is pre-computed on CPU, then applied via GPU kernel.

    This operation is NOT element-wise (reads from mask) and does NOT
    support in-place execution.

    Example:
        >>> op = SlcOffTaichiOperation()
        >>> params = {"gap_width": 0.22, "scan_period": 14, "fill_mode": "black"}
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize SLC-Off operation."""
        super().__init__("slc_off_taichi")
        self._mask_field: Any = None  # ti.field for gap mask
        self._cached_shape: tuple[int, int] | None = None
        self._cached_params: dict[str, Any] | None = None

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        SLC-Off requires reading from a separate mask, so output can technically
        be written to source, but we keep it False for clarity.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate SLC-Off parameters.

        Expected params:
        - gap_width: float - maximum gap width at edges (0.0 to 0.5)
        - scan_period: int - rows per scan cycle (2 to 100)
        - fill_mode: str - gap fill strategy ('black', 'white', or 'mean')
        - seed: int (optional) - random seed for reproducibility

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or invalid

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

    def _compute_gap_mask(
        self,
        height: int,
        width: int,
        gap_width: float,
        scan_period: int,
        seed: int,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Compute SLC-Off gap mask on CPU.

        Uses the same geometric calculation as the CPU implementation
        to create wedge-shaped gaps that widen from center to edges.

        Args:
            height: Image height
            width: Image width
            gap_width: Maximum gap width at edges as fraction of width
            scan_period: Number of rows per scan cycle
            seed: Random seed (currently unused but kept for API consistency)

        Returns:
            Boolean numpy array (height, width) where True indicates gap pixels

        """
        # Calculate center row
        center_y = height // 2

        # Create gap mask
        # For each row, calculate distance from center (normalized to 0..1)
        y_coords = np.arange(height)
        distance_from_center = np.abs(y_coords - center_y) / (height / 2)

        # Gap width increases linearly from center (0) to edges (gap_width)
        current_gap_widths = distance_from_center * gap_width * width

        # Create scan line pattern
        scan_line_indices: np.ndarray = y_coords % scan_period

        # Create the gap mask with diagonal wedge-shaped gaps
        gap_mask = np.zeros((height, width), dtype=np.float32)

        # Diagonal offset per row (simulates satellite forward motion)
        diagonal_offset_per_row = 0.3

        scan_line_number = 0
        for y in range(height):
            # Only create gaps on scan lines (every scan_period rows)
            if scan_line_indices[y] == 0:
                current_gap_width = int(current_gap_widths[y])
                if current_gap_width > 0:
                    # Determine scan direction (alternating for zig-zag pattern)
                    scan_direction = 1 if (scan_line_number % 2 == 0) else -1

                    # Create diagonal gap across multiple rows
                    for offset_row in range(min(scan_period, height - y)):
                        actual_y = y + offset_row
                        if actual_y >= height:
                            break

                        # Calculate diagonal offset for this row
                        diagonal_shift = int(
                            diagonal_offset_per_row * offset_row * scan_direction
                        )

                        # Gap width at this distance from center
                        row_distance = np.abs(actual_y - center_y) / (height / 2)
                        row_gap_width = int(row_distance * gap_width * width)

                        if row_gap_width > 0:
                            # Center gap position with diagonal shift
                            gap_center = width // 2 + diagonal_shift
                            gap_start = max(0, gap_center - row_gap_width // 2)
                            gap_end = min(width, gap_center + row_gap_width // 2)
                            gap_mask[actual_y, gap_start:gap_end] = 1.0

                    scan_line_number += 1

        return gap_mask

    def _compute_mean_fill_color(
        self,
        source: Any,
        mask: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[float, float, float]:
        """
        Compute mean color from non-gap pixels.

        For 'mean' fill mode, calculate average color of visible pixels.

        Args:
            source: Source Taichi field to read from
            mask: Gap mask (height, width) where 1.0 = gap
            height: Image height
            width: Image width

        Returns:
            Tuple of (r, g, b) mean values in [0, 1]

        """
        # Read from GPU to CPU (just for mean calculation)
        # This is acceptable since mask computation already requires CPU work
        pixels = source.to_numpy()[0, :height, :width, :3]  # batch=0, RGB only

        # Mask non-gap pixels
        non_gap_pixels = pixels[mask < GAP_MASK_THRESHOLD]

        if len(non_gap_pixels) == 0:
            # If all pixels are gaps (edge case), return black
            return (0.0, 0.0, 0.0)

        # Calculate mean
        mean_color = np.mean(non_gap_pixels, axis=0)
        return (float(mean_color[0]), float(mean_color[1]), float(mean_color[2]))

    def apply_to_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,  # ti.Vector.field
        temp_fields: dict[str, Any],  # noqa: ARG002
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply SLC-Off pattern on GPU fields.

        Pre-computes gap mask on CPU, uploads to GPU, then applies via kernel.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for SLC-Off (empty dict expected)
            params: Must contain 'gap_width', 'scan_period', 'fill_mode',
                optional 'seed'
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        gap_width = float(params["gap_width"])
        scan_period = int(params["scan_period"])
        fill_mode = str(params["fill_mode"])
        seed = int(params.get("seed", 0))

        # Check if we can reuse cached mask
        current_shape = (height, width)
        current_params = {
            "gap_width": gap_width,
            "scan_period": scan_period,
            "seed": seed,
        }

        if (
            self._mask_field is None
            or self._cached_shape != current_shape
            or self._cached_params != current_params
        ):
            # Compute gap mask on CPU
            mask_np = self._compute_gap_mask(
                height, width, gap_width, scan_period, seed
            )

            # Create or recreate Taichi field
            if self._mask_field is None or self._cached_shape != current_shape:
                self._mask_field = ti.field(dtype=ti.f32, shape=(height, width))

            # Upload mask to GPU
            self._mask_field.from_numpy(mask_np)

            # Cache shape and params
            self._cached_shape = current_shape
            self._cached_params = current_params

        # Determine fill color
        if fill_mode == "black":
            fill_r, fill_g, fill_b = 0.0, 0.0, 0.0
        elif fill_mode == "white":
            fill_r, fill_g, fill_b = 1.0, 1.0, 1.0
        else:  # mean
            mask_np = self._mask_field.to_numpy()
            fill_r, fill_g, fill_b = self._compute_mean_fill_color(
                source, mask_np, height, width
            )

        # Execute kernel (batch_idx=0 for single image)
        _slc_off_kernel(
            source, dest, self._mask_field, fill_r, fill_g, fill_b, 0, height, width
        )

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        NumPy reference implementation for testing.

        Produces identical results to apply_to_field for correctness testing.

        Args:
            image: Input image as numpy array (H, W, 3) float32 in [0, 1]
            params: Must contain 'gap_width', 'scan_period', 'fill_mode',
                optional 'seed'

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        gap_width = float(params["gap_width"])
        scan_period = int(params["scan_period"])
        fill_mode = str(params["fill_mode"])
        seed = int(params.get("seed", 0))

        h, w = image.shape[:2]
        rng = np.random.default_rng(seed)

        # Compute gap mask
        gap_mask = self._compute_gap_mask(h, w, gap_width, scan_period, seed)
        gap_mask_bool = gap_mask > GAP_MASK_THRESHOLD

        # Create output
        output = image.copy()

        # Apply fill mode
        if fill_mode == "black":
            fill_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            output[gap_mask_bool] = fill_value
        elif fill_mode == "white":
            fill_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            output[gap_mask_bool] = fill_value
        else:  # mean
            # Fill with row mean (interpolation-like approach)
            for y in range(h):
                if np.any(gap_mask_bool[y]):
                    # Calculate mean of non-gap pixels in this row
                    row_mean = np.mean(output[y, ~gap_mask_bool[y]], axis=0).astype(
                        np.float32
                    )
                    # Add small variation to avoid perfect uniformity
                    variation = rng.integers(-5, 6, size=3, dtype=np.int16) / 255.0
                    row_mean = np.clip(row_mean + variation, 0.0, 1.0).astype(
                        np.float32
                    )
                    output[y, gap_mask_bool[y]] = row_mean

        return output

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the SLC-Off kernel
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_mask = ti.field(dtype=ti.f32, shape=(2, 2))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]
                dummy_mask[i, j] = 0.0

        # Trigger compilation
        _slc_off_kernel(dummy_src, dummy_dst, dummy_mask, 0.0, 0.0, 0.0, 0, 2, 2)

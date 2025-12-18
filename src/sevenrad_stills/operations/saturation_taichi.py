"""
Taichi end-to-end pipeline saturation operation.

Saturation adjustment for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

import random
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
HUE_MODULO = 360.0


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _saturation_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        factor: ti.f32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for saturation adjustment using HSV color space.

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            factor: Saturation multiplier (0=grayscale, 1=original, >1=saturated)
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            # Read RGBA from source
            pixel = source[batch, i, j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            a = pixel[3]  # Preserve alpha

            # RGB to HSV conversion
            cmax = ti.max(ti.max(r, g), b)
            cmin = ti.min(ti.min(r, g), b)
            delta = cmax - cmin

            # Calculate Hue
            h = 0.0
            if delta > EPSILON:
                if ti.abs(cmax - r) < EPSILON:
                    h = HUE_SECTOR_0 * (((g - b) / delta) % HUE_SECTORS)
                elif ti.abs(cmax - g) < EPSILON:
                    h = HUE_SECTOR_0 * (((b - r) / delta) + 2.0)
                else:
                    h = HUE_SECTOR_0 * (((r - g) / delta) + 4.0)

            # Calculate Saturation
            s = 0.0
            if cmax > EPSILON:
                s = delta / cmax

            # Value
            v = cmax

            # Adjust saturation by factor
            s = ti.max(0.0, ti.min(1.0, s * factor))

            # HSV to RGB conversion
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

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r_new + m, g_new + m, b_new + m, a])


class SaturationTaichiOperation(BaseTaichiOperation):
    """
    Taichi saturation adjustment for end-to-end GPU pipeline.

    Adjusts image saturation using HSV color space conversion.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation is element-wise and supports in-place execution.

    Example:
        >>> op = SaturationTaichiOperation()
        >>> op.apply_to_field(source, dest, {}, {"factor": 1.5}, height, width)

    """

    def __init__(self) -> None:
        """Initialize saturation operation."""
        super().__init__("saturation_taichi")

    def _resolve_factor(self, params: dict[str, Any]) -> float:
        """
        Resolve saturation factor from parameters.

        Supports two parameter formats:
        1. Legacy: {"factor": 1.5} - Direct factor value
        2. New API: {"mode": "fixed", "value": 0.5} or {"mode": "random", "range": [-0.5, 0.5]}

        Args:
            params: Operation parameters

        Returns:
            Resolved saturation factor (0.0=grayscale, 1.0=original, >1=saturated)

        """
        # Legacy backward compatibility: direct factor parameter
        if "factor" in params:
            return float(params["factor"])

        # New API: mode-based parameters
        mode = params["mode"]
        if mode == "fixed":
            value = float(params["value"])
            # Convert value to factor: value is adjustment, factor is multiplier
            # value=-1.0 -> factor=0.0 (grayscale)
            # value=0.0 -> factor=1.0 (original)
            # value=0.5 -> factor=1.5 (more saturated)
            return max(0.0, 1.0 + value)
        else:  # random
            min_val, max_val = params["range"]
            value = random.uniform(min_val, max_val)  # noqa: S311
            return max(0.0, 1.0 + value)

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Saturation is element-wise: output[i,j] only depends on input[i,j].

        Returns:
            True - this operation supports in-place execution.

        """
        return True

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate saturation parameters.

        Supports two parameter formats:
        1. Legacy: {"factor": 1.5} - Direct factor value (backward compatibility)
        2. New API: {"mode": "fixed", "value": 0.5} or {"mode": "random", "range": [-0.5, 0.5]}

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid or missing

        """
        # Legacy backward compatibility: direct factor parameter
        if "factor" in params:
            factor = params["factor"]
            if not isinstance(factor, (int, float)):
                msg = f"Factor must be a number, got {type(factor)}"
                raise ValueError(msg)
            if factor < 0.0:
                msg = f"Factor must be >= 0.0, got {factor}"
                raise ValueError(msg)
            return

        # New API: mode-based parameters
        if "mode" not in params:
            msg = "Saturation requires either 'factor' or 'mode' parameter"
            raise ValueError(msg)

        mode = params["mode"]
        if mode not in ("fixed", "random"):
            msg = f"Invalid mode '{mode}'. Must be 'fixed' or 'random'"
            raise ValueError(msg)

        if mode == "fixed":
            # Validate fixed mode parameters
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
        else:  # random
            # Validate random mode parameters
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
        Apply saturation adjustment on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for saturation (empty dict expected)
            params: Saturation parameters (factor or mode/value/range)
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        factor = self._resolve_factor(params)

        # Execute kernel (batch_idx=0 for single image)
        _saturation_kernel(source, dest, factor, 0, height, width)

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
            params: Saturation parameters (factor or mode/value/range)

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        factor = self._resolve_factor(params)

        # Extract RGB channels
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        # RGB to HSV conversion (vectorized)
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
        result = np.zeros_like(image)

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

        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the saturation kernel
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Trigger compilation
        _saturation_kernel(dummy_src, dummy_dst, 1.0, 0, 2, 2)

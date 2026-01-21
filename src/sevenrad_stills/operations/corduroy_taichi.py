"""
Taichi end-to-end pipeline corduroy operation.

Corduroy striping for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
Simulates detector calibration errors in push-broom/whisk-broom scanners.
"""

from typing import Any, Literal

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
from sevenrad_stills.operations.taichi_kernels.random import rand_float

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Constants
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0
STRENGTH_SCALE = 0.2


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _corduroy_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        multipliers: ti.template(),  # type: ignore[valid-type]
        is_vertical: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for corduroy striping effect.

        Operates on ti.Vector.field(4) with RGBA channels.
        Multiplies RGB channels by pre-computed row/column multipliers.
        Preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            multipliers: Field of multipliers
                        (height for horizontal, width for vertical)
            is_vertical: 1 for vertical stripes, 0 for horizontal
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

            # Get multiplier based on orientation
            # Vertical: same multiplier for entire column, Horizontal: entire row
            multiplier = multipliers[j] if is_vertical == 1 else multipliers[i]

            # Apply multiplier to RGB channels
            r_new = ti.max(0.0, ti.min(1.0, r * multiplier))
            g_new = ti.max(0.0, ti.min(1.0, g * multiplier))
            b_new = ti.max(0.0, ti.min(1.0, b * multiplier))

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r_new, g_new, b_new, a])


class CorduroyTaichiOperation(BaseTaichiOperation):
    """
    Taichi corduroy striping for end-to-end GPU pipeline.

    Simulates "corduroy" or "banding" artifacts from push-broom/whisk-broom
    scanners where individual detector elements have slightly different
    sensitivity due to calibration drift or manufacturing variations.

    This operation is element-wise and supports in-place execution.

    Example:
        >>> op = CorduroyTaichiOperation()
        >>> params = {
        ...     "orientation": "vertical",
        ...     "strength": 0.5,
        ...     "density": 0.3
        ... }
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize corduroy operation."""
        super().__init__("corduroy_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Corduroy is element-wise: output[i,j] only depends on input[i,j]
        and its row/column multiplier.

        Returns:
            True - this operation supports in-place execution.

        """
        return True

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate corduroy parameters.

        Expected params:
        - orientation: "vertical" or "horizontal" - line direction
        - strength: float - striping intensity (0.0 to 1.0)
        - density: float - proportion of lines affected (0.0 to 1.0)
        - seed: int (optional) - random seed for reproducibility (default: 0)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or invalid

        """
        if "orientation" not in params:
            msg = "Corduroy requires 'orientation' parameter"
            raise ValueError(msg)

        orientation = params["orientation"]
        if orientation not in ("vertical", "horizontal"):
            msg = "Orientation must be 'vertical' or 'horizontal'"
            raise ValueError(msg)

        if "strength" not in params:
            msg = "Corduroy requires 'strength' parameter"
            raise ValueError(msg)

        strength = params["strength"]
        if not isinstance(strength, (int, float)):
            msg = f"Strength must be a number, got {type(strength)}"
            raise ValueError(msg)

        if not (MIN_STRENGTH <= strength <= MAX_STRENGTH):
            msg = (
                f"Strength must be between {MIN_STRENGTH} and {MAX_STRENGTH}, "
                f"got {strength}"
            )
            raise ValueError(msg)

        if "density" not in params:
            msg = "Corduroy requires 'density' parameter"
            raise ValueError(msg)

        density = params["density"]
        if not isinstance(density, (int, float)):
            msg = f"Density must be a number, got {type(density)}"
            raise ValueError(msg)

        if not (MIN_DENSITY <= density <= MAX_DENSITY):
            msg = (
                f"Density must be between {MIN_DENSITY} and {MAX_DENSITY}, "
                f"got {density}"
            )
            raise ValueError(msg)

        # Seed is optional, but validate if provided
        if "seed" in params:
            seed = params["seed"]
            if not isinstance(seed, int):
                msg = f"Seed must be an integer, got {type(seed)}"
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
        Apply corduroy striping on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for corduroy (empty dict expected)
            params: Must contain orientation, strength, density; seed is optional
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        orientation: Literal["vertical", "horizontal"] = params["orientation"]
        strength = float(params["strength"])
        density = float(params["density"])
        seed = int(params.get("seed", 0))

        # Determine dimensions
        if orientation == "vertical":
            num_lines = width
            num_affected = int(density * width)
        else:  # horizontal
            num_lines = height
            num_affected = int(density * height)

        # Create multipliers field
        multipliers = ti.field(dtype=ti.f32, shape=num_lines)

        # Initialize all multipliers to 1.0
        multipliers.fill(1.0)

        # Generate affected line indices using NumPy (deterministic)
        rng = np.random.default_rng(seed)
        if num_affected > 0:
            affected_lines = rng.choice(num_lines, size=num_affected, replace=False)

            # Generate multipliers for affected lines
            # Use rand_float to match NumPy RNG behavior for consistency
            for line_idx in affected_lines:
                # Generate random multiplier in range
                # [1.0 - strength*0.2, 1.0 + strength*0.2]
                # Use line_idx and seed to ensure determinism
                random_val = rng.uniform(
                    1.0 - strength * STRENGTH_SCALE,
                    1.0 + strength * STRENGTH_SCALE,
                )
                multipliers[int(line_idx)] = random_val

        # Execute kernel
        is_vertical = 1 if orientation == "vertical" else 0
        _corduroy_kernel(source, dest, multipliers, is_vertical, 0, height, width)

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
            params: Must contain orientation, strength, density; seed is optional

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        orientation: Literal["vertical", "horizontal"] = params["orientation"]
        strength = float(params["strength"])
        density = float(params["density"])
        seed = int(params.get("seed", 0))

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Work on a copy
        result = image.copy()

        h, w = result.shape[:2]

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
            multipliers = rng.uniform(
                1.0 - strength * STRENGTH_SCALE,
                1.0 + strength * STRENGTH_SCALE,
                size=num_lines,
            )

            # Apply multipliers using vectorized NumPy broadcasting
            is_grayscale = result.ndim == 2

            if orientation == "vertical":
                multipliers_array = np.ones(w, dtype=np.float32)
                multipliers_array[affected_lines] = multipliers
                # Broadcast across height (and channels if RGB)
                if is_grayscale:
                    # Grayscale: (1, w) * (h, w)
                    result *= multipliers_array[np.newaxis, :]
                else:
                    # RGB: (1, w, 1) * (h, w, 3)
                    result *= multipliers_array[np.newaxis, :, np.newaxis]
            else:  # horizontal
                multipliers_array = np.ones(h, dtype=np.float32)
                multipliers_array[affected_lines] = multipliers
                # Broadcast across width (and channels if RGB)
                if is_grayscale:
                    # Grayscale: (h, 1) * (h, w)
                    result *= multipliers_array[:, np.newaxis]
                else:
                    # RGB: (h, 1, 1) * (h, w, 3)
                    result *= multipliers_array[:, np.newaxis, np.newaxis]

            # Clip values to valid range
            np.clip(result, 0.0, 1.0, out=result)

        return result.astype(np.float32)

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the corduroy kernel
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_mult = ti.field(dtype=ti.f32, shape=2)

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]
            dummy_mult[i] = 1.0

        # Trigger compilation (vertical orientation)
        _corduroy_kernel(dummy_src, dummy_dst, dummy_mult, 1, 0, 2, 2)

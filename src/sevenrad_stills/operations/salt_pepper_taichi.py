"""
Taichi end-to-end pipeline salt and pepper noise operation.

Salt and pepper noise for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

from typing import Any

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    from sevenrad_stills.operations.taichi_kernels.random import rand_float

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    rand_float = None
    TAICHI_AVAILABLE = False

# Constants
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0
SALT_PEPPER_THRESHOLD = 0.5  # 50/50 split between salt and pepper


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _salt_pepper_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        density: ti.f32,
        seed: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for salt and pepper noise.

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            density: Proportion of pixels to affect (0.0 to 1.0)
            seed: Random seed for reproducibility
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

            # Determine if this pixel gets noise
            noise_rand = rand_float(i, j, seed)

            if noise_rand < density:
                # Determine if salt or pepper (use different seed offset)
                salt_rand = rand_float(i, j, seed + 1)

                if salt_rand < SALT_PEPPER_THRESHOLD:
                    # Pepper - black
                    r = 0.0
                    g = 0.0
                    b = 0.0
                else:
                    # Salt - white
                    r = 1.0
                    g = 1.0
                    b = 1.0

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r, g, b, a])


class SaltPepperTaichiOperation(BaseTaichiOperation):
    """
    Taichi salt and pepper noise for end-to-end GPU pipeline.

    Adds random black (pepper) and white (salt) pixels to simulate
    sensor defects, cosmic ray hits, or manufacturing imperfections.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation is element-wise and supports in-place execution.

    Example:
        >>> op = SaltPepperTaichiOperation()
        >>> op.apply_to_field(source, dest, {}, {"density": 0.05, "seed": 42}, h, w)

    """

    def __init__(self) -> None:
        """Initialize salt and pepper noise operation."""
        super().__init__("salt_pepper_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Salt and pepper is element-wise: output[i,j] only depends on input[i,j].

        Returns:
            True - this operation supports in-place execution.

        """
        return True

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate salt and pepper noise parameters.

        Expected params:
        - density: float - proportion of pixels affected (0.0 to 1.0)
        - seed: int (optional) - random seed for reproducibility

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If density is missing or invalid

        """
        if "density" not in params:
            msg = "Salt and pepper requires 'density' parameter"
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

        if "seed" in params and not isinstance(params["seed"], int):
            msg = f"Seed must be an integer, got {type(params['seed'])}"
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
        Apply salt and pepper noise on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for salt and pepper (empty dict expected)
            params: Must contain 'density' key, optional 'seed' key
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        density = float(params["density"])
        seed = params.get("seed", 0)

        # Execute kernel (batch_idx=0 for single image)
        _salt_pepper_kernel(source, dest, density, seed, 0, height, width)

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
            params: Must contain 'density' key, optional 'seed' key

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        density = float(params["density"])
        seed = params.get("seed", 0)

        height, width = image.shape[:2]
        result = image.copy()

        # Apply noise pixel by pixel using same logic as kernel
        for i in range(height):
            for j in range(width):
                # Determine if this pixel gets noise
                noise_rand = rand_float_numpy(j, i, seed)

                if noise_rand < density:
                    # Determine if salt or pepper
                    salt_rand = rand_float_numpy(j, i, seed + 1)

                    if salt_rand < SALT_PEPPER_THRESHOLD:
                        # Pepper - black
                        result[i, j] = [0.0, 0.0, 0.0]
                    else:
                        # Salt - white
                        result[i, j] = [1.0, 1.0, 1.0]

        return result.astype(np.float32)

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the salt and pepper kernel
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
        _salt_pepper_kernel(dummy_src, dummy_dst, 0.1, 0, 0, 2, 2)

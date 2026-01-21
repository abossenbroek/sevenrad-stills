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
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0
MIN_SALT_VS_PEPPER = 0.0
MAX_SALT_VS_PEPPER = 1.0


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _salt_pepper_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        amount: ti.f32,
        salt_vs_pepper: ti.f32,
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
            amount: Proportion of pixels to affect (0.0 to 1.0)
            salt_vs_pepper: Ratio of salt to pepper (0.0=all pepper, 1.0=all salt)
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

            if noise_rand < amount:
                # Determine if salt or pepper (use different seed offset)
                salt_rand = rand_float(i, j, seed + 1)

                if salt_rand < salt_vs_pepper:
                    # Salt - white
                    r = 1.0
                    g = 1.0
                    b = 1.0
                else:
                    # Pepper - black
                    r = 0.0
                    g = 0.0
                    b = 0.0

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
        >>> params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        >>> op.apply_to_field(source, dest, {}, params, h, w)

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
        - amount: float - proportion of pixels affected (0.0 to 1.0)
        - density: float (deprecated, use 'amount') - backward compatibility
        - salt_vs_pepper: float - ratio of salt to pepper (0.0 to 1.0)
        - seed: int (optional) - random seed for reproducibility

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If required parameters are missing or invalid

        """
        # Check for amount or density (backward compatibility)
        if "amount" not in params and "density" not in params:
            msg = "Salt and pepper requires 'amount' parameter"
            raise ValueError(msg)

        # Get amount (prefer 'amount', fallback to 'density')
        amount = params.get("amount", params.get("density"))
        if not isinstance(amount, (int, float)):
            msg = f"Amount must be a number, got {type(amount)}"
            raise ValueError(msg)

        if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
            msg = (
                f"Amount must be between {MIN_AMOUNT} and {MAX_AMOUNT}, "
                f"got {amount}"
            )
            raise ValueError(msg)

        # Check for required salt_vs_pepper parameter
        if "salt_vs_pepper" not in params:
            msg = "Salt and pepper requires 'salt_vs_pepper' parameter"
            raise ValueError(msg)

        salt_vs_pepper = params["salt_vs_pepper"]
        if not isinstance(salt_vs_pepper, (int, float)):
            msg = f"salt_vs_pepper must be a number, got {type(salt_vs_pepper)}"
            raise ValueError(msg)

        if not (MIN_SALT_VS_PEPPER <= salt_vs_pepper <= MAX_SALT_VS_PEPPER):
            msg = (
                f"salt_vs_pepper must be between {MIN_SALT_VS_PEPPER} and "
                f"{MAX_SALT_VS_PEPPER}, got {salt_vs_pepper}"
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
            params: Must contain 'amount' (or 'density' for backward compat),
                    'salt_vs_pepper', and optional 'seed' keys
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        # Get amount (prefer 'amount', fallback to 'density' for backward compat)
        amount = float(params.get("amount", params.get("density")))
        salt_vs_pepper = float(params["salt_vs_pepper"])
        seed = params.get("seed", 0)

        # Execute kernel (batch_idx=0 for single image)
        _salt_pepper_kernel(
            source, dest, amount, salt_vs_pepper, seed, 0, height, width
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
            params: Must contain 'amount' (or 'density' for backward compat),
                    'salt_vs_pepper', and optional 'seed' keys

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        # Get amount (prefer 'amount', fallback to 'density' for backward compat)
        amount = float(params.get("amount", params.get("density")))
        salt_vs_pepper = float(params["salt_vs_pepper"])
        seed = params.get("seed", 0)

        height, width = image.shape[:2]
        result = image.copy()

        # Apply noise pixel by pixel using same logic as kernel
        for i in range(height):
            for j in range(width):
                # Determine if this pixel gets noise
                noise_rand = rand_float_numpy(j, i, seed)

                if noise_rand < amount:
                    # Determine if salt or pepper
                    salt_rand = rand_float_numpy(j, i, seed + 1)

                    if salt_rand < salt_vs_pepper:
                        # Salt - white
                        result[i, j] = [1.0, 1.0, 1.0]
                    else:
                        # Pepper - black
                        result[i, j] = [0.0, 0.0, 0.0]

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
        _salt_pepper_kernel(dummy_src, dummy_dst, 0.1, 0.5, 0, 0, 2, 2)

"""
Taichi end-to-end pipeline noise operation.

Noise generation for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

from typing import Any, Literal

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    from sevenrad_stills.operations.taichi_kernels.random import (
        rand_float,
        rand_gaussian,
    )

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Constants
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0


# Define the kernels only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _noise_gaussian_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        amount: ti.f32,
        seed: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for Gaussian noise addition.

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            amount: Noise intensity (standard deviation for Gaussian)
            seed: Random seed for reproducible noise
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

            # Generate independent Gaussian noise for each RGB channel
            noise_r = rand_gaussian(i, j, seed, amount)
            noise_g = rand_gaussian(i, j, seed + 1, amount)
            noise_b = rand_gaussian(i, j, seed + 2, amount)

            # Apply noise and clamp
            r_new = ti.max(0.0, ti.min(1.0, r + noise_r))
            g_new = ti.max(0.0, ti.min(1.0, g + noise_g))
            b_new = ti.max(0.0, ti.min(1.0, b + noise_b))

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r_new, g_new, b_new, a])

    @ti.kernel  # type: ignore[misc]
    def _noise_row_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        amount: ti.f32,
        seed: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for row-based noise (horizontal scan line artifacts).

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            amount: Noise intensity (uniform range [-amount, amount])
            seed: Random seed for reproducible noise
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

            # Generate row-specific noise (same for all pixels in row)
            # Using column=0 to ensure all pixels in row get same noise
            u_r = rand_float(i, 0, seed)
            u_g = rand_float(i, 0, seed + 1)
            u_b = rand_float(i, 0, seed + 2)

            # Map [0,1] to [-amount, amount]
            noise_r = (u_r * 2.0 - 1.0) * amount
            noise_g = (u_g * 2.0 - 1.0) * amount
            noise_b = (u_b * 2.0 - 1.0) * amount

            # Apply noise and clamp
            r_new = ti.max(0.0, ti.min(1.0, r + noise_r))
            g_new = ti.max(0.0, ti.min(1.0, g + noise_g))
            b_new = ti.max(0.0, ti.min(1.0, b + noise_b))

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r_new, g_new, b_new, a])

    @ti.kernel  # type: ignore[misc]
    def _noise_column_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        amount: ti.f32,
        seed: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for column-based noise (vertical artifacts).

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            amount: Noise intensity (uniform range [-amount, amount])
            seed: Random seed for reproducible noise
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

            # Generate column-specific noise (same for all pixels in column)
            # Using row=0 to ensure all pixels in column get same noise
            u_r = rand_float(0, j, seed)
            u_g = rand_float(0, j, seed + 1)
            u_b = rand_float(0, j, seed + 2)

            # Map [0,1] to [-amount, amount]
            noise_r = (u_r * 2.0 - 1.0) * amount
            noise_g = (u_g * 2.0 - 1.0) * amount
            noise_b = (u_b * 2.0 - 1.0) * amount

            # Apply noise and clamp
            r_new = ti.max(0.0, ti.min(1.0, r + noise_r))
            g_new = ti.max(0.0, ti.min(1.0, g + noise_g))
            b_new = ti.max(0.0, ti.min(1.0, b + noise_b))

            # Write result with preserved alpha
            dest[batch, i, j] = ti.Vector([r_new, g_new, b_new, a])


class NoiseTaichiOperation(BaseTaichiOperation):
    """
    Taichi noise generation for end-to-end GPU pipeline.

    Adds noise to images using three different modes:
    - gaussian: Per-pixel Gaussian noise
    - row: Horizontal scan line artifacts
    - column: Vertical artifacts

    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.
    This operation is element-wise and supports in-place execution.

    Example:
        >>> op = NoiseTaichiOperation()
        >>> params = {"mode": "gaussian", "amount": 0.1, "seed": 42}
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize noise operation."""
        super().__init__("noise_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Noise is element-wise: output[i,j] only depends on input[i,j]
        and deterministic RNG based on coordinates.

        Returns:
            True - this operation supports in-place execution.

        """
        return True

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate noise parameters.

        Expected params:
        - mode: str - "gaussian", "row", or "column"
        - amount: float - noise intensity in [0.0, 1.0]
        - seed: int - random seed (optional, defaults to 0)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or invalid

        """
        if "mode" not in params:
            msg = "Noise requires 'mode' parameter"
            raise ValueError(msg)

        mode = params["mode"]
        if mode not in ("gaussian", "row", "column"):
            msg = f"Mode must be 'gaussian', 'row', or 'column', got {mode}"
            raise ValueError(msg)

        if "amount" not in params:
            msg = "Noise requires 'amount' parameter"
            raise ValueError(msg)

        amount = params["amount"]
        if not isinstance(amount, (int, float)):
            msg = f"Amount must be a number, got {type(amount)}"
            raise ValueError(msg)

        if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
            msg = f"Amount must be in [{MIN_AMOUNT}, {MAX_AMOUNT}], got {amount}"
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer"
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
        Apply noise on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for noise (empty dict expected)
            params: Must contain 'mode', 'amount', and optional 'seed'
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available
            ValueError: If mode is invalid

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        mode: Literal["gaussian", "row", "column"] = params["mode"]
        amount = float(params["amount"])
        seed = int(params.get("seed", 0))

        # Execute appropriate kernel (batch_idx=0 for single image)
        if mode == "gaussian":
            _noise_gaussian_kernel(source, dest, amount, seed, 0, height, width)
        elif mode == "row":
            _noise_row_kernel(source, dest, amount, seed, 0, height, width)
        elif mode == "column":
            _noise_column_kernel(source, dest, amount, seed, 0, height, width)
        else:
            msg = f"Unknown noise mode: {mode}"
            raise ValueError(msg)

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
            params: Must contain 'mode', 'amount', and optional 'seed'

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.random import (
            rand_float_numpy,
            rand_gaussian_numpy,
        )

        mode: Literal["gaussian", "row", "column"] = params["mode"]
        amount = float(params["amount"])
        seed = int(params.get("seed", 0))

        height, width = image.shape[:2]
        result = np.copy(image)

        if mode == "gaussian":
            # Per-pixel Gaussian noise
            for i in range(height):
                for j in range(width):
                    # Independent noise for each channel
                    noise_r = rand_gaussian_numpy(i, j, seed, amount)
                    noise_g = rand_gaussian_numpy(i, j, seed + 1, amount)
                    noise_b = rand_gaussian_numpy(i, j, seed + 2, amount)

                    result[i, j, 0] = np.clip(image[i, j, 0] + noise_r, 0.0, 1.0)
                    result[i, j, 1] = np.clip(image[i, j, 1] + noise_g, 0.0, 1.0)
                    result[i, j, 2] = np.clip(image[i, j, 2] + noise_b, 0.0, 1.0)

        elif mode == "row":
            # Row-based noise (same noise for all pixels in a row)
            for i in range(height):
                # Generate row-specific noise (using j=0)
                u_r = rand_float_numpy(i, 0, seed)
                u_g = rand_float_numpy(i, 0, seed + 1)
                u_b = rand_float_numpy(i, 0, seed + 2)

                # Map [0,1] to [-amount, amount]
                noise_r = (u_r * 2.0 - 1.0) * amount
                noise_g = (u_g * 2.0 - 1.0) * amount
                noise_b = (u_b * 2.0 - 1.0) * amount

                # Apply to all pixels in row
                for j in range(width):
                    result[i, j, 0] = np.clip(image[i, j, 0] + noise_r, 0.0, 1.0)
                    result[i, j, 1] = np.clip(image[i, j, 1] + noise_g, 0.0, 1.0)
                    result[i, j, 2] = np.clip(image[i, j, 2] + noise_b, 0.0, 1.0)

        elif mode == "column":
            # Column-based noise (same noise for all pixels in a column)
            for j in range(width):
                # Generate column-specific noise (using i=0)
                u_r = rand_float_numpy(0, j, seed)
                u_g = rand_float_numpy(0, j, seed + 1)
                u_b = rand_float_numpy(0, j, seed + 2)

                # Map [0,1] to [-amount, amount]
                noise_r = (u_r * 2.0 - 1.0) * amount
                noise_g = (u_g * 2.0 - 1.0) * amount
                noise_b = (u_b * 2.0 - 1.0) * amount

                # Apply to all pixels in column
                for i in range(height):
                    result[i, j, 0] = np.clip(image[i, j, 0] + noise_r, 0.0, 1.0)
                    result[i, j, 1] = np.clip(image[i, j, 1] + noise_g, 0.0, 1.0)
                    result[i, j, 2] = np.clip(image[i, j, 2] + noise_b, 0.0, 1.0)

        return result.astype(np.float32)

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the noise kernels
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

        # Trigger compilation for all three kernels
        _noise_gaussian_kernel(dummy_src, dummy_dst, 0.1, 42, 0, 2, 2)
        _noise_row_kernel(dummy_src, dummy_dst, 0.1, 42, 0, 2, 2)
        _noise_column_kernel(dummy_src, dummy_dst, 0.1, 42, 0, 2, 2)

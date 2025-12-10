"""
Taichi end-to-end pipeline chromatic aberration operation.

Chromatic aberration simulation for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).
"""

from typing import Any

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Constants
RGB_CHANNELS = 3

# Taichi imports with fallback for testing
try:
    import taichi as ti

    from sevenrad_stills.operations.taichi_kernels.sampling import bilinear_sample

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    bilinear_sample = None
    TAICHI_AVAILABLE = False


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _chromatic_aberration_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        shift_x: ti.f32,
        shift_y: ti.f32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for chromatic aberration simulation.

        Operates on ti.Vector.field(4) with RGBA channels.
        Shifts R and B channels in opposite directions, preserves G and A.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            shift_x: Horizontal pixel shift for red channel (blue is -shift_x)
            shift_y: Vertical pixel shift for red channel (blue is -shift_y)
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            # Sample R channel at shifted position (shift forward)
            r_pos_y = ti.cast(i, ti.f32) + shift_y
            r_pos_x = ti.cast(j, ti.f32) + shift_x
            r_sample = bilinear_sample(source, batch, r_pos_y, r_pos_x, height, width)
            r = r_sample[0]

            # G channel: no shift (reference)
            g = source[batch, i, j][1]

            # Sample B channel at shifted position (shift backward)
            b_pos_y = ti.cast(i, ti.f32) - shift_y
            b_pos_x = ti.cast(j, ti.f32) - shift_x
            b_sample = bilinear_sample(source, batch, b_pos_y, b_pos_x, height, width)
            b = b_sample[2]

            # Preserve alpha channel
            a = source[batch, i, j][3]

            # Write result
            dest[batch, i, j] = ti.Vector([r, g, b, a])


class ChromaticAberrationTaichiOperation(BaseTaichiOperation):
    """
    Taichi chromatic aberration for end-to-end GPU pipeline.

    Simulates chromatic aberration by shifting red and blue channels
    in opposite directions while keeping green as reference.
    Uses bilinear interpolation for sub-pixel accuracy.

    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation requires sampling from different locations and
    does not support in-place execution.

    Example:
        >>> op = ChromaticAberrationTaichiOperation()
        >>> op.apply_to_field(source, dest, {}, {"shift_x": 2, "shift_y": 1}, h, w)

    """

    def __init__(self) -> None:
        """Initialize chromatic aberration operation."""
        super().__init__("chromatic_aberration_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Chromatic aberration reads from shifted positions, so it cannot
        safely write to the same buffer it's reading from.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate chromatic aberration parameters.

        Expected params:
        - shift_x: int/float - horizontal pixel shift for red channel
        - shift_y: int/float - vertical pixel shift for red channel

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If required parameters are missing or invalid

        """
        for key in ("shift_x", "shift_y"):
            if key not in params:
                msg = f"Chromatic aberration requires '{key}' parameter"
                raise ValueError(msg)

            value = params[key]
            if not isinstance(value, (int, float)):
                msg = f"{key} must be a number, got {type(value)}"
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
        Apply chromatic aberration on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for chromatic aberration (empty dict expected)
            params: Must contain 'shift_x' and 'shift_y' keys
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        shift_x = float(params["shift_x"])
        shift_y = float(params["shift_y"])

        # Execute kernel (batch_idx=0 for single image)
        _chromatic_aberration_kernel(source, dest, shift_x, shift_y, 0, height, width)

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        NumPy reference implementation for testing.

        Produces identical results to apply_to_field for correctness testing.
        Uses bilinear interpolation for sub-pixel accuracy.

        Args:
            image: Input image as numpy array (H, W, 3) float32 in [0, 1]
            params: Must contain 'shift_x' and 'shift_y' keys

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            bilinear_sample_numpy,
        )

        shift_x = float(params["shift_x"])
        shift_y = float(params["shift_y"])

        h, w = image.shape[:2]
        result = np.zeros_like(image)

        # Add alpha channel if not present for consistent processing
        if image.shape[2] == RGB_CHANNELS:
            image_rgba = np.concatenate(
                [image, np.ones((h, w, 1), dtype=np.float32)], axis=2
            )
        else:
            image_rgba = image

        for i in range(h):
            for j in range(w):
                # Red channel: shift forward
                r_y = i + shift_y
                r_x = j + shift_x
                r_sample = bilinear_sample_numpy(image_rgba, r_y, r_x)
                result[i, j, 0] = r_sample[0]

                # Green channel: no shift
                result[i, j, 1] = image_rgba[i, j, 1]

                # Blue channel: shift backward
                b_y = i - shift_y
                b_x = j - shift_x
                b_sample = bilinear_sample_numpy(image_rgba, b_y, b_x)
                result[i, j, 2] = b_sample[2]

        # Ensure output is in valid range and correct dtype
        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the chromatic aberration kernel
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
        _chromatic_aberration_kernel(dummy_src, dummy_dst, 1.0, 1.0, 0, 2, 2)

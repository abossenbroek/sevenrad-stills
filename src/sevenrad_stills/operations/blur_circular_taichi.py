"""
Taichi end-to-end pipeline circular blur operation.

Circular blur for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
Creates bokeh-like blur effects using a circular disk kernel.
"""

from typing import Any

import numpy as np
from scipy.ndimage import convolve

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
from sevenrad_stills.operations.taichi_kernels.convolution import (
    circular_kernel,
    convolve_2d_numpy,
)

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:
    # Import convolution utilities
    from sevenrad_stills.operations.taichi_kernels.convolution import convolve_2d

    # We use the existing convolve_2d kernel from convolution.py
    # which handles 2D convolution with reflection boundaries


class BlurCircularTaichiOperation(BaseTaichiOperation):
    """
    Taichi circular blur for end-to-end GPU pipeline.

    Applies circular disk blur using 2D convolution. Creates bokeh-like effects
    by convolving with a circular kernel. Operates on ti.Vector.field(4) buffers
    without CPU↔GPU transfer.

    This operation requires neighborhood data and does not support in-place execution.

    Example:
        >>> op = BlurCircularTaichiOperation()
        >>> op.apply_to_field(source, dest, {}, {"radius": 5}, height, width)

    """

    def __init__(self) -> None:
        """Initialize circular blur operation."""
        super().__init__("blur_circular_taichi")
        self._kernel_field: Any = None  # ti.field for GPU
        self._last_radius: int | None = None

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Circular blur is a convolution: output[i,j] depends on neighborhood,
        so in-place execution would corrupt input data.

        Returns:
            False - this operation requires separate source and dest buffers.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate circular blur parameters.

        Expected params:
        - radius: int - blur disk radius (>= 0)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If radius is missing or invalid

        """
        if "radius" not in params:
            msg = "Circular blur requires 'radius' parameter"
            raise ValueError(msg)

        radius = params["radius"]
        if not isinstance(radius, int):
            msg = f"Radius must be an integer, got {type(radius)}"
            raise ValueError(msg)

        if radius < 0:
            msg = f"Radius must be non-negative, got {radius}"
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
        Apply circular blur on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for circular blur (empty dict expected)
            params: Must contain 'radius' key with integer blur radius
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        radius = int(params["radius"])

        # If radius is 0, just copy source to dest
        if radius == 0:
            dest.copy_from(source)
            return

        # Create or reuse kernel field
        if self._kernel_field is None or self._last_radius != radius:
            self._create_kernel_field(radius)
            self._last_radius = radius

        # Execute convolution kernel
        kernel_size = 2 * radius + 1
        convolve_2d(
            source,
            dest,
            self._kernel_field,
            radius,  # radius_h
            radius,  # radius_w
            0,  # batch
            height,
            width,
        )

    def _create_kernel_field(self, radius: int) -> None:
        """
        Create Taichi field for circular kernel.

        Args:
            radius: Circle radius

        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Generate circular kernel using utility
        kernel_np, _ = circular_kernel(radius)

        # Create Taichi field and populate
        kernel_size = 2 * radius + 1
        self._kernel_field = ti.field(dtype=ti.f32, shape=(kernel_size, kernel_size))

        # Copy kernel to GPU
        self._kernel_field.from_numpy(kernel_np.astype(np.float32))

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
            params: Must contain 'radius' key

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        radius = int(params["radius"])

        # If radius is 0, return copy of input
        if radius == 0:
            return image.copy()

        # Generate circular kernel
        kernel, _ = circular_kernel(radius)

        # Apply convolution using utility
        result = convolve_2d_numpy(image, kernel)

        # Ensure output is in valid range and correct dtype
        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the circular blur kernel
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

        # Create minimal kernel (radius=1)
        self._create_kernel_field(1)

        # Trigger compilation
        convolve_2d(dummy_src, dummy_dst, self._kernel_field, 1, 1, 0, 2, 2)

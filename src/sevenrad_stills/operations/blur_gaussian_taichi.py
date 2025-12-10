"""
Taichi end-to-end pipeline Gaussian blur operation.

Gaussian blur for GPU pipeline execution using separable convolution.
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

from typing import Any

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
from sevenrad_stills.pipeline.protocols import TempFieldSpec

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Import convolution kernels
if TAICHI_AVAILABLE and ti is not None:
    from sevenrad_stills.operations.taichi_kernels.convolution import (
        convolve_horizontal,
        convolve_vertical,
    )

# Import kernel generation from convolution module
from sevenrad_stills.operations.taichi_kernels.convolution import gaussian_kernel_1d

# Constants
DEFAULT_RADIUS_MULTIPLIER = 3
MAX_SIGMA_FOR_SKIP = 0.01


class BlurGaussianTaichiOperation(BaseTaichiOperation):
    """
    Taichi Gaussian blur for end-to-end GPU pipeline.

    Applies Gaussian blur using separable 2-pass convolution.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation requires a temporary field for the intermediate pass
    and does not support in-place execution.

    Example:
        >>> op = BlurGaussianTaichiOperation()
        >>> op.apply_to_field(source, dest, temp_fields, {"sigma": 2.0}, h, w)

    """

    def __init__(self) -> None:
        """Initialize Gaussian blur operation."""
        super().__init__("blur_gaussian_taichi")
        self._kernel_field: Any = None  # ti.field for kernel weights
        self._kernel_radius: int = 0

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Gaussian blur needs neighborhood data: output[i,j] depends on
        input[i-r:i+r, j-r:j+r]. Cannot safely modify source.

        Returns:
            False - this operation requires separate source and dest buffers.

        """
        return False

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """
        List of temporary fields required.

        Separable blur needs one intermediate buffer for horizontal pass result.

        Returns:
            List with one TempFieldSpec for the intermediate buffer.

        """
        return [TempFieldSpec("intermediate", (1.0, 1.0, 4))]

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate Gaussian blur parameters.

        Expected params:
        - sigma: float - standard deviation for Gaussian kernel (>= 0)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If sigma is missing or invalid

        """
        if "sigma" not in params:
            msg = "Gaussian blur requires 'sigma' parameter"
            raise ValueError(msg)

        sigma = params["sigma"]
        if not isinstance(sigma, (int, float)):
            msg = f"Sigma must be a number, got {type(sigma)}"
            raise ValueError(msg)

        if sigma < 0:
            msg = f"Sigma must be >= 0, got {sigma}"
            raise ValueError(msg)

    def apply_to_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,  # ti.Vector.field
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply Gaussian blur on GPU fields using separable convolution.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Must contain 'intermediate' key for temp buffer
            params: Must contain 'sigma' key with blur radius
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available
            ValueError: If intermediate temp field is missing

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        if "intermediate" not in temp_fields:
            msg = "Missing 'intermediate' temporary field for Gaussian blur"
            raise ValueError(msg)

        sigma = float(params["sigma"])

        # Skip blur if sigma is too small (no visible effect)
        if sigma < MAX_SIGMA_FOR_SKIP:
            # Copy source to dest
            self._copy_field(source, dest, height, width)
            return

        # Calculate kernel radius and generate weights
        radius = int(np.ceil(DEFAULT_RADIUS_MULTIPLIER * sigma))
        kernel_weights = gaussian_kernel_1d(sigma, radius)

        # Create or update kernel field
        kernel_size = 2 * radius + 1
        if self._kernel_field is None or self._kernel_radius != radius:
            self._kernel_field = ti.field(dtype=ti.f32, shape=(kernel_size,))
            self._kernel_radius = radius

        # Copy kernel weights to GPU
        self._kernel_field.from_numpy(kernel_weights.astype(np.float32))

        # Get intermediate buffer
        temp_field = temp_fields["intermediate"]

        # Two-pass separable convolution:
        # 1. Horizontal: source → temp_field
        convolve_horizontal(
            source, temp_field, self._kernel_field, radius, 0, height, width
        )

        # 2. Vertical: temp_field → dest
        convolve_vertical(
            temp_field, dest, self._kernel_field, radius, 0, height, width
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
            params: Must contain 'sigma' key

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_horizontal_numpy,
            convolve_vertical_numpy,
        )

        sigma = float(params["sigma"])

        # Skip blur if sigma is too small
        if sigma < MAX_SIGMA_FOR_SKIP:
            return image.copy()

        # Calculate kernel
        radius = int(np.ceil(DEFAULT_RADIUS_MULTIPLIER * sigma))
        kernel_weights = gaussian_kernel_1d(sigma, radius)

        # Two-pass separable convolution
        temp = convolve_horizontal_numpy(image, kernel_weights)
        result = convolve_vertical_numpy(temp, kernel_weights)

        # Clip to valid range
        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the convolution kernels
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_temp = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Create minimal kernel (radius=1, size=3)
        dummy_kernel = ti.field(dtype=ti.f32, shape=(3,))
        kernel_weights = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        dummy_kernel.from_numpy(kernel_weights)

        # Trigger compilation of both kernels
        convolve_horizontal(dummy_src, dummy_temp, dummy_kernel, 1, 0, 2, 2)
        convolve_vertical(dummy_temp, dummy_dst, dummy_kernel, 1, 0, 2, 2)

    def _copy_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,  # ti.Vector.field
        height: int,
        width: int,
    ) -> None:
        """
        Copy source field to dest field.

        Used when sigma is too small and no blur is needed.

        Args:
            source: Source Taichi field
            dest: Destination Taichi field
            height: Image height
            width: Image width

        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        @ti.kernel  # type: ignore[misc]
        def _copy_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
            src: ti.template(),  # type: ignore[valid-type]
            dst: ti.template(),  # type: ignore[valid-type]
            h: ti.i32,
            w: ti.i32,
        ):  # Taichi kernels don't use Python return type hints
            for i, j in ti.ndrange(h, w):
                dst[0, i, j] = src[0, i, j]

        _copy_kernel(source, dest, height, width)

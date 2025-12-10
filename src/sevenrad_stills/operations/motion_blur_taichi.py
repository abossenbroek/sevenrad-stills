"""
Taichi end-to-end pipeline motion blur operation.

Directional motion blur for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU-GPU data transfer.
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
MIN_KERNEL_SIZE = 1
MAX_KERNEL_SIZE = 100
MIN_ANGLE = 0.0
MAX_ANGLE = 360.0


# Define the kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:
    from sevenrad_stills.operations.taichi_kernels.convolution import convolve_2d


class MotionBlurTaichiOperation(BaseTaichiOperation):
    """
    Taichi motion blur for end-to-end GPU pipeline.

    Applies directional motion blur to simulate camera movement or shake.
    Operates on ti.Vector.field(4) buffers without CPU-GPU transfer.

    This operation requires neighborhood data and does not support in-place execution.

    Example:
        >>> op = MotionBlurTaichiOperation()
        >>> params = {"kernel_size": 15, "angle": 45.0}
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize motion blur operation."""
        super().__init__("motion_blur_taichi")
        self._kernel_field: Any | None = None  # ti.field(2, dtype=ti.f32)
        self._cached_params: tuple[int, float] | None = None

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Motion blur requires neighborhood data, so cannot be done in-place.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate motion blur parameters.

        Expected params:
        - kernel_size: int (1-100) - Length of motion blur
        - angle: float (0-360) - Direction in degrees (optional, default 0.0)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "kernel_size" not in params:
            msg = "Motion blur requires 'kernel_size' parameter"
            raise ValueError(msg)

        kernel_size = params["kernel_size"]
        if not isinstance(kernel_size, int):
            msg = f"Kernel size must be an integer, got {type(kernel_size)}"
            raise ValueError(msg)

        if not MIN_KERNEL_SIZE <= kernel_size <= MAX_KERNEL_SIZE:
            msg = (
                f"Kernel size must be between {MIN_KERNEL_SIZE} and "
                f"{MAX_KERNEL_SIZE}, got {kernel_size}"
            )
            raise ValueError(msg)

        # Validate angle if provided
        angle = params.get("angle", 0.0)
        if not isinstance(angle, (int, float)):
            msg = f"Angle must be a number, got {type(angle)}"
            raise ValueError(msg)
        if not MIN_ANGLE <= angle < MAX_ANGLE:
            msg = f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE}, got {angle}"
            raise ValueError(msg)

    def _create_motion_kernel_numpy(self, size: int, angle: float) -> np.ndarray:
        """
        Create a motion blur kernel using Bresenham-style line generation.

        Creates a sparse 2D kernel with weights along a line at the given angle.
        This is more accurate than rotation-based approach and avoids artifacts.

        Args:
            size: Length of motion blur (kernel will be size x size)
            angle: Direction of motion in degrees (0=horizontal right)

        Returns:
            Normalized motion blur kernel as numpy array (size, size)

        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)

        # Calculate line endpoints from center
        center = (size - 1) / 2.0
        dx = np.cos(angle_rad) * center
        dy = -np.sin(angle_rad) * center  # Negative because y-axis points down

        # Start and end points
        x0 = center - dx
        y0 = center - dy
        x1 = center + dx
        y1 = center + dy

        # Create kernel
        kernel = np.zeros((size, size), dtype=np.float32)

        # Bresenham-style line drawing to mark pixels
        points = self._bresenham_line(
            int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        )

        # Set weights for all points on the line
        for x, y in points:
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        # Normalize
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            # Fallback: single pixel at center
            kernel[size // 2, size // 2] = 1.0

        return kernel

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> list[tuple[int, int]]:
        """
        Generate points along a line using Bresenham's algorithm.

        Args:
            x0: Start x coordinate
            y0: Start y coordinate
            x1: End x coordinate
            y1: End y coordinate

        Returns:
            List of (x, y) points along the line

        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def _allocate_kernel_field(self, size: int) -> Any:
        """
        Allocate or reuse Taichi field for kernel.

        Args:
            size: Kernel size

        Returns:
            Taichi field for kernel (size, size)

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available"
            raise RuntimeError(msg)

        # Reuse field if size matches
        if self._kernel_field is not None:
            existing_shape = self._kernel_field.shape
            if existing_shape == (size, size):
                return self._kernel_field

        # Allocate new field
        self._kernel_field = ti.field(dtype=ti.f32, shape=(size, size))
        return self._kernel_field

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
        Apply motion blur on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for motion blur (empty dict expected)
            params: Must contain 'kernel_size' and optionally 'angle'
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        kernel_size = int(params["kernel_size"])
        angle = float(params.get("angle", 0.0))

        # Special case: kernel_size=1 is a no-op, just copy
        if kernel_size == 1:
            dest.copy_from(source)
            return

        # Create or reuse kernel
        current_params = (kernel_size, angle)
        if self._cached_params != current_params:
            # Generate kernel on CPU
            kernel_np = self._create_motion_kernel_numpy(kernel_size, angle)

            # Transfer to GPU
            kernel_field = self._allocate_kernel_field(kernel_size)
            kernel_field.from_numpy(kernel_np)

            # Cache params
            self._cached_params = current_params

        # Apply 2D convolution
        radius_h = kernel_size // 2
        radius_w = kernel_size // 2

        convolve_2d(
            source,
            dest,
            self._kernel_field,
            radius_h,
            radius_w,
            0,  # batch
            height,
            width,
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
            params: Must contain 'kernel_size' and optionally 'angle'

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        from scipy.ndimage import convolve

        kernel_size = int(params["kernel_size"])
        angle = float(params.get("angle", 0.0))

        # Special case: kernel_size=1 is no-op
        if kernel_size == 1:
            return image.copy()

        # Create motion blur kernel
        kernel = self._create_motion_kernel_numpy(kernel_size, angle)

        # Apply convolution to each channel
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve(
                image[:, :, c],
                kernel,
                mode="reflect",
            )

        # Ensure output stays in [0, 1]
        clipped: np.ndarray = np.clip(result, 0.0, 1.0).astype(np.float32)
        return clipped

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the motion blur kernels
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 4, 4))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 4, 4))

        # Initialize with dummy data
        for i in range(4):
            for j in range(4):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Create a small kernel for warmup
        dummy_kernel = ti.field(dtype=ti.f32, shape=(3, 3))
        dummy_kernel_np = np.ones((3, 3), dtype=np.float32) / 9.0
        dummy_kernel.from_numpy(dummy_kernel_np)

        # Trigger compilation
        convolve_2d(dummy_src, dummy_dst, dummy_kernel, 1, 1, 0, 4, 4)

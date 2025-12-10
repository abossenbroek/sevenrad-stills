"""
Convolution utilities for Taichi kernels.

This module provides kernels for:
- Separable 1D convolution (horizontal and vertical passes)
- General 2D convolution

Separable convolution is more efficient for filters that can be decomposed
into horizontal and vertical components (like Gaussian blur): O(n) per pixel
instead of O(n^2) for general 2D convolution.

Example:
    >>> @ti.kernel
    ... def gaussian_blur(src, dst, temp, kernel, radius, h, w):
    ...     convolve_horizontal(src, temp, kernel, radius, 0, h, w)
    ...     convolve_vertical(temp, dst, kernel, radius, 0, h, w)

Note:
    This module intentionally does NOT use `from __future__ import annotations`
    because Taichi requires actual type objects at decoration time, not string
    annotations (PEP 563 deferred evaluation breaks Taichi's type system).

"""

import numpy as np
from scipy import ndimage

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False


# Maximum kernel radius (for static array allocation)
MAX_KERNEL_RADIUS = 64

# Constants for magic values
MASK_THRESHOLD = 0.5
EPSILON = 1e-6


if TAICHI_AVAILABLE and ti is not None:

    @ti.func  # type: ignore[misc]
    def reflect_boundary(  # type: ignore[no-untyped-def]
        coord: ti.i32,
        size: ti.i32,
    ) -> ti.i32:
        """
        Reflect coordinate at boundary for edge handling.

        Args:
            coord: Coordinate value
            size: Image dimension

        Returns:
            Reflected coordinate in valid range

        """
        result = coord
        if coord < 0:
            result = -coord - 1
            result = ti.max(result, 0)
            if result >= size:
                result = size - 1
        elif coord >= size:
            result = 2 * size - coord - 1
            result = ti.max(result, 0)
        return result

    @ti.kernel  # type: ignore[misc]
    def convolve_horizontal(  # type: ignore[no-untyped-def]  # noqa: ANN201, PLR0913
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        kernel: ti.template(),  # type: ignore[valid-type]
        radius: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        Apply horizontal 1D convolution (first pass of separable filter).

        Convolves each row with the 1D kernel. Uses reflection at boundaries
        to avoid edge artifacts.

        Args:
            source: Input Vector.field(4) with shape (batch, H, W)
            dest: Output Vector.field(4) with same shape
            kernel: 1D kernel weights field with shape (2*radius+1,)
            radius: Kernel radius (kernel size = 2*radius + 1)
            batch: Batch index
            height: Image height
            width: Image width

        Note:
            Kernel is assumed to be pre-normalized (weights sum to 1.0).
            Alpha channel is convolved along with RGB.

        """
        for i, j in ti.ndrange(height, width):
            accum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

            for k in range(-radius, radius + 1):
                jj = reflect_boundary(j + k, width)
                weight = kernel[k + radius]
                accum += source[batch, i, jj] * weight

            dest[batch, i, j] = accum

    @ti.kernel  # type: ignore[misc]
    def convolve_vertical(  # type: ignore[no-untyped-def]  # noqa: ANN201, PLR0913
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        kernel: ti.template(),  # type: ignore[valid-type]
        radius: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        Apply vertical 1D convolution (second pass of separable filter).

        Convolves each column with the 1D kernel.

        Args:
            source: Input Vector.field(4) with shape (batch, H, W)
            dest: Output Vector.field(4) with same shape
            kernel: 1D kernel weights field with shape (2*radius+1,)
            radius: Kernel radius
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            accum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

            for k in range(-radius, radius + 1):
                ii = reflect_boundary(i + k, height)
                weight = kernel[k + radius]
                accum += source[batch, ii, j] * weight

            dest[batch, i, j] = accum

    @ti.kernel  # type: ignore[misc]
    def convolve_2d(  # type: ignore[no-untyped-def]  # noqa: ANN201, PLR0913
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        kernel: ti.template(),  # type: ignore[valid-type]
        radius_h: ti.i32,
        radius_w: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        Apply general 2D convolution.

        For non-separable kernels (e.g., circular blur, some edge detectors).
        More expensive than separable convolution but handles any kernel shape.

        Args:
            source: Input Vector.field(4) with shape (batch, H, W)
            dest: Output Vector.field(4) with same shape
            kernel: 2D kernel weights field with shape (2*radius_h+1, 2*radius_w+1)
            radius_h: Kernel vertical radius
            radius_w: Kernel horizontal radius
            batch: Batch index
            height: Image height
            width: Image width

        Note:
            Kernel should be pre-normalized. Uses reflection at boundaries.

        """
        for i, j in ti.ndrange(height, width):
            accum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

            for ki in range(-radius_h, radius_h + 1):
                for kj in range(-radius_w, radius_w + 1):
                    ii = reflect_boundary(i + ki, height)
                    jj = reflect_boundary(j + kj, width)
                    weight = kernel[ki + radius_h, kj + radius_w]
                    accum += source[batch, ii, jj] * weight

            dest[batch, i, j] = accum

    @ti.kernel  # type: ignore[misc]
    def convolve_2d_masked(  # type: ignore[no-untyped-def]  # noqa: ANN201, PLR0913
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        kernel: ti.template(),  # type: ignore[valid-type]
        mask: ti.template(),  # type: ignore[valid-type]
        radius_h: ti.i32,
        radius_w: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        Apply 2D convolution with kernel mask (for shaped kernels like circular).

        Only applies kernel weights where mask is non-zero. Normalizes
        based on actual weights used.

        Args:
            source: Input Vector.field(4)
            dest: Output Vector.field(4)
            kernel: 2D kernel weights
            mask: Binary mask indicating which kernel positions to use
            radius_h: Kernel vertical radius
            radius_w: Kernel horizontal radius
            batch: Batch index
            height: Image height
            width: Image width

        """
        for i, j in ti.ndrange(height, width):
            accum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            weight_sum = 0.0

            for ki in range(-radius_h, radius_h + 1):
                for kj in range(-radius_w, radius_w + 1):
                    m = mask[ki + radius_h, kj + radius_w]
                    if m > MASK_THRESHOLD:
                        ii = reflect_boundary(i + ki, height)
                        jj = reflect_boundary(j + kj, width)
                        weight = kernel[ki + radius_h, kj + radius_w]
                        accum += source[batch, ii, jj] * weight
                        weight_sum += weight

            # Normalize by actual weights used
            if weight_sum > EPSILON:
                dest[batch, i, j] = accum / weight_sum
            else:
                dest[batch, i, j] = source[batch, i, j]


# NumPy reference implementations for testing


def convolve_horizontal_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    NumPy reference implementation of horizontal 1D convolution.

    Args:
        image: Input image (H, W, C)
        kernel: 1D kernel weights

    Returns:
        Convolved image

    """
    result: np.ndarray = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = ndimage.convolve1d(
            image[:, :, c], kernel, axis=1, mode="reflect"
        )
    return result


def convolve_vertical_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    NumPy reference implementation of vertical 1D convolution.

    Args:
        image: Input image (H, W, C)
        kernel: 1D kernel weights

    Returns:
        Convolved image

    """
    result: np.ndarray = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = ndimage.convolve1d(
            image[:, :, c], kernel, axis=0, mode="reflect"
        )
    return result


def convolve_2d_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    NumPy reference implementation of 2D convolution.

    Args:
        image: Input image (H, W, C)
        kernel: 2D kernel weights

    Returns:
        Convolved image

    """
    result: np.ndarray = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = ndimage.convolve(image[:, :, c], kernel, mode="reflect")
    return result


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Create 1D Gaussian kernel.

    Args:
        sigma: Standard deviation
        radius: Kernel radius (default: ceil(3*sigma))

    Returns:
        Normalized 1D Gaussian kernel

    """
    if radius is None:
        radius = int(np.ceil(3 * sigma))

    x = np.arange(-radius, radius + 1)
    kernel: np.ndarray = np.exp(-(x**2) / (2 * sigma**2))
    normalized: np.ndarray = kernel / kernel.sum()
    return normalized


def circular_kernel(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create circular disk kernel.

    Args:
        radius: Circle radius

    Returns:
        Tuple of (kernel weights, mask)

    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x**2 + y**2 <= radius**2).astype(np.float32)
    kernel = mask / mask.sum()
    return kernel, mask

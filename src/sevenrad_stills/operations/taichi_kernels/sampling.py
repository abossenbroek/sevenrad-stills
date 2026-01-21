"""
Sampling and interpolation utilities for Taichi kernels.

This module provides functions for:
- Bilinear interpolation sampling
- Nearest neighbor sampling
- Coordinate clamping and reflection at boundaries

These are essential for operations like downscaling, chromatic aberration,
and any operation that samples from non-integer pixel locations.

Example:
    >>> @ti.kernel
    ... def shift_channel(source: ti.template(), dest: ti.template(), shift_x: ti.f32):
    ...     for i, j in ti.ndrange(height, width):
    ...         sampled = bilinear_sample(source, 0, i, j + shift_x, height, width)
    ...         dest[0, i, j] = sampled

Note:
    This module intentionally does NOT use `from __future__ import annotations`
    because Taichi requires actual type objects at decoration time, not string
    annotations (PEP 563 deferred evaluation breaks Taichi's type system).

"""

import numpy as np

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False


if TAICHI_AVAILABLE and ti is not None:

    @ti.func  # type: ignore[misc]
    def clamp_coords(  # type: ignore[no-untyped-def]
        i: ti.i32,
        j: ti.i32,
        h: ti.i32,
        w: ti.i32,
    ) -> ti.math.ivec2:
        """
        Clamp coordinates to valid image bounds.

        Args:
            i: Row coordinate (may be out of bounds)
            j: Column coordinate (may be out of bounds)
            h: Image height
            w: Image width

        Returns:
            Vector of (clamped_i, clamped_j)

        """
        ci = ti.max(0, ti.min(i, h - 1))
        cj = ti.max(0, ti.min(j, w - 1))
        return ti.math.ivec2(ci, cj)

    @ti.func  # type: ignore[misc]
    def clamp_i(  # type: ignore[no-untyped-def]
        coord: ti.i32,
        size: ti.i32,
    ) -> ti.i32:
        """
        Clamp a single coordinate to valid range [0, size-1].

        Args:
            coord: Coordinate value
            size: Maximum size (exclusive upper bound)

        Returns:
            Clamped coordinate

        """
        return ti.max(0, ti.min(coord, size - 1))

    @ti.func  # type: ignore[misc]
    def reflect_boundary(  # type: ignore[no-untyped-def]
        coord: ti.i32,
        size: ti.i32,
    ) -> ti.i32:
        """
        Reflect coordinate at boundary for mirror-like edge handling.

        For coordinates beyond the edge, reflects them back into the
        valid range. Useful for convolution operations to avoid
        edge artifacts.

        Args:
            coord: Coordinate value (may be negative or >= size)
            size: Image dimension (height or width)

        Returns:
            Reflected coordinate in range [0, size-1]

        Example:
            size=5: -1 -> 0, -2 -> 1, 5 -> 4, 6 -> 3

        """
        result = coord

        # Handle negative coordinates
        if coord < 0:
            result = -coord - 1
            result = ti.min(result, size - 1)

        # Handle coordinates beyond size
        elif coord >= size:
            result = 2 * size - coord - 1
            result = ti.max(result, 0)

        return result

    @ti.func  # type: ignore[misc]
    def nearest_sample(  # type: ignore[no-untyped-def]  # noqa: PLR0913
        field: ti.template(),  # type: ignore[valid-type]
        batch: ti.i32,
        y: ti.f32,
        x: ti.f32,
        height: ti.i32,
        width: ti.i32,
    ) -> ti.math.vec4:
        """
        Nearest neighbor sampling from a Taichi field.

        Samples the pixel closest to the given floating-point coordinates.
        Coordinates outside bounds are clamped to the edge.

        Args:
            field: Source Taichi Vector.field(4) with shape (batch, H, W)
            batch: Batch index
            y: Y-coordinate (row, can be fractional)
            x: X-coordinate (column, can be fractional)
            height: Image height
            width: Image width

        Returns:
            Sampled RGBA pixel as 4-component vector

        """
        # Round to nearest integer
        yi = ti.cast(ti.round(y), ti.i32)
        xi = ti.cast(ti.round(x), ti.i32)

        # Clamp to valid bounds
        yi = clamp_i(yi, height)
        xi = clamp_i(xi, width)

        return field[batch, yi, xi]

    @ti.func  # type: ignore[misc]
    def bilinear_sample(  # type: ignore[no-untyped-def]  # noqa: PLR0913
        field: ti.template(),  # type: ignore[valid-type]
        batch: ti.i32,
        y: ti.f32,
        x: ti.f32,
        height: ti.i32,
        width: ti.i32,
    ) -> ti.math.vec4:
        """
        Bilinear interpolation sampling from a Taichi field.

        Samples at a fractional pixel location by interpolating between
        the four nearest pixels. Produces smooth results for sub-pixel
        sampling operations.

        Args:
            field: Source Taichi Vector.field(4) with shape (batch, H, W)
            batch: Batch index
            y: Y-coordinate (row, can be fractional)
            x: X-coordinate (column, can be fractional)
            height: Image height
            width: Image width

        Returns:
            Interpolated RGBA pixel as 4-component vector

        Note:
            Coordinates outside bounds are clamped to the edge.
            Uses floor for integer coordinates, so y=0.5, x=0.5 is
            between pixels (0,0), (0,1), (1,0), (1,1).

        """
        # Get integer coordinates (floor)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x0 = ti.cast(ti.floor(x), ti.i32)
        y1 = y0 + 1
        x1 = x0 + 1

        # Get fractional parts
        fy = y - ti.cast(y0, ti.f32)
        fx = x - ti.cast(x0, ti.f32)

        # Clamp to valid bounds
        y0 = clamp_i(y0, height)
        y1 = clamp_i(y1, height)
        x0 = clamp_i(x0, width)
        x1 = clamp_i(x1, width)

        # Sample four corners
        p00 = field[batch, y0, x0]
        p01 = field[batch, y0, x1]
        p10 = field[batch, y1, x0]
        p11 = field[batch, y1, x1]

        # Bilinear interpolation weights
        w00 = (1.0 - fy) * (1.0 - fx)
        w01 = (1.0 - fy) * fx
        w10 = fy * (1.0 - fx)
        w11 = fy * fx

        # Interpolate
        result = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11

        return result

    @ti.func  # type: ignore[misc]
    def bilinear_sample_channel(  # type: ignore[no-untyped-def]  # noqa: PLR0913
        field: ti.template(),  # type: ignore[valid-type]
        batch: ti.i32,
        y: ti.f32,
        x: ti.f32,
        channel: ti.i32,
        height: ti.i32,
        width: ti.i32,
    ) -> ti.f32:
        """
        Bilinear interpolation for a single channel.

        Args:
            field: Source Taichi Vector.field(4)
            batch: Batch index
            y: Y-coordinate
            x: X-coordinate
            channel: Channel index (0-3)
            height: Image height
            width: Image width

        Returns:
            Interpolated single channel value

        """
        y0 = ti.cast(ti.floor(y), ti.i32)
        x0 = ti.cast(ti.floor(x), ti.i32)
        y1 = y0 + 1
        x1 = x0 + 1

        fy = y - ti.cast(y0, ti.f32)
        fx = x - ti.cast(x0, ti.f32)

        y0 = clamp_i(y0, height)
        y1 = clamp_i(y1, height)
        x0 = clamp_i(x0, width)
        x1 = clamp_i(x1, width)

        v00 = field[batch, y0, x0][channel]
        v01 = field[batch, y0, x1][channel]
        v10 = field[batch, y1, x0][channel]
        v11 = field[batch, y1, x1][channel]

        w00 = (1.0 - fy) * (1.0 - fx)
        w01 = (1.0 - fy) * fx
        w10 = fy * (1.0 - fx)
        w11 = fy * fx

        return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11


# NumPy reference implementations for testing


def clamp_coords_numpy(i: int, j: int, h: int, w: int) -> tuple[int, int]:
    """NumPy/Python reference implementation of clamp_coords."""
    ci = max(0, min(i, h - 1))
    cj = max(0, min(j, w - 1))
    return (ci, cj)


def reflect_boundary_numpy(coord: int, size: int) -> int:
    """NumPy/Python reference implementation of reflect_boundary."""
    if coord < 0:
        result = -coord - 1
        return min(result, size - 1)
    elif coord >= size:
        result = 2 * size - coord - 1
        return max(result, 0)
    return coord


def bilinear_sample_numpy(
    image: np.ndarray,
    y: float,
    x: float,
) -> np.ndarray:
    """
    NumPy reference implementation of bilinear sampling.

    Args:
        image: Input image (H, W, C) or (H, W)
        y: Y-coordinate (row)
        x: X-coordinate (column)

    Returns:
        Interpolated pixel value(s)

    """
    h, w = image.shape[:2]

    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1

    fy = y - y0
    fx = x - x0

    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))

    p00 = image[y0, x0]
    p01 = image[y0, x1]
    p10 = image[y1, x0]
    p11 = image[y1, x1]

    w00 = (1.0 - fy) * (1.0 - fx)
    w01 = (1.0 - fy) * fx
    w10 = fy * (1.0 - fx)
    w11 = fy * fx

    result: np.ndarray = p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11
    return result

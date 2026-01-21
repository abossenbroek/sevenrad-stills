"""
Reproducible random number generation utilities for Taichi kernels.

This module provides PCG hash-based random number generation that produces
deterministic, reproducible results based on pixel coordinates and seed.

Unlike Taichi's built-in ti.random(), these functions are reproducible
across runs when given the same inputs, making them suitable for testing
and consistent artistic effects.

Example:
    >>> @ti.kernel
    ... def add_noise(source: ti.template(), dest: ti.template(), seed: ti.i32):
    ...     for i, j in ti.ndrange(height, width):
    ...         noise = rand_gaussian(i, j, seed, 0.1)
    ...         pixel = source[0, i, j]
    ...         dest[0, i, j] = ti.Vector([pixel[0] + noise, ...])

Note:
    This module intentionally does NOT use `from __future__ import annotations`
    because Taichi requires actual type objects at decoration time, not string
    annotations (PEP 563 deferred evaluation breaks Taichi's type system).

"""

import math

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False


# PCG constants for high-quality hashing
PCG_MULT = 747796405
PCG_INC = 2891336453
PCG_FACTOR = 277803737

# Coordinate mixing primes (chosen to minimize correlation)
COORD_PRIME_X = 374761393
COORD_PRIME_Y = 668265263
COORD_PRIME_C = 73856093

# Normalization constant
UINT32_MAX_F = 4294967296.0

# Box-Muller constants
TWO_PI = 2.0 * math.pi

# Epsilon for numerical stability
EPSILON = 1e-10


if TAICHI_AVAILABLE and ti is not None:

    @ti.func  # type: ignore[misc]
    def pcg_hash(  # type: ignore[no-untyped-def]
        input_seed: ti.u32,
    ) -> ti.u32:
        """
        PCG-derived hash function for generating pseudo-random integers.

        Uses the PCG (Permuted Congruential Generator) algorithm's
        output permutation to produce well-distributed hash values.
        This provides better statistical properties than simpler hashes.

        Args:
            input_seed: Input seed value to hash

        Returns:
            Hashed 32-bit unsigned integer

        Note:
            This is a pure function - same input always produces same output.

        """
        # Cast constants to ti.u32() to avoid Taichi's i32 default interpretation
        # PCG_INC (2891336453) exceeds i32 max (2147483647) but fits in u32
        # u32 arithmetic provides natural mod 2^32 via hardware wraparound
        state = input_seed * ti.u32(PCG_MULT) + ti.u32(PCG_INC)
        word = ((state >> ((state >> 28) + 4)) ^ state) * ti.u32(PCG_FACTOR)
        return (word >> 22) ^ word

    @ti.func  # type: ignore[misc]
    def rand_float(  # type: ignore[no-untyped-def]
        x: ti.i32,
        y: ti.i32,
        seed: ti.i32,
    ) -> ti.f32:
        """
        Generate deterministic random float in [0, 1) based on position and seed.

        Produces a uniform random value that is reproducible - calling with
        the same (x, y, seed) always returns the same value. Uses PCG hash
        for high-quality randomness.

        Args:
            x: Pixel x-coordinate (column index)
            y: Pixel y-coordinate (row index)
            seed: Random seed value

        Returns:
            Uniform random float in range [0.0, 1.0)

        Example:
            >>> noise = rand_float(i, j, 42)  # Same position, same seed = same result

        """
        # Mix coordinates and seed into single hash input
        combined = ti.cast(x * COORD_PRIME_X + y * COORD_PRIME_Y + seed, ti.u32)
        h = pcg_hash(combined)
        return ti.cast(h, ti.f32) / UINT32_MAX_F

    @ti.func  # type: ignore[misc]
    def rand_float_channel(  # type: ignore[no-untyped-def]
        x: ti.i32,
        y: ti.i32,
        channel: ti.i32,
        seed: ti.i32,
    ) -> ti.f32:
        """
        Generate deterministic random float for specific channel.

        Similar to rand_float but includes channel index for generating
        independent random values per color channel.

        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            channel: Channel index (0=R, 1=G, 2=B, 3=A)
            seed: Random seed value

        Returns:
            Uniform random float in range [0.0, 1.0)

        """
        combined = ti.cast(
            x * COORD_PRIME_X + y * COORD_PRIME_Y + channel * COORD_PRIME_C + seed,
            ti.u32,
        )
        h = pcg_hash(combined)
        return ti.cast(h, ti.f32) / UINT32_MAX_F

    @ti.func  # type: ignore[misc]
    def rand_gaussian(  # type: ignore[no-untyped-def]
        x: ti.i32,
        y: ti.i32,
        seed: ti.i32,
        sigma: ti.f32,
    ) -> ti.f32:
        """
        Generate Gaussian (normal) distributed random value using Box-Muller.

        Produces normally distributed values with mean=0 and given sigma.
        Uses Box-Muller transform on two uniform random values.

        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            seed: Random seed value
            sigma: Standard deviation of the Gaussian distribution

        Returns:
            Gaussian random value with mean=0, std=sigma

        Note:
            Uses Box-Muller transform: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
            Two calls to rand_float with offset seeds for independence.

        """
        # Generate two uniform randoms with different seeds
        u1 = rand_float(x, y, seed)
        u2 = rand_float(x, y, seed + 1)

        # Avoid log(0) by clamping
        u1 = ti.max(u1, EPSILON)

        # Box-Muller transform (using only cos, discarding sin value)
        mag = ti.sqrt(-2.0 * ti.log(u1))
        z = mag * ti.cos(TWO_PI * u2)

        return z * sigma

    @ti.func  # type: ignore[misc]
    def rand_gaussian_channel(  # type: ignore[no-untyped-def]
        x: ti.i32,
        y: ti.i32,
        channel: ti.i32,
        seed: ti.i32,
        sigma: ti.f32,
    ) -> ti.f32:
        """
        Generate Gaussian random value for specific channel.

        Args:
            x: Pixel x-coordinate
            y: Pixel y-coordinate
            channel: Channel index
            seed: Random seed value
            sigma: Standard deviation

        Returns:
            Gaussian random value

        """
        u1 = rand_float_channel(x, y, channel, seed)
        u2 = rand_float_channel(x, y, channel, seed + 12345)

        u1 = ti.max(u1, EPSILON)
        mag = ti.sqrt(-2.0 * ti.log(u1))
        z = mag * ti.cos(TWO_PI * u2)

        return z * sigma


# NumPy-based reference implementations for testing


def pcg_hash_numpy(input_seed: int) -> int:
    """NumPy/Python reference implementation of pcg_hash."""
    input_seed = input_seed & 0xFFFFFFFF  # Ensure 32-bit
    state = (input_seed * PCG_MULT + PCG_INC) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * PCG_FACTOR) & 0xFFFFFFFF
    return ((word >> 22) ^ word) & 0xFFFFFFFF


def rand_float_numpy(x: int, y: int, seed: int) -> float:
    """NumPy/Python reference implementation of rand_float."""
    combined = (x * COORD_PRIME_X + y * COORD_PRIME_Y + seed) & 0xFFFFFFFF
    h = pcg_hash_numpy(combined)
    return h / UINT32_MAX_F


def rand_gaussian_numpy(x: int, y: int, seed: int, sigma: float) -> float:
    """NumPy/Python reference implementation of rand_gaussian."""
    u1 = max(rand_float_numpy(x, y, seed), EPSILON)
    u2 = rand_float_numpy(x, y, seed + 1)

    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.cos(TWO_PI * u2)

    return z * sigma

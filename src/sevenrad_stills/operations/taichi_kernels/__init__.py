"""
Shared Taichi kernel utilities for GPU operations.

This package provides reusable Taichi functions and kernels for:
- Random number generation (reproducible PCG hash-based)
- Sampling/interpolation (bilinear, nearest neighbor)
- Convolution (separable and general 2D)

These utilities are designed to be imported and used by individual
Taichi operation implementations.
"""

from sevenrad_stills.operations.taichi_kernels.convolution import (
    convolve_2d,
    convolve_horizontal,
    convolve_vertical,
)
from sevenrad_stills.operations.taichi_kernels.random import (
    pcg_hash,
    rand_float,
    rand_gaussian,
)
from sevenrad_stills.operations.taichi_kernels.sampling import (
    bilinear_sample,
    clamp_coords,
    nearest_sample,
    reflect_boundary,
)

__all__ = [
    "bilinear_sample",
    "clamp_coords",
    "convolve_2d",
    "convolve_horizontal",
    "convolve_vertical",
    "nearest_sample",
    "pcg_hash",
    "rand_float",
    "rand_gaussian",
    "reflect_boundary",
]

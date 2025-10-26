"""
GPU-accelerated band swap operation using Taichi.

Simulates errors in satellite data transmission where packet headers are corrupted,
causing band/channel data to be misinterpreted and swapped in rectangular regions.
Uses Taichi for GPU acceleration of the band permutation operations.
"""

from typing import Any

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants (same as CPU version)
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 50
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0

# Valid permutation patterns (excluding identity RGB)
VALID_PERMUTATIONS = {
    "GRB": [1, 0, 2],
    "BGR": [2, 1, 0],
    "BRG": [2, 0, 1],
    "GBR": [1, 2, 0],
    "RBG": [0, 2, 1],
}


@ti.func  # type: ignore[misc]
def swap_channels(  # type: ignore[no-untyped-def]  # noqa: ANN201, PLR0913
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    y: ti.i32,
    x: ti.i32,
    perm_0: ti.i32,
    perm_1: ti.i32,
    perm_2: ti.i32,
):
    """
    Device function to swap channels at a single pixel.

    Args:
        img: Image array (H, W, 3)
        y: Y coordinate
        x: X coordinate
        perm_0: First channel index
        perm_1: Second channel index
        perm_2: Third channel index

    """
    # Read original RGB values into temp vars
    r = img[y, x, 0]
    g = img[y, x, 1]
    b = img[y, x, 2]

    # Create channel vector and apply permutation
    channels = ti.Vector([r, g, b])

    # Write back permuted channels
    img[y, x, 0] = channels[perm_0]
    img[y, x, 1] = channels[perm_1]
    img[y, x, 2] = channels[perm_2]


@ti.kernel  # type: ignore[misc]
def apply_band_swap_batch(  # type: ignore[no-untyped-def]  # noqa: PLR0913, ANN201
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
    perm_0: ti.i32,
    perm_1: ti.i32,
    perm_2: ti.i32,
):
    """
    GPU kernel to process all tiles in a single batch.

    Processes all tiles with a single kernel launch to minimize overhead.
    Uses 2D parallelization to avoid expensive division/modulo operations.

    Args:
        img: Image array (H, W, 3) to modify in-place
        tiles: Tile coordinates array (N, 4) [y_start, y_end, x_start, x_end]
        num_tiles: Number of tiles
        max_tile_h: Maximum tile height
        max_tile_w: Maximum tile width
        perm_0: First channel index
        perm_1: Second channel index
        perm_2: Third channel index

    """
    # Parallelize over tiles and 2D pixel coordinates (avoids expensive div/mod)
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        # Check if this pixel is within the tile bounds
        y = y_start + local_y
        x = x_start + local_x

        if y < y_end and x < x_end:
            # Apply permutation
            swap_channels(img, y, x, perm_0, perm_1, perm_2)


class BandSwapGPUOperation(BaseImageOperation):
    """
    GPU-accelerated band/channel swapping operation using Taichi.

    This operation simulates packet mis-identification errors in satellite downlink
    by swapping color channels in rectangular tiles. The band permutation is
    accelerated using Taichi GPU kernels for improved performance on large images.

    Simulates a failure mode where satellite downlink packets containing spectral
    band data have corrupted metadata in their headers. The ground station receives
    the correct pixel data but interprets it with the wrong band labels.

    In multi-spectral satellite imagery:
    - Band 1 (Red) data is received but labeled as Band 2 (Green)
    - Band 2 (Green) data is received but labeled as Band 3 (Blue)
    - Band 3 (Blue) data is received but labeled as Band 1 (Red)

    This creates rectangular regions with sudden, dramatic color shifts - the spatial
    structure is preserved but colors are completely wrong. Common causes:
    - Bit flips in packet header metadata from cosmic rays
    - Software bugs in on-board packet assembly
    - Ground station decompression errors misinterpreting stream structure

    Performance: GPU acceleration provides significant speedup for large images
    and/or many tiles, as each tile's permutation can be parallelized across
    thousands of GPU threads.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated band swap operation."""
        super().__init__("band_swap_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
        """
        Validate parameters for band swap operation.

        Args:
            params: A dictionary containing:
                - tile_count (int): Number of affected tiles (1 to 50)
                - permutation (str): Channel swap pattern - one of:
                  'GRB', 'BGR', 'BRG', 'GBR', 'RBG'
                - tile_size_range (list): [min, max] tile size as fractions
                  of image dimensions (0.01 to 1.0)
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "tile_count" not in params:
            msg = "Band swap operation requires 'tile_count' parameter."
            raise ValueError(msg)
        tile_count = params["tile_count"]
        if not isinstance(tile_count, int) or not (
            MIN_TILE_COUNT <= tile_count <= MAX_TILE_COUNT
        ):
            msg = (
                f"tile_count must be an integer between {MIN_TILE_COUNT} "
                f"and {MAX_TILE_COUNT}."
            )
            raise ValueError(msg)

        if "permutation" not in params:
            msg = "Band swap operation requires 'permutation' parameter."
            raise ValueError(msg)
        permutation = params["permutation"]
        if permutation not in VALID_PERMUTATIONS:
            valid_list = ", ".join(VALID_PERMUTATIONS.keys())
            msg = f"Permutation must be one of: {valid_list}."
            raise ValueError(msg)

        if "tile_size_range" in params:
            tile_size_range = params["tile_size_range"]
            if (
                not isinstance(tile_size_range, (list, tuple))
                or len(tile_size_range) != 2  # noqa: PLR2004
            ):
                msg = "tile_size_range must be a list/tuple of two numbers [min, max]."
                raise ValueError(msg)
            min_size, max_size = tile_size_range
            if not isinstance(min_size, (int, float)) or not isinstance(
                max_size, (int, float)
            ):
                msg = "tile_size_range values must be numbers."
                raise ValueError(msg)
            if not (MIN_TILE_SIZE <= min_size <= MAX_TILE_SIZE) or not (
                MIN_TILE_SIZE <= max_size <= MAX_TILE_SIZE
            ):
                msg = (
                    f"tile_size_range values must be between {MIN_TILE_SIZE} "
                    f"and {MAX_TILE_SIZE}."
                )
                raise ValueError(msg)
            if min_size > max_size:
                msg = "tile_size_range min must be less than or equal to max."
                raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply GPU-accelerated band swapping to random tiles in the image.

        The tile positions and sizes are generated on CPU using numpy's RNG,
        while the actual band permutation within each tile is executed on GPU
        using Taichi kernels for parallel processing.

        Args:
            image: The input PIL Image (must be RGB or RGBA).
            params: A dictionary with 'tile_count', 'permutation',
                    optional 'tile_size_range', and optional 'seed'.

        Returns:
            The PIL Image with band swapping applied to random tiles.

        Raises:
            ValueError: If image is not RGB or RGBA mode.

        """
        self.validate_params(params)

        # Band swap only makes sense for RGB/RGBA images
        if image.mode not in ("RGB", "RGBA"):
            msg = f"Band swap requires RGB or RGBA image, got {image.mode}."
            raise ValueError(msg)

        tile_count: int = params["tile_count"]
        permutation: str = params["permutation"]
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed: int | None = params.get("seed")

        # Create random number generator (CPU-side)
        rng = np.random.default_rng(seed)

        # Convert to array and separate RGB from alpha if needed
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
        else:
            rgb = img_array.copy()
            alpha = None

        h, w = rgb.shape[:2]

        # Get permutation indices
        perm_indices = VALID_PERMUTATIONS[permutation]

        # Generate all tile coordinates upfront (CPU side)
        tiles: np.ndarray = np.zeros((tile_count, 4), dtype=np.int32)
        max_tile_h = 0
        max_tile_w = 0
        for i in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, h - tile_h + 1))
            x = rng.integers(0, max(1, w - tile_w + 1))

            # Store tile coordinates [y_start, y_end, x_start, x_end]
            tiles[i] = [y, y + tile_h, x, x + tile_w]

            # Track maximum tile dimensions for efficient GPU parallelization
            max_tile_h = max(max_tile_h, tile_h)
            max_tile_w = max(max_tile_w, tile_w)

        # Process all tiles in a single GPU kernel launch (no type conversion needed)
        apply_band_swap_batch(
            rgb,
            tiles,
            tile_count,
            max_tile_h,
            max_tile_w,
            perm_indices[0],
            perm_indices[1],
            perm_indices[2],
        )

        output = rgb

        # Recombine with alpha if needed
        output_array = np.dstack([output, alpha]) if alpha is not None else output

        return Image.fromarray(output_array)

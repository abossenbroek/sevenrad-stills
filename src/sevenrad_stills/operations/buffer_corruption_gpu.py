"""
GPU-accelerated buffer corruption operation using Taichi.

Simulates cosmic ray hits on satellite memory causing bit flips in image
data buffers before transmission. Uses Taichi for GPU acceleration of the
corruption operations.
"""

from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants (same as CPU version)
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 20
MIN_SEVERITY = 0.0
MAX_SEVERITY = 1.0
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0

# Valid corruption types
VALID_CORRUPTION_TYPES = {"xor", "invert", "channel_shuffle"}


@ti.kernel  # type: ignore[misc]
def apply_xor_corruption_batch(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    xor_masks: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
):
    """
    GPU kernel to apply XOR corruption to all tiles in a single batch.

    Args:
        img: Image array (H, W, 3) to modify in-place
        tiles: Tile coordinates array (N, 4) [y_start, y_end, x_start, x_end]
        xor_masks: XOR mask array (N, max_h, max_w, 3) with random byte patterns
        num_tiles: Number of tiles
        max_tile_h: Maximum tile height
        max_tile_w: Maximum tile width

    """
    # Parallelize over tiles and 2D pixel coordinates
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        # Global pixel coordinates
        y = y_start + local_y
        x = x_start + local_x

        # Check if this pixel is within the tile bounds
        if y < y_end and x < x_end:
            # Apply XOR with the pre-generated mask for this tile
            for c in ti.static(range(3)):
                img[y, x, c] = img[y, x, c] ^ xor_masks[tile_idx, local_y, local_x, c]


@ti.kernel  # type: ignore[misc]
def apply_invert_corruption_batch(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
    severity: ti.f32,
):
    """
    GPU kernel to apply invert corruption to all tiles in a single batch.

    Args:
        img: Image array (H, W, 3) to modify in-place
        tiles: Tile coordinates array (N, 4) [y_start, y_end, x_start, x_end]
        num_tiles: Number of tiles
        max_tile_h: Maximum tile height
        max_tile_w: Maximum tile width
        severity: Blend factor between original and inverted (0.0 to 1.0)

    """
    # Parallelize over tiles and 2D pixel coordinates
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        # Global pixel coordinates
        y = y_start + local_y
        x = x_start + local_x

        # Check if this pixel is within the tile bounds
        if y < y_end and x < x_end:
            # Bitwise inversion with blending based on severity
            for c in ti.static(range(3)):
                original = ti.cast(img[y, x, c], ti.f32)
                inverted = ti.cast(255 - img[y, x, c], ti.f32)
                blended = original * (1.0 - severity) + inverted * severity
                img[y, x, c] = ti.cast(blended, ti.u8)


@ti.kernel  # type: ignore[misc]
def apply_channel_shuffle_corruption_batch(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    permutations: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
):
    """
    GPU kernel to apply channel shuffle corruption to all tiles in a single batch.

    Args:
        img: Image array (H, W, 3) to modify in-place
        tiles: Tile coordinates array (N, 4) [y_start, y_end, x_start, x_end]
        permutations: Permutation array (N, 3) with channel indices for each tile
        num_tiles: Number of tiles
        max_tile_h: Maximum tile height
        max_tile_w: Maximum tile width

    """
    # Parallelize over tiles and 2D pixel coordinates
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        # Global pixel coordinates
        y = y_start + local_y
        x = x_start + local_x

        # Check if this pixel is within the tile bounds
        if y < y_end and x < x_end:
            # Read original RGB values
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            # Apply permutation for this tile
            perm_0 = permutations[tile_idx, 0]
            perm_1 = permutations[tile_idx, 1]
            perm_2 = permutations[tile_idx, 2]

            # Create channel vector and apply permutation
            # Use if-else since Taichi doesn't support dynamic array indexing
            # with ti.Vector in all cases
            if perm_0 == 0:
                img[y, x, 0] = r
            elif perm_0 == 1:
                img[y, x, 0] = g
            else:
                img[y, x, 0] = b

            if perm_1 == 0:
                img[y, x, 1] = r
            elif perm_1 == 1:
                img[y, x, 1] = g
            else:
                img[y, x, 1] = b

            if perm_2 == 0:
                img[y, x, 2] = r
            elif perm_2 == 1:
                img[y, x, 2] = g
            else:
                img[y, x, 2] = b


class BufferCorruptionGPUOperation(BaseImageOperation):
    """
    GPU-accelerated buffer corruption operation using Taichi.

    Simulates high-energy particle impacts on satellite memory chips causing
    "Single Event Upsets" (SEUs) - bit flips in the image data buffer before
    it's compressed or transmitted.

    In orbital environments, cosmic rays and solar particle events can flip
    individual bits in DRAM or SRAM. When this happens to image buffer memory:
    - A single bit flip can corrupt an entire tile of data
    - The corruption affects raw pixel values before encoding
    - Creates characteristic "glitch blocks" - rectangular regions with
      garbled pixels, wrong colors, or horizontal displacement

    Corruption modes:
    - xor: Bitwise XOR with random pattern (simulates bit flips)
    - invert: Bitwise inversion with blending (simulates register corruption)
    - channel_shuffle: Random RGB channel permutation per tile (simulates
      pointer corruption in multi-planar buffers)

    Performance: GPU acceleration provides significant speedup for large images
    and/or many tiles, as each tile's corruption can be parallelized across
    thousands of GPU threads.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated buffer corruption operation."""
        super().__init__("buffer_corruption_gpu")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for buffer corruption operation.

        Args:
            params: A dictionary containing:
                - tile_count (int): Number of corrupted tiles (1 to 20)
                - corruption_type (str): 'xor', 'invert', or 'channel_shuffle'
                - severity (float): Corruption intensity (0.0 to 1.0)
                - tile_size_range (list, optional): [min, max] tile size as
                  fractions of image dimensions (default: [0.05, 0.2])
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "tile_count" not in params:
            msg = "Buffer corruption operation requires 'tile_count' parameter."
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

        if "corruption_type" not in params:
            msg = "Buffer corruption operation requires 'corruption_type' parameter."
            raise ValueError(msg)
        corruption_type = params["corruption_type"]
        if corruption_type not in VALID_CORRUPTION_TYPES:
            valid_list = ", ".join(sorted(VALID_CORRUPTION_TYPES))
            msg = f"corruption_type must be one of: {valid_list}."
            raise ValueError(msg)

        if "severity" not in params:
            msg = "Buffer corruption operation requires 'severity' parameter."
            raise ValueError(msg)
        severity = params["severity"]
        if not isinstance(severity, (int, float)) or not (
            MIN_SEVERITY <= severity <= MAX_SEVERITY
        ):
            msg = (
                f"severity must be a number between {MIN_SEVERITY} and {MAX_SEVERITY}."
            )
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
        Apply GPU-accelerated buffer corruption to random tiles in the image.

        The tile positions and sizes are generated on CPU using numpy's RNG,
        while the actual corruption within each tile is executed on GPU
        using Taichi kernels for parallel processing.

        Args:
            image: The input PIL Image (must be RGB or RGBA).
            params: A dictionary with 'tile_count', 'corruption_type', 'severity',
                    optional 'tile_size_range', and optional 'seed'.

        Returns:
            The PIL Image with buffer corruption applied to random tiles.

        Raises:
            ValueError: If image is not RGB or RGBA mode.

        """
        self.validate_params(params)

        # Buffer corruption only makes sense for RGB/RGBA images
        if image.mode not in ("RGB", "RGBA"):
            msg = f"Buffer corruption GPU requires RGB or RGBA image, got {image.mode}."
            raise ValueError(msg)

        tile_count: int = params["tile_count"]
        corruption_type: Literal["xor", "invert", "channel_shuffle"] = params[
            "corruption_type"
        ]
        severity: float = params["severity"]
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed: int | None = params.get("seed")

        # Early exit optimization for zero severity
        if severity == 0.0:
            return image.copy()

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

        # For XOR corruption, generate tiles and masks to match CPU RNG sequence
        if corruption_type == "xor":
            xor_magnitude = int(255 * severity)

            # Generate tile positions first (vectorized for performance)
            tile_fractions = rng.uniform(
                tile_size_range[0], tile_size_range[1], size=tile_count
            )
            tile_heights = np.maximum(1, (h * tile_fractions).astype(np.int32))
            tile_widths = np.maximum(1, (w * tile_fractions).astype(np.int32))

            y_starts = rng.integers(
                0, np.maximum(1, h - tile_heights + 1), size=tile_count, dtype=np.int32
            )
            x_starts = rng.integers(
                0, np.maximum(1, w - tile_widths + 1), size=tile_count, dtype=np.int32
            )

            # Build tile array [y_start, y_end, x_start, x_end]
            tiles: np.ndarray = np.column_stack(
                [
                    y_starts,
                    y_starts + tile_heights,
                    x_starts,
                    x_starts + tile_widths,
                ]
            ).astype(np.int32)

            # Calculate max dimensions for GPU parallelization
            max_tile_h = int(tile_heights.max())
            max_tile_w = int(tile_widths.max())

            # Generate XOR masks efficiently
            if xor_magnitude > 0:
                # Pre-allocate mask array
                xor_masks = np.zeros(
                    (tile_count, max_tile_h, max_tile_w, 3), dtype=np.uint8
                )

                # Generate all masks at once for better performance
                total_pixels = int(tile_heights.sum() * tile_widths.sum())
                if total_pixels > 0:
                    # Generate random values for all tiles at once
                    for i in range(tile_count):
                        tile_h = tile_heights[i]
                        tile_w = tile_widths[i]
                        xor_masks[i, :tile_h, :tile_w, :] = rng.integers(
                            0,
                            xor_magnitude + 1,
                            size=(tile_h, tile_w, 3),
                            dtype=np.uint8,
                        )

                # Apply XOR corruption on GPU in parallel for better performance
                # Note: For overlapping tiles, behavior may differ slightly from CPU
                apply_xor_corruption_batch(
                    rgb, tiles, xor_masks, tile_count, max_tile_h, max_tile_w
                )

        elif corruption_type == "invert":
            # For other corruption types, can use vectorized tile generation
            tile_fractions = rng.uniform(
                tile_size_range[0], tile_size_range[1], size=tile_count
            )
            tile_heights = np.maximum(1, (h * tile_fractions).astype(np.int32))
            tile_widths = np.maximum(1, (w * tile_fractions).astype(np.int32))

            # Random positions for all tiles
            y_starts = rng.integers(
                0, np.maximum(1, h - tile_heights + 1), size=tile_count, dtype=np.int32
            )
            x_starts = rng.integers(
                0, np.maximum(1, w - tile_widths + 1), size=tile_count, dtype=np.int32
            )

            # Build tile array [y_start, y_end, x_start, x_end]
            tiles: np.ndarray = np.column_stack(
                [
                    y_starts,
                    y_starts + tile_heights,
                    x_starts,
                    x_starts + tile_widths,
                ]
            ).astype(np.int32)

            # Calculate max dimensions for GPU parallelization
            max_tile_h = int(tile_heights.max())
            max_tile_w = int(tile_widths.max())
            # Apply invert corruption on GPU
            apply_invert_corruption_batch(
                rgb, tiles, tile_count, max_tile_h, max_tile_w, float(severity)
            )

        elif corruption_type == "channel_shuffle":
            # Generate tile coordinates using NumPy vectorization
            tile_fractions = rng.uniform(
                tile_size_range[0], tile_size_range[1], size=tile_count
            )
            tile_heights = np.maximum(1, (h * tile_fractions).astype(np.int32))
            tile_widths = np.maximum(1, (w * tile_fractions).astype(np.int32))

            # Random positions for all tiles
            y_starts = rng.integers(
                0, np.maximum(1, h - tile_heights + 1), size=tile_count, dtype=np.int32
            )
            x_starts = rng.integers(
                0, np.maximum(1, w - tile_widths + 1), size=tile_count, dtype=np.int32
            )

            # Build tile array [y_start, y_end, x_start, x_end]
            tiles: np.ndarray = np.column_stack(
                [
                    y_starts,
                    y_starts + tile_heights,
                    x_starts,
                    x_starts + tile_widths,
                ]
            ).astype(np.int32)

            # Calculate max dimensions for GPU parallelization
            max_tile_h = int(tile_heights.max())
            max_tile_w = int(tile_widths.max())

            # Pre-generate permutations on CPU for reproducibility
            # Severity controls probability of shuffling per tile
            permutations = np.zeros((tile_count, 3), dtype=np.int32)
            for i in range(tile_count):
                if rng.random() < severity:
                    # Generate random permutation for this tile
                    permutations[i] = rng.permutation(3)
                else:
                    # Identity permutation (no shuffle)
                    permutations[i] = [0, 1, 2]

            # Apply channel shuffle on GPU
            apply_channel_shuffle_corruption_batch(
                rgb, tiles, permutations, tile_count, max_tile_h, max_tile_w
            )

        # Recombine with alpha if needed
        output_array = np.dstack([rgb, alpha]) if alpha is not None else rgb

        return Image.fromarray(output_array)

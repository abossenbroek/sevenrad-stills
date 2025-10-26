"""
Optimized GPU-accelerated buffer corruption using Taichi with GPU RNG.

This version sacrifices exact CPU RNG compatibility for maximum performance
by generating XOR masks directly on GPU.
"""

import time
from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=int(time.time()) % (2**31))

# Constants
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 20
MIN_SEVERITY = 0.0
MAX_SEVERITY = 1.0
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0

VALID_CORRUPTION_TYPES = {"xor", "invert", "channel_shuffle"}


@ti.kernel  # type: ignore[misc]
def apply_xor_corruption_optimized(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
    xor_magnitude: ti.i32,
):
    """
    Optimized GPU kernel: generates XOR masks on GPU and applies corruption.

    Fuses mask generation and XOR application into a single kernel,
    eliminating CPU preprocessing and memory transfer overhead.

    Args:
        img: Image array (H, W, 3) to modify in-place
        tiles: Tile coordinates array (N, 4) [y_start, y_end, x_start, x_end]
        num_tiles: Number of tiles
        max_tile_h: Maximum tile height
        max_tile_w: Maximum tile width
        xor_magnitude: Maximum XOR value (255 * severity)

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
            # Generate XOR mask values on GPU for each channel
            for c in ti.static(range(3)):
                # Use Taichi's built-in random() - seeded at ti.init
                rand_val = ti.random(ti.f32)
                mask_value = ti.cast(rand_val * (xor_magnitude + 1), ti.u8)
                img[y, x, c] = img[y, x, c] ^ mask_value


@ti.kernel  # type: ignore[misc]
def apply_invert_corruption_batch(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    tiles: ti.types.ndarray(),  # type: ignore[valid-type]
    num_tiles: ti.i32,
    max_tile_h: ti.i32,
    max_tile_w: ti.i32,
    severity: ti.f32,
):
    """GPU kernel to apply invert corruption to all tiles in a single batch."""
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        y = y_start + local_y
        x = x_start + local_x

        if y < y_end and x < x_end:
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
    """GPU kernel to apply channel shuffle corruption to all tiles."""
    for tile_idx, local_y, local_x in ti.ndrange(num_tiles, max_tile_h, max_tile_w):
        y_start = tiles[tile_idx, 0]
        y_end = tiles[tile_idx, 1]
        x_start = tiles[tile_idx, 2]
        x_end = tiles[tile_idx, 3]

        y = y_start + local_y
        x = x_start + local_x

        if y < y_end and x < x_end:
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            perm_0 = permutations[tile_idx, 0]
            perm_1 = permutations[tile_idx, 1]
            perm_2 = permutations[tile_idx, 2]

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


class BufferCorruptionGPUOptimizedOperation(BaseImageOperation):
    """
    Optimized GPU-accelerated buffer corruption using GPU-native RNG.

    This version generates XOR masks directly on GPU for maximum performance,
    sacrificing exact CPU RNG compatibility. Provides significant speedup
    over CPU for all corruption types, especially XOR.

    Performance optimizations:
    - Fused mask generation + XOR application (eliminates memory transfer)
    - GPU-native random number generation
    - Parallel processing across all tiles simultaneously
    - Optimized memory access patterns
    """

    def __init__(self) -> None:
        """Initialize the optimized GPU buffer corruption operation."""
        super().__init__("buffer_corruption_gpu_optimized")

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate parameters for buffer corruption operation."""
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
        """Apply optimized GPU-accelerated buffer corruption."""
        self.validate_params(params)

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

        if severity == 0.0:
            return image.copy()

        rng = np.random.default_rng(seed)

        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
        else:
            rgb = img_array.copy()
            alpha = None

        h, w = rgb.shape[:2]

        # Generate tile coordinates (vectorized for all types)
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

        tiles: np.ndarray = np.column_stack(
            [
                y_starts,
                y_starts + tile_heights,
                x_starts,
                x_starts + tile_widths,
            ]
        ).astype(np.int32)

        max_tile_h = int(tile_heights.max())
        max_tile_w = int(tile_widths.max())

        # Apply corruption based on type
        if corruption_type == "xor":
            xor_magnitude = int(255 * severity)
            if xor_magnitude > 0:
                # Use GPU-native RNG for mask generation (optimized!)
                apply_xor_corruption_optimized(
                    rgb,
                    tiles,
                    tile_count,
                    max_tile_h,
                    max_tile_w,
                    xor_magnitude,
                )

        elif corruption_type == "invert":
            apply_invert_corruption_batch(
                rgb, tiles, tile_count, max_tile_h, max_tile_w, float(severity)
            )

        elif corruption_type == "channel_shuffle":
            permutations = np.zeros((tile_count, 3), dtype=np.int32)
            for i in range(tile_count):
                if rng.random() < severity:
                    permutations[i] = rng.permutation(3)
                else:
                    permutations[i] = [0, 1, 2]

            apply_channel_shuffle_corruption_batch(
                rgb, tiles, permutations, tile_count, max_tile_h, max_tile_w
            )

        output_array = np.dstack([rgb, alpha]) if alpha is not None else rgb

        return Image.fromarray(output_array)

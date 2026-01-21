"""
Taichi end-to-end pipeline band swap operation.

Band swap operation for GPU pipeline execution using ti.Vector.field(4).
Simulates packet mis-identification in satellite downlink by swapping
RGB channels in randomly placed rectangular tiles.
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

# Import RNG utilities
try:
    from sevenrad_stills.operations.taichi_kernels.random import (
        COORD_PRIME_X,
        COORD_PRIME_Y,
        PCG_FACTOR,
        PCG_INC,
        PCG_MULT,
        UINT32_MAX_F,
    )
except ImportError:
    # Fallback values for when Taichi is not available
    COORD_PRIME_X = 374761393
    COORD_PRIME_Y = 668265263
    PCG_MULT = 747796405
    PCG_INC = 2891336453
    PCG_FACTOR = 277803737
    UINT32_MAX_F = 4294967296.0

# Constants
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


# Define helper functions and kernel only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.func  # type: ignore[misc]
    def pcg_hash(  # type: ignore[no-untyped-def]
        input_seed: ti.u32,
    ) -> ti.u32:
        """
        PCG-derived hash function for generating pseudo-random integers.

        Args:
            input_seed: Input seed value to hash

        Returns:
            Hashed 32-bit unsigned integer

        """
        state = input_seed * PCG_MULT + PCG_INC
        word = ((state >> ((state >> 28) + 4)) ^ state) * PCG_FACTOR
        return (word >> 22) ^ word

    @ti.func  # type: ignore[misc]
    def rand_float(  # type: ignore[no-untyped-def]
        x: ti.i32,
        y: ti.i32,
        seed: ti.i32,
    ) -> ti.f32:
        """
        Generate deterministic random float in [0, 1) based on position and seed.

        Args:
            x: X coordinate
            y: Y coordinate
            seed: Random seed value

        Returns:
            Uniform random float in range [0.0, 1.0)

        """
        combined = ti.cast(x * COORD_PRIME_X + y * COORD_PRIME_Y + seed, ti.u32)
        h = pcg_hash(combined)
        return ti.cast(h, ti.f32) / UINT32_MAX_F

    @ti.kernel  # type: ignore[misc]
    def _band_swap_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        tile_data: ti.template(),  # type: ignore[valid-type]
        perm_r: ti.i32,
        perm_g: ti.i32,
        perm_b: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
        tile_count: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for band swapping in rectangular tiles.

        Operates on ti.Vector.field(4) with RGBA channels.
        Only modifies RGB within tile regions, preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            tile_data: Field storing tile coordinates (tile_count, 4) as [y, x, h, w]
            perm_r: Target channel index for red (0, 1, or 2)
            perm_g: Target channel index for green (0, 1, or 2)
            perm_b: Target channel index for blue (0, 1, or 2)
            batch: Batch index
            height: Image height
            width: Image width
            tile_count: Number of tiles

        """
        for i, j in ti.ndrange(height, width):
            # Read RGBA from source
            pixel = source[batch, i, j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            a = pixel[3]  # Preserve alpha

            # Check if pixel is within any tile
            in_tile = False
            for tile_idx in range(tile_count):
                tile_y = tile_data[tile_idx, 0]
                tile_x = tile_data[tile_idx, 1]
                tile_h = tile_data[tile_idx, 2]
                tile_w = tile_data[tile_idx, 3]

                if (
                    i >= tile_y
                    and i < tile_y + tile_h
                    and j >= tile_x
                    and j < tile_x + tile_w
                ):
                    in_tile = True
                    break

            # Apply permutation if in tile
            if in_tile:
                # Store original values
                channels = ti.Vector([r, g, b])

                # Apply permutation
                r_new = channels[perm_r]
                g_new = channels[perm_g]
                b_new = channels[perm_b]

                dest[batch, i, j] = ti.Vector([r_new, g_new, b_new, a])
            else:
                # Copy unchanged
                dest[batch, i, j] = pixel


class BandSwapTaichiOperation(BaseTaichiOperation):
    """
    Taichi band swap operation for end-to-end GPU pipeline.

    Simulates packet mis-identification errors in satellite downlink by
    swapping RGB channels in randomly placed rectangular tiles.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation modifies pixels based on tile membership, so it does
    not support in-place execution (reading and writing could conflict
    at tile boundaries).

    Example:
        >>> op = BandSwapTaichiOperation()
        >>> params = {"tile_count": 5, "permutation": "BGR", "seed": 42}
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize band swap operation."""
        super().__init__("band_swap_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Band swap reads from multiple regions and writes to overlapping
        tiles, so in-place execution could cause conflicts.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate band swap parameters.

        Expected params:
        - tile_count: int - number of tiles (1 to 50)
        - permutation: str - channel swap pattern (e.g., "BGR", "GRB")
        - tile_size_range: list[float] - [min, max] tile size fractions (optional)
        - seed: int - random seed (optional)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or invalid

        """
        if "tile_count" not in params:
            msg = "Band swap requires 'tile_count' parameter"
            raise ValueError(msg)

        tile_count = params["tile_count"]
        if not isinstance(tile_count, int) or not (
            MIN_TILE_COUNT <= tile_count <= MAX_TILE_COUNT
        ):
            msg = (
                f"tile_count must be an integer between {MIN_TILE_COUNT} "
                f"and {MAX_TILE_COUNT}, got {tile_count}"
            )
            raise ValueError(msg)

        if "permutation" not in params:
            msg = "Band swap requires 'permutation' parameter"
            raise ValueError(msg)

        permutation = params["permutation"]
        if permutation not in VALID_PERMUTATIONS:
            valid_list = ", ".join(VALID_PERMUTATIONS.keys())
            msg = f"Permutation must be one of: {valid_list}, got {permutation}"
            raise ValueError(msg)

        if "tile_size_range" in params:
            tile_size_range = params["tile_size_range"]
            if (
                not isinstance(tile_size_range, (list, tuple))
                or len(tile_size_range) != 2
            ):
                msg = "tile_size_range must be a list/tuple of two numbers [min, max]"
                raise ValueError(msg)
            min_size, max_size = tile_size_range
            if not isinstance(min_size, (int, float)) or not isinstance(
                max_size, (int, float)
            ):
                msg = "tile_size_range values must be numbers"
                raise ValueError(msg)
            if not (MIN_TILE_SIZE <= min_size <= MAX_TILE_SIZE) or not (
                MIN_TILE_SIZE <= max_size <= MAX_TILE_SIZE
            ):
                msg = (
                    f"tile_size_range values must be between {MIN_TILE_SIZE} "
                    f"and {MAX_TILE_SIZE}"
                )
                raise ValueError(msg)
            if min_size > max_size:
                msg = "tile_size_range min must be <= max"
                raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer"
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
        Apply band swap on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for band swap (empty dict expected)
            params: Must contain 'tile_count', 'permutation', optional 'seed'
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        tile_count = int(params["tile_count"])
        permutation = params["permutation"]
        tile_size_range = params.get("tile_size_range", [0.05, 0.2])
        seed = params.get("seed", 0)

        # Generate tile positions on CPU
        rng = np.random.default_rng(seed)
        tile_positions = np.zeros((tile_count, 4), dtype=np.int32)

        for i in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(height * tile_fraction))
            tile_w = max(1, int(width * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, height - tile_h + 1))
            x = rng.integers(0, max(1, width - tile_w + 1))

            tile_positions[i] = [y, x, tile_h, tile_w]

        # Transfer tile data to GPU
        tile_field = ti.field(dtype=ti.i32, shape=(tile_count, 4))
        tile_field.from_numpy(tile_positions)

        # Get permutation mapping
        perm_indices = VALID_PERMUTATIONS[permutation]

        # Execute kernel (batch_idx=0 for single image)
        _band_swap_kernel(
            source,
            dest,
            tile_field,
            perm_indices[0],
            perm_indices[1],
            perm_indices[2],
            0,
            height,
            width,
            tile_count,
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
            params: Must contain 'tile_count', 'permutation', optional 'seed'

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        tile_count = int(params["tile_count"])
        permutation = params["permutation"]
        tile_size_range = params.get("tile_size_range", [0.05, 0.2])
        seed = params.get("seed", 0)

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Copy input image
        result = image.copy()
        h, w = image.shape[:2]

        # Get permutation indices
        perm_indices = VALID_PERMUTATIONS[permutation]

        # Generate and apply random tiles
        for _ in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, h - tile_h + 1))
            x = rng.integers(0, max(1, w - tile_w + 1))

            # Apply permutation to this tile
            result[y : y + tile_h, x : x + tile_w] = result[
                y : y + tile_h, x : x + tile_w, perm_indices
            ]

        return result.astype(np.float32)

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the band swap kernel
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))

        # Create minimal tile data (1 tile)
        dummy_tiles = ti.field(dtype=ti.i32, shape=(1, 4))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Initialize dummy tile (covers entire 2x2)
        dummy_tiles[0, 0] = 0  # y
        dummy_tiles[0, 1] = 0  # x
        dummy_tiles[0, 2] = 2  # height
        dummy_tiles[0, 3] = 2  # width

        # Trigger compilation with BGR permutation
        _band_swap_kernel(dummy_src, dummy_dst, dummy_tiles, 2, 1, 0, 0, 2, 2, 1)

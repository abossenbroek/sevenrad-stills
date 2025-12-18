"""
Taichi end-to-end pipeline buffer corruption operation.

Buffer corruption for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
Simulates cosmic ray hits causing single-event upsets in satellite memory.
"""

from typing import Any, Literal

import numpy as np

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    from sevenrad_stills.operations.taichi_kernels.random import rand_float

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Constants
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 20
MIN_SEVERITY = 0.0
MAX_SEVERITY = 1.0
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0

# Valid corruption types
VALID_CORRUPTION_TYPES = {"xor", "invert", "channel_shuffle"}

# Type-safe corruption type alias
CorruptionType = Literal["xor", "invert", "channel_shuffle"]


# Define kernels only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _buffer_corruption_kernel_xor(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        tiles_field: ti.template(),  # type: ignore[valid-type]
        xor_magnitude: ti.i32,
        seed: ti.i32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
        num_tiles: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for XOR-mode buffer corruption.

        Applies bitwise XOR with random values to pixels within corrupted tiles.
        Preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            tiles_field: Field containing tile bounds (num_tiles, 4) as [y, x, h, w]
            xor_magnitude: Maximum XOR value (0-255), scaled by severity
            seed: Random seed for reproducible corruption
            batch: Batch index
            height: Image height
            width: Image width
            num_tiles: Number of tiles to corrupt

        """
        for i, j in ti.ndrange(height, width):
            pixel = source[batch, i, j]
            r = ti.cast(pixel[0] * 255.0, ti.i32)
            g = ti.cast(pixel[1] * 255.0, ti.i32)
            b = ti.cast(pixel[2] * 255.0, ti.i32)
            a = pixel[3]  # Preserve alpha

            # Check if pixel is in any corrupted tile
            in_tile = 0
            for tile_idx in range(num_tiles):
                tile_y = tiles_field[tile_idx, 0]
                tile_x = tiles_field[tile_idx, 1]
                tile_h = tiles_field[tile_idx, 2]
                tile_w = tiles_field[tile_idx, 3]

                if (
                    i >= tile_y
                    and i < tile_y + tile_h
                    and j >= tile_x
                    and j < tile_x + tile_w
                ):
                    in_tile = 1

            if in_tile == 1 and xor_magnitude > 0:
                # Generate reproducible random XOR masks for each channel
                xor_r = ti.cast(rand_float(i, j, seed) * xor_magnitude, ti.i32)
                xor_g = ti.cast(rand_float(i, j, seed + 1) * xor_magnitude, ti.i32)
                xor_b = ti.cast(rand_float(i, j, seed + 2) * xor_magnitude, ti.i32)

                r = (r ^ xor_r) & 0xFF
                g = (g ^ xor_g) & 0xFF
                b = (b ^ xor_b) & 0xFF

            # Write result
            dest[batch, i, j] = ti.Vector(
                [r / 255.0, g / 255.0, b / 255.0, a], dt=ti.f32
            )

    @ti.kernel  # type: ignore[misc]
    def _buffer_corruption_kernel_invert(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        tiles_field: ti.template(),  # type: ignore[valid-type]
        severity: ti.f32,
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
        num_tiles: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for invert-mode buffer corruption.

        Applies bitwise inversion to pixels within corrupted tiles,
        blended by severity. Preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            tiles_field: Field containing tile bounds (num_tiles, 4) as [y, x, h, w]
            severity: Blending factor (0.0 to 1.0)
            batch: Batch index
            height: Image height
            width: Image width
            num_tiles: Number of tiles to corrupt

        """
        for i, j in ti.ndrange(height, width):
            pixel = source[batch, i, j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            a = pixel[3]  # Preserve alpha

            # Check if pixel is in any corrupted tile
            in_tile = 0
            for tile_idx in range(num_tiles):
                tile_y = tiles_field[tile_idx, 0]
                tile_x = tiles_field[tile_idx, 1]
                tile_h = tiles_field[tile_idx, 2]
                tile_w = tiles_field[tile_idx, 3]

                if (
                    i >= tile_y
                    and i < tile_y + tile_h
                    and j >= tile_x
                    and j < tile_x + tile_w
                ):
                    in_tile = 1

            if in_tile == 1 and severity > 0.0:
                # Bitwise inversion blended by severity
                r_inv = 1.0 - r
                g_inv = 1.0 - g
                b_inv = 1.0 - b

                r = r * (1.0 - severity) + r_inv * severity
                g = g * (1.0 - severity) + g_inv * severity
                b = b * (1.0 - severity) + b_inv * severity

            dest[batch, i, j] = ti.Vector([r, g, b, a], dt=ti.f32)

    @ti.kernel  # type: ignore[misc]
    def _buffer_corruption_kernel_shuffle(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        tiles_field: ti.template(),  # type: ignore[valid-type]
        permutations_field: ti.template(),  # type: ignore[valid-type]
        batch: ti.i32,
        height: ti.i32,
        width: ti.i32,
        num_tiles: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for shuffle-mode buffer corruption.

        Applies RGB channel permutation to pixels within corrupted tiles.
        Preserves alpha channel.

        Args:
            source: Input Vector.field(4) with shape (batch, height, width)
            dest: Output Vector.field(4) with same shape
            tiles_field: Field containing tile bounds (num_tiles, 4) as [y, x, h, w]
            permutations_field: Field containing channel permutations (num_tiles, 3)
            batch: Batch index
            height: Image height
            width: Image width
            num_tiles: Number of tiles to corrupt

        """
        for i, j in ti.ndrange(height, width):
            pixel = source[batch, i, j]
            channels = ti.Vector([pixel[0], pixel[1], pixel[2]], dt=ti.f32)
            a = pixel[3]  # Preserve alpha

            # Check if pixel is in any corrupted tile
            tile_found = -1
            for tile_idx in range(num_tiles):
                tile_y = tiles_field[tile_idx, 0]
                tile_x = tiles_field[tile_idx, 1]
                tile_h = tiles_field[tile_idx, 2]
                tile_w = tiles_field[tile_idx, 3]

                if (
                    i >= tile_y
                    and i < tile_y + tile_h
                    and j >= tile_x
                    and j < tile_x + tile_w
                ):
                    tile_found = tile_idx

            if tile_found >= 0:
                # Apply permutation for this tile
                perm_0 = permutations_field[tile_found, 0]
                perm_1 = permutations_field[tile_found, 1]
                perm_2 = permutations_field[tile_found, 2]

                r = channels[perm_0]
                g = channels[perm_1]
                b = channels[perm_2]

                dest[batch, i, j] = ti.Vector([r, g, b, a], dt=ti.f32)
            else:
                dest[batch, i, j] = ti.Vector(
                    [channels[0], channels[1], channels[2], a], dt=ti.f32
                )


class BufferCorruptionTaichiOperation(BaseTaichiOperation):
    """
    Taichi buffer corruption for end-to-end GPU pipeline.

    Simulates cosmic ray hits on satellite memory causing single-event upsets.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation does not support in-place execution because shuffle mode
    may need to read neighbor data.

    Example:
        >>> op = BufferCorruptionTaichiOperation()
        >>> params = {
        ...     "corruption_type": "xor",
        ...     "tile_count": 5,
        ...     "severity": 0.7,
        ...     "tile_size_range": [0.05, 0.2],
        ...     "seed": 42,
        ... }
        >>> op.apply_to_field(source, dest, {}, params, height, width)

    """

    def __init__(self) -> None:
        """Initialize buffer corruption operation."""
        super().__init__("buffer_corruption_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Buffer corruption does not support in-place because shuffle mode
        may need to read from source while writing to dest.

        Returns:
            False - this operation does not support in-place execution.

        """
        return False

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate buffer corruption parameters.

        Expected params:
        - corruption_type: str - "xor", "invert", or "channel_shuffle"
        - tile_count: int - number of tiles to corrupt (1-20)
        - severity: float - corruption intensity (0.0-1.0)
        - tile_size_range: list[float] - [min, max] tile size fractions (optional)
        - seed: int - random seed for reproducibility (optional)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are missing or invalid

        """
        if "corruption_type" not in params:
            msg = "Buffer corruption requires 'corruption_type' parameter"
            raise ValueError(msg)

        corruption_type = params["corruption_type"]

        # Backward compatibility: accept "shuffle" as alias for "channel_shuffle"
        if corruption_type == "shuffle":
            params["corruption_type"] = "channel_shuffle"
            corruption_type = "channel_shuffle"

        if corruption_type not in VALID_CORRUPTION_TYPES:
            valid_list = ", ".join(sorted(VALID_CORRUPTION_TYPES))
            msg = f"corruption_type must be one of: {valid_list}"
            raise ValueError(msg)

        if "tile_count" not in params:
            msg = "Buffer corruption requires 'tile_count' parameter"
            raise ValueError(msg)

        tile_count = params["tile_count"]
        if not isinstance(tile_count, int):
            msg = f"tile_count must be an integer, got {type(tile_count)}"
            raise ValueError(msg)

        if not (MIN_TILE_COUNT <= tile_count <= MAX_TILE_COUNT):
            msg = (
                f"tile_count must be between {MIN_TILE_COUNT} "
                f"and {MAX_TILE_COUNT}, got {tile_count}"
            )
            raise ValueError(msg)

        if "severity" not in params:
            msg = "Buffer corruption requires 'severity' parameter"
            raise ValueError(msg)

        severity = params["severity"]
        if not isinstance(severity, (int, float)):
            msg = f"severity must be a number, got {type(severity)}"
            raise ValueError(msg)

        if not (MIN_SEVERITY <= severity <= MAX_SEVERITY):
            msg = (
                f"severity must be between {MIN_SEVERITY} and {MAX_SEVERITY}, "
                f"got {severity}"
            )
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
                msg = "tile_size_range min must be less than or equal to max"
                raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "seed must be an integer"
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
        Apply buffer corruption on GPU fields.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with same shape
            temp_fields: Not used for buffer corruption (empty dict expected)
            params: Must contain corruption_type, tile_count, severity, etc.
            height: Image height
            width: Image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        corruption_type: CorruptionType = params["corruption_type"]
        tile_count = int(params["tile_count"])
        severity = float(params["severity"])
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed = params.get("seed", 0)

        # Generate tiles on CPU
        rng = np.random.default_rng(seed)
        tiles = []

        for _ in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(height * tile_fraction))
            tile_w = max(1, int(width * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, height - tile_h + 1))
            x = rng.integers(0, max(1, width - tile_w + 1))

            tiles.append([y, x, tile_h, tile_w])

        # Create tiles field
        tiles_array = np.array(tiles, dtype=np.int32)
        tiles_field = ti.field(dtype=ti.i32, shape=(tile_count, 4))
        tiles_field.from_numpy(tiles_array)

        # Execute appropriate kernel
        if corruption_type == "xor":
            xor_magnitude = int(255 * severity)
            _buffer_corruption_kernel_xor(
                source,
                dest,
                tiles_field,
                xor_magnitude,
                seed,
                0,
                height,
                width,
                tile_count,
            )
        elif corruption_type == "invert":
            _buffer_corruption_kernel_invert(
                source, dest, tiles_field, severity, 0, height, width, tile_count
            )
        elif corruption_type == "channel_shuffle":
            # Generate permutations for each tile
            permutations = []
            for _ in range(tile_count):
                if rng.random() < severity:
                    perm = rng.permutation(3).astype(np.int32)
                else:
                    perm = np.array([0, 1, 2], dtype=np.int32)
                permutations.append(perm)

            permutations_array = np.array(permutations, dtype=np.int32)
            permutations_field = ti.field(dtype=ti.i32, shape=(tile_count, 3))
            permutations_field.from_numpy(permutations_array)

            _buffer_corruption_kernel_shuffle(
                source,
                dest,
                tiles_field,
                permutations_field,
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
            params: Must contain corruption_type, tile_count, severity, etc.

        Returns:
            Processed image as numpy array (H, W, 3) float32 in [0, 1]

        """
        corruption_type: CorruptionType = params["corruption_type"]
        tile_count = int(params["tile_count"])
        severity = float(params["severity"])
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed = params.get("seed", 0)

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Copy input
        result = image.copy()
        h, w = result.shape[:2]

        # Generate random tiles and apply corruption
        for _ in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, h - tile_h + 1))
            x = rng.integers(0, max(1, w - tile_w + 1))

            # Extract tile
            tile = result[y : y + tile_h, x : x + tile_w]

            # Apply corruption based on type
            if corruption_type == "xor":
                # Convert to uint8, apply XOR, convert back
                xor_magnitude = int(255 * severity)
                if xor_magnitude > 0:
                    tile_uint8 = (tile * 255).astype(np.uint8)
                    xor_mask = rng.integers(
                        0, xor_magnitude + 1, size=tile.shape, dtype=np.uint8
                    )
                    corrupted_tile_uint8 = np.bitwise_xor(tile_uint8, xor_mask)
                    corrupted_tile = corrupted_tile_uint8.astype(np.float32) / 255.0
                else:
                    corrupted_tile = tile

            elif corruption_type == "invert":
                # Bitwise inversion scaled by severity
                if severity > 0:
                    inverted = 1.0 - tile
                    # Blend between original and inverted based on severity
                    corrupted_tile = tile * (1.0 - severity) + inverted * severity
                else:
                    corrupted_tile = tile

            elif corruption_type == "channel_shuffle":
                # Random channel permutation per tile
                # Severity controls probability of shuffling
                if rng.random() < severity and tile.ndim == 3 and tile.shape[2] >= 3:
                    # Generate random permutation
                    perm = rng.permutation(3)
                    corrupted_tile = tile.copy()
                    corrupted_tile[..., :3] = tile[..., perm]
                else:
                    corrupted_tile = tile
            else:
                corrupted_tile = tile

            # Put corrupted tile back
            result[y : y + tile_h, x : x + tile_w] = corrupted_tile

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 2x2 dummy fields.

        Called by warmup() to pre-compile the buffer corruption kernels
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 2x2 fields for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))

        # Initialize with dummy data
        for i in range(2):
            for j in range(2):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Create dummy tiles field (single tile covering the entire 2x2)
        dummy_tiles = ti.field(dtype=ti.i32, shape=(1, 4))
        dummy_tiles.from_numpy(np.array([[0, 0, 2, 2]], dtype=np.int32))

        # Create dummy permutations field
        dummy_perms = ti.field(dtype=ti.i32, shape=(1, 3))
        dummy_perms.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))

        # Trigger compilation for all kernels
        _buffer_corruption_kernel_xor(
            dummy_src, dummy_dst, dummy_tiles, 128, 0, 0, 2, 2, 1
        )
        _buffer_corruption_kernel_invert(
            dummy_src, dummy_dst, dummy_tiles, 0.5, 0, 2, 2, 1
        )
        _buffer_corruption_kernel_shuffle(
            dummy_src, dummy_dst, dummy_tiles, dummy_perms, 0, 2, 2, 1
        )

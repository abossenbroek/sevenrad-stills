"""
Pure Metal-accelerated band swap operation with custom Metal kernels.

Simulates errors in satellite data transmission where packet headers are corrupted,
causing band/channel data to be misinterpreted and swapped in rectangular regions.
Uses native Metal compute shaders for maximum GPU performance on Apple Silicon.
"""

from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

try:
    import Metal
    import objc

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

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

# Metal shader source code
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void apply_band_swap_tiles(
    device uchar *img [[buffer(0)]],
    device const int *tiles [[buffer(1)]],
    constant int &height [[buffer(2)]],
    constant int &width [[buffer(3)]],
    constant int &num_tiles [[buffer(4)]],
    constant int &max_tile_h [[buffer(5)]],
    constant int &max_tile_w [[buffer(6)]],
    constant int &perm_0 [[buffer(7)]],
    constant int &perm_1 [[buffer(8)]],
    constant int &perm_2 [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int tile_idx = gid.x;
    int local_y = gid.y;
    int local_x = gid.z;

    // Check if this thread is processing a valid tile
    if (tile_idx >= num_tiles) return;

    // Get tile bounds: [y_start, y_end, x_start, x_end]
    int tile_base = tile_idx * 4;
    int y_start = tiles[tile_base + 0];
    int y_end = tiles[tile_base + 1];
    int x_start = tiles[tile_base + 2];
    int x_end = tiles[tile_base + 3];

    // Calculate global coordinates
    int y = y_start + local_y;
    int x = x_start + local_x;

    // Check if within tile bounds
    if (y >= y_end || x >= x_end) return;

    // Calculate pixel index in flattened RGB array
    int idx = (y * width + x) * 3;

    // Read original RGB values
    uchar r = img[idx + 0];
    uchar g = img[idx + 1];
    uchar b = img[idx + 2];

    // Create channel array for permutation
    uchar channels[3] = {r, g, b};

    // Apply permutation in-place
    img[idx + 0] = channels[perm_0];
    img[idx + 1] = channels[perm_1];
    img[idx + 2] = channels[perm_2];
}
"""


class MetalComputeEngine:
    """Metal compute engine for band swap operation."""

    def __init__(self) -> None:
        """Initialize Metal device and compile shaders."""
        if not METAL_AVAILABLE:
            msg = "Metal framework not available. Install pyobjc-framework-Metal."
            raise RuntimeError(msg)

        # Get default Metal device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            msg = "No Metal-capable GPU found."
            raise RuntimeError(msg)

        # Create command queue
        self.command_queue = self.device.newCommandQueue()

        # Compile shader library
        try:
            options = Metal.MTLCompileOptions.new()
            self.library, error = self.device.newLibraryWithSource_options_error_(
                METAL_SHADER_SOURCE, options, None
            )
            if error:
                msg = f"Metal shader compilation failed: {error}"
                raise RuntimeError(msg)
        except Exception as e:
            msg = f"Failed to compile Metal shaders: {e}"
            raise RuntimeError(msg) from e

        # Create pipeline state
        self.pipeline = self._create_pipeline("apply_band_swap_tiles")

    def _create_pipeline(self, function_name: str) -> object:
        """Create compute pipeline state for a shader function."""
        function = self.library.newFunctionWithName_(function_name)
        if function is None:
            msg = f"Metal function '{function_name}' not found in library."
            raise RuntimeError(msg)

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if error:
            msg = f"Failed to create pipeline for '{function_name}': {error}"
            raise RuntimeError(msg)

        return pipeline

    def apply_band_swap(
        self,
        img_array: np.ndarray,
        tiles: np.ndarray,
        perm_indices: list[int],
    ) -> None:
        """
        Apply band swapping to tiles using Metal compute shader.

        Tiles are processed sequentially to match CPU's behavior with overlapping tiles.
        Within each tile, pixels are processed in parallel on GPU for performance.

        Args:
            img_array: RGB image array (H, W, 3) - modified in-place
            tiles: Tile coordinates array (N, 4) with [y_start, y_end, x_start, x_end]
            perm_indices: Permutation indices [perm_0, perm_1, perm_2]

        """
        h, w = img_array.shape[:2]
        num_tiles = tiles.shape[0]

        # Ensure contiguous array
        img_flat = np.ascontiguousarray(img_array).ravel()

        # Create Metal buffer for image (reused across tile iterations)
        img_buffer = self.device.newBufferWithBytes_length_options_(
            img_flat.tobytes(),
            img_flat.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create buffers for scalar parameters (reused across all tiles)
        height_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([h], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        width_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([w], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        perm_0_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([perm_indices[0]], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        perm_1_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([perm_indices[1]], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        perm_2_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([perm_indices[2]], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )

        # Process tiles sequentially to match CPU behavior with overlapping tiles
        for tile_idx in range(num_tiles):
            # Get single tile coords
            single_tile = tiles[tile_idx : tile_idx + 1].copy()
            tile_h = int(single_tile[0, 1] - single_tile[0, 0])
            tile_w = int(single_tile[0, 3] - single_tile[0, 2])

            # Create buffer for this tile
            tiles_buffer = self.device.newBufferWithBytes_length_options_(
                single_tile.ravel().astype(np.int32).tobytes(),
                single_tile.nbytes,
                Metal.MTLResourceStorageModeShared,
            )

            num_tiles_buffer = self.device.newBufferWithBytes_length_options_(
                np.array([1], dtype=np.int32).tobytes(),  # Always 1 tile per dispatch
                4,
                Metal.MTLResourceStorageModeShared,
            )
            max_tile_h_buffer = self.device.newBufferWithBytes_length_options_(
                np.array([tile_h], dtype=np.int32).tobytes(),
                4,
                Metal.MTLResourceStorageModeShared,
            )
            max_tile_w_buffer = self.device.newBufferWithBytes_length_options_(
                np.array([tile_w], dtype=np.int32).tobytes(),
                4,
                Metal.MTLResourceStorageModeShared,
            )

            # Create command buffer and encoder
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()

            # Set pipeline and buffers
            compute_encoder.setComputePipelineState_(self.pipeline)
            compute_encoder.setBuffer_offset_atIndex_(img_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(tiles_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(height_buffer, 0, 2)
            compute_encoder.setBuffer_offset_atIndex_(width_buffer, 0, 3)
            compute_encoder.setBuffer_offset_atIndex_(num_tiles_buffer, 0, 4)
            compute_encoder.setBuffer_offset_atIndex_(max_tile_h_buffer, 0, 5)
            compute_encoder.setBuffer_offset_atIndex_(max_tile_w_buffer, 0, 6)
            compute_encoder.setBuffer_offset_atIndex_(perm_0_buffer, 0, 7)
            compute_encoder.setBuffer_offset_atIndex_(perm_1_buffer, 0, 8)
            compute_encoder.setBuffer_offset_atIndex_(perm_2_buffer, 0, 9)

            # Calculate thread groups - 3D dispatch: (1 tile, tile_h, tile_w)
            thread_group_size = Metal.MTLSize(1, 16, 16)
            grid_size = Metal.MTLSize(
                1,  # Single tile
                (tile_h + 15) // 16 * 16,
                (tile_w + 15) // 16 * 16,
            )

            # Dispatch compute shader
            compute_encoder.dispatchThreads_threadsPerThreadgroup_(
                grid_size, thread_group_size
            )
            compute_encoder.endEncoding()

            # Execute and wait (ensures tiles are processed in order)
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

        # Copy results back after all tiles are processed
        result_bytes = img_buffer.contents().as_buffer(img_flat.nbytes)
        result_array: np.ndarray = np.frombuffer(result_bytes, dtype=np.uint8).copy()

        # Update original array in-place
        img_array[:] = result_array.reshape(img_array.shape)


class BandSwapMetalOperation(BaseImageOperation):
    """
    Metal-accelerated band/channel swapping operation.

    Simulates errors in satellite data transmission where packet headers are corrupted,
    causing band/channel data to be misinterpreted and swapped in rectangular regions.
    Uses native Metal compute shaders for maximum GPU performance on Apple Silicon.

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

    Performance: Pure Metal implementation provides maximum GPU performance by
    using native Metal compute shaders without intermediate frameworks like Taichi.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated band swap operation."""
        super().__init__("band_swap_metal")
        self._engine: MetalComputeEngine | None = None

    @property
    def engine(self) -> MetalComputeEngine:
        """Lazy-initialize Metal compute engine."""
        if self._engine is None:
            self._engine = MetalComputeEngine()
        return self._engine

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
        Apply Metal-accelerated band swapping to random tiles in the image.

        The tile positions and sizes are generated on CPU using numpy's RNG,
        while the actual band permutation within each tile is executed on GPU
        using Metal kernels for parallel processing.

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

        # Create random number generator (CPU-side for determinism)
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

        # Generate tile coordinates sequentially (matching CPU RNG order)
        # IMPORTANT: Must match CPU's for loop RNG call order
        tile_list = []
        for _ in range(tile_count):
            # Random tile size (matches CPU's rng.uniform() call order)
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position (matches CPU's rng.integers() call order)
            y_start = rng.integers(0, max(1, h - tile_h + 1))
            x_start = rng.integers(0, max(1, w - tile_w + 1))

            # Append tile as [y_start, y_end, x_start, x_end]
            tile_list.append([y_start, y_start + tile_h, x_start, x_start + tile_w])

        # Convert to numpy array
        tiles: np.ndarray = np.array(tile_list, dtype=np.int32)

        # Apply band swap using Metal (modifies rgb in-place)
        # Note: Sequential processing matches CPU behavior
        self.engine.apply_band_swap(rgb, tiles, perm_indices)

        # Recombine with alpha if needed
        output_array = np.dstack([rgb, alpha]) if alpha is not None else rgb

        return Image.fromarray(output_array)

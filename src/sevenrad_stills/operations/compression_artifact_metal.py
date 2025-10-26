"""
Pure Metal implementation of compression artifact operation.

Uses custom Metal shaders for maximum GPU performance on macOS.
Expected to be 5-10x faster than Taichi and 2-3x faster than CPU PIL.
"""

import ctypes
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants (same as other implementations)
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 30
MIN_QUALITY = 1
MAX_QUALITY = 20
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0


class MetalCompressionArtifact:
    """
    Wrapper for the Metal compression artifact implementation.

    Loads the compiled Swift dylib and provides Python interface via C FFI.
    """

    def __init__(self) -> None:
        """Initialize Metal backend."""
        if platform.system() != "Darwin":
            msg = "Metal backend is only available on macOS"
            raise RuntimeError(msg)

        # Load the Metal library
        lib_path = Path(__file__).parent.parent / "metal_kernels" / "build"
        dylib_path = lib_path / "libMetalCompressionArtifact.dylib"

        if not dylib_path.exists():
            msg = (
                f"Metal library not found at {dylib_path}. "
                "Please run: src/sevenrad_stills/metal_kernels/build.sh"
            )
            raise FileNotFoundError(msg)

        # Load the dynamic library
        self.lib = ctypes.CDLL(str(dylib_path))

        # Setup C function signature
        self._setup_c_function()

    def _setup_c_function(self) -> None:
        """Set up C function signature for metal_compression_apply."""
        self.apply_func = self.lib.metal_compression_apply
        self.apply_func.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # imageData
            ctypes.c_int32,  # width
            ctypes.c_int32,  # height
            ctypes.c_int32,  # channels
            ctypes.POINTER(ctypes.c_int32),  # tilesFlat
            ctypes.c_int32,  # numTiles
            ctypes.c_int32,  # quality
        ]
        self.apply_func.restype = ctypes.c_bool

    def apply(
        self,
        image_data: np.ndarray,
        tiles: list[list[int]],
        quality: int,
    ) -> np.ndarray:
        """
        Apply compression artifacts using Metal.

        Args:
            image_data: Image as numpy array (H, W, C) where C is 3 or 4
            tiles: List of [y_start, y_end, x_start, x_end] for each tile
            quality: JPEG quality (1-20)

        Returns:
            Modified image as numpy array (modified in-place)

        """
        # Ensure contiguous array
        if not image_data.flags["C_CONTIGUOUS"]:
            image_data = np.ascontiguousarray(image_data)

        h, w = image_data.shape[:2]
        num_dims = 3
        channels = image_data.shape[2] if len(image_data.shape) == num_dims else 1

        # Flatten tiles array for C interface
        # Format: [y_start, y_end, x_start, x_end, y_start, y_end, x_start, x_end, ...]
        tiles_flat = []
        for tile in tiles:
            tiles_flat.extend(tile)

        tiles_array = (ctypes.c_int32 * len(tiles_flat))(*tiles_flat)

        # Get pointer to image data
        image_ptr = image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # Call the C function
        success = self.apply_func(
            image_ptr,
            ctypes.c_int32(w),
            ctypes.c_int32(h),
            ctypes.c_int32(channels),
            tiles_array,
            ctypes.c_int32(len(tiles)),
            ctypes.c_int32(quality),
        )

        if not success:
            msg = "Metal operation failed"
            raise RuntimeError(msg)

        return image_data  # Modified in-place by Metal


class CompressionArtifactMetalOperation(BaseImageOperation):
    """
    Pure Metal GPU-accelerated compression artifact operation.

    Uses custom Metal shaders for maximum performance on macOS.
    This implementation is optimized for:
    - Single-pass compute pipeline (all tiles in one dispatch)
    - SIMD-optimized DCT/IDCT operations
    - Zero-copy texture operations
    - Minimal CPU-GPU data transfer

    Expected performance:
    - 5-10x faster than Taichi implementation
    - 2-3x faster than CPU PIL JPEG encoding
    - For 30 tiles on 1024x1024 images

    Architecture:
    - Metal compute shader processes all tiles in parallel
    - Each threadgroup handles one 8x8 block
    - Shared quantization matrices in constant buffers
    - In-place texture modification
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated compression artifact operation."""
        super().__init__("compression_artifact_metal")
        try:
            self._metal = MetalCompressionArtifact()
        except (FileNotFoundError, RuntimeError) as e:
            msg = f"Failed to initialize Metal backend: {e}"
            raise RuntimeError(msg) from e

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
        """
        Validate parameters for compression artifact operation.

        Args:
            params: A dictionary containing:
                - tile_count (int): Number of corrupted tiles (1 to 30)
                - quality (int): JPEG quality for corrupted tiles (1 to 20)
                - tile_size_range (list, optional): [min, max] tile size as
                  fractions of image dimensions (default: [0.05, 0.2])
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "tile_count" not in params:
            msg = "Compression artifact operation requires 'tile_count' parameter."
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

        if "quality" not in params:
            msg = "Compression artifact operation requires 'quality' parameter."
            raise ValueError(msg)
        quality = params["quality"]
        if not isinstance(quality, int) or not (MIN_QUALITY <= quality <= MAX_QUALITY):
            msg = f"quality must be an integer between {MIN_QUALITY} and {MAX_QUALITY}."
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
        Apply Metal-accelerated compression artifacts to random tiles.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'tile_count', 'quality',
                    optional 'tile_size_range', and optional 'seed'.

        Returns:
            The PIL Image with compression artifacts applied to random tiles.

        """
        self.validate_params(params)

        tile_count: int = params["tile_count"]
        quality: int = params["quality"]
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert to array
        img_array = np.array(image)

        # Handle different modes
        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            has_alpha = True
        elif image.mode == "RGB":
            rgb = img_array.copy()
            alpha = None
            has_alpha = False
        else:  # Grayscale
            rgb = np.stack([img_array] * 3, axis=-1).copy()
            alpha = None
            has_alpha = False

        h, w = rgb.shape[:2]

        # Generate tile coordinates
        tile_fractions = rng.uniform(
            tile_size_range[0], tile_size_range[1], size=tile_count
        )
        tile_heights = np.maximum(8, (h * tile_fractions).astype(np.int32))
        tile_widths = np.maximum(8, (w * tile_fractions).astype(np.int32))

        y_starts = rng.integers(
            0, np.maximum(1, h - tile_heights + 1), size=tile_count, dtype=np.int32
        )
        x_starts = rng.integers(
            0, np.maximum(1, w - tile_widths + 1), size=tile_count, dtype=np.int32
        )

        # Build tile list [y_start, y_end, x_start, x_end]
        tiles = [
            [
                int(y_starts[i]),
                int(y_starts[i] + tile_heights[i]),
                int(x_starts[i]),
                int(x_starts[i] + tile_widths[i]),
            ]
            for i in range(tile_count)
        ]

        # Apply Metal operation
        rgb = self._metal.apply(rgb, tiles, quality)

        # Convert back to grayscale if needed
        if image.mode == "L":
            output_array = rgb[..., 0]
        elif has_alpha:
            output_array = np.dstack([rgb, alpha])
        else:
            output_array = rgb

        return Image.fromarray(output_array)

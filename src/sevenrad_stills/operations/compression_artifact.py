"""
Compression artifact operation for simulating on-board JPEG encoder failures.

Simulates localized JPEG compression failures in satellite on-board processing where
specific memory regions or tiles get over-compressed due to encoder malfunctions,
buffer overflow, or rate control errors.
"""

import io
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 30
MIN_QUALITY = 1
MAX_QUALITY = 20
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0


class CompressionArtifactOperation(BaseImageOperation):
    """
    Apply localized JPEG compression artifacts to simulate encoder failures.

    Simulates failure modes in satellite on-board JPEG compression hardware:
    - Rate control failures causing localized over-compression
    - Buffer overflow forcing quality degradation in specific tiles
    - Encoder chip failures affecting random image regions
    - Memory constraints causing selective quality reduction

    Unlike global compression (which affects the entire image uniformly), this
    operation creates characteristic rectangular regions with severe JPEG
    artifacts (8x8 DCT blocking, color bleeding, detail loss) while surrounding
    areas remain intact.

    Real-world scenarios:
    - On-board processing running out of memory mid-compression
    - Faulty encoder hardware affecting specific tile buffers
    - Rate control algorithm bugs causing inconsistent quality
    - Thermal throttling reducing compression quality for portions of the image

    The result is visually distinctive: sharp boundaries between pristine and
    heavily compressed regions, creating a "patchwork" quality degradation.

    Performance Note:
    This operation involves JPEG encode/decode cycles for each tile, which is
    computationally expensive. High `tile_count` values (>15) may noticeably
    slow down pipeline execution. This is an acceptable trade-off for the
    high-quality simulation of localized encoder failures.
    """

    def __init__(self) -> None:
        """Initialize the compression artifact operation."""
        super().__init__("compression_artifact")

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
        Apply localized JPEG compression artifacts to random tiles.

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

        # Convert to array and handle different modes
        img_array = np.array(image)

        # Compression artifacts only make sense for RGB/RGBA
        # For grayscale, we'll convert to RGB, compress, and convert back
        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            has_alpha = True
        elif image.mode == "RGB":
            rgb = img_array.copy()
            alpha = None
            has_alpha = False
        else:  # Grayscale
            # Convert to RGB for compression (JPEG doesn't handle L mode artifacts well)
            rgb = np.stack([img_array] * 3, axis=-1).copy()
            alpha = None
            has_alpha = False

        h, w = rgb.shape[:2]

        # Generate random tiles and apply compression
        for _ in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, h - tile_h + 1))
            x = rng.integers(0, max(1, w - tile_w + 1))

            # Extract tile
            tile = rgb[y : y + tile_h, x : x + tile_w]

            # Compress tile using in-memory JPEG encoding
            buffer = io.BytesIO()
            tile_img = Image.fromarray(tile)
            tile_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            # Decode compressed tile
            compressed_tile = np.array(Image.open(buffer))

            # Put compressed tile back
            rgb[y : y + tile_h, x : x + tile_w] = compressed_tile

        # Convert back to grayscale if needed
        if image.mode == "L":
            # Take single channel from RGB
            # (channels are identical from the stack operation)
            output_array = rgb[..., 0]
        elif has_alpha:
            # Recombine with alpha
            output_array = np.dstack([rgb, alpha])
        else:
            output_array = rgb

        return Image.fromarray(output_array)

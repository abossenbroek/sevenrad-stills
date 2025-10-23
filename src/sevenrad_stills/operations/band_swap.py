"""
Band swap operation for simulating packet mis-identification in satellite downlink.

Simulates errors in satellite data transmission where packet headers are corrupted,
causing band/channel data to be misinterpreted and swapped in rectangular regions.
"""

from typing import Any, Literal

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

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


class BandSwapOperation(BaseImageOperation):
    """
    Apply band/channel swapping to simulate packet mis-identification errors.

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
    """

    def __init__(self) -> None:
        """Initialize the band swap operation."""
        super().__init__("band_swap")

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
        Apply band swapping to random tiles in the image.

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

        # Create random number generator
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

        # Generate random tiles and apply swaps
        for _ in range(tile_count):
            # Random tile size
            tile_fraction = rng.uniform(tile_size_range[0], tile_size_range[1])
            tile_h = max(1, int(h * tile_fraction))
            tile_w = max(1, int(w * tile_fraction))

            # Random tile position
            y = rng.integers(0, max(1, h - tile_h + 1))
            x = rng.integers(0, max(1, w - tile_w + 1))

            # Apply permutation to this tile
            rgb[y : y + tile_h, x : x + tile_w] = rgb[
                y : y + tile_h, x : x + tile_w, perm_indices
            ]

        # Recombine with alpha if needed
        output_array = np.dstack([rgb, alpha]) if alpha is not None else rgb

        return Image.fromarray(output_array)

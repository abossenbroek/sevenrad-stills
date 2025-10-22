"""
Buffer corruption operation for simulating single-event upsets in satellite memory.

Simulates cosmic ray hits on satellite on-board memory causing bit flips in image
data buffers before transmission. Creates localized rectangular "glitch blocks"
with bitwise corruptions.
"""

from typing import Any, Literal

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_TILE_COUNT = 1
MAX_TILE_COUNT = 20
MIN_SEVERITY = 0.0
MAX_SEVERITY = 1.0
MIN_TILE_SIZE = 0.01
MAX_TILE_SIZE = 1.0

# Valid corruption types
VALID_CORRUPTION_TYPES = {"xor", "invert", "channel_shuffle"}


class BufferCorruptionOperation(BaseImageOperation):
    """
    Apply buffer corruption to simulate single-event upsets from cosmic rays.

    Simulates high-energy particle impacts on satellite memory chips causing
    "Single Event Upsets" (SEUs) - bit flips in the image data buffer before
    it's compressed or transmitted.

    In orbital environments, cosmic rays and solar particle events can flip
    individual bits in DRAM or SRAM. When this happens to image buffer memory:
    - A single bit flip can corrupt an entire tile of data
    - The corruption affects raw pixel values before encoding
    - Creates characteristic "glitch blocks" - rectangular regions with
      garbled pixels, wrong colors, or horizontal displacement

    This differs from transmission errors (which create line dropouts) because
    the buffer corruption affects the actual pixel data structure, creating
    visually distinctive artifacts.

    Corruption modes:
    - xor: Bitwise XOR with random pattern (simulates bit flips)
    - invert: Bitwise inversion (simulates register corruption)
    - channel_shuffle: Random RGB channel permutation per tile (simulates
      pointer corruption in multi-planar buffers)
    """

    def __init__(self) -> None:
        """Initialize the buffer corruption operation."""
        super().__init__("buffer_corruption")

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
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
        Apply buffer corruption to random tiles in the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'tile_count', 'corruption_type', 'severity',
                    optional 'tile_size_range', and optional 'seed'.

        Returns:
            The PIL Image with buffer corruption applied to random tiles.

        """
        self.validate_params(params)

        tile_count: int = params["tile_count"]
        corruption_type: Literal["xor", "invert", "channel_shuffle"] = params[
            "corruption_type"
        ]
        severity: float = params["severity"]
        tile_size_range: list[float] = params.get("tile_size_range", [0.05, 0.2])
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert to array and separate RGB from alpha if needed
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
        elif image.mode == "RGB":
            rgb = img_array.copy()
            alpha = None
        else:  # Grayscale
            rgb = img_array.copy()
            alpha = None

        h, w = rgb.shape[:2]

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
            tile = rgb[y : y + tile_h, x : x + tile_w]

            # Apply corruption based on type
            if corruption_type == "xor":
                # XOR with random byte pattern scaled by severity
                # Severity controls the magnitude of the XOR mask
                xor_magnitude = int(255 * severity)
                if xor_magnitude > 0:
                    xor_mask = rng.integers(
                        0, xor_magnitude + 1, size=tile.shape, dtype=np.uint8
                    )
                    corrupted_tile = np.bitwise_xor(tile, xor_mask)
                else:
                    corrupted_tile = tile

            elif corruption_type == "invert":
                # Bitwise inversion scaled by severity
                # Severity controls how much inversion to apply
                if severity > 0:
                    inverted = np.bitwise_not(tile)
                    # Blend between original and inverted based on severity
                    corrupted_tile = (
                        tile * (1 - severity) + inverted * severity
                    ).astype(np.uint8)
                else:
                    corrupted_tile = tile

            # Random channel permutation per tile
            # Severity controls probability of shuffling (all-or-nothing per tile)
            elif rng.random() < severity and tile.ndim == 3 and tile.shape[2] >= 3:  # noqa: PLR2004
                # Generate random permutation
                perm = rng.permutation(3)
                # Only shuffle first 3 channels (RGB), preserve others if exist
                corrupted_tile = tile.copy()
                corrupted_tile[..., :3] = tile[..., perm]
            else:
                corrupted_tile = tile

            # Put corrupted tile back
            rgb[y : y + tile_h, x : x + tile_w] = corrupted_tile

        # Recombine with alpha if needed
        # Note: alpha is only set for RGBA images (see line 166)
        output_array = np.dstack([rgb, alpha]) if alpha is not None else rgb

        return Image.fromarray(output_array)

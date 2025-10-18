"""
JPEG compression operation with configurable quality and block artifacts.

Provides precise control over compression artifacts including block size,
color subsampling, and quality levels to achieve various levels of degradation.
"""

import io
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_QUALITY = 1
MAX_QUALITY = 100
DEFAULT_QUALITY = 85


class CompressionOperation(BaseImageOperation):
    """
    Apply JPEG compression with configurable quality and subsampling.

    Supports multiple compression levels and chroma subsampling modes to
    create various levels of block artifacts and color banding.

    Subsampling modes:
    - 0 (4:4:4): No subsampling - highest quality, minimal artifacts
    - 1 (4:2:2): Moderate subsampling - visible artifacts in color transitions
    - 2 (4:2:0): Heavy subsampling - severe 8x8 blocking, default JPEG behavior

    Quality range 1-100:
    - 1-15: Severe compression, heavy blocking and banding
    - 16-50: Moderate compression, visible artifacts
    - 51-85: Standard compression, balanced quality/size
    - 86-100: High quality, minimal artifacts
    """

    def __init__(self) -> None:
        """Initialize compression operation."""
        super().__init__("compression")

    def _validate_gamma(self, gamma: object) -> None:
        """
        Validate gamma correction parameter.

        Args:
            gamma: Gamma value to validate

        Raises:
            ValueError: If gamma is invalid

        """
        if not isinstance(gamma, (int, float)):
            msg = f"Gamma must be a number, got {type(gamma)}"
            raise ValueError(msg)
        if gamma <= 0:
            msg = f"Gamma must be positive, got {gamma}"
            raise ValueError(msg)

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate compression operation parameters.

        Expected params:
        - quality: int (1-100) - JPEG quality level
        - subsampling: int (0, 1, or 2) - Chroma subsampling mode (optional)
        - optimize: bool - Apply JPEG optimization (optional, default True)
        - gamma: float | None - Gamma correction factor (optional, default None)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "quality" not in params:
            msg = "Compression operation requires 'quality' parameter"
            raise ValueError(msg)

        quality = params["quality"]
        if not isinstance(quality, int):
            msg = f"Quality must be an integer, got {type(quality)}"
            raise ValueError(msg)
        if not MIN_QUALITY <= quality <= MAX_QUALITY:
            msg = (
                f"Quality must be between {MIN_QUALITY} and {MAX_QUALITY}, "
                f"got {quality}"
            )
            raise ValueError(msg)

        # Validate subsampling if provided
        if "subsampling" in params:
            subsampling = params["subsampling"]
            if not isinstance(subsampling, int):
                msg = f"Subsampling must be an integer, got {type(subsampling)}"
                raise ValueError(msg)
            if subsampling not in (0, 1, 2):
                msg = f"Subsampling must be 0, 1, or 2, got {subsampling}"
                raise ValueError(msg)

        # Validate optimize if provided
        if "optimize" in params:
            optimize = params["optimize"]
            if not isinstance(optimize, bool):
                msg = f"Optimize must be a boolean, got {type(optimize)}"
                raise ValueError(msg)

        # Validate gamma if provided
        if "gamma" in params and params["gamma"] is not None:
            self._validate_gamma(params["gamma"])

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply JPEG compression to image, with optional gamma correction.

        The compression is applied by saving to an in-memory buffer and
        reloading, simulating a real save/load cycle. If gamma is provided,
        it is applied before the compression step.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Compressed PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        quality: int = params["quality"]
        subsampling: int = params.get("subsampling", 2)  # Default to 4:2:0
        optimize: bool = params.get("optimize", True)
        gamma: float | None = params.get("gamma")

        # Convert to RGB if necessary (JPEG doesn't support transparency)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Apply gamma correction if specified
        if gamma is not None:
            # Gamma correction logic inspired by degradr (MIT License)
            # https://github.com/nhauber99/degradr
            img_array = np.array(image, dtype=np.float32) / 255.0
            gamma_corrected_array = np.power(img_array, gamma)
            img_uint8 = (np.clip(gamma_corrected_array, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            image = Image.fromarray(img_uint8)

        # Apply compression via in-memory save/load cycle
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=subsampling,
            optimize=optimize,
        )
        buffer.seek(0)
        compressed_image = Image.open(buffer)

        # Load the image data to ensure buffer can be closed
        compressed_image.load()
        buffer.close()

        return compressed_image

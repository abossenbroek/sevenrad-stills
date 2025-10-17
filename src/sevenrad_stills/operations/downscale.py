"""
Resolution downscaling operation for creating pixelation effects.

Provides control over scale factors and resampling methods to achieve
various levels of pixelation and loss of detail.
"""

from typing import Any

from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_SCALE = 0.01
MAX_SCALE = 1.0

# Resampling method mapping
RESAMPLING_METHODS = {
    "nearest": Image.Resampling.NEAREST,  # Harsh pixelation
    "bilinear": Image.Resampling.BILINEAR,  # Smooth downscaling
    "bicubic": Image.Resampling.BICUBIC,  # High-quality downscaling
    "lanczos": Image.Resampling.LANCZOS,  # Highest quality
    "box": Image.Resampling.BOX,  # Simple box filter
}


class DownscaleOperation(BaseImageOperation):
    """
    Downscale image resolution to create pixelation effects.

    Supports downscaling to a fraction of original size, with optional
    upscaling back to original dimensions to create visible pixelation.

    Scale factor:
    - 0.01-0.10: Extreme pixelation, heavily degraded
    - 0.10-0.25: Heavy pixelation, architectural details lost
    - 0.25-0.50: Moderate pixelation, visible block structures
    - 0.50-1.00: Subtle quality reduction

    Resampling methods:
    - nearest: Maximum pixelation, harsh block edges
    - bilinear: Softer pixelation with blended edges
    - bicubic/lanczos: Smooth downscaling, minimal harsh artifacts
    """

    def __init__(self) -> None:
        """Initialize downscale operation."""
        super().__init__("downscale")

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
        """
        Validate downscale operation parameters.

        Expected params:
        - scale: float (0.01-1.0) - Scale factor for downscaling
        - upscale: bool - Whether to upscale back to original size (optional)
        - downscale_method: str - Resampling method for downscaling (optional)
        - upscale_method: str - Resampling method for upscaling (optional)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "scale" not in params:
            msg = "Downscale operation requires 'scale' parameter"
            raise ValueError(msg)

        scale = params["scale"]
        if not isinstance(scale, (int, float)):
            msg = f"Scale must be a number, got {type(scale)}"
            raise ValueError(msg)
        if not MIN_SCALE <= scale <= MAX_SCALE:
            msg = f"Scale must be between {MIN_SCALE} and {MAX_SCALE}, got {scale}"
            raise ValueError(msg)

        # Validate upscale if provided
        if "upscale" in params:
            upscale = params["upscale"]
            if not isinstance(upscale, bool):
                msg = f"Upscale must be a boolean, got {type(upscale)}"
                raise ValueError(msg)

        # Validate downscale_method if provided
        if "downscale_method" in params:
            method = params["downscale_method"]
            if not isinstance(method, str):
                msg = f"Downscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = f"Invalid downscale method '{method}'. Available: {available}"
                raise ValueError(msg)

        # Validate upscale_method if provided
        if "upscale_method" in params:
            method = params["upscale_method"]
            if not isinstance(method, str):
                msg = f"Upscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = f"Invalid upscale method '{method}'. Available: {available}"
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply downscaling to image.

        Args:
            image: Input PIL Image
            params: Operation parameters (validated)

        Returns:
            Downscaled (and optionally upscaled) PIL Image

        """
        # Validate parameters first
        self.validate_params(params)

        scale: float = params["scale"]
        upscale: bool = params.get("upscale", True)  # Default to upscaling
        downscale_method_name: str = params.get("downscale_method", "bicubic")
        upscale_method_name: str = params.get("upscale_method", "nearest")

        # Get resampling methods
        downscale_method = RESAMPLING_METHODS[downscale_method_name]
        upscale_method = RESAMPLING_METHODS[upscale_method_name]

        # Store original size
        original_size = image.size

        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Downscale
        downscaled = image.resize((new_width, new_height), resample=downscale_method)

        # Upscale back if requested
        if upscale:
            return downscaled.resize(original_size, resample=upscale_method)

        return downscaled

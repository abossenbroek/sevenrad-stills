"""
Metal-accelerated corduroy striping operation using MLX.

Simulates "corduroy" or "banding" artifacts from push-broom and whisk-broom
scanners where individual detector elements have slightly different sensitivity
due to calibration drift or manufacturing variations. Uses MLX for Metal
acceleration with seamless numpy integration.
"""

from typing import Any, Literal

import numpy as np
from PIL import Image
from skimage.util import img_as_float32, img_as_ubyte

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError(
        "MLX is required for Metal acceleration. Install with: pip install mlx"
    ) from e

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0


class CorduroyMetalOperation(BaseImageOperation):
    """
    Apply Metal-accelerated corduroy striping to simulate detector calibration errors.

    Creates subtle vertical or horizontal banding by simulating "hot" (overly
    sensitive) and "cold" (less sensitive) detector elements in a push-broom
    or whisk-broom scanner array.

    In real satellite sensors, each detector in a linear array may have slightly
    different gain due to:
    - Manufacturing variation in sensitivity
    - Calibration drift over time
    - Temperature effects on individual detectors
    - Radiation damage accumulation

    This creates characteristic "corduroy" patterns - subtle repeating lines
    of slightly brighter or darker pixels running perpendicular to the scan
    direction.

    Performance: Uses MLX for Metal acceleration with automatic numpy/Metal
    conversion, providing excellent GPU performance on Apple Silicon.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated corduroy striping operation."""
        super().__init__("corduroy_metal")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for corduroy striping operation.

        Args:
            params: A dictionary containing:
                - strength (float): Striping intensity (0.0 to 1.0), maps to
                  multiplier of 1.0 ± strength x 0.2
                - orientation (str): 'vertical' or 'horizontal' line direction
                - density (float): Proportion of lines affected (0.0 to 1.0)
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "strength" not in params:
            msg = "Corduroy operation requires 'strength' parameter."
            raise ValueError(msg)
        strength = params["strength"]
        if not isinstance(strength, (int, float)) or not (
            MIN_STRENGTH <= strength <= MAX_STRENGTH
        ):
            msg = f"Strength must be a float between {MIN_STRENGTH} and {MAX_STRENGTH}."
            raise ValueError(msg)

        if "orientation" not in params:
            msg = "Corduroy operation requires 'orientation' parameter."
            raise ValueError(msg)
        orientation = params["orientation"]
        if orientation not in ("vertical", "horizontal"):
            msg = "Orientation must be 'vertical' or 'horizontal'."
            raise ValueError(msg)

        if "density" not in params:
            msg = "Corduroy operation requires 'density' parameter."
            raise ValueError(msg)
        density = params["density"]
        if not isinstance(density, (int, float)) or not (
            MIN_DENSITY <= density <= MAX_DENSITY
        ):
            msg = f"Density must be a float between {MIN_DENSITY} and {MAX_DENSITY}."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply Metal-accelerated corduroy striping to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'strength', 'orientation', 'density',
                    and optional 'seed'.

        Returns:
            The PIL Image with corduroy striping applied.

        """
        self.validate_params(params)
        strength: float = params["strength"]
        orientation: Literal["vertical", "horizontal"] = params["orientation"]
        density: float = params["density"]
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert to float array (0.0 to 1.0) using skimage utility
        img_float = img_as_float32(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_float[..., :3].copy()
            alpha = img_float[..., 3:4]
            h, w = rgb.shape[:2]
        else:
            rgb = img_float.copy()
            alpha = None
            h, w = rgb.shape[:2]

        # Determine number of lines to affect
        if orientation == "vertical":
            num_lines = int(density * w)
            total_lines = w
        else:  # horizontal
            num_lines = int(density * h)
            total_lines = h

        if num_lines > 0:
            # Select random lines
            affected_lines = rng.choice(total_lines, size=num_lines, replace=False)

            # Generate random multipliers for each line
            # strength maps to range [1.0 - strength*0.2, 1.0 + strength*0.2]
            multipliers_affected = rng.uniform(
                1.0 - strength * 0.2,
                1.0 + strength * 0.2,
                size=num_lines,
            ).astype(np.float32)

            # Create an array of multipliers, with 1.0 for unaffected lines
            multipliers_array = np.ones(total_lines, dtype=np.float32)
            multipliers_array[affected_lines] = multipliers_affected

            # Apply stripes using MLX for Metal acceleration
            self._apply_stripes_mlx(rgb, multipliers_array, orientation)

        # Recombine with alpha if needed
        if alpha is not None:
            output_float = np.concatenate([rgb, alpha], axis=2)
        else:
            output_float = rgb

        # Convert back to uint8 using skimage utility
        output_array = img_as_ubyte(output_float)
        return Image.fromarray(output_array)

    def _apply_stripes_mlx(
        self,
        img_array: np.ndarray,
        multipliers: np.ndarray,
        orientation: Literal["vertical", "horizontal"],
    ) -> None:
        """
        Apply corduroy striping using MLX Metal acceleration.

        Args:
            img_array: Image array (H, W) or (H, W, 3) - modified in-place
            multipliers: Array of multipliers (W,) for vertical or (H,) for horizontal
            orientation: 'vertical' or 'horizontal'

        """
        # Convert to MLX arrays
        img_mlx = mx.array(img_array)
        mult_mlx = mx.array(multipliers)

        # Apply multipliers based on orientation
        if orientation == "vertical":
            # Broadcast multipliers across height dimension
            # Shape: (H, W, C) * (W,) -> (H, W, C)
            if img_mlx.ndim == 3:  # RGB  # noqa: PLR2004
                # Reshape multipliers to (1, W, 1) for broadcasting
                mult_reshaped = mult_mlx.reshape(1, -1, 1)
            else:  # Grayscale
                # Reshape multipliers to (1, W) for broadcasting
                mult_reshaped = mult_mlx.reshape(1, -1)
        # Broadcast multipliers across width dimension
        elif img_mlx.ndim == 3:  # RGB  # noqa: PLR2004
            # Reshape multipliers to (H, 1, 1) for broadcasting
            mult_reshaped = mult_mlx.reshape(-1, 1, 1)
        else:  # Grayscale
            # Reshape multipliers to (H, 1) for broadcasting
            mult_reshaped = mult_mlx.reshape(-1, 1)

        # Apply multipliers and clamp to [0, 1]
        result_mlx = mx.clip(img_mlx * mult_reshaped, 0.0, 1.0)

        # Convert back to numpy and update in-place
        img_array[:] = np.array(result_mlx)

"""
Noise generation operations.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses NumPy instead of PyTorch for better integration
with PIL-based image processing pipeline.
"""

from typing import Any, Literal

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0


class NoiseOperation(BaseImageOperation):
    """
    Add noise to an image.

    Supports three modes:
    - gaussian: Random pixel-level noise following Gaussian distribution
    - row: Horizontal noise patterns (scan line artifacts)
    - column: Vertical noise patterns
    """

    def __init__(self) -> None:
        """Initialize the noise operation."""
        super().__init__("noise")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the noise operation.

        Args:
            params: A dictionary containing:
                - mode (str): 'gaussian', 'row', or 'column'.
                - amount (float): Noise intensity, typically 0.0 to 1.0.
                - seed (int, optional): Seed for the random number generator.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "mode" not in params:
            msg = "Noise operation requires a 'mode' parameter."
            raise ValueError(msg)
        mode = params["mode"]
        if mode not in ("gaussian", "row", "column"):
            msg = "Mode must be 'gaussian', 'row', or 'column'."
            raise ValueError(msg)

        if "amount" not in params:
            msg = "Noise operation requires an 'amount' parameter."
            raise ValueError(msg)
        amount = params["amount"]
        if not isinstance(amount, (int, float)) or not (
            MIN_AMOUNT <= amount <= MAX_AMOUNT
        ):
            msg = f"Amount must be a float between {MIN_AMOUNT} and {MAX_AMOUNT}."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Add noise to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'mode', 'amount', and optional 'seed'.

        Returns:
            The noisy PIL Image.

        """
        self.validate_params(params)
        mode: Literal["gaussian", "row", "column"] = params["mode"]
        amount: float = params["amount"]
        seed: int | None = params.get("seed")

        rng = np.random.default_rng(seed)
        img_array = np.array(image, dtype=np.float32) / 255.0

        noise = np.zeros_like(img_array)
        h, w = img_array.shape[:2]

        if mode == "gaussian":
            noise = rng.normal(loc=0, scale=amount, size=img_array.shape)
        elif mode == "row":
            # Generate one noise value per row and broadcast it
            # For grayscale, we need 1 channel; for RGB, 3 channels
            num_channels = 1 if img_array.ndim == 2 else img_array.shape[2]  # noqa: PLR2004
            row_noise = rng.uniform(-amount, amount, size=(h, 1, num_channels))
            noise = np.broadcast_to(row_noise, img_array.shape)
        elif mode == "column":
            # Generate one noise value per column and broadcast it
            num_channels = 1 if img_array.ndim == 2 else img_array.shape[2]  # noqa: PLR2004
            col_noise = rng.uniform(-amount, amount, size=(1, w, num_channels))
            noise = np.broadcast_to(col_noise, img_array.shape)

        # Add noise and clip to valid range [0, 1]
        noisy_array = np.clip(img_array + noise, 0.0, 1.0)

        # Convert back to original dtype and then to PIL Image
        output_array = (noisy_array * 255).astype(np.uint8)
        return Image.fromarray(output_array)

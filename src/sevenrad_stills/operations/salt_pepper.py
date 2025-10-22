"""
Salt and pepper noise operation for simulating cosmic ray sensor hits.

Simulates random white and black pixel defects caused by cosmic ray impacts
on CMOS sensors or manufacturing defects in detector arrays.
"""

from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0
MIN_RATIO = 0.0
MAX_RATIO = 1.0


class SaltPepperOperation(BaseImageOperation):
    """
    Add salt and pepper noise to simulate sensor defects.

    Salt and pepper noise represents random pixels outputting maximum (white)
    or minimum (black) values regardless of actual light input. This models:
    - Cosmic ray hits on orbital sensors causing single-event upsets
    - Manufacturing defects creating "hot" or "dead" pixels
    - Radiation damage accumulation in detector arrays

    The effect creates scattered white and black pixels across the image,
    distinct from Gaussian noise by its extreme binary nature.
    """

    def __init__(self) -> None:
        """Initialize the salt and pepper noise operation."""
        super().__init__("salt_pepper")

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for salt and pepper noise operation.

        Args:
            params: A dictionary containing:
                - amount (float): Proportion of pixels affected (0.0 to 1.0).
                - salt_vs_pepper (float): Ratio of salt (white) to pepper (black)
                  pixels, where 0.0 is all pepper, 1.0 is all salt, 0.5 is equal.
                - seed (int, optional): Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "amount" not in params:
            msg = "Salt and pepper operation requires 'amount' parameter."
            raise ValueError(msg)
        amount = params["amount"]
        if not isinstance(amount, (int, float)) or not (
            MIN_AMOUNT <= amount <= MAX_AMOUNT
        ):
            msg = f"Amount must be a float between {MIN_AMOUNT} and {MAX_AMOUNT}."
            raise ValueError(msg)

        if "salt_vs_pepper" not in params:
            msg = "Salt and pepper operation requires 'salt_vs_pepper' parameter."
            raise ValueError(msg)
        salt_vs_pepper = params["salt_vs_pepper"]
        if not isinstance(salt_vs_pepper, (int, float)) or not (
            MIN_RATIO <= salt_vs_pepper <= MAX_RATIO
        ):
            msg = (
                f"salt_vs_pepper must be a float between {MIN_RATIO} "
                f"and {MAX_RATIO}."
            )
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply salt and pepper noise to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'amount', 'salt_vs_pepper', and optional 'seed'.

        Returns:
            The PIL Image with salt and pepper noise applied.

        """
        self.validate_params(params)
        amount: float = params["amount"]
        salt_vs_pepper: float = params["salt_vs_pepper"]
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert image to array
        img_array = np.array(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            h, w = rgb.shape[:2]

            # Generate noise mask for pixels (not channels)
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels)
                noise_x = rng.integers(0, w, size=num_noise_pixels)

                # Determine which are salt vs pepper
                salt_mask = rng.random(num_noise_pixels) < salt_vs_pepper

                # Apply noise (entire pixel, all RGB channels)
                rgb[noise_y[salt_mask], noise_x[salt_mask]] = [255, 255, 255]
                rgb[noise_y[~salt_mask], noise_x[~salt_mask]] = [0, 0, 0]

            # Recombine with alpha
            output_array = np.dstack([rgb, alpha])
        elif image.mode == "RGB":
            rgb = img_array.copy()
            h, w = rgb.shape[:2]

            # Generate noise mask for pixels (not channels)
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels)
                noise_x = rng.integers(0, w, size=num_noise_pixels)

                # Determine which are salt vs pepper
                salt_mask = rng.random(num_noise_pixels) < salt_vs_pepper

                # Apply noise (entire pixel, all RGB channels)
                rgb[noise_y[salt_mask], noise_x[salt_mask]] = [255, 255, 255]
                rgb[noise_y[~salt_mask], noise_x[~salt_mask]] = [0, 0, 0]

            output_array = rgb
        else:  # Grayscale
            gray = img_array.copy()
            h, w = gray.shape

            # Generate noise for grayscale
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels)
                noise_x = rng.integers(0, w, size=num_noise_pixels)

                # Determine which are salt vs pepper
                salt_mask = rng.random(num_noise_pixels) < salt_vs_pepper

                # Apply noise
                gray[noise_y[salt_mask], noise_x[salt_mask]] = 255
                gray[noise_y[~salt_mask], noise_x[~salt_mask]] = 0

            output_array = gray

        return Image.fromarray(output_array)

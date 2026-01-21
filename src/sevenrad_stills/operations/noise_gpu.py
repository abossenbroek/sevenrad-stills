"""
GPU-accelerated noise generation using Taichi.

Provides GPU-accelerated noise application for image degradation operations.
Supports gaussian (per-pixel), row (horizontal banding), and column (vertical banding)
noise modes. Uses Taichi for GPU acceleration of the noise application step.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).
"""

from typing import Any, Literal

import numpy as np
import taichi as ti
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Initialize Taichi - will auto-select GPU if available, fallback to CPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0


@ti.kernel  # type: ignore[misc]
def apply_gaussian_noise_gray(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply gaussian noise to grayscale images.

    Args:
        img: Image array (H, W) to modify in-place
        noise: Noise array (H, W) with same shape as img
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        img[y, x] = ti.math.clamp(img[y, x] + noise[y, x], 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_gaussian_noise_rgb(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply gaussian noise to RGB images.

    Args:
        img: Image array (H, W, 3) to modify in-place
        noise: Noise array (H, W, 3) with same shape as img
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        for c in ti.static(range(3)):
            img[y, x, c] = ti.math.clamp(img[y, x, c] + noise[y, x, c], 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_row_noise_gray(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    row_noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply row-wise noise to grayscale images.

    Args:
        img: Image array (H, W) to modify in-place
        row_noise: Noise values per row (H, 1)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        img[y, x] = ti.math.clamp(img[y, x] + row_noise[y, 0], 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_row_noise_rgb(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    row_noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply row-wise noise to RGB images.

    Args:
        img: Image array (H, W, 3) to modify in-place
        row_noise: Noise values per row (H, 1, 3)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        for c in ti.static(range(3)):
            img[y, x, c] = ti.math.clamp(img[y, x, c] + row_noise[y, 0, c], 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_column_noise_gray(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    col_noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply column-wise noise to grayscale images.

    Args:
        img: Image array (H, W) to modify in-place
        col_noise: Noise values per column (1, W)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        img[y, x] = ti.math.clamp(img[y, x] + col_noise[0, x], 0.0, 1.0)


@ti.kernel  # type: ignore[misc]
def apply_column_noise_rgb(  # type: ignore[no-untyped-def]
    img: ti.types.ndarray(),  # type: ignore[valid-type]
    col_noise: ti.types.ndarray(),  # type: ignore[valid-type]
    height: ti.i32,
    width: ti.i32,
):
    """
    GPU kernel to apply column-wise noise to RGB images.

    Args:
        img: Image array (H, W, 3) to modify in-place
        col_noise: Noise values per column (1, W, 3)
        height: Image height
        width: Image width

    """
    for y, x in ti.ndrange(height, width):
        for c in ti.static(range(3)):
            img[y, x, c] = ti.math.clamp(img[y, x, c] + col_noise[0, x, c], 0.0, 1.0)


class NoiseGPUOperation(BaseImageOperation):
    """
    Add GPU-accelerated noise to an image.

    Supports three modes:
    - gaussian: Random pixel-level noise following Gaussian distribution
    - row: Horizontal noise patterns (scan line artifacts)
    - column: Vertical noise patterns

    Random number generation is performed on the CPU for reproducibility
    (ensuring consistent results with the same seed), while noise application
    (add + clamp operations) is parallelized on the GPU for performance.

    Performance: GPU acceleration provides significant speedup for large images
    by parallelizing the noise application across all pixels.
    """

    def __init__(self) -> None:
        """Initialize the GPU-accelerated noise operation."""
        super().__init__("noise_gpu")

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
        Add GPU-accelerated noise to the image.

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

        # Random number generation on CPU for reproducibility
        rng = np.random.default_rng(seed)
        img_array = np.array(image, dtype=np.float32) / 255.0

        h, w = img_array.shape[:2]

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3:4]

            # Generate noise for RGB channels only (CPU-side)
            if mode == "gaussian":
                noise_rgb = rng.normal(loc=0, scale=amount, size=rgb.shape).astype(
                    np.float32
                )
                apply_gaussian_noise_rgb(rgb, noise_rgb, h, w)
            elif mode == "row":
                row_noise = rng.uniform(-amount, amount, size=(h, 1, 3)).astype(
                    np.float32
                )
                apply_row_noise_rgb(rgb, row_noise, h, w)
            else:  # column
                col_noise = rng.uniform(-amount, amount, size=(1, w, 3)).astype(
                    np.float32
                )
                apply_column_noise_rgb(rgb, col_noise, h, w)

            # Recombine with alpha
            output_array = np.concatenate([rgb, alpha], axis=2)
        else:
            # For grayscale and RGB, apply noise normally
            rgb = img_array.copy()
            num_channels = 1 if img_array.ndim == 2 else img_array.shape[2]  # noqa: PLR2004

            if mode == "gaussian":
                noise = rng.normal(loc=0, scale=amount, size=rgb.shape).astype(
                    np.float32
                )
                if num_channels == 1:
                    apply_gaussian_noise_gray(rgb, noise, h, w)
                else:
                    apply_gaussian_noise_rgb(rgb, noise, h, w)
            elif mode == "row":
                if num_channels == 1:
                    row_noise = rng.uniform(-amount, amount, size=(h, 1)).astype(
                        np.float32
                    )
                    apply_row_noise_gray(rgb, row_noise, h, w)
                else:
                    row_noise = rng.uniform(
                        -amount, amount, size=(h, 1, num_channels)
                    ).astype(np.float32)
                    apply_row_noise_rgb(rgb, row_noise, h, w)
            elif num_channels == 1:
                col_noise = rng.uniform(-amount, amount, size=(1, w)).astype(np.float32)
                apply_column_noise_gray(rgb, col_noise, h, w)
            else:
                col_noise = rng.uniform(
                    -amount, amount, size=(1, w, num_channels)
                ).astype(np.float32)
                apply_column_noise_rgb(rgb, col_noise, h, w)

            output_array = rgb

        # Convert back to uint8 and return as PIL Image
        output_array = (output_array * 255).astype(np.uint8)
        return Image.fromarray(output_array)

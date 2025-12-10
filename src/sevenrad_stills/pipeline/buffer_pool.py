"""
Buffer pool management for Taichi GPU pipeline execution.

Manages shape-indexed buffer pairs for efficient GPU-based image processing
without reallocation overhead during pipeline execution.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Taichi imports with fallback for testing
try:
    import taichi as ti

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    TAICHI_AVAILABLE = False

# Type alias for Taichi fields
TaichiField = Any

# Constants for channel dimensions
RGB_CHANNELS = 3
RGBA_CHANNELS = 4


@dataclass
class BufferPair:
    """Pair of ping-pong buffers for a specific shape."""

    a: TaichiField
    b: TaichiField


class PipelineBufferPool:
    """
    Shape-indexed buffer pool for GPU pipeline execution.

    Manages pre-allocated buffer pairs indexed by (batch, height, width) to avoid
    expensive GPU memory allocations during pipeline execution. Each buffer uses
    ti.Vector.field(4, dtype=ti.f32) for RGBA channels, optimized for GPU vector
    operations.

    Example:
        >>> pool = PipelineBufferPool()
        >>> pool.ensure_shape(1, 1080, 1920)
        >>> pair = pool.get_pair(1, 1080, 1920)
        >>> # Use pair.a and pair.b for ping-pong processing
        >>> pool.release()

    """

    def __init__(self) -> None:
        """Initialize empty buffer pool."""
        self._pools: dict[tuple[int, int, int], BufferPair] = {}
        self._current_shape: tuple[int, int, int] | None = None

    def ensure_shape(self, batch: int, height: int, width: int) -> None:
        """
        Pre-allocate buffer pair for shape if not exists.

        Creates a pair of ti.Vector.field buffers (4 channels for RGBA) with the
        specified dimensions. If buffers for this shape already exist, this is a no-op.

        Args:
            batch: Batch size (number of images)
            height: Image height in pixels
            width: Image width in pixels

        Raises:
            RuntimeError: If Taichi is not initialized or not available

        """
        if ti is None:
            msg = "Taichi is not available. Cannot allocate GPU buffers."
            raise RuntimeError(msg)

        key = (batch, height, width)
        if key not in self._pools:
            # Use ti.Vector.field for RGBA (4 channels)
            # This is optimal for GPU vector processors
            buffer_a = ti.Vector.field(4, dtype=ti.f32, shape=(batch, height, width))
            buffer_b = ti.Vector.field(4, dtype=ti.f32, shape=(batch, height, width))
            self._pools[key] = BufferPair(a=buffer_a, b=buffer_b)

        self._current_shape = key

    def get_pair(self, batch: int, height: int, width: int) -> BufferPair:
        """
        Get buffer pair for shape.

        Retrieves the pre-allocated buffer pair for the specified shape. The shape
        must have been previously allocated via ensure_shape().

        Args:
            batch: Batch size
            height: Image height
            width: Image width

        Returns:
            BufferPair containing the two buffers for ping-pong processing

        Raises:
            KeyError: If no buffers exist for this shape

        """
        key = (batch, height, width)
        if key not in self._pools:
            msg = f"No buffer allocated for shape {key}. " f"Call ensure_shape() first."
            raise KeyError(msg)
        return self._pools[key]

    def load_image(
        self,
        image_array: np.ndarray,
        buffer: TaichiField,
        batch_idx: int = 0,
    ) -> None:
        """
        Load numpy image array into Taichi field.

        Converts numpy array (H, W, C) format to Taichi Vector field format and
        loads it into the specified batch index of the buffer.

        Args:
            image_array: Input image as numpy array (H, W, 3) or (H, W, 4)
            buffer: Target Taichi Vector.field buffer
            batch_idx: Index in batch dimension (default: 0)

        Raises:
            ValueError: If image shape doesn't match buffer shape or has wrong channels

        """
        if ti is None:
            msg = "Taichi is not available."
            raise RuntimeError(msg)

        # Validate image shape
        if image_array.ndim != RGB_CHANNELS:
            msg = f"Expected 3D array (H, W, C), got shape {image_array.shape}"
            raise ValueError(msg)

        height, width, channels = image_array.shape

        # Convert to RGBA if needed (add alpha channel)
        if channels == RGB_CHANNELS:
            # Add alpha channel with full opacity
            rgba = np.ones((height, width, RGBA_CHANNELS), dtype=image_array.dtype)
            rgba[:, :, :RGB_CHANNELS] = image_array
            image_array = rgba
        elif channels != RGBA_CHANNELS:
            msg = f"Image must have 3 or 4 channels, got {channels}"
            raise ValueError(msg)

        # Normalize to [0, 1] float range if needed
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        else:
            image_array = image_array.astype(np.float32)

        # Copy to GPU field
        # Taichi Vector.field expects indexing as [batch, i, j] with 4-vector values
        for i in range(height):
            for j in range(width):
                buffer[batch_idx, i, j] = image_array[i, j]

    def extract_result(
        self,
        buffer: TaichiField,
        batch_idx: int = 0,
    ) -> np.ndarray:
        """
        Extract numpy array from Taichi field.

        Converts Taichi Vector field back to numpy array format (H, W, C).
        Returns RGB (3 channels) with alpha channel discarded.

        Args:
            buffer: Source Taichi Vector.field buffer
            batch_idx: Index in batch dimension (default: 0)

        Returns:
            Image as numpy array (H, W, 3) in uint8 format [0, 255]

        """
        if ti is None:
            msg = "Taichi is not available."
            raise RuntimeError(msg)

        # Get buffer shape from the field
        batch, height, width = buffer.shape

        # Extract data from GPU
        result = np.zeros((height, width, 4), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                result[i, j] = buffer[batch_idx, i, j].to_numpy()

        # Convert back to uint8 [0, 255] range
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        # Return RGB only (discard alpha)
        rgb_result: np.ndarray = result[:, :, :3]
        return rgb_result

    def release(self) -> None:
        """
        Release all allocated GPU buffers.

        Clears the buffer pool. This should be called during cleanup or when
        switching to a different pipeline configuration.

        Note:
            Taichi manages field deallocation automatically, so this primarily
            clears Python references to allow garbage collection.

        """
        self._pools.clear()
        self._current_shape = None

    def get_allocated_shapes(self) -> list[tuple[int, int, int]]:
        """
        Get list of all allocated buffer shapes.

        Returns:
            List of (batch, height, width) tuples for allocated buffers

        """
        return list(self._pools.keys())

    @property
    def current_shape(self) -> tuple[int, int, int] | None:
        """
        Get the most recently allocated or accessed shape.

        Returns:
            Current shape as (batch, height, width) or None if no shapes allocated

        """
        return self._current_shape

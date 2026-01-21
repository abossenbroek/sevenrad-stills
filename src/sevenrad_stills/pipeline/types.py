"""
Shared types for the pipeline module.

This module contains dataclasses used across pipeline and operations modules.
It is kept separate to avoid circular imports between operations and pipeline.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TempFieldSpec:
    """
    Specification for temporary field required by an operation.

    Temporary fields are used by operations that need additional GPU buffers
    beyond the source and destination buffers. For example, separable filters
    might need a temporary buffer for intermediate results.

    Attributes:
        name: Identifier for this temporary field
        shape_factor: Tuple of (height_mult, width_mult, channels) where
                     multipliers are applied to the source image dimensions
        dtype: Data type for the field (e.g., "f32", "i32")

    """

    name: str
    shape_factor: tuple[float, float, int]  # (h_mult, w_mult, channels)
    dtype: str = "f32"


@dataclass
class BufferPair:
    """
    Ping-pong buffer pair for GPU operations.

    Manages two GPU buffers that can be swapped to enable efficient
    multi-pass operations without extra allocations. The "current"
    buffer alternates between a and b on each swap.

    Attributes:
        a: First Taichi field buffer
        b: Second Taichi field buffer
        current_is_a: True if 'a' is the current source buffer

    Example:
        >>> buffers = BufferPair(field_a, field_b)
        >>> operation.apply_to_field(buffers.source, buffers.dest, ...)
        >>> buffers.swap()
        >>> operation.apply_to_field(buffers.source, buffers.dest, ...)

    """

    a: Any  # ti.Vector.field
    b: Any  # ti.Vector.field
    current_is_a: bool = True

    def swap(self) -> None:
        """Swap the current and destination buffers."""
        self.current_is_a = not self.current_is_a

    @property
    def source(self) -> Any:  # noqa: ANN401
        """
        Get the current source buffer.

        Returns:
            The buffer marked as current (a or b)

        """
        return self.a if self.current_is_a else self.b

    @property
    def dest(self) -> Any:  # noqa: ANN401
        """
        Get the current destination buffer.

        Returns:
            The buffer not marked as current (b or a)

        """
        return self.b if self.current_is_a else self.a

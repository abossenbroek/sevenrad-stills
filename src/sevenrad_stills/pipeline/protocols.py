"""
Protocols and dataclasses for Taichi GPU pipeline operations.

This module defines the interfaces and data structures needed for GPU-accelerated
image operations in the pipeline system. It provides:
- TaichiFieldOperation protocol for GPU operations
- TempFieldSpec for temporary field requirements
- BufferPair for ping-pong buffer management
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Re-export from types module for backward compatibility
from sevenrad_stills.pipeline.types import BufferPair, TempFieldSpec

__all__ = ["BufferPair", "TaichiFieldOperation", "TempFieldSpec"]


@runtime_checkable
class TaichiFieldOperation(Protocol):
    """
    Protocol for Taichi GPU-accelerated image operations.

    This protocol defines the interface that all GPU operations must implement
    to work with the Taichi pipeline system. Operations manipulate Taichi fields
    directly for maximum performance, avoiding CPU-GPU transfers.

    Example Implementation:
        >>> class MyGPUOperation:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_operation"
        ...
        ...     @property
        ...     def supports_inplace(self) -> bool:
        ...         return True  # Can write to source buffer
        ...
        ...     @property
        ...     def output_shape_factor(self) -> tuple[float, float]:
        ...         return (1.0, 1.0)  # No dimension changes
        ...
        ...     @property
        ...     def temp_field_requirements(self) -> list[TempFieldSpec]:
        ...         return []  # No temporary fields needed
        ...
        ...     def apply_to_field(self, source, dest, temp_fields, params, height, width):
        ...         # GPU kernel implementation
        ...         ...
        ...
        ...     def reference_numpy(self, image, params):
        ...         # CPU reference for testing
        ...         ...
        ...
        ...     def validate_params(self, params):
        ...         if "required_param" not in params:
        ...             raise ValueError("Missing required_param")
    """

    @property
    def name(self) -> str:
        """
        Get the operation name.

        Returns:
            Unique identifier for this operation

        """
        ...

    @property
    def supports_inplace(self) -> bool:
        """
        Check if operation can write to the source buffer.

        Inplace operations can reuse the source buffer as destination,
        reducing memory requirements. Operations that need to read from
        multiple locations or perform non-local operations typically
        cannot be done inplace.

        Returns:
            True if operation can modify source buffer directly

        """
        ...

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        """
        Get the output dimension scaling factors.

        Most operations preserve dimensions (1.0, 1.0), but some like
        downscaling or padding change the output size.

        Returns:
            Tuple of (height_factor, width_factor) where factors are
            multiplied by input dimensions to get output dimensions

        """
        ...

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """
        Get specifications for required temporary fields.

        Some operations need additional GPU buffers for intermediate results.
        For example, a separable filter might need a buffer for the horizontal
        pass before the vertical pass.

        Returns:
            List of temporary field specifications, empty if none needed

        """
        ...

    def apply_to_field(
        self,
        source: Any,
        dest: Any,
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply operation to Taichi fields.

        This is the core GPU operation that manipulates Taichi fields directly.
        It should launch Taichi kernels to process the image data on GPU.

        Args:
            source: Source Taichi field (input image)
            dest: Destination Taichi field (output image)
            temp_fields: Dictionary of temporary fields by name
            params: Operation-specific parameters
            height: Image height in pixels
            width: Image width in pixels

        Note:
            For inplace operations, source and dest may be the same field.
            The operation must handle this case correctly.

        """
        ...

    def reference_numpy(
        self, image: NDArray[np.float32], params: dict[str, Any]
    ) -> NDArray[np.float32]:
        """
        Apply operation using NumPy for testing and validation.

        Provides a CPU-based reference implementation that can be used to
        verify GPU results. This should produce identical results to
        apply_to_field within floating-point precision.

        Args:
            image: Input image as numpy array (shape: H x W x C, values 0.0-1.0)
            params: Operation-specific parameters

        Returns:
            Output image as numpy array (same format as input)

        """
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate operation parameters.

        Should raise ValueError with descriptive message if parameters
        are invalid. This is called before apply_to_field to catch
        errors early.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid or missing required fields

        """
        ...

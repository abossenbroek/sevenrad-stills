"""
Abstract base class for Taichi GPU operations.

This module provides the foundation for all Taichi-accelerated operations,
defining the interface and common functionality for GPU-based image processing.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sevenrad_stills.pipeline.protocols import TempFieldSpec


class BaseTaichiOperation(ABC):
    """
    Abstract base class for Taichi GPU operations.

    Provides common functionality and interface for operations that execute
    on GPU using Taichi fields. All Taichi operations should extend this class.

    Example:
        >>> class MyOperation(BaseTaichiOperation):
        ...     def __init__(self):
        ...         super().__init__("my_operation")
        ...
        ...     def apply_to_field(self, source, dest, temp_fields, params, h, w):
        ...         my_kernel(source, dest, h, w)
        ...
        ...     def reference_numpy(self, image, params):
        ...         return image  # NumPy implementation

    """

    def __init__(self, name: str) -> None:
        """
        Initialize operation with name.

        Args:
            name: Unique identifier for this operation.

        """
        self._name = name
        self._kernels_compiled = False

    @property
    def name(self) -> str:
        """
        Get operation name.

        Returns:
            The operation's unique identifier.

        """
        return self._name

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can safely write to source buffer.

        Override and return True for element-wise operations where output[i,j]
        only depends on input[i,j] (e.g., color adjustments, noise).

        Returns:
            False by default. Override to return True for in-place operations.

        """
        return False

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        """
        Output shape relative to input: (height_factor, width_factor).

        Override for dimension-changing operations like downscale.
        Default (1.0, 1.0) means same dimensions as input.

        Returns:
            Tuple of (height_factor, width_factor) multipliers.

        """
        return (1.0, 1.0)

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """
        List of temporary fields required by this operation.

        Override for operations needing scratch space (e.g., separable blur).

        Returns:
            Empty list by default. Override to specify temporary field needs.

        """
        return []

    @abstractmethod
    def apply_to_field(  # noqa: PLR0913
        self,
        source: Any,  # ti.Vector.field  # noqa: ANN401
        dest: Any,  # ti.Vector.field  # noqa: ANN401
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply operation on GPU fields.

        Args:
            source: Input Taichi Vector.field (batch, height, width) with 4 channels
            dest: Output Taichi Vector.field (same shape)
            temp_fields: Pre-allocated temporary fields by name
            params: Operation-specific parameters
            height: Image height
            width: Image width

        """
        ...

    @abstractmethod
    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        NumPy reference implementation for testing.

        Must produce identical results to apply_to_field for correctness testing.

        Args:
            image: Input image as numpy array (H, W, 3) uint8
            params: Operation-specific parameters

        Returns:
            Processed image as numpy array (H, W, 3) uint8

        """
        ...

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate operation parameters.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        ...

    def warmup(self) -> None:
        """
        Force JIT compilation with tiny dummy data.

        Called by executor before processing to avoid JIT delays during execution.
        Subclasses should override _do_warmup() to trigger their kernels.
        """
        if self._kernels_compiled:
            return
        self._do_warmup()
        self._kernels_compiled = True

    def _do_warmup(self) -> None:  # noqa: B027
        """
        Perform actual warmup. Override in subclass.

        Default implementation does nothing. Subclasses should call their
        kernel with minimal data (e.g., 2x2 field) to trigger compilation.
        """
        pass

    @property
    def is_compiled(self) -> bool:
        """
        Check if kernels have been JIT compiled.

        Returns:
            True if warmup() has been called and kernels are compiled.

        """
        return self._kernels_compiled

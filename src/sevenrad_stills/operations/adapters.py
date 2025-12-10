"""
Adapters for legacy PIL-based operations to work in Taichi GPU pipeline.

This module provides adapter classes that bridge the gap between the existing
PIL-based image operations and the new Taichi GPU pipeline. The adapter handles
conversions between Taichi fields and PIL images transparently.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation
from sevenrad_stills.pipeline.protocols import TempFieldSpec


class LegacyOperationAdapter:
    """
    Adapts PIL-based operations to work in Taichi pipeline.

    This adapter wraps existing BaseImageOperation implementations and makes
    them compatible with the TaichiFieldOperation protocol. It handles all
    necessary conversions between Taichi fields and PIL images.

    The adapter works by:
    1. Extracting numpy array from source Taichi field
    2. Converting to PIL Image
    3. Applying the wrapped legacy operation
    4. Converting result back to numpy array
    5. Writing to destination Taichi field

    Example:
        >>> from sevenrad_stills.operations.saturation import SaturationOperation
        >>> legacy_op = SaturationOperation()
        >>> adapted_op = LegacyOperationAdapter(legacy_op)
        >>> adapted_op.apply_to_field(source_field, dest_field, {}, params, h, w)

    Attributes:
        _legacy: The wrapped BaseImageOperation instance

    """

    def __init__(self, legacy_operation: BaseImageOperation) -> None:
        """
        Initialize adapter with a legacy operation.

        Args:
            legacy_operation: PIL-based operation to wrap

        """
        self._legacy = legacy_operation

    @property
    def name(self) -> str:
        """
        Get the operation name with legacy prefix.

        Returns:
            Operation name prefixed with "legacy_"

        """
        return f"legacy_{self._legacy.name}"

    @property
    def supports_inplace(self) -> bool:
        """
        Check if operation supports inplace execution.

        Legacy operations always require separate buffers since they
        operate on PIL images which are immutable-ish and the conversion
        overhead makes inplace execution less beneficial.

        Returns:
            Always False for legacy operations

        """
        return False

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        """
        Get output dimension scaling factors.

        Most legacy operations preserve dimensions. Operations that change
        dimensions (like downscale) handle it internally in PIL.

        Returns:
            (1.0, 1.0) indicating no dimension change

        """
        return (1.0, 1.0)

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """
        Get temporary field requirements.

        Legacy operations don't use Taichi temporary fields since they
        operate on PIL images on the CPU.

        Returns:
            Empty list - no temporary fields needed

        """
        return []

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
        Apply legacy operation to Taichi fields.

        Performs the conversion pipeline:
        1. Extract from source Taichi field to numpy array
        2. Convert numpy array to PIL Image
        3. Apply wrapped legacy operation
        4. Convert result back to numpy array
        5. Copy into destination Taichi field

        Args:
            source: Source Taichi field (ti.Vector.field)
            dest: Destination Taichi field (ti.Vector.field)
            temp_fields: Unused for legacy operations
            params: Operation parameters passed to legacy operation
            height: Image height in pixels
            width: Image width in pixels

        Note:
            This method involves CPU-GPU transfers and is slower than
            native Taichi operations. Use for compatibility only.

        """
        # Extract from source field to numpy array
        # Taichi fields are typically stored as float32 in 0.0-1.0 range
        img_array = source.to_numpy()  # Shape: (height, width, 3 or 4)

        # Convert to uint8 for PIL
        img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Determine number of channels
        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

        # Convert to PIL Image
        if channels == 4:
            pil_image = Image.fromarray(img_uint8, mode="RGBA")
        elif channels == 3:
            pil_image = Image.fromarray(img_uint8, mode="RGB")
        elif channels == 1:
            pil_image = Image.fromarray(img_uint8.squeeze(), mode="L")
        else:
            msg = f"Unsupported channel count: {channels}"
            raise ValueError(msg)

        # Apply legacy operation
        result_pil = self._legacy.apply(pil_image, params)

        # Convert result back to numpy
        result_array = np.array(result_pil, dtype=np.float32) / 255.0

        # Ensure result has correct shape (add channel dimension if grayscale)
        if len(result_array.shape) == 2:
            result_array = result_array[:, :, np.newaxis]

        # Handle RGB vs RGBA conversion if needed
        result_channels = result_array.shape[2] if len(result_array.shape) == 3 else 1
        if channels != result_channels:
            if channels == 4 and result_channels == 3:
                # Add alpha channel
                alpha: NDArray[np.float32] = np.ones(
                    (height, width, 1), dtype=np.float32
                )
                result_array = np.concatenate([result_array, alpha], axis=2)
            elif channels == 3 and result_channels == 4:
                # Drop alpha channel
                result_array = result_array[:, :, :3]

        # Copy result into destination field
        dest.from_numpy(result_array)

    def reference_numpy(
        self, image: NDArray[np.float32], params: dict[str, Any]
    ) -> NDArray[np.float32]:
        """
        Apply operation using NumPy reference implementation.

        Delegates to the wrapped legacy operation by converting through PIL.
        This provides a reference implementation for testing GPU operations.

        Args:
            image: Input image array (H x W x C, float32, 0.0-1.0)
            params: Operation parameters

        Returns:
            Output image array (same format as input)

        """
        # Convert to uint8 PIL image
        img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

        channels = image.shape[2] if len(image.shape) == 3 else 1

        if channels == 4:
            pil_image = Image.fromarray(img_uint8, mode="RGBA")
        elif channels == 3:
            pil_image = Image.fromarray(img_uint8, mode="RGB")
        else:
            pil_image = Image.fromarray(img_uint8.squeeze(), mode="L")

        # Apply operation
        result_pil = self._legacy.apply(pil_image, params)

        # Convert back to float32 numpy
        result_array = np.array(result_pil, dtype=np.float32) / 255.0

        # Ensure shape matches input
        if len(result_array.shape) == 2 and len(image.shape) == 3:
            result_array = result_array[:, :, np.newaxis]

        return result_array

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate operation parameters.

        Delegates to the wrapped legacy operation's validation.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        self._legacy.validate_params(params)

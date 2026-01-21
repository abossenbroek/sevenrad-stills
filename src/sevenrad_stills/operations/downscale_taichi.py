"""
Taichi end-to-end pipeline downscale operation.

Resolution downscaling for GPU pipeline execution using ti.Vector.field(4).
Operates on pre-allocated buffers without CPU↔GPU data transfer.
"""

from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.taichi_base import BaseTaichiOperation

# Taichi imports with fallback for testing
try:
    import taichi as ti

    from sevenrad_stills.operations.taichi_kernels.sampling import (
        bilinear_sample,
        nearest_sample,
    )

    TAICHI_AVAILABLE = True
except ImportError:
    ti = None
    bilinear_sample = None
    nearest_sample = None
    TAICHI_AVAILABLE = False

# Constants
MIN_SCALE = 0.01
MAX_SCALE = 1.0
SUPPORTED_METHODS = ["nearest", "bilinear"]


# Define kernels only if Taichi is available
if TAICHI_AVAILABLE and ti is not None:

    @ti.kernel  # type: ignore[misc]
    def _downscale_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        scale: ti.f32,
        method: ti.i32,  # 0=nearest, 1=bilinear
        batch: ti.i32,
        in_height: ti.i32,
        in_width: ti.i32,
        out_height: ti.i32,
        out_width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for downscaling images.

        Operates on ti.Vector.field(4) with RGBA channels.
        Output dimensions are smaller than input.

        Args:
            source: Input Vector.field(4) with shape (batch, in_height, in_width)
            dest: Output Vector.field(4) with shape (batch, out_height, out_width)
            scale: Scale factor (0.01-1.0)
            method: Sampling method (0=nearest, 1=bilinear)
            batch: Batch index
            in_height: Input image height
            in_width: Input image width
            out_height: Output image height
            out_width: Output image width

        """
        for out_i, out_j in ti.ndrange(out_height, out_width):
            # Map output pixel to source coordinates
            # Center-aligned: output pixel center maps to source pixel center
            src_y = (out_i + 0.5) / scale - 0.5
            src_x = (out_j + 0.5) / scale - 0.5

            # Sample based on method
            # Taichi requires variable initialization before conditional use
            result = ti.Vector([0.0, 0.0, 0.0, 0.0])

            if method == 0:  # nearest
                result = nearest_sample(
                    source, batch, src_y, src_x, in_height, in_width
                )
            else:  # bilinear
                result = bilinear_sample(
                    source, batch, src_y, src_x, in_height, in_width
                )

            dest[batch, out_i, out_j] = result

    @ti.kernel  # type: ignore[misc]
    def _upscale_kernel(  # type: ignore[no-untyped-def]  # noqa: ANN202
        source: ti.template(),  # type: ignore[valid-type]
        dest: ti.template(),  # type: ignore[valid-type]
        scale: ti.f32,
        method: ti.i32,  # 0=nearest, 1=bilinear
        batch: ti.i32,
        in_height: ti.i32,
        in_width: ti.i32,
        out_height: ti.i32,
        out_width: ti.i32,
    ):  # Taichi kernels don't use Python return type hints
        """
        GPU kernel for upscaling images back to original size.

        Operates on ti.Vector.field(4) with RGBA channels.
        Output dimensions are larger than input (restore original size).

        Args:
            source: Input Vector.field(4) with shape (batch, in_height, in_width)
            dest: Output Vector.field(4) with shape (batch, out_height, out_width)
            scale: Scale factor used for downscaling (for coordinate mapping)
            method: Sampling method (0=nearest, 1=bilinear)
            batch: Batch index
            in_height: Input (downscaled) image height
            in_width: Input (downscaled) image width
            out_height: Output (original) image height
            out_width: Output (original) image width

        """
        for out_i, out_j in ti.ndrange(out_height, out_width):
            # Map output pixel to downscaled source coordinates
            # Center-aligned: output pixel center maps to source pixel center
            src_y = (out_i + 0.5) * scale - 0.5
            src_x = (out_j + 0.5) * scale - 0.5

            # Sample based on method
            # Taichi requires variable initialization before conditional use
            result = ti.Vector([0.0, 0.0, 0.0, 0.0])

            if method == 0:  # nearest
                result = nearest_sample(
                    source, batch, src_y, src_x, in_height, in_width
                )
            else:  # bilinear
                result = bilinear_sample(
                    source, batch, src_y, src_x, in_height, in_width
                )

            dest[batch, out_i, out_j] = result


class DownscaleTaichiOperation(BaseTaichiOperation):
    """
    Taichi downscale operation for end-to-end GPU pipeline.

    Downscales image resolution to create pixelation effects.
    Operates on ti.Vector.field(4) buffers without CPU↔GPU transfer.

    This operation changes output dimensions based on scale factor.
    Supports optional re-upscaling to original size for visible pixelation.

    Supported methods:
    - nearest: Maximum pixelation, harsh block edges
    - bilinear: Softer pixelation with blended edges

    Example:
        >>> op = DownscaleTaichiOperation()
        >>> # Using new separate methods
        >>> params = {"scale": 0.5, "upscale": True,
        ...           "downscale_method": "bilinear", "upscale_method": "nearest"}
        >>> op.apply_to_field(source, dest, temp_fields, params, 1080, 1920)
        >>> # Legacy method parameter still supported
        >>> params = {"scale": 0.5, "upscale": False, "method": "bilinear"}
        >>> op.apply_to_field(source, dest, {}, params, 1080, 1920)

    """

    def __init__(self) -> None:
        """Initialize downscale operation."""
        super().__init__("downscale_taichi")

    @property
    def supports_inplace(self) -> bool:
        """
        Whether operation can write to source buffer.

        Downscale changes dimensions, so cannot be done in-place.

        Returns:
            False - this operation requires separate output buffer.

        """
        return False

    def output_shape_factor(self, params: dict[str, Any]) -> tuple[float, float]:
        """
        Output shape relative to input: (height_factor, width_factor).

        CRITICAL: This operation changes dimensions based on scale parameter.

        Args:
            params: Must contain 'scale' and 'upscale' parameters

        Returns:
            Tuple of (height_factor, width_factor) multipliers.
            If upscale=True, returns (1.0, 1.0) - same as input.
            If upscale=False, returns (scale, scale) - smaller output.

        """
        scale = params.get("scale", 1.0)
        upscale = params.get("upscale", False)

        if upscale:
            return (1.0, 1.0)  # Returns to original size
        return (scale, scale)  # Smaller output

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate downscale parameters.

        Expected params:
        - scale: float (0.01-1.0) - Scale factor for downscaling
        - upscale: bool - Whether to upscale back to original size (default: False)
        - downscale_method: str - Downscaling method: "nearest" or "bilinear" (default: "bilinear")
        - upscale_method: str - Upscaling method: "nearest" or "bilinear" (default: "bilinear")
        - method: str - Legacy parameter for both methods (backward compatibility)

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "scale" not in params:
            msg = "Downscale requires 'scale' parameter"
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

        # Validate method parameters (both legacy and new)
        for method_key in ["method", "downscale_method", "upscale_method"]:
            if method_key in params:
                method = params[method_key]
                if not isinstance(method, str):
                    msg = f"{method_key.capitalize()} must be a string, got {type(method)}"
                    raise ValueError(msg)
                if method not in SUPPORTED_METHODS:
                    available = ", ".join(SUPPORTED_METHODS)
                    msg = f"Invalid {method_key} '{method}'. GPU supports: {available}"
                    raise ValueError(msg)

    def apply_to_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,  # ti.Vector.field
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """
        Apply downscale operation on GPU fields.

        IMPORTANT: If upscale=True, this operation requires a temporary field
        for the intermediate downscaled image. The temp_fields dict must contain
        a field with key matching the downscaled dimensions.

        Args:
            source: Input Taichi Vector.field(4) with shape (batch, height, width)
            dest: Output Taichi Vector.field(4) with shape (batch, out_h, out_w)
            temp_fields: Pre-allocated temporary fields (required if upscale=True)
            params: Must contain 'scale', optional 'upscale' and 'method'
            height: Input image height
            width: Input image width

        Raises:
            RuntimeError: If Taichi is not available

        """
        if not TAICHI_AVAILABLE or ti is None:
            msg = "Taichi is not available. Cannot execute GPU operation."
            raise RuntimeError(msg)

        scale = float(params["scale"])
        upscale = params.get("upscale", False)

        # Resolve methods with backward compatibility
        # Priority: specific method > legacy method > default
        downscale_method = params.get(
            "downscale_method", params.get("method", "bilinear")
        )
        upscale_method = params.get("upscale_method", params.get("method", "bilinear"))

        # Map method names to integers
        downscale_int = 1 if downscale_method == "bilinear" else 0
        upscale_int = 1 if upscale_method == "bilinear" else 0

        # Calculate downscaled dimensions
        down_height = max(1, int(height * scale))
        down_width = max(1, int(width * scale))

        if upscale:
            # Two-pass: downscale then upscale back to original
            # Need temporary field for downscaled image
            temp_key = f"downscale_{down_height}x{down_width}"
            if temp_key not in temp_fields:
                msg = f"Missing temporary field '{temp_key}' for upscale operation"
                raise RuntimeError(msg)

            temp_field = temp_fields[temp_key]

            # Pass 1: Downscale to temporary field
            _downscale_kernel(
                source,
                temp_field,
                scale,
                downscale_int,
                0,  # batch_idx
                height,
                width,
                down_height,
                down_width,
            )

            # Pass 2: Upscale back to original size
            _upscale_kernel(
                temp_field,
                dest,
                scale,
                upscale_int,
                0,  # batch_idx
                down_height,
                down_width,
                height,
                width,
            )
        else:
            # Single pass: just downscale
            _downscale_kernel(
                source,
                dest,
                scale,
                downscale_int,
                0,  # batch_idx
                height,
                width,
                down_height,
                down_width,
            )

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """
        NumPy reference implementation for testing.

        Implements the same sampling logic as the Taichi kernels
        to ensure correctness testing is accurate.

        Args:
            image: Input image as numpy array (H, W, 3) float32 in [0, 1]
            params: Must contain 'scale', optional 'upscale' and 'method'

        Returns:
            Processed image as numpy array (H_out, W_out, 3) float32 in [0, 1]

        """
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            bilinear_sample_numpy,
        )

        scale = float(params["scale"])
        upscale = params.get("upscale", False)

        # Resolve methods with backward compatibility
        # Priority: specific method > legacy method > default
        downscale_method = params.get(
            "downscale_method", params.get("method", "bilinear")
        )
        upscale_method = params.get("upscale_method", params.get("method", "bilinear"))

        in_height, in_width = image.shape[:2]

        # Calculate downscaled dimensions
        down_height = max(1, int(in_height * scale))
        down_width = max(1, int(in_width * scale))

        # Downscale pass
        downscaled = np.zeros((down_height, down_width, 3), dtype=np.float32)

        for out_i in range(down_height):
            for out_j in range(down_width):
                # Same coordinate mapping as Taichi kernel
                src_y = (out_i + 0.5) / scale - 0.5
                src_x = (out_j + 0.5) / scale - 0.5

                if downscale_method == "nearest":
                    # Nearest neighbor
                    yi = int(round(src_y))
                    xi = int(round(src_x))
                    yi = max(0, min(yi, in_height - 1))
                    xi = max(0, min(xi, in_width - 1))
                    downscaled[out_i, out_j] = image[yi, xi]
                else:  # bilinear
                    downscaled[out_i, out_j] = bilinear_sample_numpy(
                        image, src_y, src_x
                    )

        if not upscale:
            return downscaled

        # Upscale pass (back to original size)
        upscaled = np.zeros((in_height, in_width, 3), dtype=np.float32)

        for out_i in range(in_height):
            for out_j in range(in_width):
                # Map output pixel to downscaled coordinates
                src_y = (out_i + 0.5) * scale - 0.5
                src_x = (out_j + 0.5) * scale - 0.5

                if upscale_method == "nearest":
                    # Nearest neighbor
                    yi = int(round(src_y))
                    xi = int(round(src_x))
                    yi = max(0, min(yi, down_height - 1))
                    xi = max(0, min(xi, down_width - 1))
                    upscaled[out_i, out_j] = downscaled[yi, xi]
                else:  # bilinear
                    upscaled[out_i, out_j] = bilinear_sample_numpy(
                        downscaled, src_y, src_x
                    )

        return upscaled

    def _do_warmup(self) -> None:
        """
        Trigger JIT compilation with minimal 4x4 dummy fields.

        Called by warmup() to pre-compile the downscale kernels
        before actual processing begins.
        """
        if not TAICHI_AVAILABLE or ti is None:
            return

        # Create minimal 4x4 source and 2x2 destination for compilation
        dummy_src = ti.Vector.field(4, dtype=ti.f32, shape=(1, 4, 4))
        dummy_dst = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        dummy_temp = ti.Vector.field(4, dtype=ti.f32, shape=(1, 4, 4))

        # Initialize with dummy data
        for i in range(4):
            for j in range(4):
                dummy_src[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        # Trigger compilation for downscale kernel (both methods)
        _downscale_kernel(dummy_src, dummy_dst, 0.5, 0, 0, 4, 4, 2, 2)  # nearest
        _downscale_kernel(dummy_src, dummy_dst, 0.5, 1, 0, 4, 4, 2, 2)  # bilinear

        # Trigger compilation for upscale kernel
        _upscale_kernel(dummy_dst, dummy_temp, 0.5, 1, 0, 2, 2, 4, 4)

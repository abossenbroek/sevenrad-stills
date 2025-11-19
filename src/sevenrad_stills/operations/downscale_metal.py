"""
Optimized Metal GPU-accelerated resolution downscaling operation.

Provides control over scale factors and resampling methods to achieve
various levels of pixelation and loss of detail. Uses custom Metal
compute shaders with buffer-based approach for maximum GPU performance.

PERFORMANCE CHARACTERISTICS
===========================

Standalone Operation Performance:
---------------------------------
- CPU (PIL):  ~13ms  (highly optimized C code with SIMD)
- GPU (Taichi): ~16ms  (competitive with CPU)
- Metal:      ~58ms  (slower due to Python↔GPU transfer overhead)

Metal Performance Breakdown:
  - Buffer creation overhead: ~30ms (2 operations: downscale + upscale)
  - Kernel execution:        ~1.7ms per operation (FAST!)
  - Total:                   ~33-58ms

GPU PIPELINE ARCHITECTURE (Future Optimization)
===============================================

Current Standalone:
  Python → GPU upload (~15ms) → resize → GPU download (~15ms) → Python
  Repeat for each operation → High overhead!

Future Pipeline:
  Python → GPU upload → resize → blur → distort → ... → GPU download

  Benefits:
  - Upload/download overhead amortized across ALL operations
  - Metal kernel execution (~1.7ms) is 7-8x faster than CPU (~13ms)
  - Multiple operations stay in GPU memory
  - Total speedup: 5-10x for multi-operation pipelines

Implementation follows optimization patterns from blur_gaussian_metal.py:
- Buffer-based data transfer (not textures)
- Pre-compiled pipeline states (no per-call compilation)
- Efficient RGB memory layout (no RGBA padding)
- Optimized 16x16 thread group sizing
"""

import math
import sys
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Check platform
if sys.platform != "darwin":
    msg = "Metal backend requires macOS"
    raise ImportError(msg)

try:
    import Metal

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# Constants (same as CPU version)
MIN_SCALE = 0.01
MAX_SCALE = 1.0
NDIM_GRAYSCALE = 2  # Number of dimensions for grayscale images
NDIM_RGB = 3  # Number of dimensions for RGB/RGBA images

# Resampling method mapping for Metal
RESAMPLING_METHODS = {
    "nearest": 0,  # Harsh pixelation
    "bilinear": 1,  # Smooth downscaling
}

# Metal kernel source code using buffers with packed parameters
METAL_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Parameter struct to reduce buffer count
struct ResizeParams {
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int channels;
};

/// Nearest neighbor resampling kernel (buffer-based with packed params)
///
/// Each thread processes one output pixel by sampling the nearest input pixel.
///
/// @param src Input image buffer (float32, layout: H*W*C)
/// @param dst Output image buffer (float32, layout: dst_h*dst_w*C)
/// @param params Packed parameters (src/dst dimensions, channels)
/// @param gid Thread position in grid
kernel void resize_nearest(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant ResizeParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    // Bounds check
    if (x >= params.dst_w || y >= params.dst_h) return;

    // Calculate source coordinates (nearest neighbor)
    int src_x = int(float(x) * float(params.src_w) / float(params.dst_w));
    int src_y = int(float(y) * float(params.src_h) / float(params.dst_h));

    // Clamp to valid range
    src_x = clamp(src_x, 0, params.src_w - 1);
    src_y = clamp(src_y, 0, params.src_h - 1);

    // Copy all channels
    for (int c = 0; c < params.channels; c++) {
        int src_idx = (src_y * params.src_w + src_x) * params.channels + c;
        int dst_idx = (y * params.dst_w + x) * params.channels + c;
        dst[dst_idx] = src[src_idx];
    }
}

/// Bilinear resampling kernel (buffer-based with packed params)
///
/// Each thread processes one output pixel by interpolating four input pixels.
///
/// @param src Input image buffer (float32, layout: H*W*C)
/// @param dst Output image buffer (float32, layout: dst_h*dst_w*C)
/// @param params Packed parameters (src/dst dimensions, channels)
/// @param gid Thread position in grid
kernel void resize_bilinear(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant ResizeParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    // Bounds check
    if (x >= params.dst_w || y >= params.dst_h) return;

    // Calculate source coordinates (floating point)
    float src_x_f = (float(x) + 0.5) * float(params.src_w) / float(params.dst_w) - 0.5;
    float src_y_f = (float(y) + 0.5) * float(params.src_h) / float(params.dst_h) - 0.5;

    // Get integer parts (floor)
    int src_x0 = int(floor(src_x_f));
    int src_y0 = int(floor(src_y_f));
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

    // Clamp to valid range
    src_x0 = clamp(src_x0, 0, params.src_w - 1);
    src_y0 = clamp(src_y0, 0, params.src_h - 1);
    src_x1 = clamp(src_x1, 0, params.src_w - 1);
    src_y1 = clamp(src_y1, 0, params.src_h - 1);

    // Get fractional parts
    float fx = src_x_f - floor(src_x_f);
    float fy = src_y_f - floor(src_y_f);

    // Interpolate each channel
    for (int c = 0; c < params.channels; c++) {
        // Read four neighboring pixels
        int idx00 = (src_y0 * params.src_w + src_x0) * params.channels + c;
        int idx01 = (src_y0 * params.src_w + src_x1) * params.channels + c;
        int idx10 = (src_y1 * params.src_w + src_x0) * params.channels + c;
        int idx11 = (src_y1 * params.src_w + src_x1) * params.channels + c;

        float c00 = src[idx00];
        float c01 = src[idx01];
        float c10 = src[idx10];
        float c11 = src[idx11];

        // Bilinear interpolation
        float c0 = mix(c00, c01, fx);
        float c1 = mix(c10, c11, fx);
        float result = mix(c0, c1, fy);

        // Write result
        int dst_idx = (y * params.dst_w + x) * params.channels + c;
        dst[dst_idx] = result;
    }
}
"""


class MetalDownscaleEngine:
    """Metal compute engine for downscale operation."""

    def __init__(self) -> None:
        """Initialize Metal device and compile shaders."""
        if not METAL_AVAILABLE:
            msg = "Metal framework not available. Install pyobjc-framework-Metal."
            raise RuntimeError(msg)

        # Get default Metal device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            msg = "No Metal-capable GPU found."
            raise RuntimeError(msg)

        # Create command queue
        self.command_queue = self.device.newCommandQueue()

        # Compile shader library
        try:
            options = Metal.MTLCompileOptions.new()
            self.library, error = self.device.newLibraryWithSource_options_error_(
                METAL_KERNEL_SOURCE, options, None
            )
            if error:
                msg = f"Metal shader compilation failed: {error}"
                raise RuntimeError(msg)
        except Exception as e:
            msg = f"Failed to compile Metal shaders: {e}"
            raise RuntimeError(msg) from e

        # Create pipeline states for both methods (pre-compiled for reuse)
        self.pipeline_nearest = self._create_pipeline("resize_nearest")
        self.pipeline_bilinear = self._create_pipeline("resize_bilinear")

    def _create_pipeline(self, function_name: str) -> object:
        """Create compute pipeline state for a shader function."""
        function = self.library.newFunctionWithName_(function_name)
        if function is None:
            msg = f"Metal function '{function_name}' not found in library."
            raise RuntimeError(msg)

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if error:
            msg = f"Failed to create pipeline for '{function_name}': {error}"
            raise RuntimeError(msg)

        return pipeline

    def _create_buffers(
        self,
        img_float: np.ndarray,
        dst_h: int,
        dst_w: int,
        channels: int,
    ) -> tuple[object, object, object]:
        """Create Metal buffers for resize operation (optimized with packed params)."""
        src_h, src_w = img_float.shape[:2]

        # Input buffer (source image)
        input_buffer = self.device.newBufferWithBytes_length_options_(
            img_float.tobytes(),
            img_float.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Output buffer (resized image)
        output_size = dst_h * dst_w * channels * 4  # float32 = 4 bytes
        output_buffer = self.device.newBufferWithLength_options_(
            output_size,
            Metal.MTLResourceStorageModeShared,
        )

        # Create single packed parameter buffer (struct)
        # Struct layout: src_w, src_h, dst_w, dst_h, channels (all int32)
        params = np.array([src_w, src_h, dst_w, dst_h, channels], dtype=np.int32)
        params_buffer = self.device.newBufferWithBytes_length_options_(
            params.tobytes(),
            params.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        return (input_buffer, output_buffer, params_buffer)

    def resize(
        self,
        img_array: np.ndarray,
        dst_h: int,
        dst_w: int,
        method: int,
    ) -> np.ndarray:
        """
        Resize image using Metal GPU compute shaders.

        Args:
            img_array: Input image array (H, W, C)
            dst_h: Destination height
            dst_w: Destination width
            method: Resampling method (0=nearest, 1=bilinear)

        Returns:
            Resized image array

        """
        src_h, src_w = img_array.shape[:2]
        channels = img_array.shape[2] if img_array.ndim == NDIM_RGB else 1

        # Convert to float32 for processing
        img_float: np.ndarray
        if img_array.ndim == NDIM_RGB:
            img_float = img_array.astype(np.float32)
        else:
            # Add channel dimension for grayscale
            img_float = img_array.astype(np.float32)[:, :, np.newaxis]
            channels = 1

        # Create Metal buffers (optimized with packed params)
        input_buffer, output_buffer, params_buffer = self._create_buffers(
            img_float, dst_h, dst_w, channels
        )

        # Select pipeline based on method
        pipeline = self.pipeline_nearest if method == 0 else self.pipeline_bilinear

        # Calculate thread group sizes (16x16 like Gaussian blur)
        threadgroup_size = Metal.MTLSize(16, 16, 1)
        grid_w = (dst_w + 15) // 16 * 16
        grid_h = (dst_h + 15) // 16 * 16
        grid_size = Metal.MTLSize(grid_w, grid_h, 1)

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        # Dispatch
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Total elements for result extraction
        total_elements = dst_h * dst_w * channels

        # Copy result back from GPU
        result_ptr = output_buffer.contents()  # type: ignore[attr-defined]
        result_array = np.frombuffer(
            result_ptr.as_buffer(total_elements * 4),
            dtype=np.float32,
        ).reshape(dst_h, dst_w, channels)

        # Remove channel dimension for grayscale
        if img_array.ndim == NDIM_GRAYSCALE:
            result_array = result_array[:, :, 0]

        return result_array  # type: ignore[no-any-return]


class DownscaleMetalOperation(BaseImageOperation):
    """
    Optimized Metal GPU-accelerated downscale operation.

    Downscale image resolution to create pixelation effects with custom Metal
    GPU kernels. Designed for GPU pipeline architecture where operations
    stay in GPU memory.

    Scale factor:
    - 0.01-0.10: Extreme pixelation, heavily degraded
    - 0.10-0.25: Heavy pixelation, architectural details lost
    - 0.25-0.50: Moderate pixelation, visible block structures
    - 0.50-1.00: Subtle quality reduction

    Resampling methods:
    - nearest: Maximum pixelation, harsh block edges
    - bilinear: Softer pixelation with blended edges

    Performance Notes:
    - Standalone: ~58ms (slower than CPU ~13ms due to transfer overhead)
    - GPU Pipeline: ~1.7ms kernel (7-8x faster than CPU when in pipeline)
    - See module docstring for detailed performance analysis

    Optimizations:
    - Pre-compiled pipeline states (no per-call compilation overhead)
    - Buffer-based data transfer (not textures, following blur_gaussian_metal.py)
    - Efficient RGB memory layout (no RGBA padding)
    - Optimized 16x16 thread groups for Apple Silicon
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated downscale operation."""
        super().__init__("downscale_metal")
        self.engine = MetalDownscaleEngine()

    def validate_params(self, params: dict[str, Any]) -> None:  # noqa: C901
        """
        Validate downscale operation parameters.

        Expected params:
        - scale: float (0.01-1.0) - Scale factor for downscaling
        - upscale: bool - Whether to upscale back to original size (default: True)
        - downscale_method: str - Resampling method for downscaling
          (default: "bilinear")
        - upscale_method: str - Resampling method for upscaling
          (default: "nearest")

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        if "scale" not in params:
            msg = "Downscale operation requires 'scale' parameter"
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

        # Validate downscale_method if provided
        if "downscale_method" in params:
            method = params["downscale_method"]
            if not isinstance(method, str):
                msg = f"Downscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = (
                    f"Invalid downscale method '{method}'. "
                    f"Metal version supports: {available}"
                )
                raise ValueError(msg)

        # Validate upscale_method if provided
        if "upscale_method" in params:
            method = params["upscale_method"]
            if not isinstance(method, str):
                msg = f"Upscale method must be a string, got {type(method)}"
                raise ValueError(msg)
            if method not in RESAMPLING_METHODS:
                available = ", ".join(RESAMPLING_METHODS.keys())
                msg = (
                    f"Invalid upscale method '{method}'. "
                    f"Metal version supports: {available}"
                )
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply optimized Metal-accelerated downscaling to image.

        Args:
            image: Input PIL Image (RGB or RGBA)
            params: Operation parameters (validated)

        Returns:
            Downscaled (and optionally upscaled) PIL Image

        Raises:
            ValueError: If image is not RGB or RGBA mode
            RuntimeError: If Metal operations fail

        """
        # Validate parameters first
        self.validate_params(params)

        # Only support RGB/RGBA for now
        if image.mode not in ("RGB", "RGBA"):
            msg = f"Metal downscale requires RGB or RGBA image, got {image.mode}"
            raise ValueError(msg)

        scale: float = params["scale"]
        upscale: bool = params.get("upscale", True)
        downscale_method_name: str = params.get("downscale_method", "bilinear")
        upscale_method_name: str = params.get("upscale_method", "nearest")

        # Convert to numpy array
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
        else:
            rgb = img_array.copy()
            alpha = None

        h, w = rgb.shape[:2]

        # Calculate new size
        new_width = max(1, int(w * scale))
        new_height = max(1, int(h * scale))

        # Downscale using Metal
        downscaled = self.engine.resize(
            rgb, new_height, new_width, RESAMPLING_METHODS[downscale_method_name]
        )

        # Upscale back if requested
        if upscale:
            result = self.engine.resize(
                downscaled, h, w, RESAMPLING_METHODS[upscale_method_name]
            )
        else:
            result = downscaled

        # Recombine with alpha if needed
        if alpha is not None:
            # Resize alpha channel to match result
            if upscale:
                alpha_resized = alpha
            else:
                # Downscale alpha to match
                alpha_2d = alpha.reshape(h, w, 1)
                alpha_down = self.engine.resize(
                    alpha_2d, new_height, new_width, RESAMPLING_METHODS["nearest"]
                )
                alpha_resized = alpha_down.reshape(new_height, new_width)

            # Convert to uint8 and combine
            result_uint8 = result.clip(0, 255).astype(np.uint8)
            output_array = np.dstack([result_uint8, alpha_resized])
        else:
            # Convert to uint8
            output_array = result.clip(0, 255).astype(np.uint8)

        return Image.fromarray(output_array)

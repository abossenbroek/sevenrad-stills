"""
Pure Metal-accelerated Gaussian blur operation with custom Metal kernels.

Implements separable Gaussian blur using native Metal compute shaders
for maximum GPU performance on Apple Silicon and AMD GPUs on macOS.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).

This implementation uses a two-pass separable convolution approach:
1. Horizontal blur pass
2. Vertical blur pass

This reduces computational complexity from O(W*H*K^2) to O(W*H*K*2),
providing significant performance gains, especially for large sigma values.
"""

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

# Constants
RGB_CHANNELS = 3  # Number of color channels in RGB image

# Metal shader source code
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

/// Apply 1D Gaussian blur in horizontal direction with reflect boundary mode
///
/// Each thread processes one pixel, convolving with the 1D kernel horizontally.
/// Boundary handling: reflect mode (matches scipy.ndimage.gaussian_filter)
///
/// @param input Input image buffer (float32, layout: H*W*C)
/// @param output Output image buffer (float32, layout: H*W*C)
/// @param weights 1D Gaussian kernel weights (float32)
/// @param kernel_size Size of the kernel (must be odd)
/// @param height Image height in pixels
/// @param width Image width in pixels
/// @param channels Number of color channels (1 or 3)
/// @param gid Thread position (x, y) in grid
kernel void gaussian_blur_horizontal(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float *weights [[buffer(2)]],
    constant int &kernel_size [[buffer(3)]],
    constant int &height [[buffer(4)]],
    constant int &width [[buffer(5)]],
    constant int &channels [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    // Bounds check
    if (x >= width || y >= height) return;

    int radius = kernel_size / 2;

    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0;

        // Convolve with 1D kernel horizontally
        for (int k = 0; k < kernel_size; k++) {
            // Calculate source x coordinate
            int src_x = x + k - radius;

            // Reflect boundary handling (matches scipy)
            if (src_x < 0) {
                src_x = -src_x;
            } else if (src_x >= width) {
                src_x = 2 * width - src_x - 2;
            }

            // Clamp to valid range
            src_x = clamp(src_x, 0, width - 1);

            // Accumulate weighted value
            int src_idx = (y * width + src_x) * channels + c;
            sum += input[src_idx] * weights[k];
        }

        // Write result
        int dst_idx = (y * width + x) * channels + c;
        output[dst_idx] = sum;
    }
}

/// Apply 1D Gaussian blur in vertical direction with reflect boundary mode
///
/// Each thread processes one pixel, convolving with the 1D kernel vertically.
/// Boundary handling: reflect mode (matches scipy.ndimage.gaussian_filter)
///
/// @param input Input image buffer (float32, layout: H*W*C)
/// @param output Output image buffer (float32, layout: H*W*C)
/// @param weights 1D Gaussian kernel weights (float32)
/// @param kernel_size Size of the kernel (must be odd)
/// @param height Image height in pixels
/// @param width Image width in pixels
/// @param channels Number of color channels (1 or 3)
/// @param gid Thread position (x, y) in grid
kernel void gaussian_blur_vertical(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float *weights [[buffer(2)]],
    constant int &kernel_size [[buffer(3)]],
    constant int &height [[buffer(4)]],
    constant int &width [[buffer(5)]],
    constant int &channels [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    // Bounds check
    if (x >= width || y >= height) return;

    int radius = kernel_size / 2;

    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0;

        // Convolve with 1D kernel vertically
        for (int k = 0; k < kernel_size; k++) {
            // Calculate source y coordinate
            int src_y = y + k - radius;

            // Reflect boundary handling (matches scipy)
            if (src_y < 0) {
                src_y = -src_y;
            } else if (src_y >= height) {
                src_y = 2 * height - src_y - 2;
            }

            // Clamp to valid range
            src_y = clamp(src_y, 0, height - 1);

            // Accumulate weighted value
            int src_idx = (src_y * width + x) * channels + c;
            sum += input[src_idx] * weights[k];
        }

        // Write result
        int dst_idx = (y * width + x) * channels + c;
        output[dst_idx] = sum;
    }
}
"""


def compute_gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Compute 1D Gaussian kernel weights on CPU.

    The kernel size is determined by the kernel array size.
    Uses the standard Gaussian formula: exp(-x^2 / (2 * sigma^2))

    Args:
        kernel_size: Size of the kernel (odd number).
        sigma: Standard deviation of the Gaussian.

    Returns:
        1D array of normalized Gaussian weights.

    """
    radius = kernel_size // 2
    x: np.ndarray = np.arange(kernel_size) - radius
    weights: np.ndarray = np.exp(-(x**2) / (2.0 * sigma**2))
    return (weights / weights.sum()).astype(np.float32)  # type: ignore[no-any-return]


class MetalComputeEngine:
    """Metal compute engine for Gaussian blur operation."""

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
                METAL_SHADER_SOURCE, options, None
            )
            if error:
                msg = f"Metal shader compilation failed: {error}"
                raise RuntimeError(msg)
        except Exception as e:
            msg = f"Failed to compile Metal shaders: {e}"
            raise RuntimeError(msg) from e

        # Create pipeline states for both passes
        self.pipeline_horizontal = self._create_pipeline("gaussian_blur_horizontal")
        self.pipeline_vertical = self._create_pipeline("gaussian_blur_vertical")

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
        kernel_weights: np.ndarray,
        h: int,
        w: int,
        channels: int,
        kernel_size: int,
    ) -> tuple[object, object, object, object, object, object, object, object]:
        """Create Metal buffers for blur operation."""
        total_elements = h * w * channels

        # Input buffer
        input_buffer = self.device.newBufferWithBytes_length_options_(
            img_float.tobytes(),
            img_float.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Intermediate buffer (output of horizontal pass)
        temp_buffer = self.device.newBufferWithLength_options_(
            total_elements * 4,  # float32 = 4 bytes
            Metal.MTLResourceStorageModeShared,
        )

        # Output buffer (output of vertical pass)
        output_buffer = self.device.newBufferWithLength_options_(
            total_elements * 4,  # float32 = 4 bytes
            Metal.MTLResourceStorageModeShared,
        )

        # Kernel weights buffer
        kernel_buffer = self.device.newBufferWithBytes_length_options_(
            kernel_weights.tobytes(),
            kernel_weights.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create parameter buffers
        kernel_size_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([kernel_size], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        height_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([h], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        width_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([w], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        channels_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([channels], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )

        return (
            input_buffer,
            temp_buffer,
            output_buffer,
            kernel_buffer,
            kernel_size_buffer,
            height_buffer,
            width_buffer,
            channels_buffer,
        )

    def _execute_blur_passes(
        self,
        input_buffer: object,
        temp_buffer: object,
        output_buffer: object,
        kernel_buffer: object,
        kernel_size_buffer: object,
        height_buffer: object,
        width_buffer: object,
        channels_buffer: object,
        grid_size: object,
        threadgroup_size: object,
    ) -> None:
        """Execute horizontal and vertical blur passes."""
        # Create command buffer
        command_buffer = self.command_queue.commandBuffer()

        # ===== PASS 1: Horizontal blur =====
        encoder_h = command_buffer.computeCommandEncoder()
        encoder_h.setComputePipelineState_(self.pipeline_horizontal)

        # Set buffers for horizontal pass
        encoder_h.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder_h.setBuffer_offset_atIndex_(temp_buffer, 0, 1)
        encoder_h.setBuffer_offset_atIndex_(kernel_buffer, 0, 2)
        encoder_h.setBuffer_offset_atIndex_(kernel_size_buffer, 0, 3)
        encoder_h.setBuffer_offset_atIndex_(height_buffer, 0, 4)
        encoder_h.setBuffer_offset_atIndex_(width_buffer, 0, 5)
        encoder_h.setBuffer_offset_atIndex_(channels_buffer, 0, 6)

        # Dispatch horizontal pass
        encoder_h.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder_h.endEncoding()

        # ===== PASS 2: Vertical blur =====
        encoder_v = command_buffer.computeCommandEncoder()
        encoder_v.setComputePipelineState_(self.pipeline_vertical)

        # Set buffers for vertical pass (temp_buffer is now input)
        encoder_v.setBuffer_offset_atIndex_(temp_buffer, 0, 0)
        encoder_v.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder_v.setBuffer_offset_atIndex_(kernel_buffer, 0, 2)
        encoder_v.setBuffer_offset_atIndex_(kernel_size_buffer, 0, 3)
        encoder_v.setBuffer_offset_atIndex_(height_buffer, 0, 4)
        encoder_v.setBuffer_offset_atIndex_(width_buffer, 0, 5)
        encoder_v.setBuffer_offset_atIndex_(channels_buffer, 0, 6)

        # Dispatch vertical pass
        encoder_v.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder_v.endEncoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def apply_gaussian_blur(
        self,
        img_array: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Apply Gaussian blur using Metal compute shaders with separable convolution.

        Performs two passes:
        1. Horizontal blur pass
        2. Vertical blur pass

        Args:
            img_array: Input image array (can be 2D grayscale or 3D RGB)
            sigma: Blur sigma value

        Returns:
            Blurred image array with same shape as input

        """
        # Calculate kernel size based on sigma (rule of thumb: 6*sigma covers 99.7%)
        kernel_size = int(np.ceil(sigma * 6))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size for symmetric kernel
        kernel_size = max(3, kernel_size)  # Minimum size of 3

        # Compute Gaussian kernel weights on CPU
        kernel_weights = compute_gaussian_kernel_1d(kernel_size, sigma)

        h, w = img_array.shape[:2]
        is_color = img_array.ndim == RGB_CHANNELS
        channels = RGB_CHANNELS if is_color else 1

        # Convert to float32 for processing
        img_float: np.ndarray
        if is_color:
            img_float = img_array.astype(np.float32)
        else:
            # Add channel dimension for grayscale
            img_float = img_array.astype(np.float32)[:, :, np.newaxis]

        # Create Metal buffers
        (
            input_buffer,
            temp_buffer,
            output_buffer,
            kernel_buffer,
            kernel_size_buffer,
            height_buffer,
            width_buffer,
            channels_buffer,
        ) = self._create_buffers(img_float, kernel_weights, h, w, channels, kernel_size)

        # Calculate thread group sizes
        threadgroup_size = Metal.MTLSize(16, 16, 1)
        grid_w = (w + 15) // 16 * 16
        grid_h = (h + 15) // 16 * 16
        grid_size = Metal.MTLSize(grid_w, grid_h, 1)

        # Execute blur passes
        self._execute_blur_passes(
            input_buffer,
            temp_buffer,
            output_buffer,
            kernel_buffer,
            kernel_size_buffer,
            height_buffer,
            width_buffer,
            channels_buffer,
            grid_size,
            threadgroup_size,
        )

        # Total elements for result extraction
        total_elements = h * w * channels

        # Copy result back from GPU
        result_ptr = output_buffer.contents()  # type: ignore[attr-defined]
        result_array = np.frombuffer(
            result_ptr.as_buffer(total_elements * 4),
            dtype=np.float32,
        ).reshape(h, w, channels)

        # Remove channel dimension for grayscale
        if not is_color:
            result_array = result_array[:, :, 0]

        return result_array  # type: ignore[no-any-return]


class GaussianBlurMetalOperation(BaseImageOperation):
    """
    Apply Gaussian blur using pure Metal compute shaders.

    This implementation uses custom Metal Shading Language kernels with
    separable 1D filters for maximum performance on Apple Silicon and AMD GPUs.

    The separable approach provides ~10x reduction in operations compared to
    2D convolution, and Metal provides direct GPU acceleration without any
    intermediate framework overhead.

    Performance: Expected to provide 3-4x speedup over CPU scipy implementation
    and match or exceed Taichi GPU performance for large images.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated Gaussian blur operation."""
        super().__init__("blur_gaussian_metal")
        self.engine = MetalComputeEngine()

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for the Gaussian blur operation.

        Args:
            params: A dictionary containing:
                - sigma (float): The standard deviation for the Gaussian kernel.
                  Must be non-negative. A sigma of 0 returns the original image.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "sigma" not in params:
            msg = "Gaussian blur requires a 'sigma' parameter."
            raise ValueError(msg)

        sigma = params["sigma"]
        if not isinstance(sigma, (int, float)):
            msg = f"Sigma must be a number, got {type(sigma)}."
            raise ValueError(msg)
        if sigma < 0:
            msg = f"Sigma must be non-negative, got {sigma}."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply Metal-accelerated Gaussian blur to the image.

        Uses separable 1D filters for efficiency:
        1. Apply horizontal Gaussian blur
        2. Apply vertical Gaussian blur to the result

        Args:
            image: The input PIL Image.
            params: A dictionary with a 'sigma' key.

        Returns:
            The blurred PIL Image.

        """
        self.validate_params(params)
        sigma: float = params["sigma"]

        # If sigma is zero, no blur is applied. Return original image.
        if sigma == 0:
            return image.copy()

        # Convert PIL image to NumPy array. Preserve original dtype.
        img_array = np.array(image)
        original_dtype = img_array.dtype

        # Handle RGBA images separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = image.convert("RGB")
            alpha = image.getchannel("A")

            rgb_array = np.array(rgb)
            blurred_rgb_array = self.engine.apply_gaussian_blur(rgb_array, sigma)

            # Restore original data type and create new image
            blurred_rgb = Image.fromarray(
                np.clip(blurred_rgb_array, 0, 255).astype(original_dtype)
            )
            # Create RGBA image with blurred RGB and original alpha
            result = Image.new("RGBA", image.size)
            result.paste(blurred_rgb, (0, 0))
            result.putalpha(alpha)
            return result

        # For RGB or L images, apply blur directly
        blurred_array = self.engine.apply_gaussian_blur(img_array, sigma)

        # Convert back to PIL Image, ensuring original dtype is respected.
        return Image.fromarray(np.clip(blurred_array, 0, 255).astype(original_dtype))

"""
Pure Metal-accelerated chromatic aberration operation with custom Metal kernels.

Simulates chromatic aberration by shifting RGB color channels using native
Metal compute shaders for maximum GPU performance on Apple Silicon.

Inspired by and adapted from degradr by nhauber99
(https://github.com/nhauber99/degradr)
Original licensed under MIT License (see LICENSE_DEGRADR.txt).
"""

from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

try:
    import Metal
    import objc

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# Metal shader source code
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void apply_chromatic_aberration(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant int &height [[buffer(2)]],
    constant int &width [[buffer(3)]],
    constant int &shift_y [[buffer(4)]],
    constant int &shift_x [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Red channel: shift in positive direction
    int src_r_y = y - shift_y;
    int src_r_x = x - shift_x;
    bool in_bounds_r = (src_r_y >= 0 && src_r_y < height &&
                        src_r_x >= 0 && src_r_x < width);

    if (in_bounds_r) {
        int src_r_idx = (src_r_y * width + src_r_x) * 3;
        output[idx + 0] = input[src_r_idx + 0];
    } else {
        output[idx + 0] = 0;
    }

    // Green channel: no shift (reference channel)
    output[idx + 1] = input[idx + 1];

    // Blue channel: shift in negative direction (opposite of red)
    int src_b_y = y + shift_y;
    int src_b_x = x + shift_x;
    bool in_bounds_b = (src_b_y >= 0 && src_b_y < height &&
                        src_b_x >= 0 && src_b_x < width);

    if (in_bounds_b) {
        int src_b_idx = (src_b_y * width + src_b_x) * 3;
        output[idx + 2] = input[src_b_idx + 2];
    } else {
        output[idx + 2] = 0;
    }
}
"""


class MetalComputeEngine:
    """Metal compute engine for chromatic aberration operation."""

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

        # Create pipeline state
        self.pipeline = self._create_pipeline("apply_chromatic_aberration")

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

    def apply_chromatic_aberration(
        self,
        img_array: np.ndarray,
        shift_x: int,
        shift_y: int,
    ) -> np.ndarray:
        """
        Apply chromatic aberration using Metal compute shader.

        Args:
            img_array: RGB image array (H, W, 3)
            shift_x: Horizontal shift in pixels
            shift_y: Vertical shift in pixels

        Returns:
            Processed RGB image array (H, W, 3)

        """
        h, w = img_array.shape[:2]

        # Create Metal buffers with shared storage mode
        input_buffer = self.device.newBufferWithBytes_length_options_(
            img_array.tobytes(),
            img_array.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        output_size = h * w * 3
        output_buffer = self.device.newBufferWithLength_options_(
            output_size, Metal.MTLResourceStorageModeShared
        )

        # Create buffers for scalar parameters
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
        shift_y_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([shift_y], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )
        shift_x_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([shift_x], dtype=np.int32).tobytes(),
            4,
            Metal.MTLResourceStorageModeShared,
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        compute_encoder.setComputePipelineState_(self.pipeline)
        compute_encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(height_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(width_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(shift_y_buffer, 0, 4)
        compute_encoder.setBuffer_offset_atIndex_(shift_x_buffer, 0, 5)

        # Calculate thread groups
        thread_group_size = Metal.MTLSize(16, 16, 1)
        grid_size = Metal.MTLSize(
            (w + 15) // 16 * 16,
            (h + 15) // 16 * 16,
            1,
        )

        # Dispatch compute shader
        compute_encoder.dispatchThreads_threadsPerThreadgroup_(
            grid_size, thread_group_size
        )
        compute_encoder.endEncoding()

        # Execute and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy results back
        result_bytes = output_buffer.contents().as_buffer(output_size)
        result_array: np.ndarray = (
            np.frombuffer(result_bytes, dtype=np.uint8).reshape((h, w, 3)).copy()
        )

        return result_array


class ChromaticAberrationMetalOperation(BaseImageOperation):
    """
    Metal-accelerated chromatic aberration simulation.

    Chromatic aberration is a common optical phenomenon where a lens
    fails to focus all colors to the same point, causing color fringing
    at edges. This operation simulates the effect by shifting the red
    and blue channels in opposite directions using native Metal compute
    shaders for maximum GPU performance on Apple Silicon.

    Performance: Pure Metal implementation provides maximum GPU performance by
    using native Metal compute shaders without intermediate frameworks like Taichi.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated chromatic aberration operation."""
        super().__init__("chromatic_aberration_metal")
        self._engine: MetalComputeEngine | None = None

    @property
    def engine(self) -> MetalComputeEngine:
        """Lazy-initialize Metal compute engine."""
        if self._engine is None:
            self._engine = MetalComputeEngine()
        return self._engine

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters.

        Args:
            params: A dictionary containing:
                - shift_x (int): Horizontal shift in pixels.
                - shift_y (int): Vertical shift in pixels.

        Raises:
            ValueError: If parameters are invalid.

        """
        for key in ("shift_x", "shift_y"):
            if key not in params:
                msg = f"Parameter '{key}' is required."
                raise ValueError(msg)
            if not isinstance(params[key], int):
                msg = f"Parameter '{key}' must be an integer."
                raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply Metal-accelerated chromatic aberration.

        The green channel is kept as reference, while red and blue channels
        are shifted in opposite directions to create color fringing.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'shift_x' and 'shift_y'.

        Returns:
            The transformed PIL Image.

        """
        self.validate_params(params)
        shift_x: int = params["shift_x"]
        shift_y: int = params["shift_y"]

        if image.mode not in ("RGB", "RGBA"):
            return image.copy()  # Effect only applies to color images

        img_array = np.array(image)

        # Separate RGB from alpha if needed
        if image.mode == "RGBA":
            rgb = img_array[..., :3]
            alpha = img_array[..., 3]
        else:
            rgb = img_array
            alpha = None

        # Apply chromatic aberration using Metal
        output_rgb = self.engine.apply_chromatic_aberration(rgb, shift_x, shift_y)

        # Recombine with alpha if needed
        if alpha is not None:
            output_array = np.dstack([output_rgb, alpha])
        else:
            output_array = output_rgb

        return Image.fromarray(output_array)

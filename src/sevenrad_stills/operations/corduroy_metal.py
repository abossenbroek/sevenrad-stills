"""
Pure Metal-accelerated corduroy striping operation.

Simulates "corduroy" or "banding" artifacts from push-broom and whisk-broom
scanners where individual detector elements have slightly different sensitivity
due to calibration drift or manufacturing variations. Uses native Metal compute
shaders for maximum GPU performance.
"""

from typing import Any, Literal

import numpy as np
from PIL import Image
from skimage.util import img_as_float32, img_as_ubyte

from sevenrad_stills.operations.base import BaseImageOperation

try:
    import Metal
    import objc

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

# Constants
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
MIN_DENSITY = 0.0
MAX_DENSITY = 1.0

# Metal shader source code
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void apply_vertical_stripes(
    device float *img [[buffer(0)]],
    device const float *multipliers [[buffer(1)]],
    constant int &height [[buffer(2)]],
    constant int &width [[buffer(3)]],
    constant int &num_channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    if (x >= width || y >= height) return;

    float mult = multipliers[x];

    if (num_channels == 1) {
        // Grayscale
        int idx = y * width + x;
        img[idx] = clamp(img[idx] * mult, 0.0f, 1.0f);
    } else {
        // RGB
        for (int c = 0; c < 3; c++) {
            int idx = (y * width + x) * 3 + c;
            img[idx] = clamp(img[idx] * mult, 0.0f, 1.0f);
        }
    }
}

kernel void apply_horizontal_stripes(
    device float *img [[buffer(0)]],
    device const float *multipliers [[buffer(1)]],
    constant int &height [[buffer(2)]],
    constant int &width [[buffer(3)]],
    constant int &num_channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    if (x >= width || y >= height) return;

    float mult = multipliers[y];

    if (num_channels == 1) {
        // Grayscale
        int idx = y * width + x;
        img[idx] = clamp(img[idx] * mult, 0.0f, 1.0f);
    } else {
        // RGB
        for (int c = 0; c < 3; c++) {
            int idx = (y * width + x) * 3 + c;
            img[idx] = clamp(img[idx] * mult, 0.0f, 1.0f);
        }
    }
}
"""


class MetalComputeEngine:
    """Metal compute engine for corduroy operation."""

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

        # Create pipeline states
        self.vertical_pipeline = self._create_pipeline("apply_vertical_stripes")
        self.horizontal_pipeline = self._create_pipeline("apply_horizontal_stripes")

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

    def apply_stripes(
        self,
        img_array: np.ndarray,
        multipliers: np.ndarray,
        orientation: Literal["vertical", "horizontal"],
    ) -> None:
        """
        Apply corduroy striping using Metal compute shaders.

        Args:
            img_array: Image array (H, W) or (H, W, 3) - modified in-place
            multipliers: Array of multipliers (W,) for vertical or (H,) for horizontal
            orientation: 'vertical' or 'horizontal'

        """
        h, w = img_array.shape[:2]
        num_channels = 1 if img_array.ndim == 2 else 3  # noqa: PLR2004

        # Flatten array for Metal
        img_flat = img_array.ravel().astype(np.float32)

        # Create Metal buffers
        img_buffer = self.device.newBufferWithBytes_length_options_(
            img_flat.ctypes.data,
            img_flat.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        mult_buffer = self.device.newBufferWithBytes_length_options_(
            multipliers.ctypes.data,
            multipliers.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create buffers for scalar parameters
        height_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([h], dtype=np.int32).ctypes.data,
            4,
            Metal.MTLResourceStorageModeShared,
        )
        width_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([w], dtype=np.int32).ctypes.data,
            4,
            Metal.MTLResourceStorageModeShared,
        )
        channels_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([num_channels], dtype=np.int32).ctypes.data,
            4,
            Metal.MTLResourceStorageModeShared,
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        pipeline = (
            self.vertical_pipeline
            if orientation == "vertical"
            else self.horizontal_pipeline
        )
        compute_encoder.setComputePipelineState_(pipeline)
        compute_encoder.setBuffer_offset_atIndex_(img_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(mult_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(height_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(width_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(channels_buffer, 0, 4)

        # Calculate thread groups
        thread_group_size = Metal.MTLSize(16, 16, 1)
        grid_size = Metal.MTLSize(
            (w + 15) // 16 * 16,  # Round up to multiple of 16
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

        # Copy results back to numpy array
        result_ptr = img_buffer.contents()
        result_array = np.frombuffer(
            objc.PyObjC_PythonFromObjC(result_ptr, len(img_flat) * 4),
            dtype=np.float32,
        ).copy()

        # Reshape and copy back to original array
        img_array[:] = result_array.reshape(img_array.shape)


class CorduroyMetalOperation(BaseImageOperation):
    """
    Apply Metal-accelerated corduroy striping to simulate detector calibration errors.

    Creates subtle vertical or horizontal banding by simulating "hot" (overly
    sensitive) and "cold" (less sensitive) detector elements in a push-broom
    or whisk-broom scanner array.

    In real satellite sensors, each detector in a linear array may have slightly
    different gain due to:
    - Manufacturing variation in sensitivity
    - Calibration drift over time
    - Temperature effects on individual detectors
    - Radiation damage accumulation

    This creates characteristic "corduroy" patterns - subtle repeating lines
    of slightly brighter or darker pixels running perpendicular to the scan
    direction.

    Performance: Pure Metal implementation provides maximum GPU performance by
    using native Metal compute shaders without intermediate frameworks.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated corduroy striping operation."""
        super().__init__("corduroy_metal")
        self._engine: MetalComputeEngine | None = None

    @property
    def engine(self) -> MetalComputeEngine:
        """Lazy-initialize Metal compute engine."""
        if self._engine is None:
            self._engine = MetalComputeEngine()
        return self._engine

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for corduroy striping operation.

        Args:
            params: A dictionary containing:
                - strength (float): Striping intensity (0.0 to 1.0), maps to
                  multiplier of 1.0 ± strength x 0.2
                - orientation (str): 'vertical' or 'horizontal' line direction
                - density (float): Proportion of lines affected (0.0 to 1.0)
                - seed (int, optional): Random seed for reproducibility

        Raises:
            ValueError: If parameters are invalid.

        """
        if "strength" not in params:
            msg = "Corduroy operation requires 'strength' parameter."
            raise ValueError(msg)
        strength = params["strength"]
        if not isinstance(strength, (int, float)) or not (
            MIN_STRENGTH <= strength <= MAX_STRENGTH
        ):
            msg = f"Strength must be a float between {MIN_STRENGTH} and {MAX_STRENGTH}."
            raise ValueError(msg)

        if "orientation" not in params:
            msg = "Corduroy operation requires 'orientation' parameter."
            raise ValueError(msg)
        orientation = params["orientation"]
        if orientation not in ("vertical", "horizontal"):
            msg = "Orientation must be 'vertical' or 'horizontal'."
            raise ValueError(msg)

        if "density" not in params:
            msg = "Corduroy operation requires 'density' parameter."
            raise ValueError(msg)
        density = params["density"]
        if not isinstance(density, (int, float)) or not (
            MIN_DENSITY <= density <= MAX_DENSITY
        ):
            msg = f"Density must be a float between {MIN_DENSITY} and {MAX_DENSITY}."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply Metal-accelerated corduroy striping to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'strength', 'orientation', 'density',
                    and optional 'seed'.

        Returns:
            The PIL Image with corduroy striping applied.

        """
        self.validate_params(params)
        strength: float = params["strength"]
        orientation: Literal["vertical", "horizontal"] = params["orientation"]
        density: float = params["density"]
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert to float array (0.0 to 1.0) using skimage utility
        img_float = img_as_float32(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_float[..., :3].copy()
            alpha = img_float[..., 3:4]
            h, w = rgb.shape[:2]
        else:
            rgb = img_float.copy()
            alpha = None
            h, w = rgb.shape[:2]

        # Determine number of lines to affect
        if orientation == "vertical":
            num_lines = int(density * w)
            total_lines = w
        else:  # horizontal
            num_lines = int(density * h)
            total_lines = h

        if num_lines > 0:
            # Select random lines
            affected_lines = rng.choice(total_lines, size=num_lines, replace=False)

            # Generate random multipliers for each line
            # strength maps to range [1.0 - strength*0.2, 1.0 + strength*0.2]
            multipliers_affected = rng.uniform(
                1.0 - strength * 0.2,
                1.0 + strength * 0.2,
                size=num_lines,
            ).astype(np.float32)

            # Create an array of multipliers, with 1.0 for unaffected lines
            multipliers_array = np.ones(total_lines, dtype=np.float32)
            multipliers_array[affected_lines] = multipliers_affected

            # Apply Metal compute shader
            self.engine.apply_stripes(rgb, multipliers_array, orientation)

        # Recombine with alpha if needed
        if alpha is not None:
            output_float = np.concatenate([rgb, alpha], axis=2)
        else:
            output_float = rgb

        # Convert back to uint8 using skimage utility
        output_array = img_as_ubyte(output_float)
        return Image.fromarray(output_array)

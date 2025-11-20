"""
Pure Metal-accelerated SLC-Off operation with custom Metal kernels.

Simulates the 2003 Landsat 7 ETM+ Scan Line Corrector failure using native
Metal compute shaders for maximum GPU performance on Apple Silicon.
"""

import struct
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

# Constants
MIN_GAP_WIDTH = 0.0
MAX_GAP_WIDTH = 0.5
MIN_SCAN_PERIOD = 2
MAX_SCAN_PERIOD = 100
RGB_CHANNELS = 3

# Metal shader source code
METAL_SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct CreateGapMaskParams {
    int height;
    int width;
    int center_y;
    float gap_width;
    int scan_period;
    float diagonal_offset_per_row;
};

struct FillRgbParams {
    int height;
    int width;
    uchar fill_r;
    uchar fill_g;
    uchar fill_b;
};

struct FillGrayParams {
    int height;
    int width;
    uchar fill_value;
};

kernel void create_gap_mask(
    device uchar *gap_mask [[buffer(0)]],
    constant CreateGapMaskParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int y = gid.y;
    int x = gid.x;

    if (x >= params.width || y >= params.height) return;

    // Initialize to no gap
    int idx = y * params.width + x;
    gap_mask[idx] = 0;

    // Find the scan line that this pixel belongs to
    // Scan lines start at y % scan_period == 0
    int scan_cycle_start = (y / params.scan_period) * params.scan_period;
    int offset_in_cycle = y - scan_cycle_start;

    // Distance from center for the scan line start
    float distance_from_center = abs(
        float(scan_cycle_start - params.center_y)
    ) / (float(params.height) / 2.0f);
    int current_gap_width = int(
        distance_from_center * params.gap_width * float(params.width)
    );

    if (current_gap_width > 0) {
        // Determine scan direction (zig-zag pattern)
        int scan_line_number = scan_cycle_start / params.scan_period;
        int scan_direction = (scan_line_number % 2 == 0) ? 1 : -1;

        // Calculate diagonal offset for this row within the cycle
        int diagonal_shift = int(
            params.diagonal_offset_per_row *
            float(offset_in_cycle) *
            float(scan_direction)
        );

        // Gap width at current row's distance from center
        float row_distance = abs(float(y - params.center_y)) /
            (float(params.height) / 2.0f);
        int row_gap_width = int(
            row_distance * params.gap_width * float(params.width)
        );

        if (row_gap_width > 0) {
            // Center gap position with diagonal shift
            // Calculate gap boundaries with explicit integer division
            int gap_center = params.width / 2 + diagonal_shift;
            int half_row_gap = row_gap_width / 2;

            // Calculate start and end with explicit bounds checking
            int temp_start = gap_center - half_row_gap;
            int temp_end = gap_center + half_row_gap;

            int gap_start = (temp_start < 0) ? 0 : temp_start;
            int gap_end = (temp_end > params.width) ? params.width : temp_end;

            // Check if current pixel is within the gap
            if (x >= gap_start && x < gap_end) {
                gap_mask[idx] = 1;
            }
        }
    }
}

kernel void apply_constant_fill_rgb(
    device uchar *img [[buffer(0)]],
    device const uchar *gap_mask [[buffer(1)]],
    constant FillRgbParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    if (x >= params.width || y >= params.height) return;

    int mask_idx = y * params.width + x;
    if (gap_mask[mask_idx] == 1) {
        int base_idx = (y * params.width + x) * 3;
        img[base_idx + 0] = params.fill_r;
        img[base_idx + 1] = params.fill_g;
        img[base_idx + 2] = params.fill_b;
    }
}

kernel void apply_constant_fill_gray(
    device uchar *img [[buffer(0)]],
    device const uchar *gap_mask [[buffer(1)]],
    constant FillGrayParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = gid.x;
    int y = gid.y;

    if (x >= params.width || y >= params.height) return;

    int idx = y * params.width + x;
    if (gap_mask[idx] == 1) {
        img[idx] = params.fill_value;
    }
}
"""


class MetalComputeEngine:
    """Metal compute engine for SLC-Off operation."""

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

        # Storage for NumPy arrays to keep them alive during GPU operations
        self._buffer_refs: list[np.ndarray] = []

        # Compile shader library
        try:
            options = Metal.MTLCompileOptions.new()
            # Force fast-math OFF to preserve float32 precision
            options.setFastMathEnabled_(False)
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
        self.gap_mask_pipeline = self._create_pipeline("create_gap_mask")
        self.fill_rgb_pipeline = self._create_pipeline("apply_constant_fill_rgb")
        self.fill_gray_pipeline = self._create_pipeline("apply_constant_fill_gray")

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

    def create_gap_mask(
        self,
        height: int,
        width: int,
        center_y: int,
        gap_width: float,
        scan_period: int,
        diagonal_offset_per_row: float,
    ) -> np.ndarray:
        """
        Create gap mask using Metal compute shader.

        Args:
            height: Image height
            width: Image width
            center_y: Center row index
            gap_width: Maximum gap width at edges (fraction)
            scan_period: Number of rows per scan cycle
            diagonal_offset_per_row: Diagonal shift per row

        Returns:
            Gap mask as uint8 numpy array (H, W)

        """
        # Allocate output buffer
        gap_mask = np.zeros((height, width), dtype=np.uint8)

        # Ensure array is C-contiguous for zero-copy
        if not gap_mask.flags["C_CONTIGUOUS"]:
            gap_mask = np.ascontiguousarray(gap_mask)

        # Create Metal buffer with ZERO-COPY
        mask_buffer = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            gap_mask, gap_mask.nbytes, Metal.MTLResourceStorageModeShared, None
        )

        # CRITICAL: Keep NumPy array alive while Metal buffer exists
        self._buffer_refs.append(gap_mask)

        # Pack all scalar parameters into a single struct buffer
        # Use native alignment to match Metal's expectations
        params_struct = struct.pack(
            "@iiifif",  # Native byte order and alignment
            height,
            width,
            center_y,
            gap_width,
            scan_period,
            diagonal_offset_per_row,
        )
        params_buffer = self.device.newBufferWithBytes_length_options_(
            params_struct, len(params_struct), Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        compute_encoder.setComputePipelineState_(self.gap_mask_pipeline)
        compute_encoder.setBuffer_offset_atIndex_(mask_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 1)

        # Calculate thread groups
        thread_group_size = Metal.MTLSize(16, 16, 1)
        grid_size = Metal.MTLSize(
            (width + 15) // 16 * 16,
            (height + 15) // 16 * 16,
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

        # ZERO-COPY: gap_mask was modified in-place by GPU. No copy-back needed.
        self._buffer_refs.clear()
        return gap_mask

    def apply_constant_fill(
        self,
        img_array: np.ndarray,
        gap_mask: np.ndarray,
        fill_value: np.ndarray | int,
    ) -> np.ndarray:
        """
        Apply constant fill to gaps using Metal compute shader.

        Args:
            img_array: Image array (H, W) or (H, W, 3)
            gap_mask: Gap mask (H, W)
            fill_value: Fill value (scalar for grayscale, 3-element array for RGB)

        Returns:
            Modified image array (potentially a new array if input wasn't C-contiguous)

        """
        h, w = img_array.shape[:2]
        is_rgb = img_array.ndim == RGB_CHANNELS

        # Ensure arrays are C-contiguous for zero-copy (create copies if needed)
        if not img_array.flags["C_CONTIGUOUS"]:
            img_array = np.ascontiguousarray(img_array)
        if not gap_mask.flags["C_CONTIGUOUS"]:
            gap_mask = np.ascontiguousarray(gap_mask)

        # Create Metal buffers with ZERO-COPY
        img_buffer = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            img_array, img_array.nbytes, Metal.MTLResourceStorageModeShared, None
        )
        mask_buffer = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            gap_mask, gap_mask.nbytes, Metal.MTLResourceStorageModeShared, None
        )

        # CRITICAL: Keep NumPy arrays alive while Metal buffers exist
        self._buffer_refs.extend([img_array, gap_mask])

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        if is_rgb:
            # Pack RGB parameters into a struct
            params_struct = struct.pack(
                "@iiBBBx",  # Native alignment, pad to ensure proper alignment
                h,
                w,
                fill_value[0],
                fill_value[1],
                fill_value[2],
            )
            params_buffer = self.device.newBufferWithBytes_length_options_(
                params_struct, len(params_struct), Metal.MTLResourceStorageModeShared
            )

            compute_encoder.setComputePipelineState_(self.fill_rgb_pipeline)
            compute_encoder.setBuffer_offset_atIndex_(img_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(mask_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)
        else:
            # Pack Grayscale parameters into a struct
            params_struct = struct.pack(
                "@iiBxxx",  # Native alignment, pad to ensure proper alignment
                h,
                w,
                fill_value,
            )
            params_buffer = self.device.newBufferWithBytes_length_options_(
                params_struct, len(params_struct), Metal.MTLResourceStorageModeShared
            )

            compute_encoder.setComputePipelineState_(self.fill_gray_pipeline)
            compute_encoder.setBuffer_offset_atIndex_(img_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(mask_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

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

        # ZERO-COPY: img_array was modified in-place. No copy-back needed.
        self._buffer_refs.clear()
        return img_array


class SlcOffMetalOperation(BaseImageOperation):
    """
    Metal-accelerated SLC-Off artifacts simulating Landsat 7 ETM+ failure.

    Uses native Metal compute shaders for maximum GPU performance on Apple
    Silicon. Provides the fastest implementation for gap mask generation and
    constant fill operations.

    Performance: Pure Metal implementation provides maximum GPU performance by
    using native Metal compute shaders without intermediate frameworks like Taichi.
    """

    def __init__(self) -> None:
        """Initialize the Metal-accelerated SLC-Off operation."""
        super().__init__("slc_off_metal")
        self._engine: MetalComputeEngine | None = None

    @property
    def engine(self) -> MetalComputeEngine:
        """Lazy-initialize Metal compute engine."""
        if self._engine is None:
            self._engine = MetalComputeEngine()
        return self._engine

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for SLC-Off operation.

        Args:
            params: A dictionary containing:
                - gap_width (float): Maximum gap width at edges as fraction
                  of image width (0.0 to 0.5, represents ~0-50% loss)
                - scan_period (int): Number of rows per scan cycle (2 to 100)
                - fill_mode (str): Gap fill strategy - 'black', 'white', or 'mean'
                - seed (int, optional): Random seed for reproducibility
                  (used for mean fill variation)

        Raises:
            ValueError: If parameters are invalid.

        """
        if "gap_width" not in params:
            msg = "SLC-Off operation requires 'gap_width' parameter."
            raise ValueError(msg)
        gap_width = params["gap_width"]
        if not isinstance(gap_width, (int, float)) or not (
            MIN_GAP_WIDTH <= gap_width <= MAX_GAP_WIDTH
        ):
            msg = (
                f"gap_width must be a number between "
                f"{MIN_GAP_WIDTH} and {MAX_GAP_WIDTH}."
            )
            raise ValueError(msg)

        if "scan_period" not in params:
            msg = "SLC-Off operation requires 'scan_period' parameter."
            raise ValueError(msg)
        scan_period = params["scan_period"]
        if not isinstance(scan_period, int) or not (
            MIN_SCAN_PERIOD <= scan_period <= MAX_SCAN_PERIOD
        ):
            msg = (
                f"scan_period must be an integer between {MIN_SCAN_PERIOD} "
                f"and {MAX_SCAN_PERIOD}."
            )
            raise ValueError(msg)

        if "fill_mode" not in params:
            msg = "SLC-Off operation requires 'fill_mode' parameter."
            raise ValueError(msg)
        fill_mode = params["fill_mode"]
        if fill_mode not in ("black", "white", "mean"):
            msg = "fill_mode must be one of: 'black', 'white', 'mean'."
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply SLC-Off pattern to the image using Metal acceleration.

        Creates wedge-shaped gaps that widen from center to edges, simulating
        the Landsat 7 scan line corrector failure.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'gap_width', 'scan_period', 'fill_mode',
                    and optional 'seed'.

        Returns:
            The PIL Image with SLC-Off pattern applied.

        """
        self.validate_params(params)

        gap_width: float = params["gap_width"]
        scan_period: int = params["scan_period"]
        fill_mode: str = params["fill_mode"]
        seed: int | None = params.get("seed")

        # Create random number generator (used for mean fill variation)
        rng = np.random.default_rng(seed)

        # Convert to array and handle different modes
        img_array = np.array(image)

        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            has_alpha = True
        elif image.mode == "RGB":
            rgb = img_array.copy()
            alpha = None
            has_alpha = False
        else:  # Grayscale
            rgb = img_array.copy()
            alpha = None
            has_alpha = False

        h, w = rgb.shape[:2]
        center_y = h // 2
        diagonal_offset_per_row = 0.3

        # Create gap mask using Metal
        gap_mask = self.engine.create_gap_mask(
            h, w, center_y, gap_width, scan_period, diagonal_offset_per_row
        )

        # Apply fill mode
        if fill_mode == "black":
            fill_value = (
                np.array([0, 0, 0], dtype=np.uint8) if rgb.ndim == RGB_CHANNELS else 0
            )
            rgb = self.engine.apply_constant_fill(rgb, gap_mask, fill_value)
        elif fill_mode == "white":
            fill_value = (
                np.array([255, 255, 255], dtype=np.uint8)
                if rgb.ndim == RGB_CHANNELS
                else 255
            )
            rgb = self.engine.apply_constant_fill(rgb, gap_mask, fill_value)
        else:  # mean fill
            # For mean fill, use CPU implementation as it requires row-wise computation
            gap_mask_bool = gap_mask.astype(bool)
            for y in range(h):
                if np.any(gap_mask_bool[y]):
                    # Calculate mean of non-gap pixels in this row
                    if rgb.ndim == RGB_CHANNELS:  # RGB
                        row_mean = np.mean(rgb[y, ~gap_mask_bool[y]], axis=0).astype(
                            rgb.dtype
                        )
                        # Add small variation to avoid perfect uniformity
                        variation = rng.integers(-5, 6, size=3, dtype=np.int16)
                        row_mean = np.clip(
                            row_mean.astype(np.int16) + variation, 0, 255
                        ).astype(rgb.dtype)
                        rgb[y, gap_mask_bool[y]] = row_mean
                    else:  # Grayscale
                        row_mean = int(np.mean(rgb[y, ~gap_mask_bool[y]]))
                        # Add small variation
                        variation = rng.integers(-5, 6)
                        row_mean = np.clip(row_mean + variation, 0, 255)
                        rgb[y, gap_mask_bool[y]] = row_mean

        # Recombine with alpha if needed
        output_array = np.dstack([rgb, alpha]) if has_alpha else rgb

        return Image.fromarray(output_array)

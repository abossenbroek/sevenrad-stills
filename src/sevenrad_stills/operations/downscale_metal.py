"""
Pure Metal GPU-accelerated resolution downscaling operation.

Provides control over scale factors and resampling methods to achieve
various levels of pixelation and loss of detail. Uses custom Metal
kernels for maximum GPU performance.
"""

import math
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants (same as CPU version)
MIN_SCALE = 0.01
MAX_SCALE = 1.0

# Resampling method mapping for Metal
RESAMPLING_METHODS = {
    "nearest": 0,  # Harsh pixelation
    "bilinear": 1,  # Smooth downscaling
}

# Metal kernel source code
METAL_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Nearest neighbor resampling kernel
kernel void resize_nearest(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dst_w = dst.get_width();
    uint dst_h = dst.get_height();
    uint src_w = src.get_width();
    uint src_h = src.get_height();

    if (gid.x >= dst_w || gid.y >= dst_h) return;

    // Calculate source coordinates (nearest neighbor)
    uint src_x = uint(float(gid.x) * float(src_w) / float(dst_w));
    uint src_y = uint(float(gid.y) * float(src_h) / float(dst_h));

    // Clamp to valid range
    src_x = min(src_x, src_w - 1);
    src_y = min(src_y, src_h - 1);

    // Read and write pixel
    float4 color = src.read(uint2(src_x, src_y));
    dst.write(color, gid);
}

// Bilinear resampling kernel
kernel void resize_bilinear(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dst_w = dst.get_width();
    uint dst_h = dst.get_height();
    uint src_w = src.get_width();
    uint src_h = src.get_height();

    if (gid.x >= dst_w || gid.y >= dst_h) return;

    // Calculate source coordinates (floating point)
    float src_x_f = (float(gid.x) + 0.5) * float(src_w) / float(dst_w) - 0.5;
    float src_y_f = (float(gid.y) + 0.5) * float(src_h) / float(dst_h) - 0.5;

    // Get integer parts (floor)
    uint src_x0 = uint(floor(src_x_f));
    uint src_y0 = uint(floor(src_y_f));
    uint src_x1 = src_x0 + 1;
    uint src_y1 = src_y0 + 1;

    // Clamp to valid range
    src_x0 = min(src_x0, src_w - 1);
    src_y0 = min(src_y0, src_h - 1);
    src_x1 = min(src_x1, src_w - 1);
    src_y1 = min(src_y1, src_h - 1);

    // Get fractional parts
    float fx = src_x_f - floor(src_x_f);
    float fy = src_y_f - floor(src_y_f);

    // Read four neighboring pixels
    float4 c00 = src.read(uint2(src_x0, src_y0));
    float4 c01 = src.read(uint2(src_x1, src_y0));
    float4 c10 = src.read(uint2(src_x0, src_y1));
    float4 c11 = src.read(uint2(src_x1, src_y1));

    // Bilinear interpolation
    float4 c0 = mix(c00, c01, fx);
    float4 c1 = mix(c10, c11, fx);
    float4 result = mix(c0, c1, fy);

    dst.write(result, gid);
}
"""


class DownscaleMetalOperation(BaseImageOperation):
    """
    Pure Metal GPU-accelerated downscale operation.

    Downscale image resolution to create pixelation effects with custom Metal
    GPU kernels for maximum performance. Supports downscaling to a fraction of
    original size, with optional upscaling back to original dimensions to create
    visible pixelation.

    Scale factor:
    - 0.01-0.10: Extreme pixelation, heavily degraded
    - 0.10-0.25: Heavy pixelation, architectural details lost
    - 0.25-0.50: Moderate pixelation, visible block structures
    - 0.50-1.00: Subtle quality reduction

    Resampling methods:
    - nearest: Maximum pixelation, harsh block edges
    - bilinear: Softer pixelation with blended edges

    Note: Metal version provides the best performance on Apple Silicon by
    using custom GPU kernels with minimal overhead.

    Performance: Metal offers the fastest GPU acceleration on macOS through
    direct hardware access and optimized texture operations.
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated downscale operation."""
        super().__init__("downscale_metal")
        self._metal_device = None
        self._metal_library = None
        self._nearest_kernel = None
        self._bilinear_kernel = None
        self._command_queue = None
        self._initialized = False

    def _init_metal(self) -> None:
        """
        Initialize Metal device and kernels.

        Lazy initialization to avoid importing Metal unless needed.

        Raises:
            ImportError: If Metal is not available
            RuntimeError: If Metal initialization fails

        """
        if self._initialized:
            return

        try:
            # Import Metal - only available on macOS
            import Metal  # type: ignore[import-not-found]

            # Get default Metal device
            self._metal_device = Metal.MTLCreateSystemDefaultDevice()
            if self._metal_device is None:
                msg = "No Metal device found"
                raise RuntimeError(msg)

            # Create command queue
            self._command_queue = self._metal_device.newCommandQueue()

            # Compile Metal kernel library
            self._metal_library = (
                self._metal_device.newLibraryWithSource_options_error_(
                    METAL_KERNEL_SOURCE, None, None
                )[0]
            )

            if self._metal_library is None:
                msg = "Failed to compile Metal kernel"
                raise RuntimeError(msg)

            # Get kernel functions
            self._nearest_kernel = self._metal_library.newFunctionWithName_(
                "resize_nearest"
            )
            self._bilinear_kernel = self._metal_library.newFunctionWithName_(
                "resize_bilinear"
            )

            self._initialized = True

        except ImportError as e:
            msg = "Metal framework not available (macOS only)"
            raise ImportError(msg) from e

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
        Apply Metal-accelerated downscaling to image.

        Args:
            image: Input PIL Image (RGB or RGBA)
            params: Operation parameters (validated)

        Returns:
            Downscaled (and optionally upscaled) PIL Image

        Raises:
            ValueError: If image is not RGB or RGBA mode
            ImportError: If Metal is not available
            RuntimeError: If Metal operations fail

        """
        # Validate parameters first
        self.validate_params(params)

        # Initialize Metal on first use
        self._init_metal()

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
        downscaled = self._resize_metal(
            rgb, new_height, new_width, RESAMPLING_METHODS[downscale_method_name]
        )

        # Upscale back if requested
        if upscale:
            result = self._resize_metal(
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
                alpha_down = self._resize_metal(
                    alpha_2d, new_height, new_width, RESAMPLING_METHODS["nearest"]
                )
                alpha_resized = alpha_down.reshape(new_height, new_width)

            output_array = np.dstack([result, alpha_resized])
        else:
            output_array = result

        return Image.fromarray(output_array)

    def _resize_metal(
        self, img: np.ndarray, dst_h: int, dst_w: int, method: int
    ) -> np.ndarray:
        """
        Resize image using Metal GPU kernels.

        Args:
            img: Input image array (H, W, C)
            dst_h: Destination height
            dst_w: Destination width
            method: Resampling method (0=nearest, 1=bilinear)

        Returns:
            Resized image array

        Raises:
            RuntimeError: If Metal operations fail

        """
        import Metal  # type: ignore[import-not-found]

        src_h, src_w, channels = img.shape

        # Convert to float32 and normalize to [0, 1] for Metal textures
        img_float = img.astype(np.float32) / 255.0

        # Create Metal textures
        # fmt: off
        src_desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatRGBA32Float, src_w, src_h, False)  # noqa: E501
        # fmt: on
        src_desc.setUsage_(Metal.MTLTextureUsageShaderRead)

        # fmt: off
        dst_desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatRGBA32Float, dst_w, dst_h, False)  # noqa: E501
        # fmt: on
        dst_desc.setUsage_(Metal.MTLTextureUsageShaderWrite)

        src_texture = self._metal_device.newTextureWithDescriptor_(src_desc)
        dst_texture = self._metal_device.newTextureWithDescriptor_(dst_desc)

        # Expand to RGBA if needed (Metal prefers 4 channels)
        rgba_channels = 4
        if channels == rgba_channels - 1:
            rgba = np.dstack([img_float, np.ones((src_h, src_w, 1), dtype=np.float32)])
        else:
            rgba = img_float

        # Upload to source texture
        region = Metal.MTLRegionMake2D(0, 0, src_w, src_h)
        src_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            rgba.tobytes(),
            src_w * 4 * 4,  # 4 channels * 4 bytes/float
        )

        # Create compute pipeline
        kernel_func = self._nearest_kernel if method == 0 else self._bilinear_kernel
        pipeline = self._metal_device.newComputePipelineStateWithFunction_error_(
            kernel_func, None
        )[0]

        # Create command buffer and encoder
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set textures and dispatch
        encoder.setComputePipelineState_(pipeline)
        encoder.setTexture_atIndex_(src_texture, 0)
        encoder.setTexture_atIndex_(dst_texture, 1)

        # Calculate thread group sizes
        thread_group_size = Metal.MTLSizeMake(16, 16, 1)
        thread_groups = Metal.MTLSizeMake(
            math.ceil(dst_w / 16), math.ceil(dst_h / 16), 1
        )

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, thread_group_size
        )
        encoder.endEncoding()

        # Execute
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read back result
        output_float = np.zeros((dst_h, dst_w, 4), dtype=np.float32)
        dst_region = Metal.MTLRegionMake2D(0, 0, dst_w, dst_h)
        dst_texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            output_float.ctypes.data,
            dst_w * 4 * 4,  # 4 channels * 4 bytes/float
            dst_region,
            0,
        )

        # Convert back to uint8 and extract channels
        output_uint8 = (output_float * 255.0).clip(0, 255).astype(np.uint8)
        return output_uint8[:, :, :channels]

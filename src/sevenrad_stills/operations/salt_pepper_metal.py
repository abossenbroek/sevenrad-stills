"""
Pure Metal GPU-accelerated salt and pepper noise operation.

Simulates random white and black pixel defects caused by cosmic ray impacts
on CMOS sensors or manufacturing defects in detector arrays. Uses custom Metal
kernels for maximum GPU performance.
"""

import math
from typing import Any

import numpy as np
from PIL import Image

from sevenrad_stills.operations.base import BaseImageOperation

# Constants (same as CPU version)
MIN_AMOUNT = 0.0
MAX_AMOUNT = 1.0
MIN_RATIO = 0.0
MAX_RATIO = 1.0

# Metal kernel source code
METAL_KERNEL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Apply salt and pepper noise to RGB texture
kernel void apply_salt_pepper_rgb(
    texture2d<float, access::read_write> img [[texture(0)]],
    device const int2* noise_positions [[buffer(0)]],
    device const int* is_salt [[buffer(1)]],
    constant int& num_noise [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(num_noise)) return;

    int2 pos = noise_positions[tid];
    float4 color;

    if (is_salt[tid] > 0) {
        // Salt - white pixel
        color = float4(1.0, 1.0, 1.0, 1.0);
    } else {
        // Pepper - black pixel
        color = float4(0.0, 0.0, 0.0, 1.0);
    }

    // Write to all RGB channels, preserve alpha if present
    float4 current = img.read(uint2(pos));
    color.w = current.w;  // Preserve alpha
    img.write(color, uint2(pos));
}

// Apply salt and pepper noise to grayscale texture
kernel void apply_salt_pepper_gray(
    texture2d<float, access::read_write> img [[texture(0)]],
    device const int2* noise_positions [[buffer(0)]],
    device const int* is_salt [[buffer(1)]],
    constant int& num_noise [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(num_noise)) return;

    int2 pos = noise_positions[tid];
    float value;

    if (is_salt[tid] > 0) {
        value = 1.0;  // Salt - white
    } else {
        value = 0.0;  // Pepper - black
    }

    img.write(float4(value, value, value, 1.0), uint2(pos));
}
"""


class SaltPepperMetalOperation(BaseImageOperation):
    """
    Pure Metal GPU-accelerated salt and pepper noise operation.

    Add salt and pepper noise to simulate sensor defects using custom Metal
    GPU kernels for maximum performance. Salt and pepper noise represents
    random pixels outputting maximum (white) or minimum (black) values
    regardless of actual light input. This models:
    - Cosmic ray hits on orbital sensors causing single-event upsets
    - Manufacturing defects creating "hot" or "dead" pixels
    - Radiation damage accumulation in detector arrays

    The effect creates scattered white and black pixels across the image,
    distinct from Gaussian noise by its extreme binary nature.

    Note: Metal version provides the best performance on Apple Silicon by
    using custom GPU kernels with minimal overhead.

    Performance: Metal offers the fastest GPU acceleration on macOS through
    direct hardware access and optimized texture/buffer operations.
    """

    def __init__(self) -> None:
        """Initialize Metal-accelerated salt and pepper noise operation."""
        super().__init__("salt_pepper_metal")
        self._metal_device = None
        self._metal_library = None
        self._rgb_kernel = None
        self._gray_kernel = None
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
            import Metal

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
            self._rgb_kernel = self._metal_library.newFunctionWithName_(
                "apply_salt_pepper_rgb"
            )
            self._gray_kernel = self._metal_library.newFunctionWithName_(
                "apply_salt_pepper_gray"
            )

            self._initialized = True

        except ImportError as e:
            msg = "Metal framework not available (macOS only)"
            raise ImportError(msg) from e

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate parameters for salt and pepper noise operation.

        Args:
            params: A dictionary containing:
                - amount (float): Proportion of pixels affected (0.0 to 1.0).
                - salt_vs_pepper (float): Ratio of salt (white) to pepper (black)
                  pixels, where 0.0 is all pepper, 1.0 is all salt, 0.5 is equal.
                - seed (int, optional): Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.

        """
        if "amount" not in params:
            msg = "Salt and pepper operation requires 'amount' parameter."
            raise ValueError(msg)
        amount = params["amount"]
        if not isinstance(amount, (int, float)) or not (
            MIN_AMOUNT <= amount <= MAX_AMOUNT
        ):
            msg = f"Amount must be a float between {MIN_AMOUNT} and {MAX_AMOUNT}."
            raise ValueError(msg)

        if "salt_vs_pepper" not in params:
            msg = "Salt and pepper operation requires 'salt_vs_pepper' parameter."
            raise ValueError(msg)
        salt_vs_pepper = params["salt_vs_pepper"]
        if not isinstance(salt_vs_pepper, (int, float)) or not (
            MIN_RATIO <= salt_vs_pepper <= MAX_RATIO
        ):
            msg = (
                f"salt_vs_pepper must be a float between {MIN_RATIO} "
                f"and {MAX_RATIO}."
            )
            raise ValueError(msg)

        if "seed" in params and not isinstance(params["seed"], int):
            msg = "Seed must be an integer."
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply Metal-accelerated salt and pepper noise to the image.

        Args:
            image: The input PIL Image.
            params: A dictionary with 'amount', 'salt_vs_pepper', and optional 'seed'.

        Returns:
            The PIL Image with salt and pepper noise applied.

        Raises:
            ImportError: If Metal is not available
            RuntimeError: If Metal operations fail

        """
        self.validate_params(params)

        # Initialize Metal on first use
        self._init_metal()

        amount: float = params["amount"]
        salt_vs_pepper: float = params["salt_vs_pepper"]
        seed: int | None = params.get("seed")

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Convert image to array
        img_array = np.array(image)

        # Handle RGBA separately to preserve alpha channel
        if image.mode == "RGBA":
            rgb = img_array[..., :3].copy()
            alpha = img_array[..., 3]
            h, w = rgb.shape[:2]

            # Generate noise positions and salt/pepper mask
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels, dtype=np.int32)
                noise_x = rng.integers(0, w, size=num_noise_pixels, dtype=np.int32)
                noise_positions = np.stack([noise_x, noise_y], axis=1)

                # Determine which are salt vs pepper
                salt_mask = (rng.random(num_noise_pixels) < salt_vs_pepper).astype(
                    np.int32
                )

                # Apply noise using Metal kernel
                rgb = self._apply_noise_metal(
                    rgb, noise_positions, salt_mask, num_noise_pixels, is_rgb=True
                )

            # Recombine with alpha
            output_array = np.dstack([rgb, alpha])
        elif image.mode == "RGB":
            rgb = img_array.copy()
            h, w = rgb.shape[:2]

            # Generate noise positions and salt/pepper mask
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels, dtype=np.int32)
                noise_x = rng.integers(0, w, size=num_noise_pixels, dtype=np.int32)
                noise_positions = np.stack([noise_x, noise_y], axis=1)

                # Determine which are salt vs pepper
                salt_mask = (rng.random(num_noise_pixels) < salt_vs_pepper).astype(
                    np.int32
                )

                # Apply noise using Metal kernel
                rgb = self._apply_noise_metal(
                    rgb, noise_positions, salt_mask, num_noise_pixels, is_rgb=True
                )

            output_array = rgb
        else:  # Grayscale
            gray = img_array.copy()
            h, w = gray.shape

            # Generate noise positions and salt/pepper mask
            num_noise_pixels = int(amount * h * w)
            if num_noise_pixels > 0:
                # Random pixel positions
                noise_y = rng.integers(0, h, size=num_noise_pixels, dtype=np.int32)
                noise_x = rng.integers(0, w, size=num_noise_pixels, dtype=np.int32)
                noise_positions = np.stack([noise_x, noise_y], axis=1)

                # Determine which are salt vs pepper
                salt_mask = (rng.random(num_noise_pixels) < salt_vs_pepper).astype(
                    np.int32
                )

                # Apply noise using Metal kernel
                gray = self._apply_noise_metal(
                    gray, noise_positions, salt_mask, num_noise_pixels, is_rgb=False
                )

            output_array = gray

        return Image.fromarray(output_array)

    def _apply_noise_metal(
        self,
        img: np.ndarray,
        noise_positions: np.ndarray,
        is_salt: np.ndarray,
        num_noise: int,
        is_rgb: bool,
    ) -> np.ndarray:
        """
        Apply salt and pepper noise using Metal GPU kernels.

        Args:
            img: Input image array (H, W, C) or (H, W) for grayscale
            noise_positions: Array of (x, y) positions (num_noise, 2)
            is_salt: Boolean array indicating salt vs pepper
            num_noise: Number of noise pixels
            is_rgb: True for RGB/RGBA, False for grayscale

        Returns:
            Image array with noise applied

        Raises:
            RuntimeError: If Metal operations fail

        """
        import Metal

        if is_rgb:
            h, w, channels = img.shape
        else:
            h, w = img.shape
            channels = 1

        # Convert to float32 and normalize to [0, 1] for Metal textures
        img_float = img.astype(np.float32) / 255.0

        # Create Metal texture
        if is_rgb:
            # fmt: off
            tex_desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatRGBA32Float, w, h, False)  # noqa: E501
            # fmt: on
            tex_desc.setUsage_(
                Metal.MTLTextureUsageShaderRead | Metal.MTLTextureUsageShaderWrite
            )

            texture = self._metal_device.newTextureWithDescriptor_(tex_desc)  # type: ignore[attr-defined]

            # Expand to RGBA if needed (Metal prefers 4 channels)
            rgba_channels = 4
            if channels == rgba_channels - 1:
                rgba = np.dstack([img_float, np.ones((h, w, 1), dtype=np.float32)])
            else:
                rgba = img_float

            # Upload to texture
            region = Metal.MTLRegionMake2D(0, 0, w, h)
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                rgba.tobytes(),
                w * 4 * 4,  # 4 channels * 4 bytes/float
            )
        else:
            # Grayscale - use single channel texture
            # fmt: off
            tex_desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(Metal.MTLPixelFormatRGBA32Float, w, h, False)  # noqa: E501
            # fmt: on
            tex_desc.setUsage_(
                Metal.MTLTextureUsageShaderRead | Metal.MTLTextureUsageShaderWrite
            )

            texture = self._metal_device.newTextureWithDescriptor_(tex_desc)  # type: ignore[attr-defined]

            # Expand to RGBA for Metal
            rgba = np.stack([img_float] * 4, axis=2).astype(np.float32)

            # Upload to texture
            region = Metal.MTLRegionMake2D(0, 0, w, h)
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                rgba.tobytes(),
                w * 4 * 4,  # 4 channels * 4 bytes/float
            )

        # Create Metal buffers for noise data
        positions_buffer = self._metal_device.newBufferWithBytes_length_options_(  # type: ignore[attr-defined]
            noise_positions.tobytes(),
            noise_positions.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        salt_buffer = self._metal_device.newBufferWithBytes_length_options_(  # type: ignore[attr-defined]
            is_salt.tobytes(),
            is_salt.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create buffer for num_noise parameter
        num_noise_array = np.array([num_noise], dtype=np.int32)
        num_noise_buffer = self._metal_device.newBufferWithBytes_length_options_(  # type: ignore[attr-defined]
            num_noise_array.tobytes(),
            num_noise_array.nbytes,
            Metal.MTLResourceStorageModeShared,
        )

        # Create compute pipeline
        kernel_func = self._rgb_kernel if is_rgb else self._gray_kernel
        pipeline = self._metal_device.newComputePipelineStateWithFunction_error_(  # type: ignore[attr-defined]
            kernel_func, None
        )[0]

        # Create command buffer and encoder
        command_buffer = self._command_queue.commandBuffer()  # type: ignore[attr-defined]
        encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and resources
        encoder.setComputePipelineState_(pipeline)
        encoder.setTexture_atIndex_(texture, 0)
        encoder.setBuffer_offset_atIndex_(positions_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(salt_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(num_noise_buffer, 0, 2)

        # Calculate thread group sizes
        thread_group_size = Metal.MTLSizeMake(256, 1, 1)
        thread_groups = Metal.MTLSizeMake(math.ceil(num_noise / 256), 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, thread_group_size
        )
        encoder.endEncoding()

        # Execute
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read back result
        output_float = np.zeros((h, w, 4), dtype=np.float32)
        dst_region = Metal.MTLRegionMake2D(0, 0, w, h)
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            output_float.ctypes.data,
            w * 4 * 4,  # 4 channels * 4 bytes/float
            dst_region,
            0,
        )

        # Convert back to uint8 and extract channels
        output_uint8 = (output_float * 255.0).clip(0, 255).astype(np.uint8)

        if is_rgb:
            return output_uint8[:, :, :channels]  # type: ignore[no-any-return]
        return output_uint8[:, :, 0]  # type: ignore[no-any-return]

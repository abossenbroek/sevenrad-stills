"""Image operation framework for pipeline processing."""

from typing import Literal

from sevenrad_stills.operations.backend import (
    BackendNotAvailableError,
    BackendType,
    get_backend_implementation,
    get_taichi_operation,
    has_taichi_operation,
    list_taichi_operations,
    register_backend,
    register_taichi_operation,
)
from sevenrad_stills.operations.band_swap import BandSwapOperation

# Import GPU variants
from sevenrad_stills.operations.band_swap_gpu import BandSwapGPUOperation
from sevenrad_stills.operations.band_swap_metal import BandSwapMetalOperation

# Taichi pipeline operations
from sevenrad_stills.operations.band_swap_taichi import BandSwapTaichiOperation
from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation

# Import Metal variants
from sevenrad_stills.operations.bayer_filter_metal import BayerFilterMetalOperation
from sevenrad_stills.operations.bayer_filter_taichi import BayerFilterTaichiOperation
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_circular_gpu import CircularBlurGPUOperation
from sevenrad_stills.operations.blur_circular_metal import CircularBlurMetalOperation
from sevenrad_stills.operations.blur_circular_taichi import BlurCircularTaichiOperation
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation
from sevenrad_stills.operations.blur_gaussian_mlx import GaussianBlurMLXOperation
from sevenrad_stills.operations.blur_gaussian_taichi import BlurGaussianTaichiOperation
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.buffer_corruption_gpu import (
    BufferCorruptionGPUOperation,
)
from sevenrad_stills.operations.buffer_corruption_metal import (
    BufferCorruptionMetalOperation,
)
from sevenrad_stills.operations.buffer_corruption_taichi import (
    BufferCorruptionTaichiOperation,
)
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.chromatic_aberration_gpu import (
    ChromaticAberrationGPUOperation,
)
from sevenrad_stills.operations.chromatic_aberration_metal import (
    ChromaticAberrationMetalOperation,
)
from sevenrad_stills.operations.chromatic_aberration_taichi import (
    ChromaticAberrationTaichiOperation,
)
from sevenrad_stills.operations.compression import CompressionOperation
from sevenrad_stills.operations.compression_artifact import CompressionArtifactOperation
from sevenrad_stills.operations.compression_artifact_gpu import (
    CompressionArtifactGPUOperation,
)
from sevenrad_stills.operations.compression_artifact_metal import (
    CompressionArtifactMetalOperation,
)
from sevenrad_stills.operations.compression_gpu import CompressionGPUOperation
from sevenrad_stills.operations.compression_metal import CompressionMetalOperation
from sevenrad_stills.operations.corduroy import CorduroyOperation
from sevenrad_stills.operations.corduroy_gpu import CorduroyGPUOperation
from sevenrad_stills.operations.corduroy_metal import CorduroyMetalOperation
from sevenrad_stills.operations.corduroy_taichi import CorduroyTaichiOperation
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.downscale_gpu import DownscaleGPUOperation
from sevenrad_stills.operations.downscale_metal import DownscaleMetalOperation
from sevenrad_stills.operations.downscale_taichi import DownscaleTaichiOperation
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.motion_blur_gpu import MotionBlurGPUOperation
from sevenrad_stills.operations.motion_blur_metal import MotionBlurMetalOperation
from sevenrad_stills.operations.motion_blur_taichi import MotionBlurTaichiOperation
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.multi_compress_gpu import MultiCompressGPUOperation
from sevenrad_stills.operations.multi_compress_metal import MultiCompressMetalOperation
from sevenrad_stills.operations.noise import NoiseOperation
from sevenrad_stills.operations.noise_gpu import NoiseGPUOperation
from sevenrad_stills.operations.noise_metal import NoiseMetalOperation
from sevenrad_stills.operations.noise_taichi import NoiseTaichiOperation
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.salt_pepper import SaltPepperOperation
from sevenrad_stills.operations.salt_pepper_gpu import SaltPepperGPUOperation
from sevenrad_stills.operations.salt_pepper_metal import SaltPepperMetalOperation
from sevenrad_stills.operations.salt_pepper_taichi import SaltPepperTaichiOperation
from sevenrad_stills.operations.saturation import SaturationOperation
from sevenrad_stills.operations.saturation_gpu import SaturationGPUOperation
from sevenrad_stills.operations.saturation_metal import SaturationMetalOperation
from sevenrad_stills.operations.saturation_taichi import SaturationTaichiOperation
from sevenrad_stills.operations.slc_off import SlcOffOperation
from sevenrad_stills.operations.slc_off_gpu import SlcOffGPUOperation
from sevenrad_stills.operations.slc_off_metal import SlcOffMetalOperation
from sevenrad_stills.operations.slc_off_taichi import SlcOffTaichiOperation

# Register built-in operations (legacy registry for backward compatibility)
register_operation(BandSwapOperation)
register_operation(BayerFilterOperation)
register_operation(BufferCorruptionOperation)
register_operation(ChromaticAberrationOperation)
register_operation(CircularBlurOperation)
register_operation(CompressionOperation)
register_operation(CompressionArtifactOperation)
register_operation(CorduroyOperation)
register_operation(DownscaleOperation)
register_operation(GaussianBlurOperation)
register_operation(MotionBlurOperation)
register_operation(MultiCompressOperation)
register_operation(NoiseOperation)
register_operation(SaltPepperOperation)
register_operation(SaturationOperation)
register_operation(SlcOffOperation)

# Register backend-specific implementations
# band_swap: CPU + GPU + Metal
register_backend("band_swap", "cpu", BandSwapOperation)
register_backend("band_swap", "gpu", BandSwapGPUOperation)
register_backend("band_swap", "metal", BandSwapMetalOperation)

# bayer_filter: CPU + GPU + Metal
register_backend("bayer_filter", "cpu", BayerFilterOperation)
register_backend("bayer_filter", "gpu", BayerFilterGPUOperation)
register_backend("bayer_filter", "metal", BayerFilterMetalOperation)

# blur_circular: CPU + GPU + Metal
register_backend("blur_circular", "cpu", CircularBlurOperation)
register_backend("blur_circular", "gpu", CircularBlurGPUOperation)
register_backend("blur_circular", "metal", CircularBlurMetalOperation)

# blur_gaussian: CPU + GPU + Metal (MLX implementation for Metal)
register_backend("blur_gaussian", "cpu", GaussianBlurOperation)
register_backend("blur_gaussian", "gpu", GaussianBlurGPUOperation)
register_backend("blur_gaussian", "metal", GaussianBlurMLXOperation)

# buffer_corruption: CPU + GPU + Metal
register_backend("buffer_corruption", "cpu", BufferCorruptionOperation)
register_backend("buffer_corruption", "gpu", BufferCorruptionGPUOperation)
register_backend("buffer_corruption", "metal", BufferCorruptionMetalOperation)

# chromatic_aberration: CPU + GPU + Metal
register_backend("chromatic_aberration", "cpu", ChromaticAberrationOperation)
register_backend("chromatic_aberration", "gpu", ChromaticAberrationGPUOperation)
register_backend("chromatic_aberration", "metal", ChromaticAberrationMetalOperation)

# compression: CPU + GPU + Metal
register_backend("compression", "cpu", CompressionOperation)
register_backend("compression", "gpu", CompressionGPUOperation)
register_backend("compression", "metal", CompressionMetalOperation)

# compression_artifact: CPU + GPU + Metal
register_backend("compression_artifact", "cpu", CompressionArtifactOperation)
register_backend("compression_artifact", "gpu", CompressionArtifactGPUOperation)
register_backend("compression_artifact", "metal", CompressionArtifactMetalOperation)

# corduroy: CPU + GPU + Metal
register_backend("corduroy", "cpu", CorduroyOperation)
register_backend("corduroy", "gpu", CorduroyGPUOperation)
register_backend("corduroy", "metal", CorduroyMetalOperation)

# downscale: CPU + GPU + Metal
register_backend("downscale", "cpu", DownscaleOperation)
register_backend("downscale", "gpu", DownscaleGPUOperation)
register_backend("downscale", "metal", DownscaleMetalOperation)

# motion_blur: CPU + GPU + Metal
register_backend("motion_blur", "cpu", MotionBlurOperation)
register_backend("motion_blur", "gpu", MotionBlurGPUOperation)
register_backend("motion_blur", "metal", MotionBlurMetalOperation)

# multi_compress: CPU + GPU + Metal
register_backend("multi_compress", "cpu", MultiCompressOperation)
register_backend("multi_compress", "gpu", MultiCompressGPUOperation)
register_backend("multi_compress", "metal", MultiCompressMetalOperation)

# noise: CPU + GPU + Metal
register_backend("noise", "cpu", NoiseOperation)
register_backend("noise", "gpu", NoiseGPUOperation)
register_backend("noise", "metal", NoiseMetalOperation)

# salt_pepper: CPU + GPU + Metal
register_backend("salt_pepper", "cpu", SaltPepperOperation)
register_backend("salt_pepper", "gpu", SaltPepperGPUOperation)
register_backend("salt_pepper", "metal", SaltPepperMetalOperation)

# saturation: CPU + GPU + Metal
register_backend("saturation", "cpu", SaturationOperation)
register_backend("saturation", "gpu", SaturationGPUOperation)
register_backend("saturation", "metal", SaturationMetalOperation)

# slc_off: CPU + GPU + Metal
register_backend("slc_off", "cpu", SlcOffOperation)
register_backend("slc_off", "gpu", SlcOffGPUOperation)
register_backend("slc_off", "metal", SlcOffMetalOperation)

# ============================================================================
# Taichi Pipeline Operations Registration (end-to-end GPU execution)
# ============================================================================

# Simple operations
register_taichi_operation("noise", NoiseTaichiOperation)
register_taichi_operation("salt_pepper", SaltPepperTaichiOperation)
register_taichi_operation("corduroy", CorduroyTaichiOperation)
register_taichi_operation("band_swap", BandSwapTaichiOperation)
register_taichi_operation("buffer_corruption", BufferCorruptionTaichiOperation)
register_taichi_operation("saturation", SaturationTaichiOperation)

# Spatial operations (blur, chromatic aberration)
register_taichi_operation("blur_gaussian", BlurGaussianTaichiOperation)
register_taichi_operation("blur_circular", BlurCircularTaichiOperation)
register_taichi_operation("motion_blur", MotionBlurTaichiOperation)
register_taichi_operation("chromatic_aberration", ChromaticAberrationTaichiOperation)

# Complex operations (dimension changing, multi-pass)
register_taichi_operation("downscale", DownscaleTaichiOperation)
register_taichi_operation("bayer_filter", BayerFilterTaichiOperation)
register_taichi_operation("slc_off", SlcOffTaichiOperation)

__all__ = [
    "BackendNotAvailableError",
    "BackendType",
    "BandSwapOperation",
    "BandSwapTaichiOperation",
    "BayerFilterOperation",
    "BayerFilterTaichiOperation",
    "BlurCircularTaichiOperation",
    "BlurGaussianTaichiOperation",
    "BufferCorruptionOperation",
    "BufferCorruptionTaichiOperation",
    "ChromaticAberrationOperation",
    "ChromaticAberrationTaichiOperation",
    "CircularBlurOperation",
    "CompressionArtifactOperation",
    "CompressionOperation",
    "CorduroyOperation",
    "CorduroyTaichiOperation",
    "DownscaleOperation",
    "DownscaleTaichiOperation",
    "GaussianBlurOperation",
    "ImageOperation",
    "MotionBlurOperation",
    "MotionBlurTaichiOperation",
    "MultiCompressOperation",
    "NoiseOperation",
    "NoiseTaichiOperation",
    "OperationRegistry",
    "SaltPepperOperation",
    "SaltPepperTaichiOperation",
    "SaturationOperation",
    "SaturationTaichiOperation",
    "SlcOffOperation",
    "SlcOffTaichiOperation",
    "get_backend_implementation",
    "get_operation",
    "get_taichi_operation",
    "has_taichi_operation",
    "list_operations",
    "list_taichi_operations",
    "register_backend",
    "register_operation",
    "register_taichi_operation",
]

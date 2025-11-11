"""Image operation framework for pipeline processing."""

from typing import Literal

from sevenrad_stills.operations.backend import (
    BackendNotAvailableError,
    BackendType,
    get_backend_implementation,
    register_backend,
)
from sevenrad_stills.operations.band_swap import BandSwapOperation

# Import GPU variants
from sevenrad_stills.operations.band_swap_gpu import BandSwapGPUOperation
from sevenrad_stills.operations.band_swap_metal import BandSwapMetalOperation
from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation

# Import Metal variants
from sevenrad_stills.operations.bayer_filter_metal import BayerFilterMetalOperation
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_circular_gpu import CircularBlurGPUOperation
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.buffer_corruption_gpu import (
    BufferCorruptionGPUOperation,
)
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.chromatic_aberration_gpu import (
    ChromaticAberrationGPUOperation,
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
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.downscale_gpu import DownscaleGPUOperation
from sevenrad_stills.operations.downscale_metal import DownscaleMetalOperation
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.motion_blur_gpu import MotionBlurGPUOperation
from sevenrad_stills.operations.motion_blur_metal import MotionBlurMetalOperation
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.multi_compress_gpu import MultiCompressGPUOperation
from sevenrad_stills.operations.multi_compress_metal import MultiCompressMetalOperation
from sevenrad_stills.operations.noise import NoiseOperation
from sevenrad_stills.operations.noise_gpu import NoiseGPUOperation
from sevenrad_stills.operations.noise_metal import NoiseMetalOperation
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.salt_pepper import SaltPepperOperation
from sevenrad_stills.operations.salt_pepper_gpu import SaltPepperGPUOperation
from sevenrad_stills.operations.salt_pepper_metal import SaltPepperMetalOperation
from sevenrad_stills.operations.saturation import SaturationOperation
from sevenrad_stills.operations.saturation_gpu import SaturationGPUOperation
from sevenrad_stills.operations.saturation_metal import SaturationMetalOperation
from sevenrad_stills.operations.slc_off import SlcOffOperation
from sevenrad_stills.operations.slc_off_gpu import SlcOffGPUOperation
from sevenrad_stills.operations.slc_off_metal import SlcOffMetalOperation

# Note: buffer_corruption_metal needs wrapper class - TODO
# from sevenrad_stills.operations.buffer_corruption_metal import (
#     BufferCorruptionMetalOperation,
# )

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

# blur_circular: CPU + GPU
register_backend("blur_circular", "cpu", CircularBlurOperation)
register_backend("blur_circular", "gpu", CircularBlurGPUOperation)

# blur_gaussian: CPU + GPU
register_backend("blur_gaussian", "cpu", GaussianBlurOperation)
register_backend("blur_gaussian", "gpu", GaussianBlurGPUOperation)

# buffer_corruption: CPU + GPU (Metal TODO: needs Operation wrapper)
register_backend("buffer_corruption", "cpu", BufferCorruptionOperation)
register_backend("buffer_corruption", "gpu", BufferCorruptionGPUOperation)
# TODO: Add BufferCorruptionMetalOperation wrapper class
# register_backend("buffer_corruption", "metal", BufferCorruptionMetalOperation)

# chromatic_aberration: CPU + GPU
register_backend("chromatic_aberration", "cpu", ChromaticAberrationOperation)
register_backend("chromatic_aberration", "gpu", ChromaticAberrationGPUOperation)

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

__all__ = [
    "BackendNotAvailableError",
    "BackendType",
    "BandSwapOperation",
    "BayerFilterOperation",
    "BufferCorruptionOperation",
    "ChromaticAberrationOperation",
    "CircularBlurOperation",
    "CompressionArtifactOperation",
    "CompressionOperation",
    "CorduroyOperation",
    "DownscaleOperation",
    "GaussianBlurOperation",
    "ImageOperation",
    "MotionBlurOperation",
    "MultiCompressOperation",
    "NoiseOperation",
    "OperationRegistry",
    "SaltPepperOperation",
    "SaturationOperation",
    "SlcOffOperation",
    "get_backend_implementation",
    "get_operation",
    "list_operations",
    "register_backend",
    "register_operation",
]

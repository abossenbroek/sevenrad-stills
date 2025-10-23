"""Image operation framework for pipeline processing."""

from sevenrad_stills.operations.band_swap import BandSwapOperation
from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.compression import CompressionOperation
from sevenrad_stills.operations.compression_artifact import CompressionArtifactOperation
from sevenrad_stills.operations.corduroy import CorduroyOperation
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.noise import NoiseOperation
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.salt_pepper import SaltPepperOperation
from sevenrad_stills.operations.saturation import SaturationOperation
from sevenrad_stills.operations.slc_off import SlcOffOperation

# Register built-in operations
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

__all__ = [
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
    "get_operation",
    "list_operations",
    "register_operation",
]

"""Image operation framework for pipeline processing."""

from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.compression import CompressionOperation
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.noise import NoiseOperation
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.saturation import SaturationOperation

# Register built-in operations
register_operation(BayerFilterOperation)
register_operation(ChromaticAberrationOperation)
register_operation(CircularBlurOperation)
register_operation(CompressionOperation)
register_operation(DownscaleOperation)
register_operation(GaussianBlurOperation)
register_operation(MotionBlurOperation)
register_operation(MultiCompressOperation)
register_operation(NoiseOperation)
register_operation(SaturationOperation)

__all__ = [
    "BayerFilterOperation",
    "ChromaticAberrationOperation",
    "CircularBlurOperation",
    "CompressionOperation",
    "DownscaleOperation",
    "GaussianBlurOperation",
    "ImageOperation",
    "MotionBlurOperation",
    "MultiCompressOperation",
    "NoiseOperation",
    "OperationRegistry",
    "SaturationOperation",
    "get_operation",
    "list_operations",
    "register_operation",
]

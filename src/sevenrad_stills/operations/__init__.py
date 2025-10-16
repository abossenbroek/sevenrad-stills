"""Image operation framework for pipeline processing."""

from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.compression import CompressionOperation
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.saturation import SaturationOperation

# Register built-in operations
register_operation(SaturationOperation)
register_operation(CompressionOperation)
register_operation(DownscaleOperation)
register_operation(MotionBlurOperation)
register_operation(MultiCompressOperation)

__all__ = [
    "CompressionOperation",
    "DownscaleOperation",
    "ImageOperation",
    "MotionBlurOperation",
    "MultiCompressOperation",
    "OperationRegistry",
    "SaturationOperation",
    "get_operation",
    "list_operations",
    "register_operation",
]

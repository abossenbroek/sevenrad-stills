"""Image operation framework for pipeline processing."""

from sevenrad_stills.operations.base import ImageOperation, OperationRegistry
from sevenrad_stills.operations.registry import (
    get_operation,
    list_operations,
    register_operation,
)
from sevenrad_stills.operations.saturation import SaturationOperation

# Register built-in operations
register_operation(SaturationOperation)

__all__ = [
    "ImageOperation",
    "OperationRegistry",
    "SaturationOperation",
    "get_operation",
    "list_operations",
    "register_operation",
]

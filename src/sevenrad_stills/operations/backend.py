"""
Backend resolution system for image operations.

Maps operation names to their CPU, GPU, and Metal implementations
and handles backend selection with validation.
"""

from typing import Literal

from sevenrad_stills.operations.base import ImageOperation

# Backend type definition
BackendType = Literal["cpu", "gpu", "metal"]

# Backend support matrix: operation_name -> {backend -> class}
# This will be populated by the operation registry
_BACKEND_REGISTRY: dict[str, dict[str, type[ImageOperation]]] = {}


class BackendNotAvailableError(Exception):
    """Raised when requested backend is not available for an operation."""

    pass


def register_backend(
    operation_name: str, backend: BackendType, operation_class: type[ImageOperation]
) -> None:
    """
    Register an operation class for a specific backend.

    Args:
        operation_name: Base name of the operation (e.g., 'chromatic_aberration')
        backend: Backend type ('cpu', 'gpu', or 'metal')
        operation_class: Implementation class for this backend

    """
    if operation_name not in _BACKEND_REGISTRY:
        _BACKEND_REGISTRY[operation_name] = {}
    _BACKEND_REGISTRY[operation_name][backend] = operation_class


def get_backend_implementation(
    operation_name: str, backend: BackendType
) -> ImageOperation:
    """
    Get the implementation of an operation for a specific backend.

    Args:
        operation_name: Base name of the operation
        backend: Requested backend type

    Returns:
        Instance of the operation for the requested backend

    Raises:
        KeyError: If operation not found
        BackendNotAvailableError: If backend not available for this operation

    """
    if operation_name not in _BACKEND_REGISTRY:
        msg = f"Operation '{operation_name}' not found in backend registry"
        raise KeyError(msg)

    backends = _BACKEND_REGISTRY[operation_name]
    if backend not in backends:
        available = list(backends.keys())
        msg = (
            f"Backend '{backend}' not available for operation '{operation_name}'. "
            f"Available backends: {', '.join(available)}"
        )
        raise BackendNotAvailableError(msg)

    return backends[backend]()


def get_available_backends(operation_name: str) -> list[str]:
    """
    Get list of available backends for an operation.

    Args:
        operation_name: Base name of the operation

    Returns:
        List of available backend types

    Raises:
        KeyError: If operation not found

    """
    if operation_name not in _BACKEND_REGISTRY:
        msg = f"Operation '{operation_name}' not found in backend registry"
        raise KeyError(msg)

    return list(_BACKEND_REGISTRY[operation_name].keys())


def has_backend(operation_name: str, backend: BackendType) -> bool:
    """
    Check if an operation has an implementation for a specific backend.

    Args:
        operation_name: Base name of the operation
        backend: Backend to check

    Returns:
        True if backend is available, False otherwise

    """
    if operation_name not in _BACKEND_REGISTRY:
        return False
    return backend in _BACKEND_REGISTRY[operation_name]


def list_all_operations() -> list[str]:
    """
    List all operations registered in the backend registry.

    Returns:
        Sorted list of operation names

    """
    return sorted(_BACKEND_REGISTRY.keys())


def get_backend_matrix() -> dict[str, list[str]]:
    """
    Get the complete backend support matrix.

    Returns:
        Dictionary mapping operation names to their available backends

    """
    return {op: list(backends.keys()) for op, backends in _BACKEND_REGISTRY.items()}

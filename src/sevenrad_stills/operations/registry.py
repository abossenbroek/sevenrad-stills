"""
Global operation registry.

Provides a centralized registry for all image operations.
"""

from sevenrad_stills.operations.base import ImageOperation, OperationRegistry

# Global registry instance
_registry = OperationRegistry()


def register_operation(operation_class: type[ImageOperation]) -> None:
    """
    Register an operation class globally.

    Args:
        operation_class: Operation class to register

    """
    _registry.register(operation_class)


def get_operation(name: str) -> ImageOperation:
    """
    Get an operation instance by name from global registry.

    Args:
        name: Operation name

    Returns:
        Operation instance

    Raises:
        KeyError: If operation not found

    """
    return _registry.get(name)


def list_operations() -> list[str]:
    """
    List all registered operation names.

    Returns:
        Sorted list of operation names

    """
    return _registry.list_operations()


def is_operation_registered(name: str) -> bool:
    """
    Check if an operation is registered.

    Args:
        name: Operation name to check

    Returns:
        True if registered, False otherwise

    """
    return _registry.is_registered(name)

"""
Base classes and protocols for image operations.

Defines the interface that all image operations must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class ImageOperation(Protocol):
    """Protocol for image operations in the pipeline."""

    @property
    def name(self) -> str:
        """
        Get the operation name.

        Returns:
            Operation identifier

        """
        ...

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply the operation to an image.

        Args:
            image: Input PIL Image
            params: Operation-specific parameters

        Returns:
            Transformed PIL Image

        """
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate operation parameters.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """
        ...

    def save_image(
        self, image: Image.Image, output_path: Path, quality: int = 95
    ) -> None:
        """
        Save image to disk.

        Args:
            image: PIL Image to save
            output_path: Output file path
            quality: JPEG quality (if applicable)

        """
        ...


class BaseImageOperation(ABC):
    """
    Abstract base class for image operations.

    Provides common functionality for all operations.
    """

    def __init__(self, operation_name: str) -> None:
        """
        Initialize base operation.

        Args:
            operation_name: Name/identifier for this operation

        """
        self._name = operation_name

    @property
    def name(self) -> str:
        """Get the operation name."""
        return self._name

    @abstractmethod
    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Apply the operation to an image.

        Args:
            image: Input PIL Image
            params: Operation-specific parameters

        Returns:
            Transformed PIL Image

        """

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Validate operation parameters.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If parameters are invalid

        """

    def save_image(
        self, image: Image.Image, output_path: Path, quality: int = 95
    ) -> None:
        """
        Save image to disk.

        Args:
            image: PIL Image to save
            output_path: Output file path
            quality: JPEG quality (if applicable)

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in (".jpg", ".jpeg"):
            image.save(output_path, quality=quality, optimize=True)
        else:
            image.save(output_path)


class OperationRegistry:
    """Registry for managing available image operations."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._operations: dict[str, type[ImageOperation]] = {}

    def register(self, operation_class: type[ImageOperation]) -> None:
        """
        Register an operation class.

        Args:
            operation_class: Operation class to register

        """
        # Instantiate to get the name
        instance = operation_class()
        self._operations[instance.name] = operation_class

    def get(self, name: str) -> ImageOperation:
        """
        Get an operation instance by name.

        Args:
            name: Operation name

        Returns:
            Operation instance

        Raises:
            KeyError: If operation not found

        """
        if name not in self._operations:
            available = ", ".join(self._operations.keys())
            msg = f"Operation '{name}' not found. Available: {available}"
            raise KeyError(msg)
        return self._operations[name]()

    def list_operations(self) -> list[str]:
        """
        List all registered operation names.

        Returns:
            List of operation names

        """
        return sorted(self._operations.keys())

    def is_registered(self, name: str) -> bool:
        """
        Check if an operation is registered.

        Args:
            name: Operation name to check

        Returns:
            True if registered, False otherwise

        """
        return name in self._operations

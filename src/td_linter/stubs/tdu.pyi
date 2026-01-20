"""
TouchDesigner tdu module stubs for type checking.

The tdu module provides utility functions for TouchDesigner Python scripts.

Note: Names follow TouchDesigner API conventions (camelCase) which differ from PEP 8.
"""
# ruff: noqa: N802, N803, ANN401

from typing import Any

class Dependency:
    """Dependency tracking object."""

    @property
    def val(self) -> Any:
        """Current value."""
        ...

    @val.setter
    def val(self, value: Any) -> None: ...
    @property
    def callbacks(self) -> list[Any]:
        """Registered callbacks."""
        ...

def remap(
    value: float, fromMin: float, fromMax: float, toMin: float, toMax: float
) -> float:
    """Remap a value from one range to another."""
    ...

def clamp(value: float, minVal: float, maxVal: float) -> float:
    """Clamp a value between min and max."""
    ...

def rand(seed: float | None = None) -> float:
    """Generate a random number."""
    ...

def base(name: str) -> str:
    """Get the base name without digits."""
    ...

def digits(name: str) -> str:
    """Get the trailing digits from a name."""
    ...

def validName(name: str) -> str:
    """Convert string to valid operator name."""
    ...

def split(path: str) -> tuple[str, str]:
    """Split path into parent and name."""
    ...

def collapsePath(path: str) -> str:
    """Collapse path for storage."""
    ...

def expandPath(path: str) -> str:
    """Expand collapsed path."""
    ...

def legacyRecook() -> None:
    """Trigger legacy recook."""
    ...

class Vector:
    """3D vector class."""

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None: ...
    def length(self) -> float:
        """Vector length."""
        ...

    def normalize(self) -> "Vector":
        """Return normalized vector."""
        ...

    def dot(self, other: "Vector") -> float:
        """Dot product."""
        ...

    def cross(self, other: "Vector") -> "Vector":
        """Cross product."""
        ...

class Matrix:
    """4x4 transformation matrix."""

    def __init__(self) -> None: ...
    def identity(self) -> "Matrix":
        """Set to identity matrix."""
        ...

    def translate(self, x: float, y: float, z: float) -> "Matrix":
        """Apply translation."""
        ...

    def rotate(self, axis: Vector, angle: float) -> "Matrix":
        """Apply rotation."""
        ...

    def scale(self, x: float, y: float, z: float) -> "Matrix":
        """Apply scale."""
        ...

    def invert(self) -> "Matrix":
        """Return inverted matrix."""
        ...

    def transpose(self) -> "Matrix":
        """Return transposed matrix."""
        ...

class Position:
    """2D/3D position."""

    x: float
    y: float
    z: float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None: ...

class Color:
    """RGBA color."""

    r: float
    g: float
    b: float
    a: float

    def __init__(
        self, r: float = 0, g: float = 0, b: float = 0, a: float = 1
    ) -> None: ...

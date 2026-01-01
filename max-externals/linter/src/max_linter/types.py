"""Core type definitions for Max linter validation.

This module defines the type system used for strict type checking of Jitter
connections in Max patchers. It includes:

- Severity: Validation message severity levels (ERROR, WARNING, INFO)
- JitterType: Types of data flowing through Jitter connections
- TYPE_COMPATIBLE: Compatibility matrix for connection type checking
- types_compatible(): Function to check if source type can connect to dest type
"""

from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Validation message severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class JitterType(Enum):
    """Jitter connection type for strict type checking.

    Represents the types of data that can flow through Jitter connections.
    Used by TYPE_COMPATIBLE to enforce strict type matching.
    """

    TEXTURE = "jit_gl_texture"  # GPU texture
    MATRIX = "jit_matrix"  # CPU matrix
    TEXTURE_NAME = "texture_name"  # String reference to named texture
    BANG = "bang"  # Trigger message
    MESSAGE = "message"  # Any Max message
    INFO = "info"  # Dump outlet metadata
    UNKNOWN = "unknown"  # Unrecognized objects


# Strict type compatibility matrix - NO implicit conversions
TYPE_COMPATIBLE: dict[tuple[JitterType, JitterType], bool] = {
    # Same type -> compatible
    (JitterType.TEXTURE, JitterType.TEXTURE): True,
    (JitterType.MATRIX, JitterType.MATRIX): True,
    # Cross-domain connections -> ERROR (strict mode)
    (JitterType.TEXTURE, JitterType.MATRIX): False,  # ERROR - can't feed GPU to CPU
    (JitterType.MATRIX, JitterType.TEXTURE): False,  # ERROR - can't feed CPU to GPU
    # Info outlet -> data inlet -> ERROR
    (JitterType.INFO, JitterType.TEXTURE): False,  # ERROR - info is metadata
    (JitterType.INFO, JitterType.MATRIX): False,  # ERROR
    # Bang/message -> data inlet -> ERROR
    (JitterType.BANG, JitterType.TEXTURE): False,  # ERROR - bang can't be image data
    (JitterType.BANG, JitterType.MATRIX): False,  # ERROR
    (JitterType.MESSAGE, JitterType.TEXTURE): False,  # ERROR
    (JitterType.MESSAGE, JitterType.MATRIX): False,  # ERROR
    # BANG and MESSAGE can go to MESSAGE inlets
    (JitterType.BANG, JitterType.MESSAGE): True,
    (JitterType.MESSAGE, JitterType.MESSAGE): True,
    # UNKNOWN is permissive (for unrecognized objects)
    (JitterType.UNKNOWN, JitterType.UNKNOWN): True,
}


def types_compatible(source: JitterType, dest: JitterType) -> bool:
    """Check if source type can connect to dest type (strict mode).

    Args:
        source: The type of the outlet (source of connection).
        dest: The expected type of the inlet (destination of connection).

    Returns:
        True if the connection is type-compatible, False otherwise.
    """
    # TEXTURE can go to "matrix_or_texture" display sinks
    if source == JitterType.TEXTURE and dest == JitterType.MATRIX:
        return False  # Strict - use explicit conversion
    return TYPE_COMPATIBLE.get((source, dest), False)

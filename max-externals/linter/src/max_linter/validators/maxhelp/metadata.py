"""Metadata validation mixin for Max help patchers.

This module provides the MetadataValidatorMixin class that validates
patcher metadata (description, tags) in Max/MSP help patchers.
"""

from __future__ import annotations

from typing import Any


class MetadataValidatorMixin:
    """Mixin providing metadata validation methods.

    Validates patcher metadata fields.

    Rules:
        metadata: Missing description or tags field
    """

    # These attributes must be provided by the composing class
    data: dict[str, Any]

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def _validate_metadata(self) -> bool:
        """Validate patcher metadata (description, tags).

        Checks that the patcher has:
        - A description field for documentation
        - A tags field for searchability

        Returns:
            True (metadata issues are only warnings, not errors).
        """
        valid = True
        patcher = self.data.get("patcher", {})

        if not patcher.get("description"):
            self.warning("metadata", "Missing 'description' field")

        if not patcher.get("tags"):
            self.warning("metadata", "Missing 'tags' field")

        return valid

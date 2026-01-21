"""Unit tests for BaseTaichiOperation abstract base class."""

from typing import Any

import numpy as np
import pytest
from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
from sevenrad_stills.pipeline.protocols import TempFieldSpec


class ConcreteTestOperation(BaseTaichiOperation):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        """Initialize test operation."""
        super().__init__("test_operation")

    def apply_to_field(  # noqa: PLR0913
        self,
        source: Any,
        dest: Any,
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """Mock implementation for testing."""
        pass

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],  # noqa: ARG002
    ) -> np.ndarray:
        """Return image unchanged for testing."""
        return image

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate that required param is present."""
        if "required" not in params:
            raise ValueError("Missing 'required' param")


class TestBaseTaichiOperation:
    """Test suite for BaseTaichiOperation abstract base class."""

    @pytest.fixture
    def operation(self) -> ConcreteTestOperation:
        """Provide a concrete test operation instance."""
        return ConcreteTestOperation()

    def test_name_property(self, operation: ConcreteTestOperation) -> None:
        """Test that name property returns correct value."""
        assert operation.name == "test_operation"

    def test_supports_inplace_default_false(
        self, operation: ConcreteTestOperation
    ) -> None:
        """Test that supports_inplace defaults to False."""
        assert operation.supports_inplace is False

    def test_output_shape_factor_default(
        self, operation: ConcreteTestOperation
    ) -> None:
        """Test that output_shape_factor defaults to (1.0, 1.0)."""
        assert operation.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_default_empty(
        self, operation: ConcreteTestOperation
    ) -> None:
        """Test that temp_field_requirements defaults to empty list."""
        assert operation.temp_field_requirements == []

    def test_warmup_sets_compiled_flag(self, operation: ConcreteTestOperation) -> None:
        """Test that warmup() sets the compiled flag."""
        assert operation.is_compiled is False
        operation.warmup()
        assert operation.is_compiled is True

    def test_warmup_is_idempotent(self, operation: ConcreteTestOperation) -> None:
        """Test that calling warmup() multiple times is safe."""
        operation.warmup()
        operation.warmup()  # Should not fail
        assert operation.is_compiled is True

    def test_validate_params_raises_on_invalid(
        self, operation: ConcreteTestOperation
    ) -> None:
        """Test that validate_params raises ValueError for invalid params."""
        with pytest.raises(ValueError, match="Missing"):
            operation.validate_params({})

    def test_validate_params_accepts_valid(
        self, operation: ConcreteTestOperation
    ) -> None:
        """Test that validate_params accepts valid parameters."""
        operation.validate_params({"required": True})  # Should not raise

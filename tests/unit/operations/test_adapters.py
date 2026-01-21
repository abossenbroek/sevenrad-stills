"""Tests for legacy operation adapters."""

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.adapters import LegacyOperationAdapter
from sevenrad_stills.operations.base import BaseImageOperation


class MockLegacyOperation(BaseImageOperation):
    """Mock legacy operation for testing."""

    def __init__(self, operation_name: str = "mock_op") -> None:
        """Initialize mock operation."""
        super().__init__(operation_name)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Mock apply that just returns the image unchanged.

        Args:
            image: Input PIL Image
            params: Operation parameters

        Returns:
            Same image unchanged

        """
        return image

    def validate_params(self, params: dict[str, Any]) -> None:
        """
        Mock validation that checks for 'required_param'.

        Args:
            params: Parameters to validate

        Raises:
            ValueError: If required_param is missing

        """
        if "required_param" not in params:
            msg = "Missing required_param"
            raise ValueError(msg)


class MockInvertOperation(BaseImageOperation):
    """Mock operation that inverts image colors."""

    def __init__(self) -> None:
        """Initialize invert operation."""
        super().__init__("invert")

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """
        Invert image colors.

        Args:
            image: Input PIL Image
            params: Operation parameters (unused)

        Returns:
            Color-inverted image

        """
        img_array = np.array(image)
        inverted = 255 - img_array
        return Image.fromarray(inverted, mode=image.mode)

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate params (no-op for invert)."""


class TestLegacyOperationAdapter:
    """Tests for LegacyOperationAdapter class."""

    def test_initialization(self) -> None:
        """Test adapter initialization with legacy operation."""
        legacy_op = MockLegacyOperation("test_op")
        adapter = LegacyOperationAdapter(legacy_op)

        assert adapter._legacy is legacy_op

    def test_name_property(self) -> None:
        """Test that name property adds 'legacy_' prefix."""
        legacy_op = MockLegacyOperation("test_op")
        adapter = LegacyOperationAdapter(legacy_op)

        assert adapter.name == "legacy_test_op"

    def test_name_property_with_different_names(self) -> None:
        """Test name property with various operation names."""
        test_cases = [
            ("saturation", "legacy_saturation"),
            ("blur", "legacy_blur"),
            ("custom_op", "legacy_custom_op"),
        ]

        for op_name, expected_name in test_cases:
            legacy_op = MockLegacyOperation(op_name)
            adapter = LegacyOperationAdapter(legacy_op)
            assert adapter.name == expected_name

    def test_supports_inplace_is_false(self) -> None:
        """Test that supports_inplace is always False for legacy operations."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        assert adapter.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output_shape_factor is (1.0, 1.0)."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        assert adapter.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements(self) -> None:
        """Test that temp_field_requirements is empty list."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        requirements = adapter.temp_field_requirements
        assert isinstance(requirements, list)
        assert len(requirements) == 0

    def test_validate_params_delegates(self) -> None:
        """Test that validate_params delegates to wrapped operation."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Should raise error from wrapped operation
        with pytest.raises(ValueError, match="Missing required_param"):
            adapter.validate_params({})

        # Should not raise with valid params
        adapter.validate_params({"required_param": "value"})

    def test_reference_numpy_with_rgb_image(self) -> None:
        """Test reference_numpy with RGB image."""
        legacy_op = MockInvertOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Create test image (3x3 RGB, solid red)
        image = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        image = np.tile(image, (3, 3, 1))

        result = adapter.reference_numpy(image, {})

        # Should be inverted (cyan)
        expected = np.array([[[0.0, 1.0, 1.0]]], dtype=np.float32)
        expected = np.tile(expected, (3, 3, 1))

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_reference_numpy_preserves_shape(self) -> None:
        """Test that reference_numpy preserves input shape."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Test various shapes
        shapes = [(10, 10, 3), (5, 15, 3), (100, 50, 3)]

        for shape in shapes:
            image = np.random.rand(*shape).astype(np.float32)
            result = adapter.reference_numpy(image, {"required_param": "value"})
            assert result.shape == shape

    def test_apply_to_field_with_mock_fields(self) -> None:
        """Test apply_to_field with mock Taichi fields."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Create mock Taichi fields
        height, width = 4, 4
        test_image = np.random.rand(height, width, 3).astype(np.float32)

        source_field = Mock()
        source_field.to_numpy.return_value = test_image

        dest_field = Mock()

        # Apply operation
        adapter.apply_to_field(
            source=source_field,
            dest=dest_field,
            temp_fields={},
            params={"required_param": "value"},
            height=height,
            width=width,
        )

        # Verify source was read
        source_field.to_numpy.assert_called_once()

        # Verify dest was written
        dest_field.from_numpy.assert_called_once()
        written_data = dest_field.from_numpy.call_args[0][0]
        assert written_data.shape == (height, width, 3)
        assert written_data.dtype == np.float32

    def test_apply_to_field_inverts_correctly(self) -> None:
        """Test that apply_to_field correctly inverts colors."""
        legacy_op = MockInvertOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Create test image (solid red)
        height, width = 4, 4
        test_image: np.ndarray[tuple[int, int, int], np.dtype[np.float32]] = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        test_image[:, :, 0] = 1.0  # Red channel = 1.0

        source_field = Mock()
        source_field.to_numpy.return_value = test_image

        dest_field = Mock()

        # Apply operation
        adapter.apply_to_field(
            source=source_field,
            dest=dest_field,
            temp_fields={},
            params={},
            height=height,
            width=width,
        )

        # Check result is inverted (should be cyan)
        written_data = dest_field.from_numpy.call_args[0][0]
        expected: np.ndarray[tuple[int, int, int], np.dtype[np.float32]] = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        expected[:, :, 1] = 1.0  # Green
        expected[:, :, 2] = 1.0  # Blue

        np.testing.assert_allclose(written_data, expected, rtol=1e-5)

    def test_apply_to_field_handles_rgba(self) -> None:
        """Test apply_to_field handles RGBA images correctly."""
        legacy_op = MockLegacyOperation()
        adapter = LegacyOperationAdapter(legacy_op)

        # Create RGBA test image
        height, width = 4, 4
        test_image = np.random.rand(height, width, 4).astype(np.float32)

        source_field = Mock()
        source_field.to_numpy.return_value = test_image

        dest_field = Mock()

        # Apply operation
        adapter.apply_to_field(
            source=source_field,
            dest=dest_field,
            temp_fields={},
            params={"required_param": "value"},
            height=height,
            width=width,
        )

        # Verify output has 4 channels
        written_data = dest_field.from_numpy.call_args[0][0]
        assert written_data.shape == (height, width, 4)

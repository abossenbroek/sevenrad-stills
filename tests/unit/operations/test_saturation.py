"""Tests for saturation operation."""

import pytest
from PIL import Image
from sevenrad_stills.operations.saturation import SaturationOperation


class TestSaturationOperation:
    """Tests for SaturationOperation class."""

    @pytest.fixture
    def operation(self) -> SaturationOperation:
        """Create a saturation operation instance."""
        return SaturationOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        # Create a simple RGB image
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: SaturationOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "saturation"

    def test_fixed_mode_validation(self, operation: SaturationOperation) -> None:
        """Test fixed mode parameter validation."""
        params = {"mode": "fixed", "value": 1.5}
        operation.validate_params(params)  # Should not raise

    def test_random_mode_validation(self, operation: SaturationOperation) -> None:
        """Test random mode parameter validation."""
        params = {"mode": "random", "range": [-0.5, 0.5]}
        operation.validate_params(params)  # Should not raise

    def test_missing_mode_raises_error(self, operation: SaturationOperation) -> None:
        """Test missing mode parameter raises error."""
        params = {"value": 1.5}
        with pytest.raises(ValueError, match="requires 'mode' parameter"):
            operation.validate_params(params)

    def test_invalid_mode_raises_error(self, operation: SaturationOperation) -> None:
        """Test invalid mode raises error."""
        params = {"mode": "invalid"}
        with pytest.raises(ValueError, match="Invalid mode"):
            operation.validate_params(params)

    def test_fixed_mode_missing_value_raises_error(
        self, operation: SaturationOperation
    ) -> None:
        """Test fixed mode without value raises error."""
        params = {"mode": "fixed"}
        with pytest.raises(ValueError, match="requires 'value' parameter"):
            operation.validate_params(params)

    def test_fixed_mode_negative_value_raises_error(
        self, operation: SaturationOperation
    ) -> None:
        """Test fixed mode with negative value raises error."""
        params = {"mode": "fixed", "value": -1.0}
        with pytest.raises(ValueError, match="must be non-negative"):
            operation.validate_params(params)

    def test_random_mode_missing_range_raises_error(
        self, operation: SaturationOperation
    ) -> None:
        """Test random mode without range raises error."""
        params = {"mode": "random"}
        with pytest.raises(ValueError, match="requires 'range' parameter"):
            operation.validate_params(params)

    def test_random_mode_invalid_range_size_raises_error(
        self, operation: SaturationOperation
    ) -> None:
        """Test random mode with wrong range size raises error."""
        params = {"mode": "random", "range": [0.5]}
        with pytest.raises(ValueError, match="must be a list/tuple of two numbers"):
            operation.validate_params(params)

    def test_random_mode_inverted_range_raises_error(
        self, operation: SaturationOperation
    ) -> None:
        """Test random mode with inverted range raises error."""
        params = {"mode": "random", "range": [0.5, -0.5]}
        with pytest.raises(ValueError, match="must be less than max"):
            operation.validate_params(params)

    def test_apply_fixed_mode(
        self, operation: SaturationOperation, test_image: Image.Image
    ) -> None:
        """Test applying fixed mode saturation."""
        params = {"mode": "fixed", "value": 1.5}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_random_mode(
        self, operation: SaturationOperation, test_image: Image.Image
    ) -> None:
        """Test applying random mode saturation."""
        params = {"mode": "random", "range": [-0.5, 0.5]}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_preserves_image_mode(
        self, operation: SaturationOperation, test_image: Image.Image
    ) -> None:
        """Test operation preserves image mode."""
        params = {"mode": "fixed", "value": 1.5}
        result = operation.apply(test_image, params)
        assert result.mode == test_image.mode

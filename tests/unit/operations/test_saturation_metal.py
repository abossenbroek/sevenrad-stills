"""Tests for Metal hardware-accelerated saturation operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image

# Only import on macOS
if platform.system() == "Darwin":
    from sevenrad_stills.operations.saturation_metal import SaturationMetalOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal saturation only available on macOS",
)
class TestSaturationMetalOperation:
    """Tests for SaturationMetalOperation class."""

    @pytest.fixture
    def operation(self) -> SaturationMetalOperation:
        """Create a Metal saturation operation instance."""
        return SaturationMetalOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    def test_operation_name(self, operation: SaturationMetalOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "saturation_metal"

    def test_valid_params_fixed(self, operation: SaturationMetalOperation) -> None:
        """Test valid parameter validation for fixed mode."""
        params = {"mode": "fixed", "value": 0.5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_random(self, operation: SaturationMetalOperation) -> None:
        """Test valid parameter validation for random mode."""
        params = {"mode": "random", "range": [-0.5, 0.5]}
        operation.validate_params(params)  # Should not raise

    def test_missing_mode_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test missing mode parameter raises error."""
        params = {"value": 0.5}
        with pytest.raises(ValueError, match="requires 'mode' parameter"):
            operation.validate_params(params)

    def test_invalid_mode_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test invalid mode raises error."""
        params = {"mode": "invalid"}
        with pytest.raises(ValueError, match="Invalid mode"):
            operation.validate_params(params)

    def test_missing_value_in_fixed_mode_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test missing value in fixed mode raises error."""
        params = {"mode": "fixed"}
        with pytest.raises(ValueError, match="Fixed mode requires 'value'"):
            operation.validate_params(params)

    def test_invalid_value_type_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test invalid value type raises error."""
        params = {"mode": "fixed", "value": "0.5"}
        with pytest.raises(ValueError, match="Value must be a number"):
            operation.validate_params(params)

    def test_value_too_low_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test value below -1.0 raises error."""
        params = {"mode": "fixed", "value": -1.5}
        with pytest.raises(ValueError, match="Value must be >= -1.0"):
            operation.validate_params(params)

    def test_missing_range_in_random_mode_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test missing range in random mode raises error."""
        params = {"mode": "random"}
        with pytest.raises(ValueError, match="Random mode requires 'range'"):
            operation.validate_params(params)

    def test_invalid_range_type_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test invalid range type raises error."""
        params = {"mode": "random", "range": 0.5}
        with pytest.raises(ValueError, match="Range must be a list/tuple"):
            operation.validate_params(params)

    def test_invalid_range_length_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test invalid range length raises error."""
        params = {"mode": "random", "range": [0.5]}
        with pytest.raises(ValueError, match="Range must be a list/tuple of two"):
            operation.validate_params(params)

    def test_range_min_greater_than_max_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test range min >= max raises error."""
        params = {"mode": "random", "range": [0.5, 0.3]}
        with pytest.raises(ValueError, match="Range min .* must be less than max"):
            operation.validate_params(params)

    def test_range_min_too_low_raises_error(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test range min below -1.0 raises error."""
        params = {"mode": "random", "range": [-1.5, 0.5]}
        with pytest.raises(ValueError, match="Range min must be >= -1"):
            operation.validate_params(params)

    def test_apply_fixed_increase_saturation(
        self, operation: SaturationMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying fixed saturation increase with Metal acceleration."""
        params = {"mode": "fixed", "value": 0.5}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_fixed_decrease_saturation(
        self, operation: SaturationMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying fixed saturation decrease with Metal acceleration."""
        params = {"mode": "fixed", "value": -0.5}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_grayscale(
        self, operation: SaturationMetalOperation, test_image: Image.Image
    ) -> None:
        """Test converting to grayscale (saturation = 0) with Metal."""
        params = {"mode": "fixed", "value": -1.0}
        result = operation.apply(test_image, params)

        # Check that all RGB channels are equal (grayscale)
        result_array = np.array(result)
        r = result_array[:, :, 0]
        g = result_array[:, :, 1]
        b = result_array[:, :, 2]

        # All channels should be approximately equal
        assert np.allclose(r, g, atol=2)
        assert np.allclose(g, b, atol=2)

    def test_apply_random_mode(
        self, operation: SaturationMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying random saturation adjustment with Metal."""
        params = {"mode": "random", "range": [-0.3, 0.3]}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_converts_rgba_to_rgb(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"mode": "fixed", "value": 0.5}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_converts_grayscale_to_rgb(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test that grayscale images are converted to RGB."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"mode": "fixed", "value": 0.5}
        result = operation.apply(gray_image, params)
        assert result.mode == "RGB"

    def test_saturation_increase_makes_more_colorful(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test that increasing saturation makes image more colorful."""
        # Create image with some color
        test_image = Image.new("RGB", (100, 100), color=(150, 100, 50))

        base_result = operation.apply(test_image, {"mode": "fixed", "value": 0.0})
        high_sat_result = operation.apply(test_image, {"mode": "fixed", "value": 1.0})

        base_array = np.array(base_result, dtype=np.float32)
        high_sat_array = np.array(high_sat_result, dtype=np.float32)

        # Calculate color variance (should be higher with more saturation)
        base_variance = np.var(base_array, axis=2).mean()
        high_sat_variance = np.var(high_sat_array, axis=2).mean()

        assert high_sat_variance >= base_variance

    def test_saturation_decrease_makes_less_colorful(
        self, operation: SaturationMetalOperation
    ) -> None:
        """Test that decreasing saturation makes image less colorful."""
        # Create image with some color
        test_image = Image.new("RGB", (100, 100), color=(150, 100, 50))

        base_result = operation.apply(test_image, {"mode": "fixed", "value": 0.0})
        low_sat_result = operation.apply(test_image, {"mode": "fixed", "value": -0.5})

        base_array = np.array(base_result, dtype=np.float32)
        low_sat_array = np.array(low_sat_result, dtype=np.float32)

        # Calculate color variance (should be lower with less saturation)
        base_variance = np.var(base_array, axis=2).mean()
        low_sat_variance = np.var(low_sat_array, axis=2).mean()

        assert low_sat_variance <= base_variance

    def test_metal_acceleration_reusable(
        self, operation: SaturationMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that Metal acceleration can be used multiple times."""
        params = {"mode": "fixed", "value": 0.5}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        assert isinstance(result1, Image.Image)
        assert isinstance(result2, Image.Image)
        assert result1.size == result2.size

"""Tests for corduroy striping operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.corduroy import CorduroyOperation


class TestCorduroyOperation:
    """Tests for CorduroyOperation class."""

    @pytest.fixture
    def operation(self) -> CorduroyOperation:
        """Create a corduroy operation instance."""
        return CorduroyOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        # Create a uniform gray image
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: CorduroyOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "corduroy"

    def test_valid_params(self, operation: CorduroyOperation) -> None:
        """Test valid parameter validation."""
        params = {"strength": 0.5, "orientation": "vertical", "density": 0.2}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_seed(self, operation: CorduroyOperation) -> None:
        """Test valid parameters with seed."""
        params = {
            "strength": 0.3,
            "orientation": "horizontal",
            "density": 0.15,
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_strength_raises_error(self, operation: CorduroyOperation) -> None:
        """Test missing strength parameter raises error."""
        params = {"orientation": "vertical", "density": 0.2}
        with pytest.raises(ValueError, match="requires 'strength' parameter"):
            operation.validate_params(params)

    def test_missing_orientation_raises_error(
        self, operation: CorduroyOperation
    ) -> None:
        """Test missing orientation parameter raises error."""
        params = {"strength": 0.5, "density": 0.2}
        with pytest.raises(ValueError, match="requires 'orientation' parameter"):
            operation.validate_params(params)

    def test_missing_density_raises_error(self, operation: CorduroyOperation) -> None:
        """Test missing density parameter raises error."""
        params = {"strength": 0.5, "orientation": "vertical"}
        with pytest.raises(ValueError, match="requires 'density' parameter"):
            operation.validate_params(params)

    def test_invalid_strength_type_raises_error(
        self, operation: CorduroyOperation
    ) -> None:
        """Test invalid strength type raises error."""
        params = {"strength": "0.5", "orientation": "vertical", "density": 0.2}
        with pytest.raises(ValueError, match="Strength must be a float"):
            operation.validate_params(params)

    def test_strength_too_low_raises_error(self, operation: CorduroyOperation) -> None:
        """Test strength below minimum raises error."""
        params = {"strength": -0.1, "orientation": "vertical", "density": 0.2}
        with pytest.raises(ValueError, match="Strength must be a float between"):
            operation.validate_params(params)

    def test_strength_too_high_raises_error(self, operation: CorduroyOperation) -> None:
        """Test strength above maximum raises error."""
        params = {"strength": 1.5, "orientation": "vertical", "density": 0.2}
        with pytest.raises(ValueError, match="Strength must be a float between"):
            operation.validate_params(params)

    def test_invalid_orientation_raises_error(
        self, operation: CorduroyOperation
    ) -> None:
        """Test invalid orientation raises error."""
        params = {"strength": 0.5, "orientation": "diagonal", "density": 0.2}
        with pytest.raises(ValueError, match="must be 'vertical' or 'horizontal'"):
            operation.validate_params(params)

    def test_invalid_density_type_raises_error(
        self, operation: CorduroyOperation
    ) -> None:
        """Test invalid density type raises error."""
        params = {"strength": 0.5, "orientation": "vertical", "density": "0.2"}
        with pytest.raises(ValueError, match="Density must be a float"):
            operation.validate_params(params)

    def test_density_too_low_raises_error(self, operation: CorduroyOperation) -> None:
        """Test density below minimum raises error."""
        params = {"strength": 0.5, "orientation": "vertical", "density": -0.1}
        with pytest.raises(ValueError, match="Density must be a float between"):
            operation.validate_params(params)

    def test_density_too_high_raises_error(self, operation: CorduroyOperation) -> None:
        """Test density above maximum raises error."""
        params = {"strength": 0.5, "orientation": "vertical", "density": 1.5}
        with pytest.raises(ValueError, match="Density must be a float between"):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(self, operation: CorduroyOperation) -> None:
        """Test invalid seed type raises error."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.2,
            "seed": "42",
        }
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_vertical(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test applying vertical striping."""
        params = {
            "strength": 0.3,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_horizontal(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test applying horizontal striping."""
        params = {
            "strength": 0.3,
            "orientation": "horizontal",
            "density": 0.2,
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test that striping actually changes the image."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Convert to arrays and check that they differ
        original_array = np.array(test_image)
        result_array = np.array(result)

        # Should have differences due to striping
        assert not np.array_equal(original_array, result_array)

    def test_vertical_affects_columns(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test that vertical striping affects entire columns consistently."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Check if there are columns where all pixels changed similarly
        column_diffs = np.mean(result_array - original_array, axis=0)
        # Some columns should have non-zero mean difference
        changed_columns = np.any(column_diffs != 0, axis=1)
        assert np.any(changed_columns)

    def test_horizontal_affects_rows(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test that horizontal striping affects entire rows consistently."""
        params = {
            "strength": 0.5,
            "orientation": "horizontal",
            "density": 0.2,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Check if there are rows where all pixels changed similarly
        row_diffs = np.mean(result_array - original_array, axis=1)
        # Some rows should have non-zero mean difference
        changed_rows = np.any(row_diffs != 0, axis=1)
        assert np.any(changed_rows)

    def test_reproducibility_with_seed(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {
            "strength": 0.3,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 42,
        }
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Results should be pixel-identical
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {
            "strength": 0.3,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 42,
        }
        params2 = {
            "strength": 0.3,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 100,
        }
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(self, operation: CorduroyOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {
            "strength": 0.3,
            "orientation": "vertical",
            "density": 0.2,
            "seed": 42,
        }
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        # Alpha channel (channel 3) should be unchanged
        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

        # RGB channels may have striping
        # (but not necessarily if density is low and RNG didn't select many lines)

    def test_apply_grayscale(self, operation: CorduroyOperation) -> None:
        """Test that grayscale images work correctly."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have striping added (with high density)
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_zero_density_produces_no_change(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test density=0.0 produces no striping."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should be identical to original
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_zero_strength_produces_no_visible_change(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test strength=0.0 produces minimal change."""
        params = {
            "strength": 0.0,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # With strength=0.0, multipliers are all 1.0, so no change
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_high_strength_produces_visible_change(
        self, operation: CorduroyOperation, test_image: Image.Image
    ) -> None:
        """Test high strength produces noticeable changes."""
        params = {
            "strength": 1.0,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Calculate mean absolute difference
        mean_diff = np.mean(
            np.abs(result_array.astype(float) - original_array.astype(float))
        )

        # With strength=1.0 and density=0.5, we expect significant changes
        # Multipliers range from [0.8, 1.2], so changes should be noticeable
        assert mean_diff > 1.0  # At least 1 intensity level difference on average

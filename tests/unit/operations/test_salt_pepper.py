"""Tests for salt and pepper noise operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.salt_pepper import SaltPepperOperation


class TestSaltPepperOperation:
    """Tests for SaltPepperOperation class."""

    @pytest.fixture
    def operation(self) -> SaltPepperOperation:
        """Create a salt and pepper operation instance."""
        return SaltPepperOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        # Create a uniform gray image to easily detect salt and pepper
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: SaltPepperOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "salt_pepper"

    def test_valid_params(self, operation: SaltPepperOperation) -> None:
        """Test valid parameter validation."""
        params = {"amount": 0.05, "salt_vs_pepper": 0.5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_seed(self, operation: SaltPepperOperation) -> None:
        """Test valid parameters with seed."""
        params = {"amount": 0.1, "salt_vs_pepper": 0.7, "seed": 42}
        operation.validate_params(params)  # Should not raise

    def test_missing_amount_raises_error(self, operation: SaltPepperOperation) -> None:
        """Test missing amount parameter raises error."""
        params = {"salt_vs_pepper": 0.5}
        with pytest.raises(ValueError, match="requires 'amount' parameter"):
            operation.validate_params(params)

    def test_missing_salt_vs_pepper_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test missing salt_vs_pepper parameter raises error."""
        params = {"amount": 0.1}
        with pytest.raises(ValueError, match="requires 'salt_vs_pepper' parameter"):
            operation.validate_params(params)

    def test_invalid_amount_type_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test invalid amount type raises error."""
        params = {"amount": "0.1", "salt_vs_pepper": 0.5}
        with pytest.raises(ValueError, match="Amount must be a float"):
            operation.validate_params(params)

    def test_amount_too_low_raises_error(self, operation: SaltPepperOperation) -> None:
        """Test amount below minimum raises error."""
        params = {"amount": -0.1, "salt_vs_pepper": 0.5}
        with pytest.raises(ValueError, match="Amount must be a float between"):
            operation.validate_params(params)

    def test_amount_too_high_raises_error(self, operation: SaltPepperOperation) -> None:
        """Test amount above maximum raises error."""
        params = {"amount": 1.5, "salt_vs_pepper": 0.5}
        with pytest.raises(ValueError, match="Amount must be a float between"):
            operation.validate_params(params)

    def test_invalid_salt_vs_pepper_type_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test invalid salt_vs_pepper type raises error."""
        params = {"amount": 0.1, "salt_vs_pepper": "0.5"}
        with pytest.raises(ValueError, match="salt_vs_pepper must be a float"):
            operation.validate_params(params)

    def test_salt_vs_pepper_too_low_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test salt_vs_pepper below minimum raises error."""
        params = {"amount": 0.1, "salt_vs_pepper": -0.1}
        with pytest.raises(ValueError, match="salt_vs_pepper must be a float between"):
            operation.validate_params(params)

    def test_salt_vs_pepper_too_high_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test salt_vs_pepper above maximum raises error."""
        params = {"amount": 0.1, "salt_vs_pepper": 1.5}
        with pytest.raises(ValueError, match="salt_vs_pepper must be a float between"):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, operation: SaltPepperOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {"amount": 0.1, "salt_vs_pepper": 0.5, "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test applying salt and pepper noise."""
        params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_noise(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test that noise is actually added."""
        params = {"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42}
        result = operation.apply(test_image, params)

        # Convert to arrays and check that they differ
        original_array = np.array(test_image)
        result_array = np.array(result)

        # Should have differences due to noise
        assert not np.array_equal(original_array, result_array)

        # Check for both black and white pixels (salt and pepper)
        # Original is uniform gray (128, 128, 128)
        black_pixels = np.all(result_array == [0, 0, 0], axis=2).sum()
        white_pixels = np.all(result_array == [255, 255, 255], axis=2).sum()

        # With amount=0.1 and 10000 pixels, expect some of each
        assert black_pixels > 0
        assert white_pixels > 0

    def test_reproducibility_with_seed(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Results should be pixel-identical
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        params2 = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 100}
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(self, operation: SaltPepperOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        # Alpha channel (channel 3) should be unchanged
        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

        # RGB channels should have noise
        assert not np.array_equal(original_array[:, :, :3], result_array[:, :, :3])

    def test_apply_grayscale(self, operation: SaltPepperOperation) -> None:
        """Test that grayscale images work correctly."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"amount": 0.05, "salt_vs_pepper": 0.5, "seed": 42}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have noise added
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_salt_vs_pepper_all_salt(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test salt_vs_pepper=1.0 produces only white pixels."""
        params = {"amount": 0.1, "salt_vs_pepper": 1.0, "seed": 42}
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Find pixels that changed
        changed_mask = np.any(result_array != original_array, axis=2)
        changed_pixels = result_array[changed_mask]

        # All changed pixels should be white (salt)
        assert np.all(changed_pixels == [255, 255, 255])

    def test_salt_vs_pepper_all_pepper(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test salt_vs_pepper=0.0 produces only black pixels."""
        params = {"amount": 0.1, "salt_vs_pepper": 0.0, "seed": 42}
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Find pixels that changed
        changed_mask = np.any(result_array != original_array, axis=2)
        changed_pixels = result_array[changed_mask]

        # All changed pixels should be black (pepper)
        assert np.all(changed_pixels == [0, 0, 0])

    def test_zero_amount_produces_no_change(
        self, operation: SaltPepperOperation, test_image: Image.Image
    ) -> None:
        """Test amount=0.0 produces no noise."""
        params = {"amount": 0.0, "salt_vs_pepper": 0.5, "seed": 42}
        result = operation.apply(test_image, params)

        # Should be identical to original
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

"""Tests for noise operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.noise import NoiseOperation


@pytest.fixture
def noise_op() -> NoiseOperation:
    """Create a noise operation instance."""
    return NoiseOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image."""
    return Image.new("RGB", (100, 100), color=(128, 128, 128))


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    return Image.new("L", (100, 100), color=128)


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image with varying alpha."""
    img = Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))
    # Create a pattern with different alpha values
    pixels = img.load()
    for y in range(100):
        for x in range(100):
            if pixels is not None:  # Type guard
                alpha = int((x / 100) * 255)
                pixels[x, y] = (128, 128, 128, alpha)
    return img


class TestNoiseValidation:
    """Test parameter validation for noise operation."""

    def test_validation_missing_mode(self, noise_op: NoiseOperation) -> None:
        """Test that missing mode parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires a 'mode' parameter"):
            noise_op.validate_params({"amount": 0.1})

    def test_validation_invalid_mode(self, noise_op: NoiseOperation) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Mode must be"):
            noise_op.validate_params({"mode": "invalid", "amount": 0.1})

    def test_validation_missing_amount(self, noise_op: NoiseOperation) -> None:
        """Test that missing amount parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires an 'amount' parameter"):
            noise_op.validate_params({"mode": "gaussian"})

    def test_validation_amount_out_of_range(self, noise_op: NoiseOperation) -> None:
        """Test that amount outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Amount must be a float"):
            noise_op.validate_params({"mode": "gaussian", "amount": 1.5})
        with pytest.raises(ValueError, match="Amount must be a float"):
            noise_op.validate_params({"mode": "gaussian", "amount": -0.1})

    def test_validation_invalid_seed(self, noise_op: NoiseOperation) -> None:
        """Test that non-integer seed raises ValueError."""
        with pytest.raises(ValueError, match="Seed must be an integer"):
            noise_op.validate_params({"mode": "gaussian", "amount": 0.1, "seed": "42"})

    def test_validation_valid_params(self, noise_op: NoiseOperation) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        noise_op.validate_params({"mode": "gaussian", "amount": 0.1})
        noise_op.validate_params({"mode": "row", "amount": 0.5, "seed": 42})
        noise_op.validate_params({"mode": "column", "amount": 0.0})


class TestNoiseApply:
    """Test applying noise to images."""

    def test_deterministic_output_with_seed(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}
        result1 = noise_op.apply(test_image_rgb, params)
        result2 = noise_op.apply(test_image_rgb, params)

        array1 = np.array(result1)
        array2 = np.array(result2)
        np.testing.assert_array_equal(array1, array2)

    def test_gaussian_noise_rgb(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test Gaussian noise on RGB image."""
        result = noise_op.apply(
            test_image_rgb, {"mode": "gaussian", "amount": 0.1, "seed": 123}
        )

        # Verify dimensions and mode preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify noise is present (variance should increase)
        original_var = np.var(original_array.astype(float))
        result_var = np.var(result_array.astype(float))
        assert result_var > original_var

    def test_gaussian_noise_grayscale(
        self, noise_op: NoiseOperation, test_image_grayscale: Image.Image
    ) -> None:
        """Test Gaussian noise on grayscale image."""
        result = noise_op.apply(
            test_image_grayscale, {"mode": "gaussian", "amount": 0.1, "seed": 123}
        )

        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_row_noise_pattern(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that row noise creates horizontal patterns."""
        result = noise_op.apply(
            test_image_rgb, {"mode": "row", "amount": 0.2, "seed": 456}
        )

        result_array = np.array(result)

        # Verify that all pixels in a given row have the same noise
        # Check first row
        first_row = result_array[0, :, :]
        # All pixels in the row should have the same value
        assert np.all(first_row == first_row[0, :])

        # Check that different rows have different values
        second_row = result_array[1, :, :]
        assert not np.array_equal(first_row[0, :], second_row[0, :])

    def test_column_noise_pattern(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that column noise creates vertical patterns."""
        result = noise_op.apply(
            test_image_rgb, {"mode": "column", "amount": 0.2, "seed": 789}
        )

        result_array = np.array(result)

        # Verify that all pixels in a given column have the same noise
        # Check first column
        first_col = result_array[:, 0, :]
        # All pixels in the column should have the same value
        assert np.all(first_col == first_col[0, :])

        # Check that different columns have different values
        second_col = result_array[:, 1, :]
        assert not np.array_equal(first_col[0, :], second_col[0, :])

    def test_zero_amount_no_change(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that amount=0 produces no change."""
        result = noise_op.apply(test_image_rgb, {"mode": "gaussian", "amount": 0.0})

        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_different_amounts(
        self, noise_op: NoiseOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different amounts produce different noise levels."""
        result_low = noise_op.apply(
            test_image_rgb, {"mode": "gaussian", "amount": 0.05, "seed": 999}
        )
        result_high = noise_op.apply(
            test_image_rgb, {"mode": "gaussian", "amount": 0.3, "seed": 999}
        )

        array_low = np.array(result_low).astype(float)
        array_high = np.array(result_high).astype(float)
        original = np.array(test_image_rgb).astype(float)

        # Calculate deviation from original
        dev_low = np.abs(array_low - original).mean()
        dev_high = np.abs(array_high - original).mean()

        # Higher amount should produce more deviation
        assert dev_high > dev_low

    def test_operation_name(self, noise_op: NoiseOperation) -> None:
        """Test that operation has correct name."""
        assert noise_op.name == "noise"

    def test_rgba_alpha_preservation_gaussian(
        self, noise_op: NoiseOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test that Gaussian noise preserves alpha channel in RGBA images."""
        result = noise_op.apply(
            test_image_rgba, {"mode": "gaussian", "amount": 0.2, "seed": 42}
        )

        # Verify mode is preserved
        assert result.mode == "RGBA"
        assert result.size == test_image_rgba.size

        # Extract alpha channels
        original_alpha = np.array(test_image_rgba)[:, :, 3]
        result_alpha = np.array(result)[:, :, 3]

        # Alpha channel should be completely unchanged
        np.testing.assert_array_equal(result_alpha, original_alpha)

        # RGB channels should have changed
        original_rgb = np.array(test_image_rgba)[:, :, :3]
        result_rgb = np.array(result)[:, :, :3]
        assert not np.array_equal(result_rgb, original_rgb)

    def test_rgba_alpha_preservation_row(
        self, noise_op: NoiseOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test that row noise preserves alpha channel in RGBA images."""
        result = noise_op.apply(
            test_image_rgba, {"mode": "row", "amount": 0.2, "seed": 42}
        )

        # Extract alpha channels
        original_alpha = np.array(test_image_rgba)[:, :, 3]
        result_alpha = np.array(result)[:, :, 3]

        # Alpha channel should be completely unchanged
        np.testing.assert_array_equal(result_alpha, original_alpha)

    def test_rgba_alpha_preservation_column(
        self, noise_op: NoiseOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test that column noise preserves alpha channel in RGBA images."""
        result = noise_op.apply(
            test_image_rgba, {"mode": "column", "amount": 0.2, "seed": 42}
        )

        # Extract alpha channels
        original_alpha = np.array(test_image_rgba)[:, :, 3]
        result_alpha = np.array(result)[:, :, 3]

        # Alpha channel should be completely unchanged
        np.testing.assert_array_equal(result_alpha, original_alpha)

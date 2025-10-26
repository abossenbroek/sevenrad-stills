"""Tests for MLX-accelerated Gaussian blur operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian_mlx import GaussianBlurMLXOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLX tests only run on Mac (Metal backend)",
)
class TestGaussianBlurMLXOperation:
    """Tests for GaussianBlurMLXOperation class."""

    @pytest.fixture
    def operation(self) -> GaussianBlurMLXOperation:
        """Create an MLX Gaussian blur operation instance."""
        return GaussianBlurMLXOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with a simple pattern."""
        # Create a test pattern: white center, black edges
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        # Add white square in center
        for y in range(40, 60):
            for x in range(40, 60):
                img.putpixel((x, y), (255, 255, 255))
        return img

    @pytest.fixture
    def grayscale_image(self) -> Image.Image:
        """Create a grayscale test image."""
        img = Image.new("L", (100, 100), color=0)
        # Add white square in center
        for y in range(40, 60):
            for x in range(40, 60):
                img.putpixel((x, y), 255)
        return img

    def test_operation_name(self, operation: GaussianBlurMLXOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "blur_gaussian_mlx"

    def test_valid_params(self, operation: GaussianBlurMLXOperation) -> None:
        """Test valid parameter validation."""
        params = {"sigma": 2.0}
        operation.validate_params(params)  # Should not raise

    def test_missing_sigma_raises_error(
        self, operation: GaussianBlurMLXOperation
    ) -> None:
        """Test missing sigma parameter raises error."""
        params = {}
        with pytest.raises(ValueError, match="requires a 'sigma' parameter"):
            operation.validate_params(params)

    def test_zero_sigma_returns_copy(
        self, operation: GaussianBlurMLXOperation, test_image: Image.Image
    ) -> None:
        """Test that zero sigma returns a copy of the original image."""
        params = {"sigma": 0}
        result = operation.apply(test_image, params)

        # Should be equal to original
        assert result.size == test_image.size
        assert result.mode == test_image.mode
        assert np.array_equal(np.array(result), np.array(test_image))
        # But not the same object
        assert result is not test_image

    def test_apply_blur_rgb(
        self, operation: GaussianBlurMLXOperation, test_image: Image.Image
    ) -> None:
        """Test applying blur to RGB image."""
        params = {"sigma": 2.0}
        result = operation.apply(test_image, params)

        # Result should have same dimensions and mode
        assert result.size == test_image.size
        assert result.mode == test_image.mode

        # Image should be blurred (edges should be softer)
        original = np.array(test_image)
        blurred = np.array(result)

        # Check that sharp edge is now blurred
        assert original[50, 39, 0] == 0
        assert original[50, 40, 0] == 255
        assert 0 < blurred[50, 39, 0] < 255
        assert blurred[50, 40, 0] < 255

    def test_apply_blur_grayscale(
        self, operation: GaussianBlurMLXOperation, grayscale_image: Image.Image
    ) -> None:
        """Test applying blur to grayscale image."""
        params = {"sigma": 2.0}
        result = operation.apply(grayscale_image, params)

        # Result should have same dimensions and mode
        assert result.size == grayscale_image.size
        assert result.mode == grayscale_image.mode

        # Image should be blurred
        original = np.array(grayscale_image)
        blurred = np.array(result)

        assert original[50, 39] == 0
        assert original[50, 40] == 255
        assert 0 < blurred[50, 39] < 255
        assert blurred[50, 40] < 255

    def test_apply_blur_rgba(self, operation: GaussianBlurMLXOperation) -> None:
        """Test applying blur to RGBA image preserves alpha."""
        # Create RGBA image with semi-transparent white square
        img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 255))
        for y in range(40, 60):
            for x in range(40, 60):
                img.putpixel((x, y), (255, 255, 255, 128))

        params = {"sigma": 2.0}
        result = operation.apply(img, params)

        # Result should have same dimensions and mode
        assert result.size == img.size
        assert result.mode == img.mode

        # RGB channels should be blurred
        blurred = np.array(result)
        assert 0 < blurred[50, 39, 0] < 255

        # Alpha should be preserved (not blurred)
        original_alpha = np.array(img)[:, :, 3]
        result_alpha = blurred[:, :, 3]
        assert np.array_equal(result_alpha, original_alpha)

    def test_consistency_across_runs(
        self, operation: GaussianBlurMLXOperation, test_image: Image.Image
    ) -> None:
        """Test that MLX blur produces consistent results across runs."""
        params = {"sigma": 2.5}

        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Should produce identical results
        arr1 = np.array(result1)
        arr2 = np.array(result2)

        # Allow for tiny floating point differences due to GPU computation
        assert np.allclose(arr1, arr2, atol=0.01)

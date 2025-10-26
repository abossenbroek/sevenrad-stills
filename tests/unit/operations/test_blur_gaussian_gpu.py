"""Tests for GPU-accelerated Gaussian blur operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestGaussianBlurGPUOperation:
    """Tests for GaussianBlurGPUOperation class."""

    @pytest.fixture
    def operation(self) -> GaussianBlurGPUOperation:
        """Create a GPU Gaussian blur operation instance."""
        return GaussianBlurGPUOperation()

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

    def test_operation_name(self, operation: GaussianBlurGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "blur_gaussian_gpu"

    def test_valid_params(self, operation: GaussianBlurGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"sigma": 2.0}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_integer(
        self, operation: GaussianBlurGPUOperation
    ) -> None:
        """Test valid parameters with integer sigma."""
        params = {"sigma": 5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_zero(self, operation: GaussianBlurGPUOperation) -> None:
        """Test valid parameters with zero sigma."""
        params = {"sigma": 0}
        operation.validate_params(params)  # Should not raise

    def test_missing_sigma_raises_error(
        self, operation: GaussianBlurGPUOperation
    ) -> None:
        """Test missing sigma parameter raises error."""
        params = {}
        with pytest.raises(ValueError, match="requires a 'sigma' parameter"):
            operation.validate_params(params)

    def test_invalid_sigma_type_raises_error(
        self, operation: GaussianBlurGPUOperation
    ) -> None:
        """Test invalid sigma type raises error."""
        params = {"sigma": "2.0"}
        with pytest.raises(ValueError, match="Sigma must be a number"):
            operation.validate_params(params)

    def test_negative_sigma_raises_error(
        self, operation: GaussianBlurGPUOperation
    ) -> None:
        """Test negative sigma raises error."""
        params = {"sigma": -1.0}
        with pytest.raises(ValueError, match="Sigma must be non-negative"):
            operation.validate_params(params)

    def test_zero_sigma_returns_copy(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
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
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
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

        # Check that sharp edge at (39, 50) -> (40, 50) is now blurred
        # Original has sharp transition from 0 to 255
        assert original[50, 39, 0] == 0
        assert original[50, 40, 0] == 255

        # Blurred should have gradual transition
        assert 0 < blurred[50, 39, 0] < 255
        assert blurred[50, 40, 0] < 255

    def test_apply_blur_grayscale(
        self, operation: GaussianBlurGPUOperation, grayscale_image: Image.Image
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

        # Check blur effect
        assert original[50, 39] == 0
        assert original[50, 40] == 255
        assert 0 < blurred[50, 39] < 255
        assert blurred[50, 40] < 255

    def test_apply_blur_rgba(self, operation: GaussianBlurGPUOperation) -> None:
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
        assert 0 < blurred[50, 39, 0] < 255  # Edge should be blurred

        # Alpha should be preserved (not blurred)
        original_alpha = np.array(img)[:, :, 3]
        result_alpha = blurred[:, :, 3]
        assert np.array_equal(result_alpha, original_alpha)

    def test_larger_sigma_produces_more_blur(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that larger sigma produces stronger blur."""
        result_small = operation.apply(test_image, {"sigma": 1.0})
        result_large = operation.apply(test_image, {"sigma": 5.0})

        small_blur = np.array(result_small)
        large_blur = np.array(result_large)

        # At edge position, larger sigma should produce more blur
        # (value further from original sharp edge)
        # Original: black (0) at (39,50), white (255) at (40,50)
        # Larger blur should spread the white further into black region
        assert small_blur[50, 35, 0] < large_blur[50, 35, 0]

    def test_small_sigma_preserves_details(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that small sigma preserves image details."""
        params = {"sigma": 0.5}
        result = operation.apply(test_image, params)

        original = np.array(test_image)
        blurred = np.array(result)

        # With very small sigma, center should remain close to white
        # and edges should remain close to black
        assert blurred[50, 50, 0] > 200  # Center still bright
        assert blurred[10, 10, 0] < 50  # Corner still dark

    def test_result_dtype_preservation(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that result preserves original dtype."""
        params = {"sigma": 2.0}
        result = operation.apply(test_image, params)

        original_array = np.array(test_image)
        result_array = np.array(result)

        assert result_array.dtype == original_array.dtype

    def test_symmetry(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that Gaussian blur is symmetric."""
        params = {"sigma": 3.0}
        result = operation.apply(test_image, params)

        blurred = np.array(result)

        # Due to symmetric input (white square in center), output should
        # be approximately symmetric. Check a few symmetric pairs.
        # Center row, left vs right
        assert np.abs(blurred[50, 30, 0] - blurred[50, 70, 0]) < 2

        # Center column, top vs bottom
        assert np.abs(blurred[30, 50, 0] - blurred[70, 50, 0]) < 2

    def test_consistency_across_runs(
        self, operation: GaussianBlurGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that GPU blur produces consistent results across runs."""
        params = {"sigma": 2.5}

        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Should produce identical results
        assert np.array_equal(np.array(result1), np.array(result2))

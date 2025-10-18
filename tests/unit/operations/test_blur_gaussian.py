"""Tests for Gaussian blur operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation


@pytest.fixture
def blur_op() -> GaussianBlurOperation:
    """Create a Gaussian blur operation instance."""
    return GaussianBlurOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image."""
    # Create a 100x100 RGB image with a white square on black background
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # Draw white square in the middle
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 255, 255)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    img = Image.new("L", (100, 100), color=0)
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = 255  # type: ignore[index]
    return img


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image."""
    img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 255))
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 255, 255, 200)  # type: ignore[index]
    return img


class TestGaussianBlurValidation:
    """Test parameter validation for Gaussian blur operation."""

    def test_validation_missing_sigma(self, blur_op: GaussianBlurOperation) -> None:
        """Test that missing sigma parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires a 'sigma' parameter"):
            blur_op.validate_params({})

    def test_validation_invalid_sigma_type(
        self, blur_op: GaussianBlurOperation
    ) -> None:
        """Test that non-numeric sigma raises ValueError."""
        with pytest.raises(ValueError, match="Sigma must be a number"):
            blur_op.validate_params({"sigma": "invalid"})

    def test_validation_negative_sigma(self, blur_op: GaussianBlurOperation) -> None:
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="Sigma must be non-negative"):
            blur_op.validate_params({"sigma": -1.0})

    def test_validation_valid_sigma(self, blur_op: GaussianBlurOperation) -> None:
        """Test that valid sigma passes validation."""
        # Should not raise
        blur_op.validate_params({"sigma": 2.5})
        blur_op.validate_params({"sigma": 0})
        blur_op.validate_params({"sigma": 10})


class TestGaussianBlurApply:
    """Test applying Gaussian blur to images."""

    def test_apply_zero_sigma(
        self, blur_op: GaussianBlurOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that sigma=0 returns an identical image."""
        result = blur_op.apply(test_image_rgb, {"sigma": 0})

        # Verify dimensions and mode are unchanged
        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        # Verify pixel values are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self, blur_op: GaussianBlurOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying blur to RGB image."""
        result = blur_op.apply(test_image_rgb, {"sigma": 2.0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (blur was applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify edges of the white square are now blurred (not pure white/black)
        # Check a pixel just outside the square - should have some gray value
        edge_pixel = result_array[38, 50]
        assert np.all(edge_pixel > 0), "Edge should be blurred (not pure black)"
        assert np.all(edge_pixel < 255), "Edge should be blurred (not pure white)"

    def test_apply_grayscale_image(
        self, blur_op: GaussianBlurOperation, test_image_grayscale: Image.Image
    ) -> None:
        """Test applying blur to grayscale image."""
        result = blur_op.apply(test_image_grayscale, {"sigma": 2.0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify blur was applied
        edge_pixel = result_array[38, 50]
        assert 0 < edge_pixel < 255, "Edge should be blurred"

    def test_apply_rgba_image(
        self, blur_op: GaussianBlurOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying blur to RGBA image preserves alpha channel."""
        result = blur_op.apply(test_image_rgba, {"sigma": 2.0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

        # Verify RGB channels are blurred
        original_array = np.array(test_image_rgba)
        result_array = np.array(result)
        assert not np.array_equal(
            result_array[..., :3], original_array[..., :3]
        ), "RGB should be blurred"

        # Verify alpha channel is preserved
        np.testing.assert_array_equal(
            result_array[..., 3],
            original_array[..., 3],
            err_msg="Alpha should be unchanged",
        )

    def test_apply_various_sigma_values(
        self, blur_op: GaussianBlurOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different sigma values produce different blur intensities."""
        result_small = blur_op.apply(test_image_rgb, {"sigma": 1.0})
        result_large = blur_op.apply(test_image_rgb, {"sigma": 5.0})

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

        # Larger sigma should produce more blur (more smoothing)
        # Check variance in a small region - higher blur = lower variance
        region_small = array_small[35:45, 35:45, 0].astype(float)
        region_large = array_large[35:45, 35:45, 0].astype(float)

        var_small = np.var(region_small)
        var_large = np.var(region_large)

        assert var_large < var_small, "Larger sigma should produce more smoothing"

    def test_operation_name(self, blur_op: GaussianBlurOperation) -> None:
        """Test that operation has correct name."""
        assert blur_op.name == "blur_gaussian"

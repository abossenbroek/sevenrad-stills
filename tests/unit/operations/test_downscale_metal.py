"""Tests for Metal-accelerated downscale operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.downscale_metal import DownscaleMetalOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on Mac",
)
class TestDownscaleMetalOperation:
    """Tests for DownscaleMetalOperation class."""

    @pytest.fixture
    def operation(self) -> DownscaleMetalOperation:
        """Create a Metal downscale operation instance."""
        return DownscaleMetalOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with gradient pattern."""
        # Create an image with gradient to test interpolation
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                img[y, x] = [
                    int(x * 2.55),  # R: 0-255 gradient left-to-right
                    int(y * 2.55),  # G: 0-255 gradient top-to-bottom
                    128,  # B: constant
                ]
        return Image.fromarray(img)

    def test_operation_name(self, operation: DownscaleMetalOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "downscale_metal"

    def test_valid_params(self, operation: DownscaleMetalOperation) -> None:
        """Test valid parameter validation."""
        params = {"scale": 0.5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "scale": 0.25,
            "upscale": False,
            "downscale_method": "nearest",
            "upscale_method": "bilinear",
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_scale_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test missing scale parameter raises error."""
        params = {}
        with pytest.raises(ValueError, match="requires 'scale' parameter"):
            operation.validate_params(params)

    def test_invalid_scale_type_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test invalid scale type raises error."""
        params = {"scale": "0.5"}
        with pytest.raises(ValueError, match="Scale must be a number"):
            operation.validate_params(params)

    def test_scale_too_low_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test scale below minimum raises error."""
        params = {"scale": 0.001}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

    def test_scale_too_high_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test scale above maximum raises error."""
        params = {"scale": 1.5}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

    def test_invalid_upscale_type_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test invalid upscale type raises error."""
        params = {"scale": 0.5, "upscale": "yes"}
        with pytest.raises(ValueError, match="Upscale must be a boolean"):
            operation.validate_params(params)

    def test_invalid_downscale_method_type_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test invalid downscale_method type raises error."""
        params = {"scale": 0.5, "downscale_method": 1}
        with pytest.raises(ValueError, match="Downscale method must be a string"):
            operation.validate_params(params)

    def test_unsupported_downscale_method_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test unsupported downscale method raises error."""
        params = {"scale": 0.5, "downscale_method": "lanczos"}
        with pytest.raises(ValueError, match="Metal version supports"):
            operation.validate_params(params)

    def test_invalid_upscale_method_type_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test invalid upscale_method type raises error."""
        params = {"scale": 0.5, "upscale_method": 1}
        with pytest.raises(ValueError, match="Upscale method must be a string"):
            operation.validate_params(params)

    def test_unsupported_upscale_method_raises_error(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test unsupported upscale method raises error."""
        params = {"scale": 0.5, "upscale_method": "bicubic"}
        with pytest.raises(ValueError, match="Metal version supports"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: DownscaleMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying Metal downscale."""
        params = {"scale": 0.5}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size  # Upscaled back to original
        assert result.mode == "RGB"

    def test_apply_without_upscale(
        self, operation: DownscaleMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying downscale without upscaling."""
        params = {"scale": 0.5, "upscale": False}
        result = operation.apply(test_image, params)
        assert result.size == (50, 50)  # Half the original size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: DownscaleMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that Metal downscale actually changes the image."""
        params = {"scale": 0.1, "upscale_method": "nearest"}
        result = operation.apply(test_image, params)

        # Should have differences due to pixelation
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_nearest_method_creates_pixelation(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test that nearest neighbor creates visible pixelation."""
        test_img = Image.new("RGB", (100, 100), color=(100, 150, 200))

        params = {
            "scale": 0.1,
            "downscale_method": "nearest",
            "upscale_method": "nearest",
        }
        result = operation.apply(test_img, params)

        # With such extreme downscaling, we should see block artifacts
        result_array = np.array(result)
        assert result_array.shape == (100, 100, 3)

        # Check for uniform blocks (pixelation pattern)
        # In a 10x10 downscaled image upscaled to 100x100,
        # each 10x10 block should be identical
        block = result_array[0:10, 0:10]
        assert np.all(block == block[0, 0])

    def test_bilinear_method_smoother(
        self, operation: DownscaleMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that bilinear interpolation is smoother than nearest."""
        # Use gradient image (test_image fixture) to show interpolation differences

        params_nearest = {
            "scale": 0.25,
            "downscale_method": "nearest",
            "upscale_method": "nearest",
        }
        params_bilinear = {
            "scale": 0.25,
            "downscale_method": "bilinear",
            "upscale_method": "bilinear",
        }

        result_nearest = operation.apply(test_image, params_nearest)
        result_bilinear = operation.apply(test_image, params_bilinear)

        # Results should differ due to interpolation method
        assert not np.array_equal(np.array(result_nearest), np.array(result_bilinear))

    def test_rgba_preserves_alpha(self, operation: DownscaleMetalOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(100, 150, 200, 200))
        params = {"scale": 0.5}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"
        assert result.size == (100, 100)

    def test_grayscale_raises_error(self, operation: DownscaleMetalOperation) -> None:
        """Test that grayscale images raise an error."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"scale": 0.5}

        with pytest.raises(ValueError, match="Metal downscale requires RGB or RGBA"):
            operation.apply(gray_image, params)

    def test_extreme_downscale(self, operation: DownscaleMetalOperation) -> None:
        """Test extreme downscaling to minimum size."""
        test_img = Image.new("RGB", (100, 100), color=(100, 150, 200))
        params = {"scale": 0.01, "upscale": False}
        result = operation.apply(test_img, params)

        # Should result in 1x1 image (minimum size)
        assert result.size == (1, 1)

    def test_no_downscale(
        self, operation: DownscaleMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that scale=1.0 produces similar output."""
        params = {"scale": 1.0}
        result = operation.apply(test_image, params)

        # Should be very similar (allowing for minor rounding differences)
        diff = np.abs(
            np.array(test_image, dtype=np.int16) - np.array(result, dtype=np.int16)
        )
        assert np.mean(diff) < 2.0  # Average difference less than 2 pixel values

    def test_multiple_channels(self, operation: DownscaleMetalOperation) -> None:
        """Test that all RGB channels are processed correctly."""
        # Create image with different values per channel
        test_img = Image.new("RGB", (100, 100), color=(255, 0, 0))  # Pure red

        params = {"scale": 0.5, "upscale": False}
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        # Red channel should be dominant
        assert np.all(result_array[:, :, 0] > 200)  # R high
        assert np.all(result_array[:, :, 1] < 50)  # G low
        assert np.all(result_array[:, :, 2] < 50)  # B low

    def test_metal_produces_valid_output(self, test_image: Image.Image) -> None:
        """Test that Metal version produces valid, reasonable output."""
        metal_op = DownscaleMetalOperation()

        params = {"scale": 0.25}
        metal_result = metal_op.apply(test_image, params)
        metal_array = np.array(metal_result)

        # Verify output is valid
        assert metal_array.shape == (100, 100, 3)
        assert metal_array.dtype == np.uint8

        # Result should contain values in expected range
        assert metal_array.min() >= 0
        assert metal_array.max() <= 255

        # Due to downscaling/upscaling, we should see changes
        original_array = np.array(test_image)
        assert not np.array_equal(original_array, metal_array)

    def test_downscale_maintains_aspect_ratio(
        self, operation: DownscaleMetalOperation
    ) -> None:
        """Test that downscaling maintains aspect ratio."""
        # Create rectangular image
        test_img = Image.new("RGB", (200, 100), color=(100, 150, 200))
        params = {"scale": 0.5, "upscale": False}
        result = operation.apply(test_img, params)

        # Should be 100x50 (maintaining 2:1 aspect ratio)
        assert result.size == (100, 50)

"""Tests for downscale operation."""

import pytest
from PIL import Image
from sevenrad_stills.operations.downscale import DownscaleOperation


class TestDownscaleOperation:
    """Tests for DownscaleOperation class."""

    @pytest.fixture
    def operation(self) -> DownscaleOperation:
        """Create a downscale operation instance."""
        return DownscaleOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (200, 200), color=(100, 150, 200))

    def test_operation_name(self, operation: DownscaleOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "downscale"

    def test_valid_params(self, operation: DownscaleOperation) -> None:
        """Test valid parameter validation."""
        params = {"scale": 0.5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_full_options(self, operation: DownscaleOperation) -> None:
        """Test valid parameters with all options."""
        params = {
            "scale": 0.25,
            "upscale": True,
            "downscale_method": "bicubic",
            "upscale_method": "nearest",
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_scale_raises_error(self, operation: DownscaleOperation) -> None:
        """Test missing scale parameter raises error."""
        params = {"upscale": True}
        with pytest.raises(ValueError, match="requires 'scale' parameter"):
            operation.validate_params(params)

    def test_invalid_scale_type_raises_error(
        self, operation: DownscaleOperation
    ) -> None:
        """Test invalid scale type raises error."""
        params = {"scale": "0.5"}
        with pytest.raises(ValueError, match="Scale must be a number"):
            operation.validate_params(params)

    def test_scale_too_low_raises_error(self, operation: DownscaleOperation) -> None:
        """Test scale below minimum raises error."""
        params = {"scale": 0.005}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

    def test_scale_too_high_raises_error(self, operation: DownscaleOperation) -> None:
        """Test scale above maximum raises error."""
        params = {"scale": 1.5}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

    def test_invalid_upscale_type_raises_error(
        self, operation: DownscaleOperation
    ) -> None:
        """Test invalid upscale type raises error."""
        params = {"scale": 0.5, "upscale": "yes"}
        with pytest.raises(ValueError, match="Upscale must be a boolean"):
            operation.validate_params(params)

    def test_invalid_downscale_method_type_raises_error(
        self, operation: DownscaleOperation
    ) -> None:
        """Test invalid downscale method type raises error."""
        params = {"scale": 0.5, "downscale_method": 123}
        with pytest.raises(ValueError, match="Downscale method must be a string"):
            operation.validate_params(params)

    def test_invalid_downscale_method_value_raises_error(
        self, operation: DownscaleOperation
    ) -> None:
        """Test invalid downscale method value raises error."""
        params = {"scale": 0.5, "downscale_method": "invalid"}
        with pytest.raises(ValueError, match="Invalid downscale method"):
            operation.validate_params(params)

    def test_invalid_upscale_method_raises_error(
        self, operation: DownscaleOperation
    ) -> None:
        """Test invalid upscale method raises error."""
        params = {"scale": 0.5, "upscale_method": "supersampling"}
        with pytest.raises(ValueError, match="Invalid upscale method"):
            operation.validate_params(params)

    def test_apply_with_upscale(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test downscale with upscale back to original size."""
        params = {"scale": 0.5, "upscale": True}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size  # Upscaled back to original

    def test_apply_without_upscale(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test downscale without upscaling."""
        params = {"scale": 0.5, "upscale": False}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)  # 50% of 200x200

    def test_apply_extreme_downscale(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test extreme downscaling."""
        params = {"scale": 0.1, "upscale": True, "upscale_method": "nearest"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_nearest_upscale_creates_pixelation(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test that nearest neighbor upscaling creates harsh pixelation."""
        params = {
            "scale": 0.25,
            "upscale": True,
            "downscale_method": "bicubic",
            "upscale_method": "nearest",
        }
        result = operation.apply(test_image, params)
        assert result.size == test_image.size

    def test_apply_different_resampling_methods(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test different resampling methods work."""
        for method in ["nearest", "bilinear", "bicubic", "lanczos", "box"]:
            params = {"scale": 0.5, "downscale_method": method}
            result = operation.apply(test_image, params)
            assert isinstance(result, Image.Image)

    def test_apply_preserves_image_mode(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test operation preserves image mode."""
        params = {"scale": 0.5}
        result = operation.apply(test_image, params)
        assert result.mode == test_image.mode

    def test_apply_minimum_size_constraint(self, operation: DownscaleOperation) -> None:
        """Test that downscaling enforces minimum 1x1 size."""
        tiny_image = Image.new("RGB", (10, 10), color=(100, 100, 100))
        params = {"scale": 0.01, "upscale": False}  # Would be 0x0
        result = operation.apply(tiny_image, params)
        # Should be at least 1x1
        assert result.width >= 1
        assert result.height >= 1

    def test_deterministic_output(
        self, operation: DownscaleOperation, test_image: Image.Image
    ) -> None:
        """Test that downscaling produces deterministic output."""
        params = {"scale": 0.5, "upscale": True, "upscale_method": "nearest"}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)
        assert list(result1.getdata()) == list(result2.getdata())

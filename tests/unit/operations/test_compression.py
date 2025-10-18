"""Tests for compression operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression import CompressionOperation


class TestCompressionOperation:
    """Tests for CompressionOperation class."""

    @pytest.fixture
    def operation(self) -> CompressionOperation:
        """Create a compression operation instance."""
        return CompressionOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        # Create a colorful RGB image to test compression artifacts
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    def test_operation_name(self, operation: CompressionOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "compression"

    def test_valid_params(self, operation: CompressionOperation) -> None:
        """Test valid parameter validation."""
        params = {"quality": 50}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_subsampling(
        self, operation: CompressionOperation
    ) -> None:
        """Test valid parameters with subsampling."""
        params = {"quality": 50, "subsampling": 2}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_optimize(self, operation: CompressionOperation) -> None:
        """Test valid parameters with optimize flag."""
        params = {"quality": 85, "subsampling": 0, "optimize": False}
        operation.validate_params(params)  # Should not raise

    def test_missing_quality_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test missing quality parameter raises error."""
        params = {"subsampling": 2}
        with pytest.raises(ValueError, match="requires 'quality' parameter"):
            operation.validate_params(params)

    def test_invalid_quality_type_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test invalid quality type raises error."""
        params = {"quality": "50"}
        with pytest.raises(ValueError, match="Quality must be an integer"):
            operation.validate_params(params)

    def test_quality_too_low_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test quality below minimum raises error."""
        params = {"quality": 0}
        with pytest.raises(ValueError, match="Quality must be between"):
            operation.validate_params(params)

    def test_quality_too_high_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test quality above maximum raises error."""
        params = {"quality": 101}
        with pytest.raises(ValueError, match="Quality must be between"):
            operation.validate_params(params)

    def test_invalid_subsampling_type_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test invalid subsampling type raises error."""
        params = {"quality": 50, "subsampling": "2"}
        with pytest.raises(ValueError, match="Subsampling must be an integer"):
            operation.validate_params(params)

    def test_invalid_subsampling_value_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test invalid subsampling value raises error."""
        params = {"quality": 50, "subsampling": 3}
        with pytest.raises(ValueError, match="Subsampling must be 0, 1, or 2"):
            operation.validate_params(params)

    def test_invalid_optimize_type_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test invalid optimize type raises error."""
        params = {"quality": 50, "optimize": "true"}
        with pytest.raises(ValueError, match="Optimize must be a boolean"):
            operation.validate_params(params)

    def test_apply_high_quality(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test applying high quality compression."""
        params = {"quality": 95}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_low_quality(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test applying low quality compression."""
        params = {"quality": 10}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_severe_compression(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test severe compression with heavy subsampling."""
        params = {"quality": 5, "subsampling": 2}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_no_subsampling(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test compression without subsampling."""
        params = {"quality": 85, "subsampling": 0}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_converts_rgba_to_rgb(self, operation: CompressionOperation) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"quality": 85}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_preserves_grayscale(self, operation: CompressionOperation) -> None:
        """Test that grayscale images remain grayscale."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"quality": 85}
        result = operation.apply(gray_image, params)
        assert result.mode == "L"

    def test_deterministic_output(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test that compression produces deterministic output."""
        params = {"quality": 50, "subsampling": 2}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Images should be the same size and mode
        assert result1.size == result2.size
        assert result1.mode == result2.mode

        # Pixel values should be very similar (JPEG can have minor variations)
        pixels1 = list(result1.getdata())
        pixels2 = list(result2.getdata())
        assert len(pixels1) == len(pixels2)

        # Check that most pixels match (allow for minor JPEG variations)
        matching_pixels = sum(
            1 for p1, p2 in zip(pixels1, pixels2, strict=False) if p1 == p2
        )
        match_ratio = matching_pixels / len(pixels1)
        assert match_ratio > 0.99  # At least 99% of pixels should match

    def test_invalid_gamma_type_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test that non-numeric gamma raises error."""
        params = {"quality": 80, "gamma": "1.5"}
        with pytest.raises(ValueError, match="Gamma must be a number"):
            operation.validate_params(params)

    def test_invalid_gamma_value_raises_error(
        self, operation: CompressionOperation
    ) -> None:
        """Test that zero or negative gamma raises error."""
        params = {"quality": 80, "gamma": 0}
        with pytest.raises(ValueError, match="Gamma must be positive"):
            operation.validate_params(params)

        params = {"quality": 80, "gamma": -1.0}
        with pytest.raises(ValueError, match="Gamma must be positive"):
            operation.validate_params(params)

    def test_valid_gamma_passes_validation(
        self, operation: CompressionOperation
    ) -> None:
        """Test that valid gamma values pass validation."""
        operation.validate_params({"quality": 80, "gamma": 1.5})
        operation.validate_params({"quality": 80, "gamma": 0.5})
        operation.validate_params({"quality": 80, "gamma": None})

    def test_apply_with_gamma_none_is_unchanged(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma=None behaves identically to no gamma parameter."""
        result_no_gamma = operation.apply(test_image, {"quality": 80})
        result_gamma_none = operation.apply(test_image, {"quality": 80, "gamma": None})
        np.testing.assert_array_equal(
            np.array(result_no_gamma), np.array(result_gamma_none)
        )

    def test_apply_gamma_darkens_image(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma > 1.0 darkens the image."""
        base_result = operation.apply(test_image, {"quality": 95})
        dark_result = operation.apply(test_image, {"quality": 95, "gamma": 2.0})

        mean_base = np.mean(np.array(base_result))
        mean_dark = np.mean(np.array(dark_result))

        assert mean_dark < mean_base

    def test_apply_gamma_brightens_image(
        self, operation: CompressionOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma < 1.0 brightens the image."""
        base_result = operation.apply(test_image, {"quality": 95})
        bright_result = operation.apply(test_image, {"quality": 95, "gamma": 0.5})

        mean_base = np.mean(np.array(base_result))
        mean_bright = np.mean(np.array(bright_result))

        assert mean_bright > mean_base

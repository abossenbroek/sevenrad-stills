"""Tests for GPU-accelerated compression operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression_gpu import CompressionGPUOperation


class TestCompressionGPUOperation:
    """Tests for CompressionGPUOperation class."""

    @pytest.fixture
    def operation(self) -> CompressionGPUOperation:
        """Create a GPU compression operation instance."""
        return CompressionGPUOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    def test_operation_name(self, operation: CompressionGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "compression_gpu"

    def test_valid_params(self, operation: CompressionGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"quality": 50}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_gamma(self, operation: CompressionGPUOperation) -> None:
        """Test valid parameters with gamma correction."""
        params = {"quality": 80, "gamma": 2.2}
        operation.validate_params(params)  # Should not raise

    def test_missing_quality_raises_error(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test missing quality parameter raises error."""
        params = {"gamma": 1.5}
        with pytest.raises(ValueError, match="requires 'quality' parameter"):
            operation.validate_params(params)

    def test_invalid_quality_type_raises_error(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test invalid quality type raises error."""
        params = {"quality": "50"}
        with pytest.raises(ValueError, match="Quality must be an integer"):
            operation.validate_params(params)

    def test_quality_out_of_range_raises_error(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test quality outside valid range raises error."""
        params = {"quality": 0}
        with pytest.raises(ValueError, match="Quality must be between"):
            operation.validate_params(params)

        params = {"quality": 101}
        with pytest.raises(ValueError, match="Quality must be between"):
            operation.validate_params(params)

    def test_invalid_gamma_type_raises_error(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test that non-numeric gamma raises error."""
        params = {"quality": 80, "gamma": "1.5"}
        with pytest.raises(ValueError, match="Gamma must be a number"):
            operation.validate_params(params)

    def test_invalid_gamma_value_raises_error(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test that zero or negative gamma raises error."""
        params = {"quality": 80, "gamma": 0}
        with pytest.raises(ValueError, match="Gamma must be positive"):
            operation.validate_params(params)

        params = {"quality": 80, "gamma": -1.0}
        with pytest.raises(ValueError, match="Gamma must be positive"):
            operation.validate_params(params)

    def test_apply_high_quality(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying high quality compression."""
        params = {"quality": 95}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_gamma_correction(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying compression with GPU gamma correction."""
        params = {"quality": 85, "gamma": 2.2}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_gamma_none_vs_no_gamma(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma=None behaves identically to no gamma parameter."""
        result_no_gamma = operation.apply(test_image, {"quality": 80})
        result_gamma_none = operation.apply(test_image, {"quality": 80, "gamma": None})
        np.testing.assert_array_equal(
            np.array(result_no_gamma), np.array(result_gamma_none)
        )

    def test_apply_gamma_darkens_image(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma > 1.0 darkens the image."""
        base_result = operation.apply(test_image, {"quality": 95})
        dark_result = operation.apply(test_image, {"quality": 95, "gamma": 2.0})

        mean_base = np.mean(np.array(base_result))
        mean_dark = np.mean(np.array(dark_result))

        assert mean_dark < mean_base

    def test_apply_gamma_brightens_image(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that gamma < 1.0 brightens the image."""
        base_result = operation.apply(test_image, {"quality": 95})
        bright_result = operation.apply(test_image, {"quality": 95, "gamma": 0.5})

        mean_base = np.mean(np.array(base_result))
        mean_bright = np.mean(np.array(bright_result))

        assert mean_bright > mean_base

    def test_apply_converts_rgba_to_rgb(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"quality": 85}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_grayscale_with_gamma(
        self, operation: CompressionGPUOperation
    ) -> None:
        """Test grayscale image with gamma correction."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"quality": 85, "gamma": 1.5}
        result = operation.apply(gray_image, params)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_subsampling_parameter_accepted(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that subsampling parameter is accepted."""
        params = {"quality": 80, "subsampling": 0}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)

    def test_optimize_parameter_accepted(
        self, operation: CompressionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that optimize parameter is accepted."""
        params = {"quality": 80, "optimize": False}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)

"""Tests for multi-compression operation."""

import pytest
from PIL import Image
from sevenrad_stills.operations.multi_compress import MultiCompressOperation


class TestMultiCompressOperation:
    """Tests for MultiCompressOperation class."""

    @pytest.fixture
    def operation(self) -> MultiCompressOperation:
        """Create a multi-compress operation instance."""
        return MultiCompressOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    def test_operation_name(self, operation: MultiCompressOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "multi_compress"

    def test_valid_params_fixed_decay(self, operation: MultiCompressOperation) -> None:
        """Test valid parameters with fixed decay."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_linear_decay(self, operation: MultiCompressOperation) -> None:
        """Test valid parameters with linear decay."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
        operation.validate_params(params)  # Should not raise

    def test_valid_params_exponential_decay(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test valid parameters with exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 15,
            "decay": "exponential",
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_iterations_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test missing iterations parameter raises error."""
        params = {"quality_start": 50}
        with pytest.raises(ValueError, match="requires 'iterations' parameter"):
            operation.validate_params(params)

    def test_invalid_iterations_type_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test invalid iterations type raises error."""
        params = {"iterations": "5", "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be an integer"):
            operation.validate_params(params)

    def test_iterations_too_low_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test iterations below minimum raises error."""
        params = {"iterations": 0, "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be between"):
            operation.validate_params(params)

    def test_iterations_too_high_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test iterations above maximum raises error."""
        params = {"iterations": 51, "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be between"):
            operation.validate_params(params)

    def test_missing_quality_start_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test missing quality_start parameter raises error."""
        params = {"iterations": 5}
        with pytest.raises(ValueError, match="requires 'quality_start' parameter"):
            operation.validate_params(params)

    def test_invalid_quality_start_type_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test invalid quality_start type raises error."""
        params = {"iterations": 5, "quality_start": "50"}
        with pytest.raises(ValueError, match="Quality start must be an integer"):
            operation.validate_params(params)

    def test_quality_start_out_of_range_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test quality_start out of range raises error."""
        params = {"iterations": 5, "quality_start": 101}
        with pytest.raises(ValueError, match="Quality start must be between"):
            operation.validate_params(params)

    def test_invalid_decay_type_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test invalid decay type raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "logarithmic"}
        with pytest.raises(ValueError, match="Decay must be"):
            operation.validate_params(params)

    def test_linear_decay_missing_quality_end_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test linear decay without quality_end raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "linear"}
        with pytest.raises(ValueError, match="requires 'quality_end' parameter"):
            operation.validate_params(params)

    def test_exponential_decay_missing_quality_end_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test exponential decay without quality_end raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "exponential"}
        with pytest.raises(ValueError, match="requires 'quality_end' parameter"):
            operation.validate_params(params)

    def test_quality_end_not_less_than_start_raises_error(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test quality_end >= quality_start raises error."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 60,
            "decay": "linear",
        }
        with pytest.raises(ValueError, match=r"Quality end.*must be less than"):
            operation.validate_params(params)

    def test_apply_single_iteration(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test applying single compression iteration."""
        params = {"iterations": 1, "quality_start": 50}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_fixed_decay(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with fixed quality."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_linear_decay(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with linear decay."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_exponential_decay(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 10,
            "decay": "exponential",
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_with_subsampling(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test multi-compression with subsampling."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
            "subsampling": 2,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_converts_rgba_to_rgb(
        self, operation: MultiCompressOperation
    ) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_preserves_grayscale(self, operation: MultiCompressOperation) -> None:
        """Test that grayscale images remain grayscale."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(gray_image, params)
        assert result.mode == "L"

    def test_more_iterations_completes_successfully(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test that multiple iterations complete successfully."""
        # Apply few iterations
        params_few = {"iterations": 2, "quality_start": 50}
        result_few = operation.apply(test_image, params_few)

        # Apply many iterations
        params_many = {"iterations": 10, "quality_start": 50}
        result_many = operation.apply(test_image, params_many)

        # Both should be valid images with correct properties
        assert isinstance(result_few, Image.Image)
        assert isinstance(result_many, Image.Image)
        assert result_few.size == test_image.size
        assert result_many.size == test_image.size
        assert result_few.mode == "RGB"
        assert result_many.mode == "RGB"

    def test_deterministic_output(
        self, operation: MultiCompressOperation, test_image: Image.Image
    ) -> None:
        """Test that multi-compression produces deterministic output."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
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

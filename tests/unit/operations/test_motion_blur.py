"""Tests for motion blur operation."""

import pytest
from PIL import Image
from sevenrad_stills.operations.motion_blur import MotionBlurOperation


class TestMotionBlurOperation:
    """Tests for MotionBlurOperation class."""

    @pytest.fixture
    def operation(self) -> MotionBlurOperation:
        """Create a motion blur operation instance."""
        return MotionBlurOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: MotionBlurOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "motion_blur"

    def test_valid_params(self, operation: MotionBlurOperation) -> None:
        """Test valid parameter validation."""
        params = {"kernel_size": 5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_angle(self, operation: MotionBlurOperation) -> None:
        """Test valid parameters with angle."""
        params = {"kernel_size": 10, "angle": 45.0}
        operation.validate_params(params)  # Should not raise

    def test_missing_kernel_size_raises_error(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test missing kernel_size parameter raises error."""
        params = {"angle": 90.0}
        with pytest.raises(ValueError, match="requires 'kernel_size' parameter"):
            operation.validate_params(params)

    def test_invalid_kernel_size_type_raises_error(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test invalid kernel_size type raises error."""
        params = {"kernel_size": "5"}
        with pytest.raises(ValueError, match="Kernel size must be an integer"):
            operation.validate_params(params)

    def test_kernel_size_too_low_raises_error(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test kernel_size below minimum raises error."""
        params = {"kernel_size": 0}
        with pytest.raises(ValueError, match="Kernel size must be between"):
            operation.validate_params(params)

    def test_kernel_size_too_high_raises_error(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test kernel_size above maximum raises error."""
        params = {"kernel_size": 101}
        with pytest.raises(ValueError, match="Kernel size must be between"):
            operation.validate_params(params)

    def test_invalid_angle_type_raises_error(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test invalid angle type raises error."""
        params = {"kernel_size": 5, "angle": "45"}
        with pytest.raises(ValueError, match="Angle must be a number"):
            operation.validate_params(params)

    def test_angle_too_low_raises_error(self, operation: MotionBlurOperation) -> None:
        """Test angle below minimum raises error."""
        params = {"kernel_size": 5, "angle": -10.0}
        with pytest.raises(ValueError, match="Angle must be between"):
            operation.validate_params(params)

    def test_angle_too_high_raises_error(self, operation: MotionBlurOperation) -> None:
        """Test angle at/above maximum raises error."""
        params = {"kernel_size": 5, "angle": 360.0}
        with pytest.raises(ValueError, match="Angle must be between"):
            operation.validate_params(params)

    def test_apply_minimal_blur(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test applying minimal blur."""
        params = {"kernel_size": 2}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == test_image.mode

    def test_apply_kernel_size_one_returns_copy(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test that kernel size 1 returns a copy of the image."""
        params = {"kernel_size": 1}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        # Should be effectively unchanged
        assert list(result.getdata()) == list(test_image.getdata())

    def test_apply_horizontal_blur(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test applying horizontal blur (0 degrees)."""
        params = {"kernel_size": 10, "angle": 0.0}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_vertical_blur(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test applying vertical blur (90 degrees)."""
        params = {"kernel_size": 10, "angle": 90.0}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_diagonal_blur(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test applying diagonal blur (45 degrees)."""
        params = {"kernel_size": 8, "angle": 45.0}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_heavy_blur(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test applying heavy blur."""
        params = {"kernel_size": 30}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_preserves_image_mode(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test operation preserves image mode."""
        params = {"kernel_size": 5}
        result = operation.apply(test_image, params)
        assert result.mode == test_image.mode

    def test_apply_works_with_grayscale(self, operation: MotionBlurOperation) -> None:
        """Test that motion blur works with grayscale images."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"kernel_size": 5}
        result = operation.apply(gray_image, params)
        assert result.mode == "L"
        assert result.size == gray_image.size

    def test_apply_works_with_rgba(self, operation: MotionBlurOperation) -> None:
        """Test that motion blur works with RGBA images."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))
        params = {"kernel_size": 5}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGBA"
        assert result.size == rgba_image.size

    def test_deterministic_output(
        self, operation: MotionBlurOperation, test_image: Image.Image
    ) -> None:
        """Test that motion blur produces deterministic output."""
        params = {"kernel_size": 10, "angle": 45.0}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)
        assert list(result1.getdata()) == list(result2.getdata())

    def test_different_angles_produce_different_results(
        self, operation: MotionBlurOperation
    ) -> None:
        """Test that different angles produce different blur effects."""
        # Create an image with distinct content to see blur differences
        content_image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        # Add a vertical line to make blur direction visible
        for y in range(100):
            content_image.putpixel((50, y), (0, 0, 0))

        result_h = operation.apply(content_image, {"kernel_size": 10, "angle": 0.0})
        result_v = operation.apply(content_image, {"kernel_size": 10, "angle": 90.0})

        # Results should be different
        assert list(result_h.getdata()) != list(result_v.getdata())

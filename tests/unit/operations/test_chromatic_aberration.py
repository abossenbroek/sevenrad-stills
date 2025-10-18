"""Tests for chromatic aberration operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.chromatic_aberration import (
    ChromaticAberrationOperation,
)


@pytest.fixture
def chroma_op() -> ChromaticAberrationOperation:
    """Create a chromatic aberration operation instance."""
    return ChromaticAberrationOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create an RGB test image with distinct regions."""
    img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    pixels = img.load()
    # Create a white vertical line in the middle
    for i in range(50):
        pixels[i, 25] = (255, 255, 255)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a grayscale test image."""
    return Image.new("L", (50, 50), color=128)


class TestChromaticAberrationValidation:
    """Test parameter validation for chromatic aberration operation."""

    def test_validation_missing_shift_x(
        self, chroma_op: ChromaticAberrationOperation
    ) -> None:
        """Test that missing shift_x raises ValueError."""
        with pytest.raises(ValueError, match="'shift_x' is required"):
            chroma_op.validate_params({"shift_y": 0})

    def test_validation_missing_shift_y(
        self, chroma_op: ChromaticAberrationOperation
    ) -> None:
        """Test that missing shift_y raises ValueError."""
        with pytest.raises(ValueError, match="'shift_y' is required"):
            chroma_op.validate_params({"shift_x": 0})

    def test_validation_invalid_shift_x_type(
        self, chroma_op: ChromaticAberrationOperation
    ) -> None:
        """Test that non-integer shift_x raises ValueError."""
        with pytest.raises(ValueError, match="'shift_x' must be an integer"):
            chroma_op.validate_params({"shift_x": 2.5, "shift_y": 0})

    def test_validation_invalid_shift_y_type(
        self, chroma_op: ChromaticAberrationOperation
    ) -> None:
        """Test that non-integer shift_y raises ValueError."""
        with pytest.raises(ValueError, match="'shift_y' must be an integer"):
            chroma_op.validate_params({"shift_x": 0, "shift_y": "2"})

    def test_validation_valid_params(
        self, chroma_op: ChromaticAberrationOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        chroma_op.validate_params({"shift_x": 0, "shift_y": 0})
        chroma_op.validate_params({"shift_x": 5, "shift_y": -3})
        chroma_op.validate_params({"shift_x": -10, "shift_y": 10})


class TestChromaticAberrationApply:
    """Test applying chromatic aberration to images."""

    def test_no_shift(
        self, chroma_op: ChromaticAberrationOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that zero shift produces an identical image."""
        result = chroma_op.apply(test_image_rgb, {"shift_x": 0, "shift_y": 0})

        # Verify dimensions and mode preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify pixels are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_horizontal_shift(
        self, chroma_op: ChromaticAberrationOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test horizontal channel shifting."""
        result = chroma_op.apply(test_image_rgb, {"shift_x": 3, "shift_y": 0})

        result_array = np.array(result)
        original_array = np.array(test_image_rgb)

        # Verify dimensions preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed due to channel shifts
        assert not np.array_equal(result_array, original_array)

        # Red should be shifted in x direction (+3 in column index)
        # Blue should be shifted opposite (-3 in column index)
        # Check that red channel at shifted position matches original
        # Original white line is at column 25
        # Red channel (shift_x=+3) should have white at column 25+3=28
        assert np.any(result_array[:, 28, 0] == 255), "Red shifted right"
        # Blue channel (shift_x=-3) should have white at column 25-3=22
        assert np.any(result_array[:, 22, 2] == 255), "Blue shifted left"

    def test_vertical_shift(
        self, chroma_op: ChromaticAberrationOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test vertical channel shifting."""
        result = chroma_op.apply(test_image_rgb, {"shift_x": 0, "shift_y": 3})

        # Verify dimensions preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_non_color_image(
        self,
        chroma_op: ChromaticAberrationOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test that grayscale images are returned unmodified."""
        result = chroma_op.apply(test_image_grayscale, {"shift_x": 5, "shift_y": 5})

        # Verify image is unchanged
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_rgba_image(self, chroma_op: ChromaticAberrationOperation) -> None:
        """Test RGBA image preserves alpha channel."""
        img = Image.new("RGBA", (50, 50), color=(128, 128, 128, 200))
        result = chroma_op.apply(img, {"shift_x": 2, "shift_y": 2})

        # Verify mode preserved
        assert result.mode == "RGBA"

        # Verify alpha channel unchanged
        original_array = np.array(img)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array[..., 3], original_array[..., 3])

    def test_operation_name(self, chroma_op: ChromaticAberrationOperation) -> None:
        """Test that operation has correct name."""
        assert chroma_op.name == "chromatic_aberration"

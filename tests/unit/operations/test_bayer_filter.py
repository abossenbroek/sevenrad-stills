"""Tests for Bayer filter operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation


@pytest.fixture
def bayer_op() -> BayerFilterOperation:
    """Create a Bayer filter operation instance."""
    return BayerFilterOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with a colored square."""
    img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 128, 64)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    return Image.new("L", (100, 100), color=128)


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image with a semi-transparent colored square."""
    img = Image.new("RGBA", (100, 100), color=(10, 20, 30, 255))
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 128, 64, 200)  # type: ignore[index]
    return img


class TestBayerFilterValidation:
    """Test parameter validation for Bayer filter operation."""

    def test_validation_missing_pattern_is_ok(
        self, bayer_op: BayerFilterOperation
    ) -> None:
        """Test that missing pattern parameter is valid (uses default)."""
        try:
            bayer_op.validate_params({})
        except ValueError:
            pytest.fail("validate_params should not raise for missing 'pattern'")

    def test_validation_invalid_pattern_type(
        self, bayer_op: BayerFilterOperation
    ) -> None:
        """Test that non-string pattern raises ValueError."""
        with pytest.raises(ValueError, match="Pattern must be a string"):
            bayer_op.validate_params({"pattern": 123})

    def test_validation_invalid_pattern_string(
        self, bayer_op: BayerFilterOperation
    ) -> None:
        """Test that an invalid pattern string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pattern 'INVALID'"):
            bayer_op.validate_params({"pattern": "INVALID"})

    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_validation_valid_patterns(
        self, bayer_op: BayerFilterOperation, pattern: str
    ) -> None:
        """Test that valid patterns pass validation."""
        try:
            bayer_op.validate_params({"pattern": pattern})
        except ValueError:
            pytest.fail(
                f"validate_params should not raise for valid pattern '{pattern}'"
            )


class TestBayerFilterApply:
    """Test applying Bayer filter to images."""

    def test_operation_name(self, bayer_op: BayerFilterOperation) -> None:
        """Test that operation has correct name."""
        assert bayer_op.name == "bayer_filter"

    def test_apply_grayscale_returns_copy(
        self, bayer_op: BayerFilterOperation, test_image_grayscale: Image.Image
    ) -> None:
        """Test that grayscale image is returned unchanged, but as a copy."""
        result = bayer_op.apply(test_image_grayscale, {})
        assert result.mode == test_image_grayscale.mode
        assert result.size == test_image_grayscale.size
        assert np.array_equal(np.array(result), np.array(test_image_grayscale))
        assert result is not test_image_grayscale, "Should be a copy"

    def test_apply_rgb_image_changes_pixels(
        self, bayer_op: BayerFilterOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying filter to an RGB image alters it."""
        result = bayer_op.apply(test_image_rgb, {"pattern": "RGGB"})
        assert result.mode == "RGB"
        assert result.size == test_image_rgb.size
        assert not np.array_equal(np.array(result), np.array(test_image_rgb))

    def test_apply_rgba_preserves_alpha(
        self, bayer_op: BayerFilterOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying filter to RGBA image preserves alpha channel."""
        result = bayer_op.apply(test_image_rgba, {"pattern": "RGGB"})
        assert result.mode == "RGBA"
        assert result.size == test_image_rgba.size

        original_array = np.array(test_image_rgba)
        result_array = np.array(result)

        # RGB channels should be different
        assert not np.array_equal(result_array[..., :3], original_array[..., :3])

        # Alpha channel should be the same
        np.testing.assert_array_equal(result_array[..., 3], original_array[..., 3])

    def test_different_patterns_produce_different_results(
        self, bayer_op: BayerFilterOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different Bayer patterns produce different output images."""
        result_rggb = bayer_op.apply(test_image_rgb, {"pattern": "RGGB"})
        result_bggr = bayer_op.apply(test_image_rgb, {"pattern": "BGGR"})

        assert not np.array_equal(np.array(result_rggb), np.array(result_bggr))

    def test_default_pattern_is_rggb(
        self, bayer_op: BayerFilterOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that omitting the pattern parameter uses the RGGB default."""
        result_default = bayer_op.apply(test_image_rgb, {})
        result_rggb = bayer_op.apply(test_image_rgb, {"pattern": "RGGB"})
        np.testing.assert_array_equal(np.array(result_default), np.array(result_rggb))

    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_all_valid_patterns_run_successfully(
        self, bayer_op: BayerFilterOperation, test_image_rgb: Image.Image, pattern: str
    ) -> None:
        """Test that all valid patterns execute without error and produce an image."""
        result = bayer_op.apply(test_image_rgb, {"pattern": pattern})
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == test_image_rgb.size

    def test_effect_creates_visible_artifacts(
        self, bayer_op: BayerFilterOperation
    ) -> None:
        """Test that the effect creates color artifacts at sharp edges."""
        img = Image.new("RGB", (50, 50), "black")
        pixels = img.load()
        for i in range(50):
            for j in range(25, 50):
                pixels[i, j] = (255, 255, 255)  # type: ignore[index]

        result = bayer_op.apply(img, {"pattern": "RGGB"})
        result_array = np.array(result).astype(np.float32)

        # Check for color fringing along the vertical edge (x=24)
        edge_pixels = result_array[:, 24, :]

        # Check if any pixel on the edge has significant color deviation
        # (i.e., is not grayscale). A standard deviation > 5 across R,G,B
        # indicates color is present.
        color_deviation = np.std(edge_pixels, axis=1)
        assert np.any(
            color_deviation > 5
        ), "Expected color fringing (non-grayscale pixels) at the edge"

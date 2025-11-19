"""Tests for Metal-accelerated compression artifact operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression_artifact import CompressionArtifactOperation
from sevenrad_stills.operations.compression_artifact_metal import (
    CompressionArtifactMetalOperation,
)


@pytest.fixture
def artifact_op_metal() -> CompressionArtifactMetalOperation:
    """Create a Metal compression artifact operation instance."""
    return CompressionArtifactMetalOperation()


@pytest.fixture
def artifact_op_cpu() -> CompressionArtifactOperation:
    """Create a CPU compression artifact operation instance for comparison."""
    return CompressionArtifactOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with color gradients."""
    # Create a 200x200 RGB image with gradients
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    pixels = img.load()
    for i in range(200):
        for j in range(200):
            pixels[j, i] = (i % 256, j % 256, (i + j) % 256)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    img = Image.new("L", (200, 200), color=128)
    pixels = img.load()
    for i in range(200):
        for j in range(200):
            pixels[j, i] = (i + j) % 256  # type: ignore[index]
    return img


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image."""
    img = Image.new("RGBA", (200, 200), color=(128, 128, 128, 255))
    pixels = img.load()
    for i in range(200):
        for j in range(200):
            pixels[j, i] = (i % 256, j % 256, (i + j) % 256, 200)  # type: ignore[index]
    return img


class TestCompressionArtifactMetalValidation:
    """Test parameter validation for Metal compression artifact operation."""

    def test_validation_missing_tile_count(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that missing tile_count parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            artifact_op_metal.validate_params({"quality": 5})

    def test_validation_missing_quality(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that missing quality parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'quality' parameter"):
            artifact_op_metal.validate_params({"tile_count": 5})

    def test_validation_invalid_tile_count_type(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that non-integer tile_count raises ValueError."""
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            artifact_op_metal.validate_params({"tile_count": 2.5, "quality": 5})

    def test_validation_tile_count_too_low(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that tile_count below minimum raises ValueError."""
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            artifact_op_metal.validate_params({"tile_count": 0, "quality": 5})

    def test_validation_tile_count_too_high(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that tile_count above maximum raises ValueError."""
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            artifact_op_metal.validate_params({"tile_count": 100, "quality": 5})

    def test_validation_invalid_quality_type(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that non-integer quality raises ValueError."""
        with pytest.raises(ValueError, match="quality must be an integer"):
            artifact_op_metal.validate_params({"tile_count": 5, "quality": 2.5})

    def test_validation_quality_too_low(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that quality below minimum raises ValueError."""
        with pytest.raises(ValueError, match="quality must be an integer between"):
            artifact_op_metal.validate_params({"tile_count": 5, "quality": 0})

    def test_validation_quality_too_high(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that quality above maximum raises ValueError."""
        with pytest.raises(ValueError, match="quality must be an integer between"):
            artifact_op_metal.validate_params({"tile_count": 5, "quality": 100})

    def test_validation_invalid_tile_size_range_type(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that invalid tile_size_range type raises ValueError."""
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two"
        ):
            artifact_op_metal.validate_params(
                {"tile_count": 5, "quality": 5, "tile_size_range": 0.1}
            )

    def test_validation_invalid_tile_size_range_length(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that wrong length tile_size_range raises ValueError."""
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two"
        ):
            artifact_op_metal.validate_params(
                {"tile_count": 5, "quality": 5, "tile_size_range": [0.1]}
            )

    def test_validation_tile_size_range_min_greater_than_max(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that min > max in tile_size_range raises ValueError."""
        with pytest.raises(
            ValueError, match="tile_size_range min must be less than or equal to max"
        ):
            artifact_op_metal.validate_params(
                {"tile_count": 5, "quality": 5, "tile_size_range": [0.5, 0.2]}
            )

    def test_validation_valid_params(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        artifact_op_metal.validate_params({"tile_count": 1, "quality": 1})
        artifact_op_metal.validate_params({"tile_count": 15, "quality": 10})
        artifact_op_metal.validate_params({"tile_count": 30, "quality": 20})
        artifact_op_metal.validate_params(
            {"tile_count": 5, "quality": 5, "tile_size_range": [0.05, 0.2]}
        )
        artifact_op_metal.validate_params({"tile_count": 5, "quality": 5, "seed": 42})


class TestCompressionArtifactMetalApply:
    """Test applying Metal compression artifact to images."""

    def test_apply_rgb_image(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying compression artifacts to RGB image."""
        result = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 3, "quality": 5, "seed": 42}
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (artifacts were applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_grayscale_image(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying compression artifacts to grayscale image."""
        result = artifact_op_metal.apply(
            test_image_grayscale, {"tile_count": 3, "quality": 5, "seed": 42}
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test applying artifacts to RGBA image."""
        result = artifact_op_metal.apply(
            test_image_rgba, {"tile_count": 3, "quality": 5, "seed": 42}
        )

        # Verify dimensions are preserved (mode may change to RGB)
        assert result.size == test_image_rgba.size

        # Verify image has changed
        original_array = np.array(test_image_rgba)
        result_array = np.array(result)
        # Compare only RGB channels if modes differ
        if result.mode == "RGB":
            assert not np.array_equal(result_array, original_array[..., :3])
        else:
            assert not np.array_equal(result_array, original_array)

    def test_apply_reproducible_with_seed(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that same seed produces identical results."""
        result1 = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 5, "seed": 123}
        )
        result2 = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 5, "seed": 123}
        )

        array1 = np.array(result1)
        array2 = np.array(result2)

        # Results should be identical with same seed
        np.testing.assert_array_equal(array1, array2)

    def test_apply_different_with_different_seeds(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different seeds produce different results."""
        result1 = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 5, "seed": 123}
        )
        result2 = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 5, "seed": 456}
        )

        array1 = np.array(result1)
        array2 = np.array(result2)

        # Results should be different with different seeds
        assert not np.array_equal(array1, array2)

    def test_apply_various_tile_counts(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different tile counts produce different results."""
        result_small = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 1, "quality": 5, "seed": 42}
        )
        result_large = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 10, "quality": 5, "seed": 42}
        )

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

    def test_apply_various_quality_values(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different quality values produce different results."""
        result_low = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 1, "seed": 42}
        )
        result_high = artifact_op_metal.apply(
            test_image_rgb, {"tile_count": 5, "quality": 15, "seed": 42}
        )

        array_low = np.array(result_low)
        array_high = np.array(result_high)

        # Results should be different
        assert not np.array_equal(array_low, array_high)

    def test_operation_name(
        self, artifact_op_metal: CompressionArtifactMetalOperation
    ) -> None:
        """Test that operation has correct name."""
        assert artifact_op_metal.name == "compression_artifact_metal"


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_consistency(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        artifact_op_cpu: CompressionArtifactOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for RGB."""
        params = {"tile_count": 3, "quality": 5, "seed": 42}
        result_metal = artifact_op_metal.apply(test_image_rgb, params)
        result_cpu = artifact_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be similar (JPEG compression implementations differ)
        # Metal uses custom DCT/IDCT vs PIL's optimized C encoder
        # Use atol=35.0 to account for numerical differences at low quality
        # (quality=5 is aggressive compression where differences are more visible)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=35.0,
            rtol=0.10,
            err_msg="Metal and CPU results should be similar",
        )

    def test_grayscale_consistency(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        artifact_op_cpu: CompressionArtifactOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for grayscale."""
        params = {"tile_count": 3, "quality": 5, "seed": 42}
        result_metal = artifact_op_metal.apply(test_image_grayscale, params)
        result_cpu = artifact_op_cpu.apply(test_image_grayscale, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Grayscale: slightly lower tolerance than RGB (max diff ~25)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=30.0,
            rtol=0.10,
            err_msg="Metal and CPU results should match for grayscale",
        )

    def test_single_tile_consistency(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        artifact_op_cpu: CompressionArtifactOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with single tile."""
        params = {"tile_count": 1, "quality": 10, "seed": 123}
        result_metal = artifact_op_metal.apply(test_image_rgb, params)
        result_cpu = artifact_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Single tile with higher quality (10): smaller differences (max diff ~21)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=25.0,
            rtol=0.10,
            err_msg="Metal and CPU should match for single tile",
        )

    def test_many_tiles_consistency(
        self,
        artifact_op_metal: CompressionArtifactMetalOperation,
        artifact_op_cpu: CompressionArtifactOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with many tiles."""
        params = {"tile_count": 15, "quality": 5, "seed": 456}
        result_metal = artifact_op_metal.apply(test_image_rgb, params)
        result_cpu = artifact_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Many tiles: higher tolerance due to accumulated differences (max diff ~41)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=45.0,
            rtol=0.10,
            err_msg="Metal and CPU should match for many tiles",
        )

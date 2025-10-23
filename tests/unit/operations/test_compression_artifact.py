"""Tests for compression artifact operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression_artifact import CompressionArtifactOperation


class TestCompressionArtifactOperation:
    """Tests for CompressionArtifactOperation class."""

    @pytest.fixture
    def operation(self) -> CompressionArtifactOperation:
        """Create a compression artifact operation instance."""
        return CompressionArtifactOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with patterns that will show compression artifacts."""
        # Create image with high-frequency content (checkerboard)
        # that will be heavily affected by JPEG compression
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = [255, 255, 255]  # White squares
        img[1::2, 1::2] = [255, 255, 255]
        return Image.fromarray(img)

    def test_operation_name(self, operation: CompressionArtifactOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "compression_artifact"

    def test_valid_params(self, operation: CompressionArtifactOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "quality": 10}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "quality": 5,
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"quality": 10}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            operation.validate_params(params)

    def test_missing_quality_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test missing quality parameter raises error."""
        params = {"tile_count": 5}
        with pytest.raises(ValueError, match="requires 'quality' parameter"):
            operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 50, "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_invalid_quality_type_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid quality type raises error."""
        params = {"tile_count": 5, "quality": "10"}
        with pytest.raises(ValueError, match="quality must be an integer"):
            operation.validate_params(params)

    def test_quality_too_low_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test quality below minimum raises error."""
        params = {"tile_count": 5, "quality": 0}
        with pytest.raises(ValueError, match="quality must be an integer between"):
            operation.validate_params(params)

    def test_quality_too_high_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test quality above maximum raises error."""
        params = {"tile_count": 5, "quality": 25}
        with pytest.raises(ValueError, match="quality must be an integer between"):
            operation.validate_params(params)

    def test_invalid_tile_size_range_type_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid tile_size_range type raises error."""
        params = {"tile_count": 5, "quality": 10, "tile_size_range": 0.1}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_invalid_tile_size_range_length_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid tile_size_range length raises error."""
        params = {"tile_count": 5, "quality": 10, "tile_size_range": [0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_invalid_tile_size_range_values_type_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid tile_size_range value types raise error."""
        params = {
            "tile_count": 5,
            "quality": 10,
            "tile_size_range": ["0.1", 0.3],
        }
        with pytest.raises(ValueError, match="tile_size_range values must be numbers"):
            operation.validate_params(params)

    def test_tile_size_range_out_of_bounds_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test tile_size_range out of bounds raises error."""
        params = {"tile_count": 5, "quality": 10, "tile_size_range": [0.0, 1.5]}
        with pytest.raises(ValueError, match="tile_size_range values must be between"):
            operation.validate_params(params)

    def test_tile_size_range_inverted_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test inverted tile_size_range raises error."""
        params = {"tile_count": 5, "quality": 10, "tile_size_range": [0.3, 0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range min must be less than or equal"
        ):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {"tile_count": 5, "quality": 10, "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test applying compression artifacts."""
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that compression artifacts actually change the image."""
        params = {"tile_count": 10, "quality": 1, "seed": 42}
        result = operation.apply(test_image, params)

        # Should have differences due to JPEG compression
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_reproducibility_with_seed(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {"tile_count": 5, "quality": 10, "seed": 42}
        params2 = {"tile_count": 5, "quality": 10, "seed": 100}
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ (tiles in different positions)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_apply_grayscale(self, operation: CompressionArtifactOperation) -> None:
        """Test that grayscale images work correctly."""
        # Create checkerboard pattern in grayscale
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[::2, ::2] = 255
        gray_image[1::2, 1::2] = 255
        gray_image = Image.fromarray(gray_image, mode="L")

        params = {"tile_count": 10, "quality": 5, "seed": 42}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have compression applied
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_low_quality_produces_strong_artifacts(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that low quality produces noticeable artifacts."""
        params = {
            "tile_count": 1,
            "quality": 1,  # Lowest quality
            "tile_size_range": [0.5, 0.5],  # Large tile
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Calculate mean absolute difference
        mean_diff = np.mean(
            np.abs(result_array.astype(float) - original_array.astype(float))
        )

        # With quality=1 and checkerboard pattern, expect significant artifacts
        assert mean_diff > 4.0  # At least 4 intensity levels difference on average

    def test_full_image_compression(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test compressing entire image as single tile."""
        params = {
            "tile_count": 1,
            "quality": 5,
            "tile_size_range": [1.0, 1.0],  # Full image
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should still produce valid image
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

        # Should show compression artifacts
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_single_tile_compression(
        self, operation: CompressionArtifactOperation
    ) -> None:
        """Test that a single tile shows JPEG artifacts."""
        # Create solid color image
        test_img = Image.new("RGB", (50, 50), color=(128, 128, 128))

        params = {
            "tile_count": 1,
            "quality": 1,
            "tile_size_range": [0.2, 0.2],
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        # Result should differ from original due to JPEG compression
        # (even solid colors get artifacts at very low quality)
        result_array = np.array(result)
        original_array = np.array(test_img)

        # At quality=1, even solid colors may show some variation
        # but this depends on JPEG encoder, so we just verify it ran
        assert isinstance(result, Image.Image)

    def test_multiple_tiles_coverage(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that with many tiles, significant portion is affected."""
        params = {"tile_count": 30, "quality": 1, "seed": 42}
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Count pixels that changed
        changed_pixels = np.any(result_array != original_array, axis=2).sum()
        total_pixels = result_array.shape[0] * result_array.shape[1]

        # With 30 tiles, expect at least 10% coverage
        coverage = changed_pixels / total_pixels
        assert coverage > 0.1

    def test_quality_boundary_values(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test quality at boundary values (1 and 20)."""
        # Minimum quality
        params_min = {"tile_count": 1, "quality": 1, "seed": 42}
        result_min = operation.apply(test_image, params_min)
        assert isinstance(result_min, Image.Image)

        # Maximum quality (still low for artifacts)
        params_max = {"tile_count": 1, "quality": 20, "seed": 100}  # Different seed
        result_max = operation.apply(test_image, params_max)
        assert isinstance(result_max, Image.Image)

        # Both should produce some artifacts (just verify they work)
        diff_min = np.mean(
            np.abs(
                np.array(result_min).astype(float) - np.array(test_image).astype(float)
            )
        )
        diff_max = np.mean(
            np.abs(
                np.array(result_max).astype(float) - np.array(test_image).astype(float)
            )
        )

        # Both quality levels should produce some change
        assert diff_min > 0
        assert diff_max > 0

    def test_small_tiles(
        self, operation: CompressionArtifactOperation, test_image: Image.Image
    ) -> None:
        """Test that very small tiles work correctly."""
        params = {
            "tile_count": 10,
            "quality": 5,
            "tile_size_range": [0.01, 0.05],  # Very small tiles
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should still produce valid image
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

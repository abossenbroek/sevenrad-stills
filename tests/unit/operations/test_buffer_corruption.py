"""Tests for buffer corruption operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation


class TestBufferCorruptionOperation:
    """Tests for BufferCorruptionOperation class."""

    @pytest.fixture
    def operation(self) -> BufferCorruptionOperation:
        """Create a buffer corruption operation instance."""
        return BufferCorruptionOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: BufferCorruptionOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "buffer_corruption"

    def test_valid_params(self, operation: BufferCorruptionOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": 0.5}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "corruption_type": "invert",
            "severity": 0.7,
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            operation.validate_params(params)

    def test_missing_corruption_type_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test missing corruption_type parameter raises error."""
        params = {"tile_count": 5, "severity": 0.5}
        with pytest.raises(ValueError, match="requires 'corruption_type' parameter"):
            operation.validate_params(params)

    def test_missing_severity_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test missing severity parameter raises error."""
        params = {"tile_count": 5, "corruption_type": "xor"}
        with pytest.raises(ValueError, match="requires 'severity' parameter"):
            operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 50, "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_invalid_corruption_type_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test invalid corruption_type raises error."""
        params = {"tile_count": 5, "corruption_type": "invalid", "severity": 0.5}
        with pytest.raises(ValueError, match="corruption_type must be one of"):
            operation.validate_params(params)

    def test_invalid_severity_type_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test invalid severity type raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": "0.5"}
        with pytest.raises(ValueError, match="severity must be a number"):
            operation.validate_params(params)

    def test_severity_too_low_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test severity below minimum raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": -0.1}
        with pytest.raises(ValueError, match="severity must be a number between"):
            operation.validate_params(params)

    def test_severity_too_high_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test severity above maximum raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": 1.5}
        with pytest.raises(ValueError, match="severity must be a number between"):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": "42",
        }
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_xor_corruption(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test applying XOR corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_invert_corruption(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test applying invert corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "invert",
            "severity": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_channel_shuffle_corruption(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test applying channel shuffle corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,  # High severity for guaranteed shuffle
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_xor_produces_change(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that XOR corruption actually changes the image."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.8,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_invert_produces_change(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that invert corruption actually changes the image."""
        params = {
            "tile_count": 10,
            "corruption_type": "invert",
            "severity": 0.8,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_severity_produces_no_change_xor(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test severity=0.0 for XOR produces no change."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should be identical
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_zero_severity_produces_no_change_invert(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test severity=0.0 for invert produces no change."""
        params = {
            "tile_count": 10,
            "corruption_type": "invert",
            "severity": 0.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should be identical
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_reproducibility_with_seed(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        params2 = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 100,
        }
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ (tiles in different positions)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_apply_grayscale(self, operation: BufferCorruptionOperation) -> None:
        """Test that grayscale images work correctly."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have corruption applied
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_all_corruption_types_are_valid(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test that all documented corruption types are valid."""
        valid_types = ["xor", "invert", "channel_shuffle"]
        for ctype in valid_types:
            params = {"tile_count": 1, "corruption_type": ctype, "severity": 0.5}
            operation.validate_params(params)  # Should not raise

    def test_high_severity_produces_strong_effects(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that high severity produces noticeable changes."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 1.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Calculate mean absolute difference
        mean_diff = np.mean(
            np.abs(result_array.astype(float) - original_array.astype(float))
        )

        # With severity=1.0 and multiple tiles, expect significant changes
        assert mean_diff > 5.0  # At least 5 intensity levels difference on average

    def test_invert_with_full_severity(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test full inversion with severity=1.0 on a simple case."""
        # Create image with known value
        test_img = Image.new("RGB", (10, 10), color=(100, 150, 200))

        params = {
            "tile_count": 1,
            "corruption_type": "invert",
            "severity": 1.0,
            "tile_size_range": [1.0, 1.0],  # Full image
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        # Full inversion: 100 → ~155, 150 → ~105, 200 → ~55
        # Check that values changed significantly
        mean_result = np.mean(result_array, axis=(0, 1))
        mean_original = np.array([100, 150, 200])

        # Values should be different after inversion
        assert not np.allclose(mean_result, mean_original, atol=10)

    def test_channel_shuffle_changes_colors(
        self, operation: BufferCorruptionOperation
    ) -> None:
        """Test that channel shuffle produces color changes."""
        # Create image with distinct RGB values
        test_img = Image.new("RGB", (50, 50), color=(100, 150, 200))

        params = {
            "tile_count": 1,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,  # Guarantee shuffle
            "tile_size_range": [0.5, 0.5],  # Cover significant area
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        original_array = np.array(test_img)

        # Check that some pixels have different color orderings
        # (not all if tile doesn't cover entire image)
        changed_pixels = np.any(result_array != original_array, axis=2).sum()

        # Should have some changed pixels from the shuffle
        assert changed_pixels > 0

    def test_tile_size_range_parameter(
        self, operation: BufferCorruptionOperation, test_image: Image.Image
    ) -> None:
        """Test that tile_size_range parameter is respected."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "tile_size_range": [0.01, 0.05],  # Very small tiles
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should still produce valid image
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

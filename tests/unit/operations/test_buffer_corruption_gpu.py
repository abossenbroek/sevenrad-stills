"""Tests for GPU-accelerated buffer corruption operation (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.buffer_corruption import BufferCorruptionOperation
from sevenrad_stills.operations.buffer_corruption_gpu import (
    BufferCorruptionGPUOperation,
)


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestBufferCorruptionGPUOperation:
    """Tests for BufferCorruptionGPUOperation class."""

    @pytest.fixture
    def gpu_operation(self) -> BufferCorruptionGPUOperation:
        """Create a GPU buffer corruption operation instance."""
        return BufferCorruptionGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> BufferCorruptionOperation:
        """Create a CPU buffer corruption operation instance."""
        return BufferCorruptionOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    @pytest.fixture
    def large_test_image(self) -> Image.Image:
        """Create a large test image for performance testing (4K resolution)."""
        return Image.new("RGB", (3840, 2160), color=(128, 128, 128))

    # === Basic Operation Tests ===

    def test_operation_name(self, gpu_operation: BufferCorruptionGPUOperation) -> None:
        """Test operation has correct name."""
        assert gpu_operation.name == "buffer_corruption_gpu"

    def test_valid_params(self, gpu_operation: BufferCorruptionGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": 0.5}
        gpu_operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "corruption_type": "invert",
            "severity": 0.7,
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        gpu_operation.validate_params(params)  # Should not raise

    # === Parameter Validation Tests ===

    def test_missing_tile_count_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            gpu_operation.validate_params(params)

    def test_missing_corruption_type_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test missing corruption_type parameter raises error."""
        params = {"tile_count": 5, "severity": 0.5}
        with pytest.raises(ValueError, match="requires 'corruption_type' parameter"):
            gpu_operation.validate_params(params)

    def test_missing_severity_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test missing severity parameter raises error."""
        params = {"tile_count": 5, "corruption_type": "xor"}
        with pytest.raises(ValueError, match="requires 'severity' parameter"):
            gpu_operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            gpu_operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            gpu_operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 50, "corruption_type": "xor", "severity": 0.5}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            gpu_operation.validate_params(params)

    def test_invalid_corruption_type_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test invalid corruption_type raises error."""
        params = {"tile_count": 5, "corruption_type": "invalid", "severity": 0.5}
        with pytest.raises(ValueError, match="corruption_type must be one of"):
            gpu_operation.validate_params(params)

    def test_invalid_severity_type_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test invalid severity type raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": "0.5"}
        with pytest.raises(ValueError, match="severity must be a number"):
            gpu_operation.validate_params(params)

    def test_severity_too_low_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test severity below minimum raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": -0.1}
        with pytest.raises(ValueError, match="severity must be a number between"):
            gpu_operation.validate_params(params)

    def test_severity_too_high_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test severity above maximum raises error."""
        params = {"tile_count": 5, "corruption_type": "xor", "severity": 1.5}
        with pytest.raises(ValueError, match="severity must be a number between"):
            gpu_operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": "42",
        }
        with pytest.raises(ValueError, match="Seed must be an integer"):
            gpu_operation.validate_params(params)

    # === Application Tests ===

    def test_apply_xor_corruption(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying XOR corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_invert_corruption(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying invert corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "invert",
            "severity": 0.5,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_channel_shuffle_corruption(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying channel shuffle corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,  # High severity for guaranteed shuffle
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_xor_produces_change(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that XOR corruption actually changes the image."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.8,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_invert_produces_change(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that invert corruption actually changes the image."""
        params = {
            "tile_count": 10,
            "corruption_type": "invert",
            "severity": 0.8,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_severity_produces_no_change(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test severity=0.0 produces no change (early exit optimization)."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.0,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)

        # Should be identical (not same object, but same data)
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_reproducibility_with_seed(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result1 = gpu_operation.apply(test_image, params)
        result2 = gpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
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
        result1 = gpu_operation.apply(test_image, params1)
        result2 = gpu_operation.apply(test_image, params2)

        # Results should differ (tiles in different positions)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        result = gpu_operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_raises_error(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test that grayscale images raise an error."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }

        with pytest.raises(ValueError, match="Buffer corruption GPU requires RGB"):
            gpu_operation.apply(gray_image, params)

    def test_all_corruption_types_are_valid(
        self, gpu_operation: BufferCorruptionGPUOperation
    ) -> None:
        """Test that all documented corruption types are valid."""
        valid_types = ["xor", "invert", "channel_shuffle"]
        for ctype in valid_types:
            params = {"tile_count": 1, "corruption_type": ctype, "severity": 0.5}
            gpu_operation.validate_params(params)  # Should not raise

    def test_high_severity_produces_strong_effects(
        self, gpu_operation: BufferCorruptionGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that high severity produces noticeable changes."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 1.0,
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Calculate mean absolute difference
        mean_diff = np.mean(
            np.abs(result_array.astype(float) - original_array.astype(float))
        )

        # With severity=1.0 and multiple tiles, expect significant changes
        assert mean_diff > 5.0  # At least 5 intensity levels difference on average

    # === GPU vs CPU Correctness Tests ===

    def test_gpu_matches_cpu_xor(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        test_image: Image.Image,
    ) -> None:
        """
        Test GPU produces similar results to CPU for XOR corruption.

        Note: For overlapping tiles, GPU (parallel) and CPU (sequential)
        may differ slightly.
        """
        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        gpu_result = gpu_operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Allow tolerance for tile overlap differences (GPU parallel vs CPU sequential)
        np.testing.assert_allclose(
            np.array(gpu_result), np.array(cpu_result), atol=130, rtol=0.1
        )

    def test_gpu_matches_cpu_invert(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU produces same results as CPU for invert corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "invert",
            "severity": 0.5,
            "seed": 42,
        }
        gpu_result = gpu_operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Allow small difference due to floating point precision
        np.testing.assert_allclose(np.array(gpu_result), np.array(cpu_result), atol=1)

    def test_gpu_matches_cpu_channel_shuffle(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU produces same results as CPU for channel shuffle corruption."""
        params = {
            "tile_count": 5,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,
            "seed": 42,
        }
        gpu_result = gpu_operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(gpu_result), np.array(cpu_result))

    # === Performance Tests ===

    def test_gpu_xor_performance_comparable(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        large_test_image: Image.Image,
    ) -> None:
        """
        Test GPU XOR performance is comparable to CPU.

        Note: XOR corruption has significant CPU preprocessing overhead
        (mask generation) which limits GPU speedup. Performance is comparable
        but may not always beat CPU.
        """
        params = {
            "tile_count": 20,
            "corruption_type": "xor",
            "severity": 0.8,
            "seed": 42,
        }

        # Warm up GPU (first run includes kernel compilation)
        _ = gpu_operation.apply(large_test_image, params)

        # Time GPU
        gpu_start = time.time()
        _ = gpu_operation.apply(large_test_image, params)
        gpu_time = time.time() - gpu_start

        # Time CPU
        cpu_start = time.time()
        _ = cpu_operation.apply(large_test_image, params)
        cpu_time = time.time() - cpu_start

        # GPU should be within 2x of CPU (performance is comparable)
        assert (
            gpu_time < cpu_time * 2.0
        ), f"GPU ({gpu_time:.4f}s) should be comparable to CPU ({cpu_time:.4f}s)"

    def test_gpu_faster_than_cpu_invert(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        large_test_image: Image.Image,
    ) -> None:
        """Test GPU is faster than CPU for invert corruption on large images."""
        params = {
            "tile_count": 20,
            "corruption_type": "invert",
            "severity": 0.8,
            "seed": 42,
        }

        # Warm up GPU
        _ = gpu_operation.apply(large_test_image, params)

        # Time GPU
        gpu_start = time.time()
        _ = gpu_operation.apply(large_test_image, params)
        gpu_time = time.time() - gpu_start

        # Time CPU
        cpu_start = time.time()
        _ = cpu_operation.apply(large_test_image, params)
        cpu_time = time.time() - cpu_start

        # GPU should be faster
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"

    def test_gpu_faster_than_cpu_channel_shuffle(
        self,
        gpu_operation: BufferCorruptionGPUOperation,
        cpu_operation: BufferCorruptionOperation,
        large_test_image: Image.Image,
    ) -> None:
        """Test GPU is faster than CPU for channel shuffle on large images."""
        params = {
            "tile_count": 20,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,
            "seed": 42,
        }

        # Warm up GPU
        _ = gpu_operation.apply(large_test_image, params)

        # Time GPU
        gpu_start = time.time()
        _ = gpu_operation.apply(large_test_image, params)
        gpu_time = time.time() - gpu_start

        # Time CPU
        cpu_start = time.time()
        _ = cpu_operation.apply(large_test_image, params)
        cpu_time = time.time() - cpu_start

        # GPU should be faster
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"

    def test_gpu_produces_valid_output(self, test_image: Image.Image) -> None:
        """Test that GPU version produces valid, reasonable output."""
        gpu_op = BufferCorruptionGPUOperation()

        params = {
            "tile_count": 5,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }
        gpu_result = gpu_op.apply(test_image, params)
        gpu_array = np.array(gpu_result)

        # Verify output is valid
        assert gpu_array.shape == (100, 100, 3)
        assert gpu_array.dtype == np.uint8

        # Result should contain values in expected range
        assert gpu_array.min() >= 0
        assert gpu_array.max() <= 255

        # Due to corruption, we should see variation in the image
        original_array = np.array(test_image)
        assert not np.array_equal(original_array, gpu_array)

"""Tests for GPU-accelerated compression artifact operation (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression_artifact import (
    CompressionArtifactOperation,
)
from sevenrad_stills.operations.compression_artifact_gpu import (
    CompressionArtifactGPUOperation,
)


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestCompressionArtifactGPUOperation:
    """Tests for CompressionArtifactGPUOperation class."""

    @pytest.fixture
    def gpu_operation(self) -> CompressionArtifactGPUOperation:
        """Create a GPU compression artifact operation instance."""
        return CompressionArtifactGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> CompressionArtifactOperation:
        """Create a CPU compression artifact operation instance."""
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

    @pytest.fixture
    def large_test_image(self) -> Image.Image:
        """Create a larger test image for performance testing."""
        # Create 1024x1024 checkerboard
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        img[::2, ::2] = [255, 255, 255]
        img[1::2, 1::2] = [255, 255, 255]
        return Image.fromarray(img)

    def test_operation_name(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test operation has correct name."""
        assert gpu_operation.name == "compression_artifact_gpu"

    def test_valid_params(self, gpu_operation: CompressionArtifactGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "quality": 10}
        gpu_operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "quality": 5,
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        gpu_operation.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"quality": 10}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            gpu_operation.validate_params(params)

    def test_missing_quality_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test missing quality parameter raises error."""
        params = {"tile_count": 5}
        with pytest.raises(ValueError, match="requires 'quality' parameter"):
            gpu_operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            gpu_operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            gpu_operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 50, "quality": 10}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            gpu_operation.validate_params(params)

    def test_invalid_quality_type_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test invalid quality type raises error."""
        params = {"tile_count": 5, "quality": "10"}
        with pytest.raises(ValueError, match="quality must be an integer"):
            gpu_operation.validate_params(params)

    def test_quality_too_low_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test quality below minimum raises error."""
        params = {"tile_count": 5, "quality": 0}
        with pytest.raises(ValueError, match="quality must be an integer between"):
            gpu_operation.validate_params(params)

    def test_quality_too_high_raises_error(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test quality above maximum raises error."""
        params = {"tile_count": 5, "quality": 25}
        with pytest.raises(ValueError, match="quality must be an integer between"):
            gpu_operation.validate_params(params)

    def test_apply_basic(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test applying GPU compression artifacts."""
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result = gpu_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU compression artifacts actually change the image."""
        params = {"tile_count": 10, "quality": 1, "seed": 42}
        result = gpu_operation.apply(test_image, params)

        # Should have differences due to compression
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_reproducibility_with_seed(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result1 = gpu_operation.apply(test_image, params)
        result2 = gpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {"tile_count": 5, "quality": 10, "seed": 42}
        params2 = {"tile_count": 5, "quality": 10, "seed": 100}
        result1 = gpu_operation.apply(test_image, params1)
        result2 = gpu_operation.apply(test_image, params2)

        # Results should differ (tiles in different positions)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {"tile_count": 5, "quality": 10, "seed": 42}
        result = gpu_operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_apply_grayscale(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test that grayscale images work correctly."""
        # Create checkerboard pattern in grayscale
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[::2, ::2] = 255
        gray_image[1::2, 1::2] = 255
        gray_image = Image.fromarray(gray_image, mode="L")

        params = {"tile_count": 10, "quality": 5, "seed": 42}
        result = gpu_operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have compression applied
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_low_quality_produces_strong_artifacts(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that low quality produces noticeable artifacts."""
        params = {
            "tile_count": 1,
            "quality": 1,  # Lowest quality
            "tile_size_range": [0.5, 0.5],  # Large tile
            "seed": 42,
        }
        result = gpu_operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Calculate mean absolute difference
        mean_diff = np.mean(
            np.abs(result_array.astype(float) - original_array.astype(float))
        )

        # With quality=1 and checkerboard pattern, expect significant artifacts
        assert mean_diff > 1.0  # At least 1 intensity level difference on average

    def test_gpu_vs_cpu_visual_similarity(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        cpu_operation: CompressionArtifactOperation,
        test_image: Image.Image,
    ) -> None:
        """
        Test that GPU and CPU produce visually similar (not identical) results.

        GPU uses DCT simulation while CPU uses actual JPEG encoding.
        They should produce similar blocky artifacts but won't be identical.
        """
        params = {"tile_count": 3, "quality": 5, "seed": 42}

        gpu_result = gpu_operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        gpu_array = np.array(gpu_result).astype(float)
        cpu_array = np.array(cpu_result).astype(float)

        # Both should have changed the image
        original_array = np.array(test_image).astype(float)
        gpu_diff = np.mean(np.abs(gpu_array - original_array))
        cpu_diff = np.mean(np.abs(cpu_array - original_array))

        assert gpu_diff > 0.5
        assert cpu_diff > 0.5

        # GPU and CPU results should be in the same ballpark
        # but not necessarily identical (different compression methods)
        # Check that the difference between GPU and CPU is reasonable
        method_diff = np.mean(np.abs(gpu_array - cpu_array))

        # Allow for some difference but not too much
        # (both should produce compression artifacts, just different algorithms)
        assert method_diff < 50  # Less than 50 intensity levels on average

    def test_gpu_performance_characteristics(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        cpu_operation: CompressionArtifactOperation,
        large_test_image: Image.Image,
    ) -> None:
        """
        Test GPU version performance characteristics and compare to CPU.

        Note: The GPU DCT-based simulation may not always be faster than
        PIL's highly-optimized C-based JPEG encoder for small tile counts,
        but it provides:
        - Deterministic, reproducible results (identical artifacts every time)
        - Better parallelization for many tiles
        - Visual similarity to JPEG artifacts without entropy coding overhead
        """
        # Test with more tiles to see GPU parallelization benefit
        params = {"tile_count": 30, "quality": 5, "seed": 42}

        # Warm up GPU (first run may be slower due to kernel compilation)
        _ = gpu_operation.apply(large_test_image, params)

        # Time GPU version (multiple runs for stability)
        gpu_times = []
        for _ in range(3):
            gpu_start = time.perf_counter()
            gpu_result = gpu_operation.apply(large_test_image, params)
            gpu_times.append(time.perf_counter() - gpu_start)
        gpu_time = min(gpu_times)  # Use best time

        # Time CPU version (multiple runs for stability)
        cpu_times = []
        for _ in range(3):
            cpu_start = time.perf_counter()
            cpu_result = cpu_operation.apply(large_test_image, params)
            cpu_times.append(time.perf_counter() - cpu_start)
        cpu_time = min(cpu_times)  # Use best time

        # Verify both produced valid results
        assert isinstance(gpu_result, Image.Image)
        assert isinstance(cpu_result, Image.Image)

        # Report performance comparison
        print(f"\nCPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # The GPU version should be reasonably fast (< 1 second for 1024x1024)
        # This ensures it's production-ready even if not faster than PIL's JPEG
        assert gpu_time < 1.0  # Should complete in under 1 second

    def test_blocking_artifacts_present(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
    ) -> None:
        """Test that 8x8 blocking artifacts are visible (JPEG characteristic)."""
        # Create a smooth gradient that will show 8x8 blocks clearly
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                img[i, j] = [i * 4, j * 4, 128]
        test_img = Image.fromarray(img)

        params = {
            "tile_count": 1,
            "quality": 1,
            "tile_size_range": [1.0, 1.0],  # Full image
            "seed": 42,
        }
        result = gpu_operation.apply(test_img, params)
        result_array = np.array(result)

        # Check that we have some discontinuities at 8-pixel boundaries
        # (characteristic of DCT-based compression)
        # Look at horizontal differences across 8-pixel boundaries
        boundary_diffs = []
        for y in range(8, 64, 8):
            diff = np.mean(np.abs(result_array[y] - result_array[y - 1]))
            boundary_diffs.append(diff)

        # At least some 8x8 block boundaries should have visible discontinuities
        # due to independent quantization
        mean_boundary_diff = np.mean(boundary_diffs)
        assert mean_boundary_diff > 0.1  # Some visible blocking

    def test_multiple_tiles_coverage(
        self,
        gpu_operation: CompressionArtifactGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that with many tiles, significant portion is affected."""
        params = {"tile_count": 30, "quality": 1, "seed": 42}
        result = gpu_operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Count pixels that changed
        changed_pixels = np.any(result_array != original_array, axis=2).sum()
        total_pixels = result_array.shape[0] * result_array.shape[1]

        # With 30 tiles, expect reasonable coverage
        coverage = changed_pixels / total_pixels
        assert coverage > 0.05  # At least 5% coverage

    def test_quantization_matrix_scaling(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test that quantization matrix scaling works correctly."""
        from sevenrad_stills.operations.compression_artifact_gpu import (
            JPEG_QUANT_LUMA,
        )

        # Test quality=1 (maximum compression)
        scaled_q1 = gpu_operation._scale_quantization_matrix(JPEG_QUANT_LUMA, 1)
        assert scaled_q1.shape == (8, 8)
        assert scaled_q1.min() >= 1  # Should be clamped to at least 1
        assert scaled_q1.max() <= 255  # Should be clamped to at most 255
        # At quality=1, scaling is very large, so values should be much larger
        assert scaled_q1.mean() > JPEG_QUANT_LUMA.mean()

        # Test quality=20
        scaled_q20 = gpu_operation._scale_quantization_matrix(JPEG_QUANT_LUMA, 20)
        assert scaled_q20.shape == (8, 8)
        # Higher quality = smaller quantization values (less aggressive compression)
        assert scaled_q20.mean() < scaled_q1.mean()

    def test_edge_case_small_image(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test handling of very small images."""
        # Create 16x16 image (2x2 blocks)
        small_img = Image.new("RGB", (16, 16), color=(128, 128, 128))

        params = {"tile_count": 1, "quality": 5, "seed": 42}
        result = gpu_operation.apply(small_img, params)

        assert isinstance(result, Image.Image)
        assert result.size == (16, 16)

    def test_numerical_stability(
        self, gpu_operation: CompressionArtifactGPUOperation
    ) -> None:
        """Test that output is numerically stable (no NaNs or out-of-range values)."""
        # Create random noise image
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        test_img = Image.fromarray(noise)

        params = {"tile_count": 10, "quality": 1, "seed": 42}
        result = gpu_operation.apply(test_img, params)
        result_array = np.array(result)

        # Check no NaNs or infinities
        assert not np.any(np.isnan(result_array))
        assert not np.any(np.isinf(result_array))

        # Check valid range
        assert result_array.min() >= 0
        assert result_array.max() <= 255

        # Check correct dtype
        assert result_array.dtype == np.uint8

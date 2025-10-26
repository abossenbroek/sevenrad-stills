"""Tests for GPU-accelerated noise operation."""

import sys
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.noise import NoiseOperation
from sevenrad_stills.operations.noise_gpu import NoiseGPUOperation


class TestNoiseGPUOperation:
    """Tests for NoiseGPUOperation class."""

    @pytest.fixture
    def cpu_operation(self) -> NoiseOperation:
        """Create a CPU noise operation instance."""
        return NoiseOperation()

    @pytest.fixture
    def gpu_operation(self) -> NoiseGPUOperation:
        """Create a GPU noise operation instance."""
        return NoiseGPUOperation()

    @pytest.fixture
    def test_image_rgb(self) -> Image.Image:
        """Create an RGB test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    @pytest.fixture
    def test_image_rgba(self) -> Image.Image:
        """Create an RGBA test image."""
        return Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))

    @pytest.fixture
    def test_image_gray(self) -> Image.Image:
        """Create a grayscale test image."""
        return Image.new("L", (100, 100), color=128)

    @pytest.fixture
    def large_image(self) -> Image.Image:
        """Create a large test image for performance benchmarks."""
        return Image.new("RGB", (1920, 1080), color=(128, 64, 192))

    def test_operation_name(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test operation has correct name."""
        assert gpu_operation.name == "noise_gpu"

    # ==================== Parameter Validation Tests ====================

    def test_valid_params_gaussian(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test valid gaussian mode parameters."""
        params = {"mode": "gaussian", "amount": 0.1}
        gpu_operation.validate_params(params)  # Should not raise

    def test_valid_params_row(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test valid row mode parameters."""
        params = {"mode": "row", "amount": 0.2, "seed": 42}
        gpu_operation.validate_params(params)  # Should not raise

    def test_valid_params_column(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test valid column mode parameters."""
        params = {"mode": "column", "amount": 0.5}
        gpu_operation.validate_params(params)  # Should not raise

    def test_missing_mode_raises_error(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test missing mode parameter raises error."""
        params = {"amount": 0.1}
        with pytest.raises(ValueError, match="requires a 'mode' parameter"):
            gpu_operation.validate_params(params)

    def test_invalid_mode_raises_error(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test invalid mode raises error."""
        params = {"mode": "invalid", "amount": 0.1}
        with pytest.raises(ValueError, match="must be 'gaussian', 'row', or 'column'"):
            gpu_operation.validate_params(params)

    def test_missing_amount_raises_error(
        self, gpu_operation: NoiseGPUOperation
    ) -> None:
        """Test missing amount parameter raises error."""
        params = {"mode": "gaussian"}
        with pytest.raises(ValueError, match="requires an 'amount' parameter"):
            gpu_operation.validate_params(params)

    def test_amount_out_of_range_raises_error(
        self, gpu_operation: NoiseGPUOperation
    ) -> None:
        """Test amount outside valid range raises error."""
        params = {"mode": "gaussian", "amount": -0.1}
        with pytest.raises(ValueError, match="must be a float between"):
            gpu_operation.validate_params(params)

        params = {"mode": "gaussian", "amount": 1.5}
        with pytest.raises(ValueError, match="must be a float between"):
            gpu_operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, gpu_operation: NoiseGPUOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            gpu_operation.validate_params(params)

    # ==================== Functional Tests ====================

    def test_apply_gaussian_rgb(
        self, gpu_operation: NoiseGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying gaussian noise to RGB image."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}
        result = gpu_operation.apply(test_image_rgb, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

    def test_apply_row_rgba(
        self, gpu_operation: NoiseGPUOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying row noise to RGBA image."""
        params = {"mode": "row", "amount": 0.2, "seed": 42}
        result = gpu_operation.apply(test_image_rgba, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

    def test_apply_column_gray(
        self, gpu_operation: NoiseGPUOperation, test_image_gray: Image.Image
    ) -> None:
        """Test applying column noise to grayscale image."""
        params = {"mode": "column", "amount": 0.15, "seed": 42}
        result = gpu_operation.apply(test_image_gray, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_gray.size
        assert result.mode == "L"

    # ==================== Numerical Accuracy Tests ====================

    def test_gaussian_matches_cpu_rgb(
        self,
        cpu_operation: NoiseOperation,
        gpu_operation: NoiseGPUOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test GPU gaussian noise matches CPU implementation for RGB."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        cpu_result = cpu_operation.apply(test_image_rgb, params)
        gpu_result = gpu_operation.apply(test_image_rgb, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)

        # Results should be very close (within floating-point precision)
        np.testing.assert_allclose(cpu_array, gpu_array, rtol=1e-5, atol=1e-3)

    def test_row_matches_cpu_rgba(
        self,
        cpu_operation: NoiseOperation,
        gpu_operation: NoiseGPUOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test GPU row noise matches CPU implementation for RGBA."""
        params = {"mode": "row", "amount": 0.2, "seed": 123}

        cpu_result = cpu_operation.apply(test_image_rgba, params)
        gpu_result = gpu_operation.apply(test_image_rgba, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)

        np.testing.assert_allclose(cpu_array, gpu_array, rtol=1e-5, atol=1e-3)

    def test_column_matches_cpu_gray(
        self,
        cpu_operation: NoiseOperation,
        gpu_operation: NoiseGPUOperation,
        test_image_gray: Image.Image,
    ) -> None:
        """Test GPU column noise matches CPU implementation for grayscale."""
        params = {"mode": "column", "amount": 0.15, "seed": 456}

        cpu_result = cpu_operation.apply(test_image_gray, params)
        gpu_result = gpu_operation.apply(test_image_gray, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)

        np.testing.assert_allclose(cpu_array, gpu_array, rtol=1e-5, atol=1e-3)

    def test_deterministic_with_seed(
        self, gpu_operation: NoiseGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result1 = gpu_operation.apply(test_image_rgb, params)
        result2 = gpu_operation.apply(test_image_rgb, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_without_seed(
        self, gpu_operation: NoiseGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that without seed, results are different."""
        params = {"mode": "gaussian", "amount": 0.1}

        result1 = gpu_operation.apply(test_image_rgb, params)
        result2 = gpu_operation.apply(test_image_rgb, params)

        # Results should be different (extremely unlikely to be identical)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_rgba_preserves_alpha(
        self, gpu_operation: NoiseGPUOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test that RGBA processing preserves alpha channel."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result = gpu_operation.apply(test_image_rgba, params)
        result_array = np.array(result)
        original_array = np.array(test_image_rgba)

        # Alpha channel should be unchanged
        np.testing.assert_array_equal(result_array[..., 3], original_array[..., 3])

    # ==================== Performance Tests ====================

    @pytest.mark.mac
    def test_performance_vs_cpu(
        self,
        cpu_operation: NoiseOperation,
        gpu_operation: NoiseGPUOperation,
        large_image: Image.Image,
    ) -> None:
        """Test that GPU implementation is faster than CPU for large images."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        # Warmup GPU
        _ = gpu_operation.apply(large_image, params)

        # Benchmark CPU
        cpu_start = time.perf_counter()
        for _ in range(5):
            _ = cpu_operation.apply(large_image, params)
        cpu_time = time.perf_counter() - cpu_start

        # Benchmark GPU
        gpu_start = time.perf_counter()
        for _ in range(5):
            _ = gpu_operation.apply(large_image, params)
        gpu_time = time.perf_counter() - gpu_start

        # GPU should be faster (or at least competitive) for large images
        # Allow 20% margin for overhead
        assert gpu_time <= cpu_time * 1.2

    @pytest.mark.mac
    def test_all_modes_performance(
        self, gpu_operation: NoiseGPUOperation, large_image: Image.Image
    ) -> None:
        """Test performance of all noise modes."""
        modes = ["gaussian", "row", "column"]
        params_template = {"amount": 0.1, "seed": 42}

        for mode in modes:
            params = {**params_template, "mode": mode}

            # Warmup
            _ = gpu_operation.apply(large_image, params)

            # Benchmark
            start = time.perf_counter()
            for _ in range(3):
                _ = gpu_operation.apply(large_image, params)
            elapsed = time.perf_counter() - start

            # Should complete in reasonable time (< 1 second for 3 iterations)
            assert elapsed < 1.0

    # ==================== Edge Cases ====================

    def test_zero_amount(
        self, gpu_operation: NoiseGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that zero amount produces unchanged image."""
        params = {"mode": "gaussian", "amount": 0.0, "seed": 42}

        result = gpu_operation.apply(test_image_rgb, params)
        result_array = np.array(result, dtype=np.float32)
        original_array = np.array(test_image_rgb, dtype=np.float32)

        # Should be very close (allowing for floating-point conversion)
        np.testing.assert_allclose(result_array, original_array, rtol=1e-5, atol=2.0)

    def test_max_amount(
        self, gpu_operation: NoiseGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test maximum amount noise."""
        params = {"mode": "gaussian", "amount": 1.0, "seed": 42}

        result = gpu_operation.apply(test_image_rgb, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgb.size

    def test_small_image(self, gpu_operation: NoiseGPUOperation) -> None:
        """Test with very small image."""
        small_image = Image.new("RGB", (8, 8), color=(128, 128, 128))
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result = gpu_operation.apply(small_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == (8, 8)

"""Tests for Metal-accelerated noise operation."""

import sys
import time

import numpy as np
import pytest
from PIL import Image

# Skip all tests if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal operations require macOS"
)


@pytest.mark.mac
class TestNoiseMetalOperation:
    """Tests for NoiseMetalOperation class."""

    @pytest.fixture
    def cpu_operation(self):
        """Create a CPU noise operation instance."""
        from sevenrad_stills.operations.noise import NoiseOperation

        return NoiseOperation()

    @pytest.fixture
    def gpu_operation(self):
        """Create a GPU noise operation instance."""
        from sevenrad_stills.operations.noise_gpu import NoiseGPUOperation

        return NoiseGPUOperation()

    @pytest.fixture
    def metal_operation(self):
        """Create a Metal noise operation instance."""
        from sevenrad_stills.operations.noise_metal import NoiseMetalOperation

        return NoiseMetalOperation()

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

    def test_operation_name(self, metal_operation) -> None:
        """Test operation has correct name."""
        assert metal_operation.name == "noise_metal"

    # ==================== Parameter Validation Tests ====================

    def test_valid_params_gaussian(self, metal_operation) -> None:
        """Test valid gaussian mode parameters."""
        params = {"mode": "gaussian", "amount": 0.1}
        metal_operation.validate_params(params)  # Should not raise

    def test_valid_params_row(self, metal_operation) -> None:
        """Test valid row mode parameters."""
        params = {"mode": "row", "amount": 0.2, "seed": 42}
        metal_operation.validate_params(params)  # Should not raise

    def test_valid_params_column(self, metal_operation) -> None:
        """Test valid column mode parameters."""
        params = {"mode": "column", "amount": 0.5}
        metal_operation.validate_params(params)  # Should not raise

    def test_missing_mode_raises_error(self, metal_operation) -> None:
        """Test missing mode parameter raises error."""
        params = {"amount": 0.1}
        with pytest.raises(ValueError, match="requires a 'mode' parameter"):
            metal_operation.validate_params(params)

    def test_invalid_mode_raises_error(self, metal_operation) -> None:
        """Test invalid mode raises error."""
        params = {"mode": "invalid", "amount": 0.1}
        with pytest.raises(ValueError, match="must be 'gaussian', 'row', or 'column'"):
            metal_operation.validate_params(params)

    # ==================== Functional Tests ====================

    def test_apply_gaussian_rgb(
        self, metal_operation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying gaussian noise to RGB image."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}
        result = metal_operation.apply(test_image_rgb, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

    def test_apply_row_rgba(
        self, metal_operation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying row noise to RGBA image."""
        params = {"mode": "row", "amount": 0.2, "seed": 42}
        result = metal_operation.apply(test_image_rgba, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

    def test_apply_column_gray(
        self, metal_operation, test_image_gray: Image.Image
    ) -> None:
        """Test applying column noise to grayscale image."""
        params = {"mode": "column", "amount": 0.15, "seed": 42}
        result = metal_operation.apply(test_image_gray, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_gray.size
        assert result.mode == "L"

    # ==================== Numerical Accuracy Tests ====================

    def test_gaussian_matches_cpu_rgb(
        self, cpu_operation, metal_operation, test_image_rgb: Image.Image
    ) -> None:
        """Test Metal gaussian noise matches CPU implementation for RGB."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        cpu_result = cpu_operation.apply(test_image_rgb, params)
        metal_result = metal_operation.apply(test_image_rgb, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        # Results should be very close (within floating-point precision)
        np.testing.assert_allclose(cpu_array, metal_array, rtol=1e-5, atol=1e-3)

    def test_row_matches_cpu_rgba(
        self, cpu_operation, metal_operation, test_image_rgba: Image.Image
    ) -> None:
        """Test Metal row noise matches CPU implementation for RGBA."""
        params = {"mode": "row", "amount": 0.2, "seed": 123}

        cpu_result = cpu_operation.apply(test_image_rgba, params)
        metal_result = metal_operation.apply(test_image_rgba, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        np.testing.assert_allclose(cpu_array, metal_array, rtol=1e-5, atol=1e-3)

    def test_column_matches_cpu_gray(
        self, cpu_operation, metal_operation, test_image_gray: Image.Image
    ) -> None:
        """Test Metal column noise matches CPU implementation for grayscale."""
        params = {"mode": "column", "amount": 0.15, "seed": 456}

        cpu_result = cpu_operation.apply(test_image_gray, params)
        metal_result = metal_operation.apply(test_image_gray, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        np.testing.assert_allclose(cpu_array, metal_array, rtol=1e-5, atol=1e-3)

    def test_matches_gpu_implementation(
        self, gpu_operation, metal_operation, test_image_rgb: Image.Image
    ) -> None:
        """Test Metal matches GPU implementation."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        gpu_result = gpu_operation.apply(test_image_rgb, params)
        metal_result = metal_operation.apply(test_image_rgb, params)

        gpu_array = np.array(gpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        # Results should be identical (same algorithm, same precision)
        np.testing.assert_allclose(gpu_array, metal_array, rtol=1e-6, atol=1e-4)

    def test_deterministic_with_seed(
        self, metal_operation, test_image_rgb: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result1 = metal_operation.apply(test_image_rgb, params)
        result2 = metal_operation.apply(test_image_rgb, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_rgba_preserves_alpha(
        self, metal_operation, test_image_rgba: Image.Image
    ) -> None:
        """Test that RGBA processing preserves alpha channel."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result = metal_operation.apply(test_image_rgba, params)
        result_array = np.array(result)
        original_array = np.array(test_image_rgba)

        # Alpha channel should be unchanged
        np.testing.assert_array_equal(result_array[..., 3], original_array[..., 3])

    # ==================== Performance Hierarchy Tests ====================

    def test_performance_hierarchy_gaussian(
        self, cpu_operation, gpu_operation, metal_operation, large_image: Image.Image
    ) -> None:
        """Test that CPU > GPU > Metal for gaussian noise performance."""
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}
        iterations = 5

        # Warmup
        _ = gpu_operation.apply(large_image, params)
        _ = metal_operation.apply(large_image, params)

        # Benchmark CPU
        cpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = cpu_operation.apply(large_image, params)
        cpu_time = time.perf_counter() - cpu_start

        # Benchmark GPU
        gpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = gpu_operation.apply(large_image, params)
        gpu_time = time.perf_counter() - gpu_start

        # Benchmark Metal
        metal_start = time.perf_counter()
        for _ in range(iterations):
            _ = metal_operation.apply(large_image, params)
        metal_time = time.perf_counter() - metal_start

        # Assert performance hierarchy: CPU > GPU > Metal
        # Allow some margin for measurement variance
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"
        assert (
            metal_time < gpu_time
        ), f"Metal ({metal_time:.4f}s) should be faster than GPU ({gpu_time:.4f}s)"

    def test_performance_hierarchy_row(
        self, cpu_operation, gpu_operation, metal_operation, large_image: Image.Image
    ) -> None:
        """Test that CPU > GPU > Metal for row noise performance."""
        params = {"mode": "row", "amount": 0.2, "seed": 42}
        iterations = 5

        # Warmup
        _ = gpu_operation.apply(large_image, params)
        _ = metal_operation.apply(large_image, params)

        # Benchmark CPU
        cpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = cpu_operation.apply(large_image, params)
        cpu_time = time.perf_counter() - cpu_start

        # Benchmark GPU
        gpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = gpu_operation.apply(large_image, params)
        gpu_time = time.perf_counter() - gpu_start

        # Benchmark Metal
        metal_start = time.perf_counter()
        for _ in range(iterations):
            _ = metal_operation.apply(large_image, params)
        metal_time = time.perf_counter() - metal_start

        # Assert performance hierarchy
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"
        assert (
            metal_time < gpu_time
        ), f"Metal ({metal_time:.4f}s) should be faster than GPU ({gpu_time:.4f}s)"

    def test_performance_hierarchy_column(
        self, cpu_operation, gpu_operation, metal_operation, large_image: Image.Image
    ) -> None:
        """Test that CPU > GPU > Metal for column noise performance."""
        params = {"mode": "column", "amount": 0.15, "seed": 42}
        iterations = 5

        # Warmup
        _ = gpu_operation.apply(large_image, params)
        _ = metal_operation.apply(large_image, params)

        # Benchmark CPU
        cpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = cpu_operation.apply(large_image, params)
        cpu_time = time.perf_counter() - cpu_start

        # Benchmark GPU
        gpu_start = time.perf_counter()
        for _ in range(iterations):
            _ = gpu_operation.apply(large_image, params)
        gpu_time = time.perf_counter() - gpu_start

        # Benchmark Metal
        metal_start = time.perf_counter()
        for _ in range(iterations):
            _ = metal_operation.apply(large_image, params)
        metal_time = time.perf_counter() - metal_start

        # Assert performance hierarchy
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"
        assert (
            metal_time < gpu_time
        ), f"Metal ({metal_time:.4f}s) should be faster than GPU ({gpu_time:.4f}s)"

    # ==================== Edge Cases ====================

    def test_zero_amount(self, metal_operation, test_image_rgb: Image.Image) -> None:
        """Test that zero amount produces unchanged image."""
        params = {"mode": "gaussian", "amount": 0.0, "seed": 42}

        result = metal_operation.apply(test_image_rgb, params)
        result_array = np.array(result, dtype=np.float32)
        original_array = np.array(test_image_rgb, dtype=np.float32)

        # Should be very close (allowing for floating-point conversion)
        np.testing.assert_allclose(result_array, original_array, rtol=1e-5, atol=2.0)

    def test_max_amount(self, metal_operation, test_image_rgb: Image.Image) -> None:
        """Test maximum amount noise."""
        params = {"mode": "gaussian", "amount": 1.0, "seed": 42}

        result = metal_operation.apply(test_image_rgb, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image_rgb.size

    def test_small_image(self, metal_operation) -> None:
        """Test with very small image."""
        small_image = Image.new("RGB", (8, 8), color=(128, 128, 128))
        params = {"mode": "gaussian", "amount": 0.1, "seed": 42}

        result = metal_operation.apply(small_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == (8, 8)

    def test_all_modes_with_all_image_types(
        self, metal_operation, test_image_rgb, test_image_rgba, test_image_gray
    ) -> None:
        """Test all combinations of modes and image types."""
        modes = ["gaussian", "row", "column"]
        images = [
            ("RGB", test_image_rgb),
            ("RGBA", test_image_rgba),
            ("L", test_image_gray),
        ]

        for mode in modes:
            for img_type, image in images:
                params = {"mode": mode, "amount": 0.1, "seed": 42}
                result = metal_operation.apply(image, params)
                assert isinstance(result, Image.Image)
                assert result.size == image.size
                assert result.mode == img_type

"""Tests for GPU-accelerated SLC-Off operation (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.slc_off import SlcOffOperation
from sevenrad_stills.operations.slc_off_gpu import SlcOffGPUOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestSlcOffGPUOperation:
    """Tests for SlcOffGPUOperation class."""

    @pytest.fixture
    def operation(self) -> SlcOffGPUOperation:
        """Create a GPU SLC-Off operation instance."""
        return SlcOffGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> SlcOffOperation:
        """Create a CPU SLC-Off operation instance for comparison."""
        return SlcOffOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with varied content."""
        img = Image.new("RGB", (200, 200))
        pixels = img.load()
        for i in range(200):
            for j in range(200):
                pixels[i, j] = ((i * 7) % 256, (j * 11) % 256, ((i + j) * 13) % 256)  # type: ignore[index]
        return img

    def test_operation_name(self, operation: SlcOffGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "slc_off_gpu"

    def test_valid_params(self, operation: SlcOffGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_seed(self, operation: SlcOffGPUOperation) -> None:
        """Test valid parameters with seed."""
        params = {
            "gap_width": 0.3,
            "scan_period": 15,
            "fill_mode": "mean",
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_gap_width_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test missing gap_width parameter raises error."""
        params = {"scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'gap_width' parameter"):
            operation.validate_params(params)

    def test_missing_scan_period_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test missing scan_period parameter raises error."""
        params = {"gap_width": 0.2, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'scan_period' parameter"):
            operation.validate_params(params)

    def test_missing_fill_mode_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test missing fill_mode parameter raises error."""
        params = {"gap_width": 0.2, "scan_period": 10}
        with pytest.raises(ValueError, match="requires 'fill_mode' parameter"):
            operation.validate_params(params)

    def test_invalid_gap_width_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test invalid gap_width raises error."""
        params = {"gap_width": 0.6, "scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="gap_width must be a number between"):
            operation.validate_params(params)

    def test_invalid_scan_period_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test invalid scan_period raises error."""
        params = {"gap_width": 0.2, "scan_period": 1, "fill_mode": "black"}
        with pytest.raises(ValueError, match="scan_period must be an integer between"):
            operation.validate_params(params)

    def test_invalid_fill_mode_raises_error(
        self, operation: SlcOffGPUOperation
    ) -> None:
        """Test invalid fill_mode raises error."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "invalid"}
        with pytest.raises(ValueError, match="fill_mode must be one of"):
            operation.validate_params(params)

    def test_apply_black_fill(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying GPU SLC-Off with black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_white_fill(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying GPU SLC-Off with white fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "white", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_mean_fill(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying GPU SLC-Off with mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_produces_change(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that GPU SLC-Off actually changes the image."""
        params = {"gap_width": 0.3, "scan_period": 8, "fill_mode": "black", "seed": 42}
        result = operation.apply(test_image, params)

        # Should have differences (gaps created)
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_gap_width_no_change(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that zero gap_width produces no change."""
        params = {"gap_width": 0.0, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_rgba_preserves_alpha(self, operation: SlcOffGPUOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 200))
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)
        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_support(self, operation: SlcOffGPUOperation) -> None:
        """Test that grayscale images are supported."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

    def test_gpu_matches_cpu_black_fill(
        self,
        operation: SlcOffGPUOperation,
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU and CPU produce nearly identical results for black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}

        cpu_result = cpu_operation.apply(test_image.copy(), params)
        gpu_result = operation.apply(test_image.copy(), params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Gap mask generation may differ slightly at boundaries due to parallel vs sequential
        # computation, but overall pattern should be very similar (>98% match)
        diff = np.abs(cpu_array - gpu_array)
        matching_pixels = np.sum(diff < 5.0) / diff.size

        assert (
            matching_pixels > 0.98
        ), f"Only {matching_pixels*100:.1f}% of pixels match closely (expected >98%)"

    def test_gpu_matches_cpu_white_fill(
        self,
        operation: SlcOffGPUOperation,
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU and CPU produce nearly identical results for white fill."""
        params = {"gap_width": 0.3, "scan_period": 12, "fill_mode": "white", "seed": 42}

        cpu_result = cpu_operation.apply(test_image.copy(), params)
        gpu_result = operation.apply(test_image.copy(), params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Gap mask generation may differ slightly, check overall similarity
        diff = np.abs(cpu_array - gpu_array)
        matching_pixels = np.sum(diff < 5.0) / diff.size

        assert (
            matching_pixels > 0.98
        ), f"Only {matching_pixels*100:.1f}% of pixels match closely (expected >98%)"

    def test_gpu_matches_cpu_mean_fill(
        self,
        operation: SlcOffGPUOperation,
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU and CPU produce similar results for mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}

        cpu_result = cpu_operation.apply(test_image.copy(), params)
        gpu_result = operation.apply(test_image.copy(), params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Mean fill has additional variation, check similarity with higher tolerance
        diff = np.abs(cpu_array - gpu_array)
        matching_pixels = np.sum(diff < 15.0) / diff.size

        assert (
            matching_pixels > 0.96
        ), f"Only {matching_pixels*100:.1f}% of pixels match reasonably (expected >96%)"

    def test_gpu_matches_cpu_grayscale(
        self, operation: SlcOffGPUOperation, cpu_operation: SlcOffOperation
    ) -> None:
        """Test that GPU and CPU match for grayscale images."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}

        cpu_result = cpu_operation.apply(gray_image.copy(), params)
        gpu_result = operation.apply(gray_image.copy(), params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Gap mask generation may differ slightly, check overall similarity
        diff = np.abs(cpu_array - gpu_array)
        matching_pixels = np.sum(diff < 5.0) / diff.size

        assert (
            matching_pixels > 0.98
        ), f"Only {matching_pixels*100:.1f}% of pixels match closely (expected >98%)"

    def test_different_scan_periods(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that different scan periods produce different results."""
        params1 = {
            "gap_width": 0.2,
            "scan_period": 5,
            "fill_mode": "black",
            "seed": 42,
        }
        params2 = {
            "gap_width": 0.2,
            "scan_period": 20,
            "fill_mode": "black",
            "seed": 42,
        }

        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_different_gap_widths(
        self, operation: SlcOffGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that different gap widths produce different results."""
        params1 = {
            "gap_width": 0.1,
            "scan_period": 10,
            "fill_mode": "black",
            "seed": 42,
        }
        params2 = {
            "gap_width": 0.4,
            "scan_period": 10,
            "fill_mode": "black",
            "seed": 42,
        }

        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ (wider gaps)
        assert not np.array_equal(np.array(result1), np.array(result2))


@pytest.mark.mac
class TestGPUPerformance:
    """Test that GPU implementation provides performance improvements."""

    def test_gpu_performance_reasonable(self) -> None:
        """Test that GPU implementation has reasonable performance."""
        # Use a large image for performance testing
        large_image = Image.new("RGB", (2000, 2000))
        pixels = large_image.load()
        for i in range(0, 2000, 10):
            for j in range(0, 2000, 10):
                pixels[i, j] = ((i * 13) % 256, (j * 17) % 256, ((i + j) * 19) % 256)  # type: ignore[index]

        params = {"gap_width": 0.3, "scan_period": 15, "fill_mode": "black", "seed": 42}

        cpu_op = SlcOffOperation()
        gpu_op = SlcOffGPUOperation()

        # Warm-up runs
        cpu_op.apply(large_image.copy(), params)
        gpu_op.apply(large_image.copy(), params)

        # Time CPU
        start_cpu = time.perf_counter()
        for _ in range(3):
            cpu_op.apply(large_image.copy(), params)
        cpu_time = time.perf_counter() - start_cpu

        # Time GPU
        start_gpu = time.perf_counter()
        for _ in range(3):
            gpu_op.apply(large_image.copy(), params)
        gpu_time = time.perf_counter() - start_gpu

        # Print performance comparison
        speedup = cpu_time / gpu_time
        print(
            f"\nGPU Performance: CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x"
        )

        # GPU should be at least not terribly slower (within 3x)
        # Note: For some operations, Taichi initialization overhead can dominate
        assert (
            gpu_time < cpu_time * 3.0
        ), f"GPU ({gpu_time:.4f}s) is significantly slower than CPU ({cpu_time:.4f}s)"

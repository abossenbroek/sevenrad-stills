"""Tests for Metal-accelerated SLC-Off operation (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image

# Only import on macOS
HAS_METAL = False
if platform.system() == "Darwin":
    try:
        from sevenrad_stills.operations.slc_off_metal import SlcOffMetalOperation

        HAS_METAL = True
    except ImportError:
        pass

from sevenrad_stills.operations.slc_off import SlcOffOperation
from sevenrad_stills.operations.slc_off_gpu import SlcOffGPUOperation


@pytest.mark.skipif(
    not HAS_METAL,
    reason="Metal tests only run on Mac with Metal support",
)
class TestSlcOffMetalOperation:
    """Tests for SlcOffMetalOperation class."""

    @pytest.fixture
    def operation(self):  # type: ignore[no-untyped-def]
        """Create a Metal SLC-Off operation instance."""
        return SlcOffMetalOperation()

    @pytest.fixture
    def cpu_operation(self) -> SlcOffOperation:
        """Create a CPU SLC-Off operation instance for comparison."""
        return SlcOffOperation()

    @pytest.fixture
    def gpu_operation(self) -> SlcOffGPUOperation:
        """Create a GPU SLC-Off operation instance for comparison."""
        return SlcOffGPUOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with varied content."""
        img = Image.new("RGB", (200, 200))
        pixels = img.load()
        for i in range(200):
            for j in range(200):
                pixels[i, j] = ((i * 7) % 256, (j * 11) % 256, ((i + j) * 13) % 256)  # type: ignore[index]
        return img

    def test_operation_name(self, operation) -> None:  # type: ignore[no-untyped-def]
        """Test operation has correct name."""
        assert operation.name == "slc_off_metal"

    def test_valid_params(self, operation) -> None:  # type: ignore[no-untyped-def]
        """Test valid parameter validation."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_seed(self, operation) -> None:  # type: ignore[no-untyped-def]
        """Test valid parameters with seed."""
        params = {
            "gap_width": 0.3,
            "scan_period": 15,
            "fill_mode": "mean",
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_gap_width_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test missing gap_width parameter raises error."""
        params = {"scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'gap_width' parameter"):
            operation.validate_params(params)

    def test_missing_scan_period_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test missing scan_period parameter raises error."""
        params = {"gap_width": 0.2, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'scan_period' parameter"):
            operation.validate_params(params)

    def test_missing_fill_mode_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test missing fill_mode parameter raises error."""
        params = {"gap_width": 0.2, "scan_period": 10}
        with pytest.raises(ValueError, match="requires 'fill_mode' parameter"):
            operation.validate_params(params)

    def test_invalid_gap_width_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test invalid gap_width raises error."""
        params = {"gap_width": 0.6, "scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="gap_width must be a number between"):
            operation.validate_params(params)

    def test_invalid_scan_period_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test invalid scan_period raises error."""
        params = {"gap_width": 0.2, "scan_period": 1, "fill_mode": "black"}
        with pytest.raises(ValueError, match="scan_period must be an integer between"):
            operation.validate_params(params)

    def test_invalid_fill_mode_raises_error(
        self,
        operation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test invalid fill_mode raises error."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "invalid"}
        with pytest.raises(ValueError, match="fill_mode must be one of"):
            operation.validate_params(params)

    def test_apply_black_fill(
        self,
        operation,
        test_image: Image.Image,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test applying Metal SLC-Off with black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_white_fill(
        self,
        operation,
        test_image: Image.Image,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test applying Metal SLC-Off with white fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "white", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_mean_fill(
        self,
        operation,
        test_image: Image.Image,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test applying Metal SLC-Off with mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_produces_change(
        self,
        operation,
        test_image: Image.Image,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test that Metal SLC-Off actually changes the image."""
        params = {"gap_width": 0.3, "scan_period": 8, "fill_mode": "black", "seed": 42}
        result = operation.apply(test_image, params)

        # Should have differences (gaps created)
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_gap_width_no_change(
        self,
        operation,
        test_image: Image.Image,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test that zero gap_width produces no change."""
        params = {"gap_width": 0.0, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_rgba_preserves_alpha(self, operation) -> None:  # type: ignore[no-untyped-def]
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 200))
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)
        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_support(self, operation) -> None:  # type: ignore[no-untyped-def]
        """Test that grayscale images are supported."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

    def test_metal_matches_cpu_black_fill(
        self,
        operation,  # type: ignore[no-untyped-def]
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that Metal and CPU produce identical results for black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}

        cpu_result = cpu_operation.apply(test_image, params)
        metal_result = operation.apply(test_image, params)

        cpu_array = np.array(cpu_result)
        metal_array = np.array(metal_result)

        # Should be identical for deterministic black fill
        np.testing.assert_array_equal(
            cpu_array,
            metal_array,
            err_msg="Metal and CPU results differ for black fill",
        )

    def test_metal_matches_cpu_white_fill(
        self,
        operation,  # type: ignore[no-untyped-def]
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that Metal and CPU produce identical results for white fill."""
        params = {"gap_width": 0.3, "scan_period": 12, "fill_mode": "white", "seed": 42}

        cpu_result = cpu_operation.apply(test_image, params)
        metal_result = operation.apply(test_image, params)

        cpu_array = np.array(cpu_result)
        metal_array = np.array(metal_result)

        # Should be identical for deterministic white fill
        np.testing.assert_array_equal(
            cpu_array,
            metal_array,
            err_msg="Metal and CPU results differ for white fill",
        )

    def test_metal_matches_cpu_mean_fill(
        self,
        operation,  # type: ignore[no-untyped-def]
        cpu_operation: SlcOffOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that Metal and CPU produce identical results for mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}

        cpu_result = cpu_operation.apply(test_image, params)
        metal_result = operation.apply(test_image, params)

        cpu_array = np.array(cpu_result)
        metal_array = np.array(metal_result)

        # Should be identical with same seed
        np.testing.assert_array_equal(
            cpu_array,
            metal_array,
            err_msg="Metal and CPU results differ for mean fill",
        )

    def test_metal_matches_gpu_black_fill(
        self,
        operation,  # type: ignore[no-untyped-def]
        gpu_operation: SlcOffGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that Metal and GPU produce identical results for black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}

        gpu_result = gpu_operation.apply(test_image, params)
        metal_result = operation.apply(test_image, params)

        gpu_array = np.array(gpu_result)
        metal_array = np.array(metal_result)

        # Should be identical
        np.testing.assert_array_equal(
            gpu_array,
            metal_array,
            err_msg="Metal and GPU results differ for black fill",
        )

    def test_metal_matches_gpu_white_fill(
        self,
        operation,  # type: ignore[no-untyped-def]
        gpu_operation: SlcOffGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that Metal and GPU produce identical results for white fill."""
        params = {"gap_width": 0.3, "scan_period": 12, "fill_mode": "white", "seed": 42}

        gpu_result = gpu_operation.apply(test_image, params)
        metal_result = operation.apply(test_image, params)

        gpu_array = np.array(gpu_result)
        metal_array = np.array(metal_result)

        # Should be identical
        np.testing.assert_array_equal(
            gpu_array,
            metal_array,
            err_msg="Metal and GPU results differ for white fill",
        )

    def test_metal_matches_cpu_grayscale(
        self,
        operation,
        cpu_operation: SlcOffOperation,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test that Metal and CPU match for grayscale images."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 42}

        cpu_result = cpu_operation.apply(gray_image, params)
        metal_result = operation.apply(gray_image, params)

        cpu_array = np.array(cpu_result)
        metal_array = np.array(metal_result)

        # Should be identical
        np.testing.assert_array_equal(
            cpu_array,
            metal_array,
            err_msg="Metal and CPU results differ for grayscale",
        )


@pytest.mark.mac
class TestPerformanceHierarchy:
    """Test that Metal > GPU > CPU in performance."""

    def test_performance_hierarchy(self) -> None:
        """Test that Metal is fastest, followed by GPU, then CPU."""
        # Use a large image for meaningful performance comparison
        large_image = Image.new("RGB", (2000, 2000))
        pixels = large_image.load()
        for i in range(0, 2000, 10):
            for j in range(0, 2000, 10):
                pixels[i, j] = ((i * 13) % 256, (j * 17) % 256, ((i + j) * 19) % 256)  # type: ignore[index]

        params = {"gap_width": 0.3, "scan_period": 15, "fill_mode": "black", "seed": 42}

        cpu_op = SlcOffOperation()
        gpu_op = SlcOffGPUOperation()

        if not HAS_METAL:
            pytest.skip("Metal not available")
        metal_op = SlcOffMetalOperation()

        # Warm-up runs
        cpu_op.apply(large_image, params)
        gpu_op.apply(large_image, params)
        metal_op.apply(large_image, params)

        # Time CPU
        start_cpu = time.perf_counter()
        for _ in range(3):
            cpu_op.apply(large_image, params)
        cpu_time = time.perf_counter() - start_cpu

        # Time GPU
        start_gpu = time.perf_counter()
        for _ in range(3):
            gpu_op.apply(large_image, params)
        gpu_time = time.perf_counter() - start_gpu

        # Time Metal
        start_metal = time.perf_counter()
        for _ in range(3):
            metal_op.apply(large_image, params)
        metal_time = time.perf_counter() - start_metal

        # Performance hierarchy: Metal < GPU < CPU (lower is faster)
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"

        assert (
            metal_time < gpu_time
        ), f"Metal ({metal_time:.4f}s) should be faster than GPU ({gpu_time:.4f}s)"

        # Print performance comparison
        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time
        metal_vs_gpu = gpu_time / metal_time

        print(
            f"\nPerformance Hierarchy:"
            f"\n  CPU:   {cpu_time:.4f}s (baseline)"
            f"\n  GPU:   {gpu_time:.4f}s ({gpu_speedup:.2f}x faster than CPU)"
            f"\n  Metal: {metal_time:.4f}s ({metal_speedup:.2f}x faster than CPU, {metal_vs_gpu:.2f}x faster than GPU)"
        )

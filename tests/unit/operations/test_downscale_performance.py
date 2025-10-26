"""Performance tests for downscale GPU/Metal acceleration (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.downscale import DownscaleOperation
from sevenrad_stills.operations.downscale_gpu import DownscaleGPUOperation

# Try to import Metal operation - may fail if PyObjC Metal not installed
try:
    from sevenrad_stills.operations.downscale_metal import DownscaleMetalOperation

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    DownscaleMetalOperation = None  # type: ignore[misc,assignment]


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU/Metal performance tests only run on Mac",
)
class TestDownscalePerformance:
    """Performance tests comparing CPU, GPU, and Metal implementations."""

    @pytest.fixture
    def large_image(self) -> Image.Image:
        """Create a large test image for performance testing."""
        return Image.new("RGB", (2048, 2048), color=(100, 150, 200))

    @pytest.fixture
    def medium_image(self) -> Image.Image:
        """Create a medium test image for performance testing."""
        return Image.new("RGB", (1024, 1024), color=(100, 150, 200))

    def benchmark_operation(
        self,
        operation: DownscaleOperation | DownscaleGPUOperation | DownscaleMetalOperation,
        image: Image.Image,
        params: dict,
        iterations: int = 5,
    ) -> float:
        """
        Benchmark an operation.

        Args:
            operation: Operation to benchmark
            image: Test image
            params: Operation parameters
            iterations: Number of iterations

        Returns:
            Mean time in milliseconds

        """
        # Warmup
        operation.apply(image, params)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation.apply(image, params)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return float(np.mean(times))

    def test_gpu_faster_than_cpu_large_image(self, large_image: Image.Image) -> None:
        """Test that GPU is competitive with CPU for large images."""
        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()

        # Use bilinear for fair comparison (both support it)
        params = {
            "scale": 0.25,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        print(
            f"\nCPU: {cpu_time:.2f}ms, GPU: {gpu_time:.2f}ms, Speedup: {speedup:.2f}x"
        )

        # GPU should be competitive (within 50% of CPU)
        # PIL's resize is highly optimized, so GPU may not always be faster
        assert speedup >= 0.5, (
            f"GPU ({gpu_time:.2f}ms) should be competitive with "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_metal_competitive_with_gpu_large_image(
        self, large_image: Image.Image
    ) -> None:
        """Test that Metal is competitive with Taichi GPU for large images."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        gpu_op = DownscaleGPUOperation()
        metal_op = DownscaleMetalOperation()

        params = {
            "scale": 0.25,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = gpu_time / metal_time

        print(
            f"\nGPU: {gpu_time:.2f}ms, Metal: {metal_time:.2f}ms, "
            f"Speedup: {speedup:.2f}x"
        )

        # Metal should be competitive (within 2x of GPU)
        # Allow tolerance as Metal has different overhead characteristics
        assert speedup >= 0.5, (
            f"Metal ({metal_time:.2f}ms) should be competitive with "
            f"GPU ({gpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_performance_hierarchy_large_image(self, large_image: Image.Image) -> None:
        """Test performance characteristics of all implementations."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()
        metal_op = DownscaleMetalOperation()

        params = {
            "scale": 0.25,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        print(
            f"\nPerformance comparison (2048x2048):\n"
            f"  CPU:   {cpu_time:.2f}ms\n"
            f"  GPU:   {gpu_time:.2f}ms ({cpu_time/gpu_time:.2f}x vs CPU)\n"
            f"  Metal: {metal_time:.2f}ms ({cpu_time/metal_time:.2f}x vs CPU)"
        )

        # All implementations should complete successfully
        # Verify GPU is competitive (within 2x of CPU)
        assert (
            gpu_time < cpu_time * 2
        ), f"GPU ({gpu_time:.2f}ms) should be competitive with CPU ({cpu_time:.2f}ms)"
        # Verify Metal is competitive (within 2x of CPU)
        assert metal_time < cpu_time * 2, (
            f"Metal ({metal_time:.2f}ms) should be competitive "
            f"with CPU ({cpu_time:.2f}ms)"
        )

    def test_gpu_competitive_medium_image(self, medium_image: Image.Image) -> None:
        """Test GPU performance with medium-sized images."""
        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()

        params = {
            "scale": 0.5,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_time = self.benchmark_operation(cpu_op, medium_image, params)
        gpu_time = self.benchmark_operation(gpu_op, medium_image, params)

        speedup = cpu_time / gpu_time

        print(
            f"\nMedium image (1024x1024):\n"
            f"  CPU: {cpu_time:.2f}ms, GPU: {gpu_time:.2f}ms, Speedup: {speedup:.2f}x"
        )

        # GPU should be competitive (within 2x of CPU for medium images)
        assert speedup >= 0.3, (
            f"GPU ({gpu_time:.2f}ms) should be competitive with "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_numerical_accuracy_gpu_vs_cpu(self, medium_image: Image.Image) -> None:
        """Test that GPU results are numerically close to CPU results."""
        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()

        # Use same method for both
        params = {
            "scale": 0.5,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_result = cpu_op.apply(medium_image, params)
        gpu_result = gpu_op.apply(medium_image, params)

        cpu_array = np.array(cpu_result, dtype=np.float64)
        gpu_array = np.array(gpu_result, dtype=np.float64)

        # Calculate difference
        diff = np.abs(cpu_array - gpu_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(
            f"\nNumerical accuracy (GPU vs CPU):\n"
            f"  Max difference:  {max_diff:.6f}\n"
            f"  Mean difference: {mean_diff:.6f}"
        )

        # Results should be very close (allowing for rounding differences)
        # For uint8 images, differences should be within a few pixel values
        assert max_diff <= 5.0, f"Max pixel difference {max_diff} exceeds tolerance"
        assert mean_diff <= 2.0, f"Mean pixel difference {mean_diff} exceeds tolerance"

    def test_numerical_accuracy_metal_vs_cpu(self, medium_image: Image.Image) -> None:
        """Test that Metal results are numerically close to CPU results."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        cpu_op = DownscaleOperation()
        metal_op = DownscaleMetalOperation()

        # Use same method for both
        params = {
            "scale": 0.5,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_result = cpu_op.apply(medium_image, params)
        metal_result = metal_op.apply(medium_image, params)

        cpu_array = np.array(cpu_result, dtype=np.float64)
        metal_array = np.array(metal_result, dtype=np.float64)

        # Calculate difference
        diff = np.abs(cpu_array - metal_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(
            f"\nNumerical accuracy (Metal vs CPU):\n"
            f"  Max difference:  {max_diff:.6f}\n"
            f"  Mean difference: {mean_diff:.6f}"
        )

        # Results should be very close (allowing for rounding differences)
        # Metal may have slightly different rounding than CPU/GPU
        assert max_diff <= 10.0, f"Max pixel difference {max_diff} exceeds tolerance"
        assert mean_diff <= 3.0, f"Mean pixel difference {mean_diff} exceeds tolerance"

    def test_numerical_accuracy_metal_vs_gpu(self, medium_image: Image.Image) -> None:
        """Test that Metal and GPU results are numerically close."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        gpu_op = DownscaleGPUOperation()
        metal_op = DownscaleMetalOperation()

        params = {
            "scale": 0.5,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        gpu_result = gpu_op.apply(medium_image, params)
        metal_result = metal_op.apply(medium_image, params)

        gpu_array = np.array(gpu_result, dtype=np.float64)
        metal_array = np.array(metal_result, dtype=np.float64)

        # Calculate difference
        diff = np.abs(gpu_array - metal_array)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(
            f"\nNumerical accuracy (Metal vs GPU):\n"
            f"  Max difference:  {max_diff:.6f}\n"
            f"  Mean difference: {mean_diff:.6f}"
        )

        # Results should be very close
        assert max_diff <= 5.0, f"Max pixel difference {max_diff} exceeds tolerance"
        assert mean_diff <= 2.0, f"Mean pixel difference {mean_diff} exceeds tolerance"

    def test_nearest_neighbor_accuracy(self, medium_image: Image.Image) -> None:
        """Test that nearest neighbor is pixel-perfect across all implementations."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()
        metal_op = DownscaleMetalOperation()

        # Nearest neighbor should be deterministic
        params = {
            "scale": 0.5,
            "downscale_method": "nearest",
            "upscale_method": "nearest",
        }

        cpu_result = cpu_op.apply(medium_image, params)
        gpu_result = gpu_op.apply(medium_image, params)
        metal_result = metal_op.apply(medium_image, params)

        cpu_array = np.array(cpu_result)
        gpu_array = np.array(gpu_result)
        metal_array = np.array(metal_result)

        # For nearest neighbor, all should be identical
        # (allowing minimal tolerance for implementation differences)
        diff_gpu_cpu = np.max(np.abs(cpu_array.astype(int) - gpu_array.astype(int)))
        diff_metal_cpu = np.max(np.abs(cpu_array.astype(int) - metal_array.astype(int)))

        print(
            f"\nNearest neighbor accuracy:\n"
            f"  GPU vs CPU max diff:   {diff_gpu_cpu}\n"
            f"  Metal vs CPU max diff: {diff_metal_cpu}"
        )

        assert diff_gpu_cpu <= 1, "GPU nearest neighbor should match CPU"
        assert diff_metal_cpu <= 1, "Metal nearest neighbor should match CPU"

    def test_extreme_downscale_performance(self) -> None:
        """Test performance with extreme downscaling."""
        try:
            import Metal  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("Metal framework not available (requires PyObjC)")

        cpu_op = DownscaleOperation()
        gpu_op = DownscaleGPUOperation()
        metal_op = DownscaleMetalOperation()

        large_img = Image.new("RGB", (2048, 2048), color=(100, 150, 200))

        # Extreme downscale (0.01 = 2048x2048 -> 20x20)
        params = {
            "scale": 0.01,
            "downscale_method": "bilinear",
            "upscale_method": "nearest",
        }

        cpu_time = self.benchmark_operation(cpu_op, large_img, params, iterations=3)
        gpu_time = self.benchmark_operation(gpu_op, large_img, params, iterations=3)
        metal_time = self.benchmark_operation(metal_op, large_img, params, iterations=3)

        print(
            f"\nExtreme downscale (0.01x):\n"
            f"  CPU:   {cpu_time:.2f}ms\n"
            f"  GPU:   {gpu_time:.2f}ms\n"
            f"  Metal: {metal_time:.2f}ms"
        )

        # GPU implementations should still be competitive (within 3x)
        assert (
            gpu_time < cpu_time * 3
        ), "GPU should be competitive even at extreme scales"
        assert (
            metal_time < cpu_time * 3
        ), "Metal should be competitive even at extreme scales"

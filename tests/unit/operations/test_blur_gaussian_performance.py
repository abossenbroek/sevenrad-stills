"""Performance comparison tests for Gaussian blur (CPU vs GPU)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation


@pytest.mark.mac
@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU performance tests only run on Mac (Metal backend)",
)
class TestGaussianBlurPerformance:
    """Performance comparison tests between CPU and GPU Gaussian blur."""

    @pytest.fixture
    def cpu_operation(self) -> GaussianBlurOperation:
        """Create CPU Gaussian blur operation."""
        return GaussianBlurOperation()

    @pytest.fixture
    def gpu_operation(self) -> GaussianBlurGPUOperation:
        """Create GPU Gaussian blur operation."""
        return GaussianBlurGPUOperation()

    def _create_test_image(self, size: int) -> Image.Image:
        """Create a test image with random pattern."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(data, mode="RGB")

    def _benchmark_operation(
        self,
        operation: GaussianBlurOperation | GaussianBlurGPUOperation,
        image: Image.Image,
        params: dict,
        iterations: int = 10,
        warmup: int = 2,
    ) -> tuple[float, float]:
        """
        Benchmark an operation.

        Args:
            operation: The blur operation to benchmark.
            image: Input image.
            params: Operation parameters.
            iterations: Number of iterations to run.
            warmup: Number of warmup iterations.

        Returns:
            Tuple of (mean_time_ms, std_time_ms).

        """
        # Warmup runs
        for _ in range(warmup):
            operation.apply(image, params)

        # Benchmark runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation.apply(image, params)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

        return float(np.mean(times)), float(np.std(times))

    def test_performance_small_image_small_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test performance on small image with small sigma."""
        image = self._create_test_image(512)
        params = {"sigma": 2.0}

        cpu_mean, cpu_std = self._benchmark_operation(cpu_operation, image, params)
        gpu_mean, gpu_std = self._benchmark_operation(gpu_operation, image, params)

        speedup = cpu_mean / gpu_mean

        print(f"\n{'='*60}")
        print(f"Small Image (512x512), Small Sigma (2.0)")
        print(f"{'='*60}")
        print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
        print(f"GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # For small images/sigma, GPU may not be faster due to overhead
        # Just verify both produce reasonable times
        assert cpu_mean > 0
        assert gpu_mean > 0

    def test_performance_medium_image_medium_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test performance on medium image with medium sigma."""
        image = self._create_test_image(1024)
        params = {"sigma": 5.0}

        cpu_mean, cpu_std = self._benchmark_operation(cpu_operation, image, params)
        gpu_mean, gpu_std = self._benchmark_operation(gpu_operation, image, params)

        speedup = cpu_mean / gpu_mean

        print(f"\n{'='*60}")
        print(f"Medium Image (1024x1024), Medium Sigma (5.0)")
        print(f"{'='*60}")
        print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
        print(f"GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # At medium scale, GPU may be slower due to overhead
        # Just verify both complete successfully
        assert cpu_mean > 0
        assert gpu_mean > 0

    def test_performance_large_image_medium_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test performance on large image with medium sigma."""
        image = self._create_test_image(2048)
        params = {"sigma": 5.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=5
        )
        gpu_mean, gpu_std = self._benchmark_operation(
            gpu_operation, image, params, iterations=5
        )

        speedup = cpu_mean / gpu_mean

        print(f"\n{'='*60}")
        print(f"Large Image (2048x2048), Medium Sigma (5.0)")
        print(f"{'='*60}")
        print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
        print(f"GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # CRITICAL: GPU MUST be faster than CPU for large images
        # This is the main requirement from the user
        assert (
            speedup > 1.0
        ), f"GPU must be faster than CPU! Got speedup: {speedup:.2f}x"

    def test_performance_large_image_large_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test performance on large image with large sigma."""
        image = self._create_test_image(2048)
        params = {"sigma": 10.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=5
        )
        gpu_mean, gpu_std = self._benchmark_operation(
            gpu_operation, image, params, iterations=5
        )

        speedup = cpu_mean / gpu_mean

        print(f"\n{'='*60}")
        print(f"Large Image (2048x2048), Large Sigma (10.0)")
        print(f"{'='*60}")
        print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
        print(f"GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # CRITICAL: GPU MUST be faster than CPU for large sigma
        assert (
            speedup > 1.0
        ), f"GPU must be faster than CPU! Got speedup: {speedup:.2f}x"

    def test_performance_very_large_image(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test performance on very large image."""
        image = self._create_test_image(4096)
        params = {"sigma": 5.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=3, warmup=1
        )
        gpu_mean, gpu_std = self._benchmark_operation(
            gpu_operation, image, params, iterations=3, warmup=1
        )

        speedup = cpu_mean / gpu_mean

        print(f"\n{'='*60}")
        print(f"Very Large Image (4096x4096), Medium Sigma (5.0)")
        print(f"{'='*60}")
        print(f"CPU: {cpu_mean:.2f} ± {cpu_std:.2f} ms")
        print(f"GPU: {gpu_mean:.2f} ± {gpu_std:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # CRITICAL: GPU MUST show significant speedup on very large images
        assert (
            speedup > 1.0
        ), f"GPU must be faster than CPU! Got speedup: {speedup:.2f}x"

    def test_results_match_cpu(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
    ) -> None:
        """Test that GPU results closely match CPU results."""
        image = self._create_test_image(512)
        params = {"sigma": 3.0}

        cpu_result = cpu_operation.apply(image, params)
        gpu_result = gpu_operation.apply(image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)

        # Results should be very close (allowing for minor numerical differences)
        # Mean absolute error should be small
        mae = np.mean(np.abs(cpu_array - gpu_array))
        print(f"\nMean Absolute Error (CPU vs GPU): {mae:.4f}")

        # Should be within reasonable tolerance
        # (some difference expected due to different implementations)
        assert mae < 2.0, f"Results differ too much: MAE = {mae:.4f}"

        # Most pixels should be very close
        max_error = np.max(np.abs(cpu_array - gpu_array))
        print(f"Max Absolute Error: {max_error:.4f}")
        assert max_error <= 10.0, f"Max error too large: {max_error:.4f}"

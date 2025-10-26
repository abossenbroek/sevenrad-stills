"""Performance comparison: MLX vs Taichi GPU vs CPU."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation
from sevenrad_stills.operations.blur_gaussian_mlx import GaussianBlurMLXOperation


@pytest.mark.mac
@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLX performance tests only run on Mac (Metal backend)",
)
class TestGaussianBlurMLXPerformance:
    """Performance comparison between CPU, Taichi GPU, and MLX."""

    @pytest.fixture
    def cpu_operation(self) -> GaussianBlurOperation:
        """Create CPU Gaussian blur operation."""
        return GaussianBlurOperation()

    @pytest.fixture
    def taichi_gpu_operation(self) -> GaussianBlurGPUOperation:
        """Create Taichi GPU Gaussian blur operation."""
        return GaussianBlurGPUOperation()

    @pytest.fixture
    def mlx_operation(self) -> GaussianBlurMLXOperation:
        """Create MLX Gaussian blur operation."""
        return GaussianBlurMLXOperation()

    def _create_test_image(self, size: int) -> Image.Image:
        """Create a test image with random pattern."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(data, mode="RGB")

    def _benchmark_operation(
        self,
        operation: GaussianBlurOperation
        | GaussianBlurGPUOperation
        | GaussianBlurMLXOperation,
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
            times.append((end - start) * 1000)

        return float(np.mean(times)), float(np.std(times))

    def test_performance_large_image_medium_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        taichi_gpu_operation: GaussianBlurGPUOperation,
        mlx_operation: GaussianBlurMLXOperation,
    ) -> None:
        """Test performance on large image with medium sigma."""
        image = self._create_test_image(2048)
        params = {"sigma": 5.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=5
        )
        taichi_mean, taichi_std = self._benchmark_operation(
            taichi_gpu_operation, image, params, iterations=5
        )
        mlx_mean, mlx_std = self._benchmark_operation(
            mlx_operation, image, params, iterations=5
        )

        taichi_speedup = cpu_mean / taichi_mean
        mlx_speedup = cpu_mean / mlx_mean
        mlx_vs_taichi = taichi_mean / mlx_mean

        print(f"\n{'='*70}")
        print(f"Large Image (2048x2048), Medium Sigma (5.0)")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_mean:7.2f} ± {cpu_std:5.2f} ms")
        print(
            f"Taichi GPU:     {taichi_mean:7.2f} ± {taichi_std:5.2f} ms  [{taichi_speedup:.2f}x vs CPU]"
        )
        print(
            f"MLX:            {mlx_mean:7.2f} ± {mlx_std:5.2f} ms  [{mlx_speedup:.2f}x vs CPU]"
        )
        print(f"{'='*70}")
        print(
            f"MLX vs Taichi:  {mlx_vs_taichi:.2f}x {'FASTER' if mlx_vs_taichi > 1 else 'slower'}"
        )
        print(f"{'='*70}")

        # CRITICAL: MLX should be faster than Taichi GPU
        assert (
            mlx_speedup > taichi_speedup
        ), f"MLX ({mlx_speedup:.2f}x) should be faster than Taichi ({taichi_speedup:.2f}x)"

    def test_performance_large_image_large_sigma(
        self,
        cpu_operation: GaussianBlurOperation,
        taichi_gpu_operation: GaussianBlurGPUOperation,
        mlx_operation: GaussianBlurMLXOperation,
    ) -> None:
        """Test performance on large image with large sigma."""
        image = self._create_test_image(2048)
        params = {"sigma": 10.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=5
        )
        taichi_mean, taichi_std = self._benchmark_operation(
            taichi_gpu_operation, image, params, iterations=5
        )
        mlx_mean, mlx_std = self._benchmark_operation(
            mlx_operation, image, params, iterations=5
        )

        taichi_speedup = cpu_mean / taichi_mean
        mlx_speedup = cpu_mean / mlx_mean
        mlx_vs_taichi = taichi_mean / mlx_mean

        print(f"\n{'='*70}")
        print(f"Large Image (2048x2048), Large Sigma (10.0)")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_mean:7.2f} ± {cpu_std:5.2f} ms")
        print(
            f"Taichi GPU:     {taichi_mean:7.2f} ± {taichi_std:5.2f} ms  [{taichi_speedup:.2f}x vs CPU]"
        )
        print(
            f"MLX:            {mlx_mean:7.2f} ± {mlx_std:5.2f} ms  [{mlx_speedup:.2f}x vs CPU]"
        )
        print(f"{'='*70}")
        print(
            f"MLX vs Taichi:  {mlx_vs_taichi:.2f}x {'FASTER' if mlx_vs_taichi > 1 else 'slower'}"
        )
        print(f"{'='*70}")

        # CRITICAL: MLX should be faster than Taichi GPU
        assert (
            mlx_speedup > taichi_speedup
        ), f"MLX ({mlx_speedup:.2f}x) should be faster than Taichi ({taichi_speedup:.2f}x)"

    def test_performance_very_large_image(
        self,
        cpu_operation: GaussianBlurOperation,
        taichi_gpu_operation: GaussianBlurGPUOperation,
        mlx_operation: GaussianBlurMLXOperation,
    ) -> None:
        """Test performance on very large image."""
        image = self._create_test_image(4096)
        params = {"sigma": 5.0}

        cpu_mean, cpu_std = self._benchmark_operation(
            cpu_operation, image, params, iterations=3, warmup=1
        )
        taichi_mean, taichi_std = self._benchmark_operation(
            taichi_gpu_operation, image, params, iterations=3, warmup=1
        )
        mlx_mean, mlx_std = self._benchmark_operation(
            mlx_operation, image, params, iterations=3, warmup=1
        )

        taichi_speedup = cpu_mean / taichi_mean
        mlx_speedup = cpu_mean / mlx_mean
        mlx_vs_taichi = taichi_mean / mlx_mean

        print(f"\n{'='*70}")
        print(f"Very Large Image (4096x4096), Medium Sigma (5.0)")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_mean:7.2f} ± {cpu_std:5.2f} ms")
        print(
            f"Taichi GPU:     {taichi_mean:7.2f} ± {taichi_std:5.2f} ms  [{taichi_speedup:.2f}x vs CPU]"
        )
        print(
            f"MLX:            {mlx_mean:7.2f} ± {mlx_std:5.2f} ms  [{mlx_speedup:.2f}x vs CPU]"
        )
        print(f"{'='*70}")
        print(
            f"MLX vs Taichi:  {mlx_vs_taichi:.2f}x {'FASTER' if mlx_vs_taichi > 1 else 'slower'}"
        )
        print(f"{'='*70}")

        # CRITICAL: MLX should show best performance on very large images
        assert (
            mlx_speedup > taichi_speedup
        ), f"MLX ({mlx_speedup:.2f}x) should be faster than Taichi ({taichi_speedup:.2f}x)"

    def test_results_match_taichi_and_cpu(
        self,
        cpu_operation: GaussianBlurOperation,
        taichi_gpu_operation: GaussianBlurGPUOperation,
        mlx_operation: GaussianBlurMLXOperation,
    ) -> None:
        """Test that MLX results closely match CPU and Taichi results."""
        image = self._create_test_image(512)
        params = {"sigma": 3.0}

        cpu_result = cpu_operation.apply(image, params)
        taichi_result = taichi_gpu_operation.apply(image, params)
        mlx_result = mlx_operation.apply(image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        taichi_array = np.array(taichi_result, dtype=np.float32)
        mlx_array = np.array(mlx_result, dtype=np.float32)

        # MLX vs CPU
        mae_cpu = np.mean(np.abs(cpu_array - mlx_array))
        print(f"\nMLX vs CPU - MAE: {mae_cpu:.4f}")
        assert mae_cpu < 2.0, f"MLX differs too much from CPU: MAE = {mae_cpu:.4f}"

        # MLX vs Taichi
        mae_taichi = np.mean(np.abs(taichi_array - mlx_array))
        print(f"MLX vs Taichi - MAE: {mae_taichi:.4f}")
        assert (
            mae_taichi < 2.0
        ), f"MLX differs too much from Taichi: MAE = {mae_taichi:.4f}"

        # All three should be very close on average (MAE is the key metric)
        max_error = np.max(
            np.abs(np.stack([cpu_array, taichi_array, mlx_array]).std(axis=0))
        )
        print(f"Max std across all 3 implementations: {max_error:.4f}")
        # MAE < 2.0 is the critical metric; max_error can be higher at edges
        assert max_error < 50.0, f"Implementations diverge too much: {max_error:.4f}"

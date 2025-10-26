"""Performance tests for band swap GPU acceleration (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.band_swap import BandSwapOperation
from sevenrad_stills.operations.band_swap_gpu import BandSwapGPUOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU performance tests only run on Mac (Metal backend)",
)
class TestBandSwapPerformance:
    """Performance tests comparing CPU and GPU implementations."""

    @pytest.fixture
    def large_image(self) -> Image.Image:
        """Create a large test image for performance testing."""
        return Image.new("RGB", (2048, 2048), color=(100, 150, 200))

    def benchmark_operation(
        self,
        operation: BandSwapOperation | BandSwapGPUOperation,
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
        """Test that GPU is faster than CPU for large images with many tiles."""
        cpu_op = BandSwapOperation()
        gpu_op = BandSwapGPUOperation()

        params = {"tile_count": 30, "permutation": "GRB", "seed": 42}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        # GPU should be faster (speedup > 1.0)
        assert speedup > 1.0, (
            f"GPU ({gpu_time:.2f}ms) should be faster than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_gpu_faster_medium_tiles(self) -> None:
        """Test GPU performance with medium-sized images and moderate tile count."""
        cpu_op = BandSwapOperation()
        gpu_op = BandSwapGPUOperation()

        test_image = Image.new("RGB", (1024, 1024), color=(100, 150, 200))
        params = {"tile_count": 20, "permutation": "BGR", "seed": 42}

        cpu_time = self.benchmark_operation(cpu_op, test_image, params)
        gpu_time = self.benchmark_operation(gpu_op, test_image, params)

        speedup = cpu_time / gpu_time

        # GPU should be competitive (within 15% of CPU for medium images)
        assert speedup >= 0.85, (
            f"GPU ({gpu_time:.2f}ms) should not be significantly slower than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_gpu_faster_many_tiles(self, large_image: Image.Image) -> None:
        """Test that GPU excels with many tiles (batch processing advantage)."""
        cpu_op = BandSwapOperation()
        gpu_op = BandSwapGPUOperation()

        # Use maximum tile count to maximize GPU batch advantage
        params = {"tile_count": 50, "permutation": "RBG", "seed": 42}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        # With many tiles, GPU batch processing should show clear advantage
        assert speedup > 1.0, (
            f"GPU ({gpu_time:.2f}ms) should be faster than "
            f"CPU ({cpu_time:.2f}ms) with many tiles, got {speedup:.2f}x"
        )

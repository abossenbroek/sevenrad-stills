"""Performance tests for corduroy GPU and Metal acceleration (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.corduroy import CorduroyOperation
from sevenrad_stills.operations.corduroy_gpu import CorduroyGPUOperation

# Conditionally import Metal operation
try:
    from sevenrad_stills.operations.corduroy_metal import CorduroyMetalOperation

    METAL_AVAILABLE = True
except (ImportError, RuntimeError):
    METAL_AVAILABLE = False


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Performance tests only run on Mac",
)
@pytest.mark.mac
class TestCorduroyPerformance:
    """Performance tests comparing CPU, GPU (Taichi), and Metal implementations."""

    @pytest.fixture
    def large_image(self) -> Image.Image:
        """Create a large test image for performance testing."""
        return Image.new("RGB", (2048, 2048), color=(128, 128, 128))

    @pytest.fixture
    def medium_image(self) -> Image.Image:
        """Create a medium test image for performance testing."""
        return Image.new("RGB", (1024, 1024), color=(128, 128, 128))

    def benchmark_operation(
        self,
        operation: CorduroyOperation | CorduroyGPUOperation,
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
        """Test that GPU is faster than CPU for large images."""
        cpu_op = CorduroyOperation()
        gpu_op = CorduroyGPUOperation()

        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        print(f"\nLarge image (2048x2048):")
        print(f"  CPU:    {cpu_time:.2f}ms")
        print(f"  GPU:    {gpu_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # GPU should be faster (speedup > 1.0)
        assert speedup > 1.0, (
            f"GPU ({gpu_time:.2f}ms) should be faster than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_gpu_faster_medium_image(self, medium_image: Image.Image) -> None:
        """Test GPU performance with medium-sized images."""
        cpu_op = CorduroyOperation()
        gpu_op = CorduroyGPUOperation()

        params = {
            "strength": 0.7,
            "orientation": "horizontal",
            "density": 0.3,
            "seed": 42,
        }

        cpu_time = self.benchmark_operation(cpu_op, medium_image, params)
        gpu_time = self.benchmark_operation(gpu_op, medium_image, params)

        speedup = cpu_time / gpu_time

        print(f"\nMedium image (1024x1024):")
        print(f"  CPU:    {cpu_time:.2f}ms")
        print(f"  GPU:    {gpu_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # GPU should be competitive (within 25% of CPU for medium images)
        assert speedup >= 0.75, (
            f"GPU ({gpu_time:.2f}ms) should not be significantly slower than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_gpu_faster_high_density(self, large_image: Image.Image) -> None:
        """Test that GPU excels with high density (more affected lines)."""
        cpu_op = CorduroyOperation()
        gpu_op = CorduroyGPUOperation()

        # High density means more lines affected
        params = {
            "strength": 0.8,
            "orientation": "vertical",
            "density": 1.0,
            "seed": 42,
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        print(f"\nHigh density (100% lines affected):")
        print(f"  CPU:    {cpu_time:.2f}ms")
        print(f"  GPU:    {gpu_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # With high density, GPU parallelization should show clear advantage
        assert speedup > 1.0, (
            f"GPU ({gpu_time:.2f}ms) should be faster than "
            f"CPU ({cpu_time:.2f}ms) with high density, got {speedup:.2f}x"
        )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_metal_faster_than_gpu(self, large_image: Image.Image) -> None:
        """Test that Metal is faster than Taichi GPU."""
        gpu_op = CorduroyGPUOperation()
        metal_op = CorduroyMetalOperation()

        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }

        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = gpu_time / metal_time

        print(f"\nMetal vs GPU (2048x2048):")
        print(f"  GPU (Taichi): {gpu_time:.2f}ms")
        print(f"  Metal:        {metal_time:.2f}ms")
        print(f"  Speedup:      {speedup:.2f}x")

        # Metal should be faster than Taichi GPU
        assert speedup > 1.0, (
            f"Metal ({metal_time:.2f}ms) should be faster than "
            f"GPU ({gpu_time:.2f}ms), got {speedup:.2f}x"
        )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_metal_faster_than_cpu(self, large_image: Image.Image) -> None:
        """Test that Metal is faster than CPU."""
        cpu_op = CorduroyOperation()
        metal_op = CorduroyMetalOperation()

        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = cpu_time / metal_time

        print(f"\nMetal vs CPU (2048x2048):")
        print(f"  CPU:     {cpu_time:.2f}ms")
        print(f"  Metal:   {metal_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Metal should be much faster than CPU
        assert speedup > 1.0, (
            f"Metal ({metal_time:.2f}ms) should be faster than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_complete_performance_comparison(self, large_image: Image.Image) -> None:
        """Test complete performance hierarchy: CPU > GPU > Metal."""
        cpu_op = CorduroyOperation()
        gpu_op = CorduroyGPUOperation()
        metal_op = CorduroyMetalOperation()

        params = {
            "strength": 0.6,
            "orientation": "horizontal",
            "density": 0.7,
            "seed": 42,
        }

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        print(f"\nComplete performance comparison (2048x2048):")
        print(f"  CPU:          {cpu_time:.2f}ms")
        print(f"  GPU (Taichi): {gpu_time:.2f}ms")
        print(f"  Metal:        {metal_time:.2f}ms")
        print(f"\nPerformance hierarchy:")
        print(f"  CPU/GPU:    {cpu_time / gpu_time:.2f}x")
        print(f"  GPU/Metal:  {gpu_time / metal_time:.2f}x")
        print(f"  CPU/Metal:  {cpu_time / metal_time:.2f}x")

        # Verify performance hierarchy: metal_time < gpu_time < cpu_time
        assert metal_time < gpu_time, (
            f"Metal ({metal_time:.2f}ms) should be faster than "
            f"GPU ({gpu_time:.2f}ms)"
        )
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.2f}ms) should be faster than " f"CPU ({cpu_time:.2f}ms)"
        )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_metal_matches_cpu_numerically(self, large_image: Image.Image) -> None:
        """Test that Metal produces numerically accurate results vs CPU."""
        cpu_op = CorduroyOperation()
        metal_op = CorduroyMetalOperation()

        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }

        cpu_result = cpu_op.apply(large_image, params)
        metal_result = metal_op.apply(large_image, params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        metal_array = np.array(metal_result).astype(np.float32)

        # Should be identical within machine epsilon
        np.testing.assert_allclose(
            cpu_array,
            metal_array,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Metal and CPU results differ beyond machine epsilon",
        )

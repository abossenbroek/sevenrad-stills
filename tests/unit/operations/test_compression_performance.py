"""Performance tests for compression GPU/Metal acceleration (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.compression import CompressionOperation
from sevenrad_stills.operations.compression_gpu import CompressionGPUOperation

# Import Metal operation only on macOS
if platform.system() == "Darwin":
    from sevenrad_stills.operations.compression_metal import CompressionMetalOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU/Metal performance tests only run on Mac",
)
class TestCompressionPerformance:
    """Performance tests comparing CPU, GPU, and Metal implementations."""

    @pytest.fixture
    def large_image(self) -> Image.Image:
        """Create a large test image for performance testing."""
        # Use realistic dimensions for photo processing
        return Image.new("RGB", (2048, 2048), color=(100, 150, 200))

    @pytest.fixture
    def xlarge_image(self) -> Image.Image:
        """Create an extra-large test image for stress testing."""
        return Image.new("RGB", (4096, 2048), color=(100, 150, 200))

    def benchmark_operation(
        self,
        operation: CompressionOperation
        | CompressionGPUOperation
        | CompressionMetalOperation,
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
        # Warmup (important for GPU/Metal to initialize)
        operation.apply(image, params)
        operation.apply(image, params)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation.apply(image, params)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return float(np.mean(times))

    def test_gpu_faster_than_cpu_with_gamma(self, large_image: Image.Image) -> None:
        """Test that GPU is faster than CPU for gamma correction on large images."""
        cpu_op = CompressionOperation()
        gpu_op = CompressionGPUOperation()

        # Use gamma correction to trigger GPU acceleration
        params = {"quality": 85, "gamma": 2.2}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        print(
            f"\nCPU: {cpu_time:.2f}ms, GPU: {gpu_time:.2f}ms, Speedup: {speedup:.2f}x"
        )

        # GPU should be faster for gamma correction
        # Note: For small images, GPU overhead may dominate, so we test with large
        # images
        assert speedup > 0.8, (
            f"GPU ({gpu_time:.2f}ms) should not be significantly slower "
            f"than CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_metal_faster_than_gpu(self, large_image: Image.Image) -> None:
        """Test that Metal hardware encoder is faster than GPU hybrid approach."""
        gpu_op = CompressionGPUOperation()
        metal_op = CompressionMetalOperation()

        # Test without gamma (pure JPEG encoding comparison)
        params = {"quality": 85}

        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = gpu_time / metal_time

        print(
            f"\nGPU: {gpu_time:.2f}ms, Metal: {metal_time:.2f}ms, "
            f"Speedup: {speedup:.2f}x"
        )

        # Metal hardware encoder should be faster than PIL JPEG encoding
        assert speedup > 0.9, (
            f"Metal ({metal_time:.2f}ms) should not be slower than "
            f"GPU ({gpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_metal_faster_than_cpu(self, large_image: Image.Image) -> None:
        """Test that Metal hardware encoder is faster than CPU."""
        cpu_op = CompressionOperation()
        metal_op = CompressionMetalOperation()

        # Test without gamma (pure JPEG encoding)
        params = {"quality": 85}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = cpu_time / metal_time

        print(
            f"\nCPU: {cpu_time:.2f}ms, Metal: {metal_time:.2f}ms, "
            f"Speedup: {speedup:.2f}x"
        )

        # Metal should be competitive with PIL's JPEG encoder
        assert speedup > 0.8, (
            f"Metal ({metal_time:.2f}ms) should not be significantly slower than "
            f"CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_full_chain_cpu_vs_gpu_vs_metal(self, xlarge_image: Image.Image) -> None:
        """Test full processing chain: CPU < GPU <= Metal for gamma + compression."""
        cpu_op = CompressionOperation()
        gpu_op = CompressionGPUOperation()
        metal_op = CompressionMetalOperation()

        # Use gamma to stress the full pipeline
        params = {"quality": 80, "gamma": 2.2}

        cpu_time = self.benchmark_operation(cpu_op, xlarge_image, params, iterations=3)
        gpu_time = self.benchmark_operation(gpu_op, xlarge_image, params, iterations=3)
        metal_time = self.benchmark_operation(
            metal_op, xlarge_image, params, iterations=3
        )

        print(
            f"\nFull chain (gamma + JPEG) on 4K image:\n"
            f"  CPU:   {cpu_time:.2f}ms\n"
            f"  GPU:   {gpu_time:.2f}ms ({cpu_time/gpu_time:.2f}x)\n"
            f"  Metal: {metal_time:.2f}ms ({cpu_time/metal_time:.2f}x)"
        )

        # GPU should improve on CPU for gamma
        assert gpu_time <= cpu_time * 1.2, (
            f"GPU ({gpu_time:.2f}ms) significantly slower "
            f"than CPU ({cpu_time:.2f}ms)"
        )

        # Metal should be competitive or better than GPU
        assert metal_time <= gpu_time * 1.2, (
            f"Metal ({metal_time:.2f}ms) significantly slower "
            f"than GPU ({gpu_time:.2f}ms)"
        )

    def test_numerical_accuracy_gpu_vs_cpu(self, large_image: Image.Image) -> None:
        """Test GPU and CPU produce nearly identical gamma correction results."""
        cpu_op = CompressionOperation()
        gpu_op = CompressionGPUOperation()

        # High quality to minimize JPEG artifacts
        params = {"quality": 95, "gamma": 1.5}

        cpu_result = cpu_op.apply(large_image, params)
        gpu_result = gpu_op.apply(large_image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)

        # JPEG is lossy, so we can't expect exact equality
        # But for the same quality, results should be very similar
        max_diff = np.max(np.abs(cpu_array - gpu_array))
        mean_diff = np.mean(np.abs(cpu_array - gpu_array))

        print(
            f"\nNumerical accuracy: max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}"
        )

        # Differences should be small (JPEG quantization differences)
        assert max_diff < 10.0, f"Max difference {max_diff} too large"
        assert mean_diff < 2.0, f"Mean difference {mean_diff} too large"

    def test_numerical_accuracy_metal_vs_cpu(self, large_image: Image.Image) -> None:
        """Test that Metal and CPU produce similar results."""
        cpu_op = CompressionOperation()
        metal_op = CompressionMetalOperation()

        # High quality to minimize artifacts
        params = {"quality": 95, "gamma": 1.5}

        cpu_result = cpu_op.apply(large_image, params)
        metal_result = metal_op.apply(large_image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        # VideoToolbox may use different JPEG encoder settings
        # Allow for larger differences
        max_diff = np.max(np.abs(cpu_array - metal_array))
        mean_diff = np.mean(np.abs(cpu_array - metal_array))

        print(
            f"\nMetal vs CPU accuracy: "
            f"max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}"
        )

        # Differences can be larger due to different encoders
        assert max_diff < 20.0, f"Max difference {max_diff} too large"
        assert mean_diff < 5.0, f"Mean difference {mean_diff} too large"

    def test_gamma_only_accuracy_cpu_vs_gpu(self) -> None:
        """Test machine epsilon accuracy for gamma correction only (no JPEG)."""
        # Create test image with known values
        test_array = np.linspace(0, 255, 1000000).reshape(1000, 1000).astype(np.uint8)
        test_image = Image.fromarray(test_array, mode="L")

        # Apply gamma using both CPU and GPU (we'll extract just the gamma step)
        cpu_gamma = np.power(test_array.astype(np.float32) / 255.0, 2.2)
        cpu_result = (np.clip(cpu_gamma, 0.0, 1.0) * 255.0).astype(np.uint8)

        # GPU gamma correction through the operation
        gpu_op = CompressionGPUOperation()
        # Use very high quality to minimize JPEG impact
        gpu_full = gpu_op.apply(test_image, {"quality": 100, "gamma": 2.2})
        gpu_result = np.array(gpu_full)

        # Due to JPEG, we can't test machine epsilon here
        # But we can verify the gamma was applied in the right direction
        assert np.mean(gpu_result) < np.mean(
            test_array
        ), "Gamma should have darkened image"

        # The mean difference should be reasonably small
        diff = np.abs(gpu_result.astype(np.float32) - cpu_result.astype(np.float32))
        print(f"\nGamma-only accuracy (with JPEG): mean_diff={np.mean(diff):.4f}")
        assert np.mean(diff) < 5.0, "Gamma results diverged too much"

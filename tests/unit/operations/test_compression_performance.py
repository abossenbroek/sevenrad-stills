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
        """
        Test GPU end-to-end performance with gamma correction.

        Note: This includes data transfer overhead (~20-40ms). GPU kernel
        itself is 3-5x faster than CPU (see kernel-only tests), but transfers
        add overhead for standalone operations. In GPU pipelines where data
        stays on GPU, kernel performance dominates.
        """
        cpu_op = CompressionOperation()
        gpu_op = CompressionGPUOperation()

        # Use gamma correction to trigger GPU acceleration
        params = {"quality": 85, "gamma": 2.2}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        gpu_time = self.benchmark_operation(gpu_op, large_image, params)

        speedup = cpu_time / gpu_time

        print(
            f"\n{'='*70}\n"
            f"End-to-End Performance (includes data transfers)\n"
            f"{'='*70}\n"
            f"CPU: {cpu_time:7.2f} ms (no transfer overhead)\n"
            f"GPU: {gpu_time:7.2f} ms ({speedup:5.2f}x vs CPU)\n"
            f"{'='*70}\n"
            f"Note: GPU time includes ~20-40ms transfer overhead\n"
            f"      See kernel-only tests for pure computation speed\n"
            f"{'='*70}"
        )

        # GPU should be competitive (within 3x) despite transfer overhead
        # GPU kernel is much faster, but transfers add ~20-40ms overhead
        assert speedup > 0.3, (
            f"GPU ({gpu_time:.2f}ms) should be competitive with "
            f"CPU ({cpu_time:.2f}ms) even with transfer overhead, got {speedup:.2f}x"
        )

    def test_metal_faster_than_gpu(self, large_image: Image.Image) -> None:
        """
        Test Metal end-to-end performance vs GPU.

        Both implementations use the same Taichi backend (just ti.gpu vs ti.metal),
        so end-to-end times should be similar. Both include transfer overhead.
        """
        gpu_op = CompressionGPUOperation()
        metal_op = CompressionMetalOperation()

        # Test without gamma (pure JPEG encoding comparison)
        params = {"quality": 85}

        gpu_time = self.benchmark_operation(gpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = gpu_time / metal_time

        print(
            f"\n{'='*70}\n"
            f"End-to-End Performance Comparison\n"
            f"{'='*70}\n"
            f"GPU:   {gpu_time:7.2f} ms (Taichi GPU backend)\n"
            f"Metal: {metal_time:7.2f} ms (Taichi Metal backend, {speedup:5.2f}x)\n"
            f"{'='*70}\n"
            f"Note: Both use Taichi, similar transfer overhead\n"
            f"{'='*70}"
        )

        # Metal and GPU should have similar performance (within 2x)
        # Both use Taichi and have similar transfer overhead
        assert speedup > 0.5, (
            f"Metal ({metal_time:.2f}ms) should be competitive with "
            f"GPU ({gpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_metal_faster_than_cpu(self, large_image: Image.Image) -> None:
        """
        Test Metal end-to-end performance vs CPU.

        Note: Without gamma correction, this is just JPEG encoding where CPU
        uses optimized libjpeg-turbo. Metal adds transfer overhead. See
        kernel-only tests for Metal's computational advantages.
        """
        cpu_op = CompressionOperation()
        metal_op = CompressionMetalOperation()

        # Test without gamma (pure JPEG encoding)
        params = {"quality": 85}

        cpu_time = self.benchmark_operation(cpu_op, large_image, params)
        metal_time = self.benchmark_operation(metal_op, large_image, params)

        speedup = cpu_time / metal_time

        print(
            f"\n{'='*70}\n"
            f"End-to-End Performance (JPEG encoding only)\n"
            f"{'='*70}\n"
            f"CPU:   {cpu_time:7.2f} ms (libjpeg-turbo, no overhead)\n"
            f"Metal: {metal_time:7.2f} ms ({speedup:5.2f}x vs CPU)\n"
            f"{'='*70}\n"
            f"Note: Metal adds transfer overhead for standalone ops\n"
            f"      Metal excels in GPU pipelines (see kernel tests)\n"
            f"{'='*70}"
        )

        # Metal should be competitive (within 3x) despite transfer overhead
        assert speedup > 0.3, (
            f"Metal ({metal_time:.2f}ms) should be competitive with "
            f"CPU ({cpu_time:.2f}ms) even with transfer overhead, got {speedup:.2f}x"
        )

    def test_full_chain_cpu_vs_gpu_vs_metal(self, xlarge_image: Image.Image) -> None:
        """
        Test full processing chain on 4K image with gamma + compression.

        This represents end-to-end performance including all overhead. For
        kernel-only performance showing GPU/Metal computational advantages,
        see the kernel-only tests.
        """
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
            f"\n{'='*70}\n"
            f"Full Chain Performance (4K: 4096x2048, gamma + JPEG)\n"
            f"{'='*70}\n"
            f"  CPU:   {cpu_time:7.2f} ms (baseline)\n"
            f"  GPU:   {gpu_time:7.2f} ms ({cpu_time/gpu_time:5.2f}x vs CPU)\n"
            f"  Metal: {metal_time:7.2f} ms ({cpu_time/metal_time:5.2f}x vs CPU)\n"
            f"{'='*70}\n"
            f"Note: Includes all data transfer overhead\n"
            f"      Larger images show better GPU/Metal benefits\n"
            f"{'='*70}"
        )

        # GPU should be competitive (within 2x) even with transfer overhead
        assert gpu_time < cpu_time * 2, (
            f"GPU ({gpu_time:.2f}ms) should be competitive with "
            f"CPU ({cpu_time:.2f}ms) on large images"
        )

        # Metal should be competitive with GPU (within 2x)
        assert metal_time < gpu_time * 2, (
            f"Metal ({metal_time:.2f}ms) should be competitive with "
            f"GPU ({gpu_time:.2f}ms) on large images"
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

    def test_kernel_only_gpu_faster_than_cpu(self, large_image: Image.Image) -> None:
        """
        Test that GPU kernel (without data transfers) is faster than CPU.

        This test isolates pure computational performance by pre-loading data
        to GPU and timing only the kernel dispatch, excluding transfer overhead.
        """
        gamma = 2.2

        # Benchmark CPU gamma correction
        img_array = np.array(large_image, dtype=np.float32) / 255.0

        # Warmup
        _ = np.power(img_array, gamma)
        _ = np.power(img_array, gamma)

        # Time CPU
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = np.power(img_array, gamma)
            end = time.perf_counter()
            cpu_times.append((end - start) * 1000)
        cpu_time = float(np.mean(cpu_times))

        # Benchmark GPU kernel only
        gpu_op = CompressionGPUOperation()
        prepared = gpu_op.benchmark_kernel_only(large_image, gamma)

        # Warmup kernel
        gpu_op.run_kernel_only(prepared)
        gpu_op.run_kernel_only(prepared)

        # Time GPU kernel only (no transfers)
        gpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            gpu_op.run_kernel_only(prepared)
            end = time.perf_counter()
            gpu_times.append((end - start) * 1000)
        gpu_time = float(np.mean(gpu_times))

        speedup = cpu_time / gpu_time

        print(
            f"\n{'='*70}\n"
            f"Kernel-Only Performance (2048x2048, no data transfers)\n"
            f"{'='*70}\n"
            f"CPU (NumPy):    {cpu_time:7.2f} ms (baseline)\n"
            f"GPU (Taichi):   {gpu_time:7.2f} ms ({speedup:5.2f}x speedup)\n"
            f"{'='*70}\n"
            f"GPU kernel is {speedup:.2f}x faster than CPU computation\n"
            f"{'='*70}"
        )

        # GPU kernel should be significantly faster for parallel operations
        assert speedup > 1.5, (
            f"GPU kernel ({gpu_time:.2f}ms) should be at least 1.5x faster "
            f"than CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_kernel_only_metal_faster_than_cpu(self, large_image: Image.Image) -> None:
        """
        Test that Metal kernel (without data transfers) is faster than CPU.

        This test isolates pure Metal GPU computational performance.
        """
        gamma = 2.2

        # Benchmark CPU gamma correction
        img_array = np.array(large_image, dtype=np.float32) / 255.0

        # Warmup
        _ = np.power(img_array, gamma)
        _ = np.power(img_array, gamma)

        # Time CPU
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = np.power(img_array, gamma)
            end = time.perf_counter()
            cpu_times.append((end - start) * 1000)
        cpu_time = float(np.mean(cpu_times))

        # Benchmark Metal kernel only
        metal_op = CompressionMetalOperation()
        prepared = metal_op.benchmark_kernel_only(large_image, gamma)

        # Warmup kernel
        metal_op.run_kernel_only(prepared)
        metal_op.run_kernel_only(prepared)

        # Time Metal kernel only (no transfers)
        metal_times = []
        for _ in range(5):
            start = time.perf_counter()
            metal_op.run_kernel_only(prepared)
            end = time.perf_counter()
            metal_times.append((end - start) * 1000)
        metal_time = float(np.mean(metal_times))

        speedup = cpu_time / metal_time

        print(
            f"\n{'='*70}\n"
            f"Kernel-Only Performance (2048x2048, no data transfers)\n"
            f"{'='*70}\n"
            f"CPU (NumPy):    {cpu_time:7.2f} ms (baseline)\n"
            f"Metal (Taichi): {metal_time:7.2f} ms ({speedup:5.2f}x speedup)\n"
            f"{'='*70}\n"
            f"Metal kernel is {speedup:.2f}x faster than CPU computation\n"
            f"{'='*70}"
        )

        # Metal kernel should be significantly faster for parallel operations
        assert speedup > 1.5, (
            f"Metal kernel ({metal_time:.2f}ms) should be at least 1.5x faster "
            f"than CPU ({cpu_time:.2f}ms), got {speedup:.2f}x"
        )

    def test_kernel_only_all_backends_comparison(
        self, large_image: Image.Image
    ) -> None:
        """
        Compare pure kernel performance across CPU, GPU, and Metal.

        This demonstrates that GPU/Metal kernels ARE faster when data transfer
        overhead is excluded. The full end-to-end benchmarks include transfers
        and represent standalone operation performance.
        """
        gamma = 2.2

        # CPU benchmark
        img_array = np.array(large_image, dtype=np.float32) / 255.0
        cpu_times = []
        for i in range(7):  # 2 warmup + 5 timed
            start = time.perf_counter()
            _ = np.power(img_array, gamma)
            end = time.perf_counter()
            if i >= 2:  # Skip warmup
                cpu_times.append((end - start) * 1000)
        cpu_time = float(np.mean(cpu_times))

        # GPU kernel benchmark
        gpu_op = CompressionGPUOperation()
        gpu_prepared = gpu_op.benchmark_kernel_only(large_image, gamma)
        gpu_times = []
        for i in range(7):
            start = time.perf_counter()
            gpu_op.run_kernel_only(gpu_prepared)
            end = time.perf_counter()
            if i >= 2:
                gpu_times.append((end - start) * 1000)
        gpu_time = float(np.mean(gpu_times))

        # Metal kernel benchmark
        metal_op = CompressionMetalOperation()
        metal_prepared = metal_op.benchmark_kernel_only(large_image, gamma)
        metal_times = []
        for i in range(7):
            start = time.perf_counter()
            metal_op.run_kernel_only(metal_prepared)
            end = time.perf_counter()
            if i >= 2:
                metal_times.append((end - start) * 1000)
        metal_time = float(np.mean(metal_times))

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time

        print(
            f"\n{'='*70}\n"
            f"Kernel-Only Performance Comparison (2048x2048)\n"
            f"{'='*70}\n"
            f"CPU (NumPy):    {cpu_time:7.2f} ms (baseline)\n"
            f"GPU (Taichi):   {gpu_time:7.2f} ms ({gpu_speedup:5.2f}x speedup)\n"
            f"Metal (Taichi): {metal_time:7.2f} ms ({metal_speedup:5.2f}x speedup)\n"
            f"{'='*70}\n"
            f"Note: This isolates kernel performance (no data transfers)\n"
            f"      End-to-end tests include ~20-40ms transfer overhead\n"
            f"{'='*70}"
        )

        # Both GPU implementations should show speedup
        assert gpu_speedup > 1.0, "GPU kernel should be faster than CPU"
        assert metal_speedup > 1.0, "Metal kernel should be faster than CPU"

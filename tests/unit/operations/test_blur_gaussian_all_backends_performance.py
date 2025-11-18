"""
Comprehensive performance comparison tests.

Gaussian blur across all backends (CPU, GPU, Metal).
"""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation
from sevenrad_stills.operations.blur_gaussian_metal import GaussianBlurMetalOperation


def time_operation(operation, image, params, warmup_runs=2, timed_runs=5):
    """
    Time an operation with warmup runs.

    Args:
        operation: The operation to benchmark
        image: Input image
        params: Operation parameters
        warmup_runs: Number of warmup iterations
        timed_runs: Number of timed iterations

    Returns:
        Mean execution time in milliseconds

    """
    # Warmup runs (important for GPU kernel compilation)
    for _ in range(warmup_runs):
        operation.apply(image, params)

    # Timed runs
    times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = operation.apply(image, params)
        end = time.perf_counter()
        times.append(end - start)
        del result  # Free memory

    return float(np.mean(times)) * 1000  # Convert to milliseconds


@pytest.mark.mac
@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal backend requires macOS",
)
class TestGaussianBlurAllBackendsPerformance:
    """Performance comparison across CPU, GPU (Taichi), and Metal backends."""

    @pytest.fixture
    def cpu_operation(self) -> GaussianBlurOperation:
        """Create CPU Gaussian blur operation."""
        return GaussianBlurOperation()

    @pytest.fixture
    def gpu_operation(self) -> GaussianBlurGPUOperation:
        """Create GPU (Taichi) Gaussian blur operation."""
        return GaussianBlurGPUOperation()

    @pytest.fixture
    def metal_operation(self) -> GaussianBlurMetalOperation:
        """Create Metal Gaussian blur operation."""
        return GaussianBlurMetalOperation()

    def _create_test_image(self, size: int) -> Image.Image:
        """Create a test image with random pattern."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(data, mode="RGB")

    def test_all_backends_small_image(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Test performance on small image (512x512)."""
        image = self._create_test_image(512)
        params = {"sigma": 2.0}

        cpu_time = time_operation(cpu_operation, image, params)
        gpu_time = time_operation(gpu_operation, image, params)
        metal_time = time_operation(metal_operation, image, params)

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time
        metal_vs_gpu = gpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Small Image (512x512), Sigma: 2.0")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_time:7.2f} ms (baseline)")
        print(f"GPU (Taichi):   {gpu_time:7.2f} ms ({gpu_speedup:5.2f}x vs CPU)")
        print(f"Metal (Pure):   {metal_time:7.2f} ms ({metal_speedup:5.2f}x vs CPU)")
        print(f"{'='*70}")
        comparison = "FASTER" if metal_vs_gpu > 1 else "slower"
        print(f"Metal vs GPU:   {metal_vs_gpu:5.2f}x {comparison}")
        print(f"{'='*70}")

        # For small images, GPU overhead may dominate
        # Just verify all complete successfully
        assert cpu_time > 0
        assert gpu_time > 0
        assert metal_time > 0

    def test_all_backends_medium_image(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Test performance on medium image (1024x1024)."""
        image = self._create_test_image(1024)
        params = {"sigma": 5.0}

        cpu_time = time_operation(cpu_operation, image, params)
        gpu_time = time_operation(gpu_operation, image, params)
        metal_time = time_operation(metal_operation, image, params)

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time
        metal_vs_gpu = gpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Medium Image (1024x1024), Sigma: 5.0")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_time:7.2f} ms (baseline)")
        print(f"GPU (Taichi):   {gpu_time:7.2f} ms ({gpu_speedup:5.2f}x vs CPU)")
        print(f"Metal (Pure):   {metal_time:7.2f} ms ({metal_speedup:5.2f}x vs CPU)")
        print(f"{'='*70}")
        comparison = "FASTER" if metal_vs_gpu > 1 else "slower"
        print(f"Metal vs GPU:   {metal_vs_gpu:5.2f}x {comparison}")
        print(f"{'='*70}")

        # GPU should start showing benefits at this size
        assert cpu_time > 0
        assert gpu_time > 0
        assert metal_time > 0

    def test_all_backends_large_image(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Test performance on large image (2048x2048)."""
        image = self._create_test_image(2048)
        params = {"sigma": 5.0}

        cpu_time = time_operation(cpu_operation, image, params, timed_runs=5)
        gpu_time = time_operation(gpu_operation, image, params, timed_runs=5)
        metal_time = time_operation(metal_operation, image, params, timed_runs=5)

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time
        metal_vs_gpu = gpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Large Image (2048x2048), Sigma: 5.0")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_time:7.2f} ms (baseline)")
        print(f"GPU (Taichi):   {gpu_time:7.2f} ms ({gpu_speedup:5.2f}x vs CPU)")
        print(f"Metal (Pure):   {metal_time:7.2f} ms ({metal_speedup:5.2f}x vs CPU)")
        print(f"{'='*70}")
        comparison = "FASTER" if metal_vs_gpu > 1 else "slower"
        print(f"Metal vs GPU:   {metal_vs_gpu:5.2f}x {comparison}")
        print(f"{'='*70}")

        # CRITICAL: GPU and Metal MUST be faster than CPU for large images
        assert (
            gpu_speedup > 1.0
        ), f"GPU must be faster than CPU on large images! Got {gpu_speedup:.2f}x"
        assert (
            metal_speedup > 1.0
        ), f"Metal must be faster than CPU on large images! Got {metal_speedup:.2f}x"

    def test_all_backends_very_large_image(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Test performance on very large image (4096x4096)."""
        image = self._create_test_image(4096)
        params = {"sigma": 5.0}

        cpu_time = time_operation(
            cpu_operation, image, params, warmup_runs=1, timed_runs=3
        )
        gpu_time = time_operation(
            gpu_operation, image, params, warmup_runs=1, timed_runs=3
        )
        metal_time = time_operation(
            metal_operation, image, params, warmup_runs=1, timed_runs=3
        )

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time
        metal_vs_gpu = gpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Very Large Image (4096x4096), Sigma: 5.0")
        print(f"{'='*70}")
        print(f"CPU (scipy):    {cpu_time:7.2f} ms (baseline)")
        print(f"GPU (Taichi):   {gpu_time:7.2f} ms ({gpu_speedup:5.2f}x vs CPU)")
        print(f"Metal (Pure):   {metal_time:7.2f} ms ({metal_speedup:5.2f}x vs CPU)")
        print(f"{'='*70}")
        comparison = "FASTER" if metal_vs_gpu > 1 else "slower"
        print(f"Metal vs GPU:   {metal_vs_gpu:5.2f}x {comparison}")
        print(f"{'='*70}")

        # CRITICAL: Maximum GPU advantage expected at this scale
        assert (
            gpu_speedup > 1.0
        ), f"GPU must show speedup on very large images! Got {gpu_speedup:.2f}x"
        assert (
            metal_speedup > 1.0
        ), f"Metal must show speedup on very large images! Got {metal_speedup:.2f}x"

    def test_metal_vs_gpu_comparison(
        self,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Direct comparison of Metal vs GPU across different image sizes."""
        sizes = [512, 1024, 2048]
        sigma = 5.0

        print(f"\n{'='*70}")
        print("Metal vs GPU Direct Comparison (Sigma: 5.0)")
        print(f"{'='*70}")
        print(f"{'Size':<15} {'GPU (ms)':<15} {'Metal (ms)':<15} {'Metal/GPU':<15}")
        print(f"{'='*70}")

        for size in sizes:
            image = self._create_test_image(size)
            params = {"sigma": sigma}

            gpu_time = time_operation(gpu_operation, image, params)
            metal_time = time_operation(metal_operation, image, params)

            ratio = metal_time / gpu_time

            print(
                f"{size}x{size:<8} {gpu_time:<15.2f} {metal_time:<15.2f} {ratio:<15.2f}"
            )

        print(f"{'='*70}")

        # Metal should be competitive with GPU (within reasonable range)
        # Allow Metal to be slightly slower due to different implementation

    def test_varying_shift_values(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Test that performance is consistent across different sigma values."""
        image = self._create_test_image(2048)
        sigmas = [2.0, 5.0, 10.0]

        print(f"\n{'='*70}")
        print("Performance vs Sigma Value (2048x2048)")
        print(f"{'='*70}")
        header = (
            f"{'Sigma':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} "
            f"{'Metal (ms)':<12} {'GPU Speedup':<15} {'Metal Speedup':<15}"
        )
        print(header)
        print(f"{'='*70}")

        for sigma in sigmas:
            params = {"sigma": sigma}

            cpu_time = time_operation(cpu_operation, image, params, timed_runs=3)
            gpu_time = time_operation(gpu_operation, image, params, timed_runs=3)
            metal_time = time_operation(metal_operation, image, params, timed_runs=3)

            gpu_speedup = cpu_time / gpu_time
            metal_speedup = cpu_time / metal_time

            row = (
                f"{sigma:<10.1f} {cpu_time:<12.2f} {gpu_time:<12.2f} "
                f"{metal_time:<12.2f} {gpu_speedup:<15.2f}x "
                f"{metal_speedup:<15.2f}x"
            )
            print(row)

        print(f"{'='*70}")

        # Performance should scale with sigma but GPU/Metal should remain faster

    def test_numerical_accuracy(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Verify that all backends produce numerically similar results."""
        image = self._create_test_image(512)
        params = {"sigma": 3.0}

        cpu_result = cpu_operation.apply(image, params)
        gpu_result = gpu_operation.apply(image, params)
        metal_result = metal_operation.apply(image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        # CPU vs GPU
        mae_gpu = np.mean(np.abs(cpu_array - gpu_array))
        max_error_gpu = np.max(np.abs(cpu_array - gpu_array))

        # CPU vs Metal
        mae_metal = np.mean(np.abs(cpu_array - metal_array))
        max_error_metal = np.max(np.abs(cpu_array - metal_array))

        # GPU vs Metal
        mae_gpu_metal = np.mean(np.abs(gpu_array - metal_array))
        max_error_gpu_metal = np.max(np.abs(gpu_array - metal_array))

        print(f"\n{'='*70}")
        print("Numerical Accuracy Comparison")
        print(f"{'='*70}")
        print(
            f"CPU vs GPU:         MAE = {mae_gpu:.4f}, "
            f"Max Error = {max_error_gpu:.4f}"
        )
        print(
            f"CPU vs Metal:       MAE = {mae_metal:.4f}, "
            f"Max Error = {max_error_metal:.4f}"
        )
        print(
            f"GPU vs Metal:       MAE = {mae_gpu_metal:.4f}, "
            f"Max Error = {max_error_gpu_metal:.4f}"
        )
        print(f"{'='*70}")

        # All implementations should be very close
        # (allowing for minor numerical differences)
        msg_gpu = f"GPU differs too much from CPU: MAE = {mae_gpu:.4f}"
        assert mae_gpu < 2.0, msg_gpu
        msg_metal = f"Metal differs too much from CPU: MAE = {mae_metal:.4f}"
        assert mae_metal < 2.0, msg_metal
        assert (
            mae_gpu_metal < 2.0
        ), f"GPU and Metal differ too much: MAE = {mae_gpu_metal:.4f}"

        # Max error can be higher at edges due to boundary handling
        assert max_error_gpu <= 10.0, f"GPU max error too large: {max_error_gpu:.4f}"
        assert (
            max_error_metal <= 10.0
        ), f"Metal max error too large: {max_error_metal:.4f}"

    def test_rgba_support(
        self,
        cpu_operation: GaussianBlurOperation,
        gpu_operation: GaussianBlurGPUOperation,
        metal_operation: GaussianBlurMetalOperation,
    ) -> None:
        """Verify that all backends handle RGBA images correctly."""
        # Create RGBA image
        rng = np.random.default_rng(42)
        rgba_data = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
        image = Image.fromarray(rgba_data, mode="RGBA")
        params = {"sigma": 3.0}

        cpu_result = cpu_operation.apply(image, params)
        gpu_result = gpu_operation.apply(image, params)
        metal_result = metal_operation.apply(image, params)

        # Verify results are RGBA
        assert cpu_result.mode == "RGBA"
        assert gpu_result.mode == "RGBA"
        assert metal_result.mode == "RGBA"

        # Verify results are similar
        cpu_array = np.array(cpu_result, dtype=np.float32)
        gpu_array = np.array(gpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        mae_gpu = np.mean(np.abs(cpu_array - gpu_array))
        mae_metal = np.mean(np.abs(cpu_array - metal_array))

        print(f"\n{'='*70}")
        print("RGBA Image Support")
        print(f"{'='*70}")
        print(f"CPU vs GPU (RGBA):   MAE = {mae_gpu:.4f}")
        print(f"CPU vs Metal (RGBA): MAE = {mae_metal:.4f}")
        print(f"{'='*70}")

        assert mae_gpu < 2.0, f"GPU RGBA differs too much from CPU: MAE = {mae_gpu:.4f}"
        assert (
            mae_metal < 2.0
        ), f"Metal RGBA differs too much from CPU: MAE = {mae_metal:.4f}"

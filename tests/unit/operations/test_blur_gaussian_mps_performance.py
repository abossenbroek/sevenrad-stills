"""Performance tests for MPS Gaussian blur vs MLX, Taichi GPU, and CPU."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_gpu import GaussianBlurGPUOperation
from sevenrad_stills.operations.blur_gaussian_mlx import GaussianBlurMLXOperation
from sevenrad_stills.operations.blur_gaussian_mps import GaussianBlurMPSOperation

# Mark all tests as slow and Mac-specific
pytestmark = [
    pytest.mark.slow,
    pytest.mark.mac,
    pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="MPS performance tests only run on Mac (Metal backend)",
    ),
]


def benchmark_operation(
    operation: object,
    image: Image.Image,
    params: dict,
    warmup: int = 2,
    iterations: int = 10,
) -> tuple[float, float]:
    """
    Benchmark an operation with warmup iterations.

    Args:
        operation: The operation instance to benchmark
        image: Input image
        params: Operation parameters
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Tuple of (mean_time_ms, std_time_ms)

    """
    # Warmup
    for _ in range(warmup):
        operation.apply(image, params)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = operation.apply(image, params)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
        del result  # Free memory

    return float(np.mean(times)), float(np.std(times))


@pytest.mark.parametrize(
    ("size", "sigma"),
    [
        (512, 2.0),
        (512, 5.0),
        (1024, 2.0),
        (1024, 5.0),
        (2048, 2.0),
        (2048, 5.0),
        (4096, 5.0),
    ],
)
def test_mps_vs_all_performance(size: int, sigma: float) -> None:
    """Compare MPS performance against MLX, Taichi GPU, and CPU implementations."""
    # Create test image
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode="RGB")
    params = {"sigma": sigma}

    # Initialize operations
    cpu_op = GaussianBlurOperation()
    taichi_op = GaussianBlurGPUOperation()
    mlx_op = GaussianBlurMLXOperation()
    mps_op = GaussianBlurMPSOperation()

    # Benchmark all implementations
    cpu_mean, cpu_std = benchmark_operation(cpu_op, image, params)
    taichi_mean, taichi_std = benchmark_operation(taichi_op, image, params)
    mlx_mean, mlx_std = benchmark_operation(mlx_op, image, params)
    mps_mean, mps_std = benchmark_operation(mps_op, image, params)

    # Calculate speedups
    taichi_speedup = cpu_mean / taichi_mean
    mlx_speedup = cpu_mean / mlx_mean
    mps_speedup = cpu_mean / mps_mean

    print(f"\n{'='*80}")
    print(f"Image size: {size}x{size}, Sigma: {sigma}")
    print(f"{'='*80}")
    print(f"CPU (scipy):        {cpu_mean:8.2f} ms ± {cpu_std:6.2f} ms (baseline)")
    print(
        f"Taichi GPU:         {taichi_mean:8.2f} ms ± {taichi_std:6.2f} ms ({taichi_speedup:5.2f}x vs CPU)"
    )
    print(
        f"MLX:                {mlx_mean:8.2f} ms ± {mlx_std:6.2f} ms ({mlx_speedup:5.2f}x vs CPU)"
    )
    print(
        f"MPS (Metal):        {mps_mean:8.2f} ms ± {mps_std:6.2f} ms ({mps_speedup:5.2f}x vs CPU)"
    )

    # MPS vs other GPU implementations
    mps_vs_taichi = taichi_mean / mps_mean
    mps_vs_mlx = mlx_mean / mps_mean

    print(f"\nGPU Comparison:")
    print(f"MPS vs Taichi:      {mps_vs_taichi:5.2f}x")
    print(f"MPS vs MLX:         {mps_vs_mlx:5.2f}x")
    print(f"{'='*80}")

    # Verify correctness - all implementations should produce similar results
    cpu_result = cpu_op.apply(image, params)
    mps_result = mps_op.apply(image, params)

    cpu_array = np.array(cpu_result, dtype=np.float32)
    mps_array = np.array(mps_result, dtype=np.float32)

    mae = np.mean(np.abs(cpu_array - mps_array))
    max_abs_error = np.max(np.abs(cpu_array - mps_array))

    print(f"\nNumerical Accuracy (MPS vs CPU):")
    print(f"Mean Absolute Error:     {mae:.2f}")
    print(f"Max Absolute Error:      {max_abs_error:.2f}")

    # MPS must maintain good accuracy (< 3.0 allows for edge handling differences)
    # MPS uses Apple's blur implementation which may handle edges differently than scipy
    assert mae < 3.0, f"MPS mean absolute error {mae:.2f} exceeds threshold"

    # MPS should be faster than CPU for all sizes
    assert mps_speedup > 1.0, f"MPS ({mps_speedup:.2f}x) should be faster than CPU"


@pytest.mark.parametrize(
    ("size", "sigma"),
    [
        (2048, 5.0),
        (4096, 5.0),
    ],
)
def test_mps_ultimate_performance(size: int, sigma: float) -> None:
    """
    Verify MPS provides excellent performance for large images.

    This test verifies that MPS uses Apple's hand-optimized Metal Performance
    Shaders to achieve significant speedup over CPU for production-scale images.

    Note: MLX may still be faster due to Apple Silicon-specific optimizations,
    but MPS provides a solid Metal-native alternative with excellent accuracy.
    """
    # Create test image
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode="RGB")
    params = {"sigma": sigma}

    # Initialize operations
    cpu_op = GaussianBlurOperation()
    mps_op = GaussianBlurMPSOperation()

    # Benchmark
    cpu_mean, _ = benchmark_operation(cpu_op, image, params, iterations=5)
    mps_mean, _ = benchmark_operation(mps_op, image, params, iterations=5)

    speedup = cpu_mean / mps_mean

    print(f"\n{'='*80}")
    print(f"MPS Performance Test: {size}x{size}, Sigma: {sigma}")
    print(f"{'='*80}")
    print(f"CPU:  {cpu_mean:8.2f} ms")
    print(f"MPS:  {mps_mean:8.2f} ms")
    print(f"Speedup: {speedup:.2f}x vs CPU")
    print(f"{'='*80}")

    # MPS should provide good speedup over CPU (2-4x range is realistic)
    min_speedup = 2.0
    assert (
        speedup >= min_speedup
    ), f"MPS speedup {speedup:.2f}x below minimum {min_speedup}x for {size}x{size} images"

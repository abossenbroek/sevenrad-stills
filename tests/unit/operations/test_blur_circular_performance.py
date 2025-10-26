"""Performance benchmarks for GPU vs CPU circular blur operations."""

import sys
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_circular_gpu import CircularBlurGPUOperation


@pytest.fixture
def blur_op_cpu() -> CircularBlurOperation:
    """Create a CPU circular blur operation instance."""
    return CircularBlurOperation()


@pytest.fixture
def blur_op_gpu() -> CircularBlurGPUOperation:
    """Create a GPU circular blur operation instance."""
    return CircularBlurGPUOperation()


def create_test_image(size: tuple[int, int]) -> Image.Image:
    """
    Create a test RGB image with random content.

    Args:
        size: Tuple of (width, height) for the image.

    Returns:
        RGB PIL Image with random pixel values.

    """
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def time_operation(
    operation: CircularBlurOperation | CircularBlurGPUOperation,
    image: Image.Image,
    radius: int,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> float:
    """
    Time an operation with warmup and multiple runs.

    Args:
        operation: The blur operation to time.
        image: The input image.
        radius: Blur radius.
        warmup_runs: Number of warmup iterations.
        timed_runs: Number of timed iterations.

    Returns:
        Average execution time in seconds.

    """
    params = {"radius": radius}

    # Warmup runs (especially important for GPU to initialize kernels)
    for _ in range(warmup_runs):
        operation.apply(image, params)

    # Timed runs
    times = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        operation.apply(image, params)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times))


@pytest.mark.mac
class TestCircularBlurPerformance:
    """Performance benchmarks comparing GPU and CPU implementations."""

    def test_gpu_baseline_small_image(
        self,
        blur_op_cpu: CircularBlurOperation,
        blur_op_gpu: CircularBlurGPUOperation,
    ) -> None:
        """Baseline test showing GPU overhead is acceptable on small images."""
        image = create_test_image((512, 512))
        radius = 5

        cpu_time = time_operation(blur_op_cpu, image, radius)
        gpu_time = time_operation(blur_op_gpu, image, radius)

        print(f"\nSmall image (512x512), radius={radius}")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        if gpu_time < cpu_time:
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            print(f"Overhead: {gpu_time / cpu_time:.2f}x (expected for small images)")

        # GPU has initialization overhead, so small images may be slower
        # Just verify it completes in reasonable time
        assert (
            gpu_time < 0.5
        ), f"GPU should complete small image in reasonable time, got {gpu_time:.4f}s"

    def test_gpu_faster_medium_image_medium_radius(
        self,
        blur_op_cpu: CircularBlurOperation,
        blur_op_gpu: CircularBlurGPUOperation,
    ) -> None:
        """Test GPU performance on medium image with medium radius."""
        image = create_test_image((1024, 1024))
        radius = 10

        cpu_time = time_operation(blur_op_cpu, image, radius)
        gpu_time = time_operation(blur_op_gpu, image, radius)

        print(f"\nMedium image (1024x1024), radius={radius}")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # GPU should be faster on medium images
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on medium images"
        )

    def test_gpu_faster_large_image_large_radius(
        self,
        blur_op_cpu: CircularBlurOperation,
        blur_op_gpu: CircularBlurGPUOperation,
    ) -> None:
        """Test GPU performance on large image with large radius."""
        image = create_test_image((2048, 2048))
        radius = 15

        cpu_time = time_operation(blur_op_cpu, image, radius)
        gpu_time = time_operation(blur_op_gpu, image, radius)

        print(f"\nLarge image (2048x2048), radius={radius}")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # GPU should show significant speedup on large images
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on large images"
        )
        assert cpu_time / gpu_time >= 1.5, (
            f"GPU should show at least 1.5x speedup on large images, "
            f"got {cpu_time / gpu_time:.2f}x"
        )

    def test_gpu_scales_better_with_radius(
        self,
        blur_op_cpu: CircularBlurOperation,
        blur_op_gpu: CircularBlurGPUOperation,
    ) -> None:
        """Test that GPU scales better than CPU as radius increases."""
        image = create_test_image((1024, 1024))

        # Test with small and large radius
        small_radius = 5
        large_radius = 20

        # Small radius
        cpu_time_small = time_operation(blur_op_cpu, image, small_radius)
        gpu_time_small = time_operation(blur_op_gpu, image, small_radius)

        # Large radius
        cpu_time_large = time_operation(blur_op_cpu, image, large_radius)
        gpu_time_large = time_operation(blur_op_gpu, image, large_radius)

        cpu_scaling = cpu_time_large / cpu_time_small
        gpu_scaling = gpu_time_large / gpu_time_small

        print(f"\nScaling test (1024x1024)")
        print(
            f"Small radius ({small_radius}): CPU={cpu_time_small:.4f}s, "
            f"GPU={gpu_time_small:.4f}s"
        )
        print(
            f"Large radius ({large_radius}): CPU={cpu_time_large:.4f}s, "
            f"GPU={gpu_time_large:.4f}s"
        )
        print(f"CPU scaling factor: {cpu_scaling:.2f}x")
        print(f"GPU scaling factor: {gpu_scaling:.2f}x")

        # GPU should scale better (lower scaling factor) as radius increases
        assert gpu_scaling <= cpu_scaling, (
            f"GPU should scale better than CPU with increasing radius. "
            f"GPU scaling: {gpu_scaling:.2f}x, CPU scaling: {cpu_scaling:.2f}x"
        )

    def test_gpu_overhead_acceptable(
        self, blur_op_gpu: CircularBlurGPUOperation
    ) -> None:
        """Test that GPU initialization overhead is acceptable."""
        # Very small image where CPU might be competitive
        image = create_test_image((100, 100))
        radius = 3

        # Time single run (including any initialization)
        start = time.perf_counter()
        blur_op_gpu.apply(image, {"radius": radius})
        end = time.perf_counter()

        single_run_time = end - start

        print(f"\nGPU overhead test (100x100, radius={radius})")
        print(f"Single run time: {single_run_time:.4f}s")

        # Should complete in reasonable time even with initialization
        assert (
            single_run_time < 1.0
        ), f"GPU should complete small image in <1s, got {single_run_time:.4f}s"


# Skip all tests if not on macOS
def pytest_collection_modifyitems(items):
    """Skip mac-only tests if not running on macOS."""
    if sys.platform != "darwin":
        skip_mac = pytest.mark.skip(reason="Only runs on macOS")
        for item in items:
            if "mac" in item.keywords:
                item.add_marker(skip_mac)

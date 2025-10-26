"""Performance benchmarks for GPU vs CPU Bayer filter operations."""
# ruff: noqa: T201

import sys
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation


@pytest.fixture
def bayer_op_cpu() -> BayerFilterOperation:
    """Create a CPU Bayer filter operation instance."""
    return BayerFilterOperation()


@pytest.fixture
def bayer_op_gpu() -> BayerFilterGPUOperation:
    """Create a GPU Bayer filter operation instance."""
    return BayerFilterGPUOperation()


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
    operation: BayerFilterOperation | BayerFilterGPUOperation,
    image: Image.Image,
    pattern: str = "RGGB",
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> float:
    """
    Time an operation with warmup and multiple runs.

    Args:
        operation: The Bayer filter operation to time.
        image: The input image.
        pattern: Bayer pattern to use.
        warmup_runs: Number of warmup iterations.
        timed_runs: Number of timed iterations.

    Returns:
        Average execution time in seconds.

    """
    params = {"pattern": pattern}

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
class TestBayerFilterPerformance:
    """Performance benchmarks comparing GPU and CPU implementations."""

    def test_gpu_baseline_small_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_gpu: BayerFilterGPUOperation,
    ) -> None:
        """Baseline test showing GPU overhead on small images."""
        image = create_test_image((512, 512))

        cpu_time = time_operation(bayer_op_cpu, image)
        gpu_time = time_operation(bayer_op_gpu, image)

        print(f"\nSmall image (512x512)")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        if gpu_time < cpu_time:
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            print(f"Overhead: {gpu_time / cpu_time:.2f}x (expected for small images)")

        # GPU has initialization overhead, but should still complete in reasonable time
        assert (
            gpu_time < 1.0
        ), f"GPU should complete small image in reasonable time, got {gpu_time:.4f}s"

    def test_gpu_performance_medium_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_gpu: BayerFilterGPUOperation,
    ) -> None:
        """Test GPU performance on medium image (overhead still dominates)."""
        image = create_test_image((1024, 1024))

        cpu_time = time_operation(bayer_op_cpu, image)
        gpu_time = time_operation(bayer_op_gpu, image)

        print(f"\nMedium image (1024x1024)")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        if gpu_time < cpu_time:
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            print(f"Overhead: {gpu_time / cpu_time:.2f}x (overhead still dominates)")

        # GPU should complete in reasonable time
        assert (
            gpu_time < 1.0
        ), f"GPU should complete in reasonable time, got {gpu_time:.4f}s"

    def test_gpu_competitive_large_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_gpu: BayerFilterGPUOperation,
    ) -> None:
        """Test GPU becomes competitive on large images."""
        image = create_test_image((2048, 2048))

        cpu_time = time_operation(bayer_op_cpu, image)
        gpu_time = time_operation(bayer_op_gpu, image)

        print(f"\nLarge image (2048x2048)")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        if gpu_time < cpu_time:
            print(f"Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            ratio = gpu_time / cpu_time
            print(f"Ratio: {ratio:.2f}x (becoming competitive)")

        # GPU should be reasonably close to CPU or faster
        assert (
            gpu_time < cpu_time * 2.0
        ), f"GPU should be competitive with CPU on large images"

    def test_gpu_faster_xlarge_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_gpu: BayerFilterGPUOperation,
    ) -> None:
        """Test GPU performance on extra-large image - MUST be faster than CPU."""
        image = create_test_image((4096, 4096))

        cpu_time = time_operation(bayer_op_cpu, image, warmup_runs=1, timed_runs=3)
        gpu_time = time_operation(bayer_op_gpu, image, warmup_runs=1, timed_runs=3)

        print(f"\nExtra-large image (4096x4096)")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

        # GPU MUST be faster on extra-large images - this is where GPU wins
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) MUST be faster than "
            f"CPU ({cpu_time:.4f}s) on extra-large images"
        )

        # Expect at least 1.5x speedup on extra-large images
        speedup = cpu_time / gpu_time
        assert speedup >= 1.5, (
            f"GPU should show at least 1.5x speedup on extra-large images, "
            f"got {speedup:.2f}x"
        )

    def test_gpu_scales_better_with_image_size(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_gpu: BayerFilterGPUOperation,
    ) -> None:
        """Test that GPU scales better than CPU as image size increases."""
        small_image = create_test_image((512, 512))
        large_image = create_test_image((2048, 2048))

        # Small image
        cpu_time_small = time_operation(bayer_op_cpu, small_image)
        gpu_time_small = time_operation(bayer_op_gpu, small_image)

        # Large image
        cpu_time_large = time_operation(bayer_op_cpu, large_image)
        gpu_time_large = time_operation(bayer_op_gpu, large_image)

        cpu_scaling = cpu_time_large / cpu_time_small
        gpu_scaling = gpu_time_large / gpu_time_small

        print(f"\nScaling test")
        print(f"Small (512x512): CPU={cpu_time_small:.4f}s, GPU={gpu_time_small:.4f}s")
        print(
            f"Large (2048x2048): CPU={cpu_time_large:.4f}s, GPU={gpu_time_large:.4f}s"
        )
        print(f"CPU scaling factor: {cpu_scaling:.2f}x")
        print(f"GPU scaling factor: {gpu_scaling:.2f}x")

        # GPU should scale better (lower scaling factor) as size increases
        # This shows GPU overhead is amortized better
        assert gpu_scaling <= cpu_scaling * 1.2, (
            f"GPU should scale as well or better than CPU. "
            f"GPU scaling: {gpu_scaling:.2f}x, CPU scaling: {cpu_scaling:.2f}x"
        )

    def test_all_patterns_perform_similarly_on_gpu(
        self, bayer_op_gpu: BayerFilterGPUOperation
    ) -> None:
        """Test that all Bayer patterns have similar GPU performance."""
        image = create_test_image((1024, 1024))
        patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]

        times = {}
        for pattern in patterns:
            times[pattern] = time_operation(bayer_op_gpu, image, pattern=pattern)

        print(f"\nPattern performance comparison (1024x1024)")
        for pattern, time_val in times.items():
            print(f"{pattern}: {time_val:.4f}s")

        # All patterns should have similar performance (within 20%)
        min_time = min(times.values())
        max_time = max(times.values())
        variation = (max_time - min_time) / min_time

        assert variation < 0.2, (
            f"Pattern performance should be consistent. "
            f"Variation {variation:.1%} exceeds 20% threshold"
        )

    def test_gpu_overhead_acceptable(
        self, bayer_op_gpu: BayerFilterGPUOperation
    ) -> None:
        """Test that GPU initialization overhead is acceptable."""
        # Very small image where CPU might be competitive
        image = create_test_image((100, 100))

        # Time single run (including any initialization)
        start = time.perf_counter()
        bayer_op_gpu.apply(image, {"pattern": "RGGB"})
        end = time.perf_counter()

        single_run_time = end - start

        print(f"\nGPU overhead test (100x100)")
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

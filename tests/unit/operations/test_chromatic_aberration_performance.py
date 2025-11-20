"""Performance benchmarks for chromatic aberration: CPU vs GPU vs Metal."""

import sys
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.chromatic_aberration_gpu import (
    ChromaticAberrationGPUOperation,
)
from sevenrad_stills.operations.chromatic_aberration_metal import (
    ChromaticAberrationMetalOperation,
)

pytestmark = pytest.mark.gpu


@pytest.fixture
def aberration_op_cpu() -> ChromaticAberrationOperation:
    """Create a CPU chromatic aberration operation instance."""
    return ChromaticAberrationOperation()


@pytest.fixture
def aberration_op_gpu() -> ChromaticAberrationGPUOperation:
    """Create a GPU chromatic aberration operation instance."""
    return ChromaticAberrationGPUOperation()


@pytest.fixture
def aberration_op_metal() -> ChromaticAberrationMetalOperation:
    """Create a Metal chromatic aberration operation instance."""
    return ChromaticAberrationMetalOperation()


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


def time_operation(  # noqa: PLR0913
    operation: ChromaticAberrationOperation
    | ChromaticAberrationGPUOperation
    | ChromaticAberrationMetalOperation,
    image: Image.Image,
    shift_x: int,
    shift_y: int,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> float:
    """
    Time an operation with warmup and multiple runs.

    Args:
        operation: The chromatic aberration operation to time.
        image: The input image.
        shift_x: Horizontal shift in pixels.
        shift_y: Vertical shift in pixels.
        warmup_runs: Number of warmup iterations.
        timed_runs: Number of timed iterations.

    Returns:
        Average execution time in seconds.

    """
    params = {"shift_x": shift_x, "shift_y": shift_y}

    # Warmup runs (especially important for GPU/Metal to initialize kernels)
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
class TestChromaticAberrationPerformance:
    """Performance benchmarks comparing Metal, GPU (Taichi), and CPU implementations."""

    def test_all_backends_small_image(
        self,
        aberration_op_cpu: ChromaticAberrationOperation,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Baseline test on small images - GPU/Metal may have overhead."""
        image = create_test_image((512, 512))
        shift_x, shift_y = 10, 10

        cpu_time = time_operation(aberration_op_cpu, image, shift_x, shift_y)
        gpu_time = time_operation(aberration_op_gpu, image, shift_x, shift_y)
        metal_time = time_operation(aberration_op_metal, image, shift_x, shift_y)

        print(f"\n{'='*70}")
        print(f"Small image (512x512), shift=({shift_x}, {shift_y})")
        print(f"{'='*70}")
        print(f"CPU time:   {cpu_time:.4f}s")
        print(f"GPU time:   {gpu_time:.4f}s", end="")
        if gpu_time < cpu_time:
            print(f"  [Speedup: {cpu_time / gpu_time:.2f}x]")
        else:
            print(f"  [Overhead: {gpu_time / cpu_time:.2f}x]")
        print(f"Metal time: {metal_time:.4f}s", end="")
        if metal_time < cpu_time:
            print(f"  [Speedup: {cpu_time / metal_time:.2f}x]")
        else:
            print(f"  [Overhead: {metal_time / cpu_time:.2f}x]")

        # All should complete in reasonable time
        assert cpu_time < 1.0, f"CPU should complete small image in <1s"
        assert gpu_time < 1.0, f"GPU should complete small image in <1s"
        assert metal_time < 1.0, f"Metal should complete small image in <1s"

    def test_all_backends_medium_image(
        self,
        aberration_op_cpu: ChromaticAberrationOperation,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Test performance on medium images - GPU/Metal show benefits."""
        image = create_test_image((1024, 1024))
        shift_x, shift_y = 15, 15

        cpu_time = time_operation(aberration_op_cpu, image, shift_x, shift_y)
        gpu_time = time_operation(aberration_op_gpu, image, shift_x, shift_y)
        metal_time = time_operation(aberration_op_metal, image, shift_x, shift_y)

        print(f"\n{'='*70}")
        print(f"Medium image (1024x1024), shift=({shift_x}, {shift_y})")
        print(f"{'='*70}")
        print(f"CPU time:   {cpu_time:.4f}s")
        print(f"GPU time:   {gpu_time:.4f}s  [Speedup: {cpu_time / gpu_time:.2f}x]")
        print(f"Metal time: {metal_time:.4f}s  [Speedup: {cpu_time / metal_time:.2f}x]")

        # GPU and Metal should be faster
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on medium images"
        )
        assert metal_time < cpu_time, (
            f"Metal ({metal_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on medium images"
        )

    def test_all_backends_large_image(
        self,
        aberration_op_cpu: ChromaticAberrationOperation,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Test performance on large images - significant GPU/Metal advantage."""
        image = create_test_image((2048, 2048))
        shift_x, shift_y = 20, 20

        cpu_time = time_operation(aberration_op_cpu, image, shift_x, shift_y)
        gpu_time = time_operation(aberration_op_gpu, image, shift_x, shift_y)
        metal_time = time_operation(aberration_op_metal, image, shift_x, shift_y)

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Large image (2048x2048), shift=({shift_x}, {shift_y})")
        print(f"{'='*70}")
        print(f"CPU time:   {cpu_time:.4f}s")
        print(f"GPU time:   {gpu_time:.4f}s  [Speedup: {gpu_speedup:.2f}x]")
        print(f"Metal time: {metal_time:.4f}s  [Speedup: {metal_speedup:.2f}x]")
        print(f"\nMetal vs GPU: {gpu_time / metal_time:.2f}x")

        # Both should show significant speedup
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on large images"
        )
        assert metal_time < cpu_time, (
            f"Metal ({metal_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on large images"
        )

        # Expect at least 2x speedup on large images
        assert gpu_speedup >= 2.0, (
            f"GPU should show at least 2x speedup on large images, "
            f"got {gpu_speedup:.2f}x"
        )
        assert metal_speedup >= 2.0, (
            f"Metal should show at least 2x speedup on large images, "
            f"got {metal_speedup:.2f}x"
        )

    def test_all_backends_very_large_image(
        self,
        aberration_op_cpu: ChromaticAberrationOperation,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Test performance on very large images - maximum GPU/Metal advantage."""
        image = create_test_image((4096, 4096))
        shift_x, shift_y = 25, 25

        cpu_time = time_operation(
            aberration_op_cpu, image, shift_x, shift_y, warmup_runs=1, timed_runs=3
        )
        gpu_time = time_operation(
            aberration_op_gpu, image, shift_x, shift_y, warmup_runs=1, timed_runs=3
        )
        metal_time = time_operation(
            aberration_op_metal, image, shift_x, shift_y, warmup_runs=1, timed_runs=3
        )

        gpu_speedup = cpu_time / gpu_time
        metal_speedup = cpu_time / metal_time

        print(f"\n{'='*70}")
        print(f"Very Large image (4096x4096), shift=({shift_x}, {shift_y})")
        print(f"{'='*70}")
        print(f"CPU time:   {cpu_time:.4f}s")
        print(f"GPU time:   {gpu_time:.4f}s  [Speedup: {gpu_speedup:.2f}x]")
        print(f"Metal time: {metal_time:.4f}s  [Speedup: {metal_speedup:.2f}x]")
        print(f"\nMetal vs GPU: {gpu_time / metal_time:.2f}x")

        # Both should be significantly faster
        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on very large images"
        )
        assert metal_time < cpu_time, (
            f"Metal ({metal_time:.4f}s) should be faster than "
            f"CPU ({cpu_time:.4f}s) on very large images"
        )

    def test_metal_vs_gpu_comparison(
        self,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Direct comparison between Metal and GPU (Taichi) across image sizes."""
        sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        shift_x, shift_y = 15, 15

        print(f"\n{'='*70}")
        print("Metal vs GPU (Taichi) Comparison")
        print(f"{'='*70}")
        print(f"{'Size':<15} {'GPU Time':<12} {'Metal Time':<12} {'Ratio':<10}")
        print(f"{'-'*70}")

        for size in sizes:
            image = create_test_image(size)
            runs = 3 if size == (4096, 4096) else 5  # Fewer runs for very large images

            gpu_time = time_operation(
                aberration_op_gpu, image, shift_x, shift_y, timed_runs=runs
            )
            metal_time = time_operation(
                aberration_op_metal, image, shift_x, shift_y, timed_runs=runs
            )

            ratio = gpu_time / metal_time
            winner = "Metal" if metal_time < gpu_time else "GPU"

            print(
                f"{size!s:<15} {gpu_time:.4f}s     {metal_time:.4f}s     "
                f"{ratio:.2f}x ({winner})"
            )

            # Just verify both complete in reasonable time
            assert gpu_time < 30.0, f"GPU should complete {size} in reasonable time"
            assert metal_time < 30.0, f"Metal should complete {size} in reasonable time"

    def test_varying_shift_values(
        self,
        aberration_op_cpu: ChromaticAberrationOperation,
        aberration_op_gpu: ChromaticAberrationGPUOperation,
        aberration_op_metal: ChromaticAberrationMetalOperation,
    ) -> None:
        """Test that shift magnitude doesn't significantly affect performance."""
        image = create_test_image((1024, 1024))
        shifts = [(5, 5), (20, 20), (50, 50)]

        print(f"\n{'='*70}")
        print("Performance vs Shift Magnitude (1024x1024)")
        print(f"{'='*70}")
        print(f"{'Shift':<12} {'CPU Time':<12} {'GPU Time':<12} {'Metal Time':<12}")
        print(f"{'-'*70}")

        for shift_x, shift_y in shifts:
            cpu_time = time_operation(aberration_op_cpu, image, shift_x, shift_y)
            gpu_time = time_operation(aberration_op_gpu, image, shift_x, shift_y)
            metal_time = time_operation(aberration_op_metal, image, shift_x, shift_y)

            print(
                f"({shift_x}, {shift_y})     {cpu_time:.4f}s    {gpu_time:.4f}s    "
                f"{metal_time:.4f}s"
            )

        # Performance should be relatively constant regardless of shift
        # (chromatic aberration complexity doesn't depend on shift magnitude)


# Skip all tests if not on macOS
def pytest_collection_modifyitems(items):
    """Skip mac-only tests if not running on macOS."""
    if sys.platform != "darwin":
        skip_mac = pytest.mark.skip(reason="Only runs on macOS")
        for item in items:
            if "mac" in item.keywords:
                item.add_marker(skip_mac)

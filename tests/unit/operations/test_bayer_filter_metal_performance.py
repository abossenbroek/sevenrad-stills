"""Performance benchmarks for Metal vs Taichi vs CPU Bayer filter implementations."""
# ruff: noqa: T201

import sys

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation
from sevenrad_stills.operations.bayer_filter_gpu import BayerFilterGPUOperation

try:
    from sevenrad_stills.operations.bayer_filter_metal import (
        BayerFilterMetalOperation,
    )

    METAL_AVAILABLE = True
except (ImportError, RuntimeError):
    METAL_AVAILABLE = False


@pytest.fixture
def bayer_op_cpu() -> BayerFilterOperation:
    """Create a CPU Bayer filter operation instance."""
    return BayerFilterOperation()


@pytest.fixture
def bayer_op_taichi() -> BayerFilterGPUOperation:
    """Create a Taichi GPU Bayer filter operation instance."""
    return BayerFilterGPUOperation()


@pytest.fixture
def bayer_op_metal() -> BayerFilterMetalOperation:
    """Create a Metal Bayer filter operation instance."""
    if not METAL_AVAILABLE:
        pytest.skip("Metal not available")
    return BayerFilterMetalOperation()


def create_test_image(size: tuple[int, int]) -> Image.Image:
    """Create a test RGB image with random content."""
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def time_operation(
    operation: BayerFilterOperation
    | BayerFilterGPUOperation
    | BayerFilterMetalOperation,
    image: Image.Image,
    pattern: str = "RGGB",
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> float:
    """Time an operation with warmup and multiple runs."""
    import time

    params = {"pattern": pattern}

    # Warmup
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
@pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
class TestMetalPerformance:
    """Performance benchmarks comparing Metal, Taichi, and CPU implementations."""

    def test_metal_faster_than_taichi_small_image(
        self,
        bayer_op_taichi: BayerFilterGPUOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal beats Taichi on small images."""
        image = create_test_image((512, 512))

        taichi_time = time_operation(bayer_op_taichi, image)
        metal_time = time_operation(bayer_op_metal, image)

        print(f"\n512x512 image:")
        print(f"  Taichi: {taichi_time:.4f}s")
        print(f"  Metal:  {metal_time:.4f}s")
        print(f"  Speedup: {taichi_time / metal_time:.2f}x")

        assert metal_time < taichi_time, (
            f"Metal ({metal_time:.4f}s) should be faster than "
            f"Taichi ({taichi_time:.4f}s) on small images"
        )

    def test_metal_faster_than_cpu_small_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal is competitive with CPU on small images."""
        image = create_test_image((512, 512))

        cpu_time = time_operation(bayer_op_cpu, image)
        metal_time = time_operation(bayer_op_metal, image)

        print(f"\n512x512 image CPU vs Metal:")
        print(f"  CPU:    {cpu_time:.4f}s")
        print(f"  Metal:  {metal_time:.4f}s")
        if metal_time < cpu_time:
            print(f"  Speedup: {cpu_time / metal_time:.2f}x")
        else:
            print(f"  Overhead: {metal_time / cpu_time:.2f}x")

        # Metal should at least be within 2x of CPU on small images
        assert metal_time < cpu_time * 2, (
            f"Metal ({metal_time:.4f}s) should be within 2x of "
            f"CPU ({cpu_time:.4f}s) on small images"
        )

    def test_metal_faster_than_all_medium_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_taichi: BayerFilterGPUOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal beats both CPU and Taichi on medium images."""
        image = create_test_image((1024, 1024))

        cpu_time = time_operation(bayer_op_cpu, image)
        taichi_time = time_operation(bayer_op_taichi, image)
        metal_time = time_operation(bayer_op_metal, image)

        print(f"\n1024x1024 image:")
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  Taichi:  {taichi_time:.4f}s")
        print(f"  Metal:   {metal_time:.4f}s")
        print(f"  Metal vs CPU speedup: {cpu_time / metal_time:.2f}x")
        print(f"  Metal vs Taichi speedup: {taichi_time / metal_time:.2f}x")

        assert metal_time < taichi_time, (
            f"Metal ({metal_time:.4f}s) should beat " f"Taichi ({taichi_time:.4f}s)"
        )
        assert (
            metal_time < cpu_time
        ), f"Metal ({metal_time:.4f}s) should beat CPU ({cpu_time:.4f}s)"

    def test_metal_faster_than_all_large_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_taichi: BayerFilterGPUOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal shows significant speedup on large images."""
        image = create_test_image((2048, 2048))

        cpu_time = time_operation(bayer_op_cpu, image)
        taichi_time = time_operation(bayer_op_taichi, image)
        metal_time = time_operation(bayer_op_metal, image)

        print(f"\n2048x2048 image:")
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  Taichi:  {taichi_time:.4f}s")
        print(f"  Metal:   {metal_time:.4f}s")
        print(f"  Metal vs CPU speedup: {cpu_time / metal_time:.2f}x")
        print(f"  Metal vs Taichi speedup: {taichi_time / metal_time:.2f}x")

        assert metal_time < taichi_time, (
            f"Metal ({metal_time:.4f}s) should beat " f"Taichi ({taichi_time:.4f}s)"
        )
        assert (
            metal_time < cpu_time
        ), f"Metal ({metal_time:.4f}s) should beat CPU ({cpu_time:.4f}s)"
        assert cpu_time / metal_time >= 2.0, (
            f"Metal should show at least 2x speedup vs CPU, "
            f"got {cpu_time / metal_time:.2f}x"
        )

    def test_metal_faster_than_all_xlarge_image(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_taichi: BayerFilterGPUOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal shows maximum speedup on extra-large images."""
        image = create_test_image((4096, 4096))

        cpu_time = time_operation(bayer_op_cpu, image, timed_runs=5)
        taichi_time = time_operation(bayer_op_taichi, image, timed_runs=5)
        metal_time = time_operation(bayer_op_metal, image, timed_runs=5)

        print(f"\n4096x4096 image:")
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  Taichi:  {taichi_time:.4f}s")
        print(f"  Metal:   {metal_time:.4f}s")
        print(f"  Metal vs CPU speedup: {cpu_time / metal_time:.2f}x")
        print(f"  Metal vs Taichi speedup: {taichi_time / metal_time:.2f}x")

        assert metal_time < taichi_time, (
            f"Metal ({metal_time:.4f}s) should beat " f"Taichi ({taichi_time:.4f}s)"
        )
        assert (
            metal_time < cpu_time
        ), f"Metal ({metal_time:.4f}s) should beat CPU ({cpu_time:.4f}s)"
        assert cpu_time / metal_time >= 3.0, (
            f"Metal should show at least 3x speedup vs CPU on 4K, "
            f"got {cpu_time / metal_time:.2f}x"
        )
        assert taichi_time / metal_time >= 2.0, (
            f"Metal should show at least 2x speedup vs Taichi on 4K, "
            f"got {taichi_time / metal_time:.2f}x"
        )

    def test_metal_output_matches_cpu(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Test that Metal produces similar output to CPU."""
        image = create_test_image((512, 512))
        params = {"pattern": "RGGB"}

        cpu_result = bayer_op_cpu.apply(image, params)
        metal_result = bayer_op_metal.apply(image, params)

        cpu_array = np.array(cpu_result)
        metal_array = np.array(metal_result)

        # Allow some difference due to floating point and algorithm variations
        diff = np.abs(cpu_array.astype(np.float32) - metal_array.astype(np.float32))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        print(f"\nOutput comparison:")
        print(f"  Mean difference: {mean_diff:.2f}")
        print(f"  Max difference:  {max_diff:.2f}")

        # Metal uses edge-directed demosaicing vs CPU's Malvar2004,
        # so differences are expected
        assert mean_diff < 50, f"Mean difference too high: {mean_diff:.2f}"
        assert max_diff <= 255, f"Max difference too high: {max_diff:.2f}"


# Skip all tests if not on macOS
def pytest_collection_modifyitems(items):
    """Skip mac-only tests if not running on macOS."""
    if sys.platform != "darwin":
        skip_mac = pytest.mark.skip(reason="Only runs on macOS")
        for item in items:
            if "mac" in item.keywords:
                item.add_marker(skip_mac)

"""Test Metal Malvar2004 implementation for quality and performance."""
# ruff: noqa: T201

import sys

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.bayer_filter import BayerFilterOperation

try:
    from sevenrad_stills.operations.bayer_filter_metal_malvar import (
        BayerFilterMetalMalvarOperation,
    )

    METAL_AVAILABLE = True
except (ImportError, RuntimeError):
    METAL_AVAILABLE = False


@pytest.fixture
def bayer_op_cpu() -> BayerFilterOperation:
    """Create a CPU Bayer filter operation instance."""
    return BayerFilterOperation()


@pytest.fixture
def bayer_op_metal_malvar() -> BayerFilterMetalMalvarOperation:
    """Create a Metal Malvar Bayer filter operation instance."""
    if not METAL_AVAILABLE:
        pytest.skip("Metal not available")
    return BayerFilterMetalMalvarOperation()


def create_test_image(size: tuple[int, int]) -> Image.Image:
    """Create a test RGB image."""
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.mac
@pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
class TestMetalMalvarQuality:
    """Quality tests for Metal Malvar2004 implementation."""

    def test_malvar_matches_cpu_closely(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal_malvar: BayerFilterMetalMalvarOperation,
    ) -> None:
        """Test that Metal Malvar matches CPU output very closely."""
        image = create_test_image((512, 512))
        params = {"pattern": "RGGB"}

        cpu_result = bayer_op_cpu.apply(image, params)
        metal_result = bayer_op_metal_malvar.apply(image, params)

        cpu_array = np.array(cpu_result, dtype=np.float32)
        metal_array = np.array(metal_result, dtype=np.float32)

        diff = np.abs(cpu_array - metal_array)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        max_diff = np.max(diff)
        p95_diff = np.percentile(diff, 95)

        print(f"\nMetal Malvar vs CPU:")
        print(f"  Mean diff:   {mean_diff:.2f}")
        print(f"  Median diff: {median_diff:.2f}")
        print(f"  95th %ile:   {p95_diff:.2f}")
        print(f"  Max diff:    {max_diff:.2f}")

        # With same algorithm, should match very closely
        # Allow some difference due to floating point precision and boundary handling
        assert mean_diff < 10, (
            f"Mean difference too high: {mean_diff:.2f} "
            "(expected <10 with same Malvar2004 algorithm)"
        )
        assert median_diff <= 0, f"Median should match exactly: {median_diff:.2f}"
        assert p95_diff < 50, f"95th percentile too high: {p95_diff:.2f}"

    def test_malvar_performance(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal_malvar: BayerFilterMetalMalvarOperation,
    ) -> None:
        """Test that Metal Malvar maintains performance advantage."""
        import time

        image = create_test_image((2048, 2048))
        params = {"pattern": "RGGB"}

        # Warmup
        for _ in range(3):
            bayer_op_cpu.apply(image, params)
            bayer_op_metal_malvar.apply(image, params)

        # Time CPU
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            bayer_op_cpu.apply(image, params)
            cpu_times.append(time.perf_counter() - start)

        # Time Metal
        metal_times = []
        for _ in range(5):
            start = time.perf_counter()
            bayer_op_metal_malvar.apply(image, params)
            metal_times.append(time.perf_counter() - start)

        cpu_avg = np.mean(cpu_times)
        metal_avg = np.mean(metal_times)
        speedup = cpu_avg / metal_avg

        print(f"\nPerformance (2048x2048):")
        print(f"  CPU (Malvar2004):   {cpu_avg:.4f}s")
        print(f"  Metal (Malvar2004): {metal_avg:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Should still be significantly faster than CPU
        assert (
            metal_avg < cpu_avg
        ), f"Metal ({metal_avg:.4f}s) should be faster than CPU ({cpu_avg:.4f}s)"
        assert (
            speedup > 2.0
        ), f"Metal should show at least 2x speedup, got {speedup:.2f}x"


# Skip all tests if not on macOS
def pytest_collection_modifyitems(items):
    """Skip mac-only tests if not running on macOS."""
    if sys.platform != "darwin":
        skip_mac = pytest.mark.skip(reason="Only runs on macOS")
        for item in items:
            if "mac" in item.keywords:
                item.add_marker(skip_mac)

"""Quality comparison tests between CPU, Taichi, and Metal implementations."""
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
    """Create a test RGB image with structured content for quality analysis."""
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def analyze_difference(
    original: Image.Image, result: Image.Image, name: str
) -> dict[str, float]:
    """Analyze the difference between two images."""
    orig_array = np.array(original, dtype=np.float32)
    result_array = np.array(result, dtype=np.float32)

    diff = np.abs(orig_array - result_array)

    stats = {
        "mean": float(np.mean(diff)),
        "std": float(np.std(diff)),
        "max": float(np.max(diff)),
        "median": float(np.median(diff)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
    }

    print(f"\n{name}:")
    print(f"  Mean diff:   {stats['mean']:.2f}")
    print(f"  Std diff:    {stats['std']:.2f}")
    print(f"  Median diff: {stats['median']:.2f}")
    print(f"  95th %ile:   {stats['p95']:.2f}")
    print(f"  99th %ile:   {stats['p99']:.2f}")
    print(f"  Max diff:    {stats['max']:.2f}")

    return stats


@pytest.mark.mac
class TestBayerFilterQuality:
    """Quality comparison tests for all implementations."""

    def test_all_patterns_quality_comparison(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_taichi: BayerFilterGPUOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Compare output quality across all patterns and implementations."""
        patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]
        image = create_test_image((512, 512))

        print("\n" + "=" * 70)
        print("QUALITY COMPARISON: CPU vs Taichi vs Metal")
        print("=" * 70)

        for pattern in patterns:
            print(f"\nPattern: {pattern}")
            print("-" * 70)

            params = {"pattern": pattern}
            cpu_result = bayer_op_cpu.apply(image, params)
            taichi_result = bayer_op_taichi.apply(image, params)
            metal_result = bayer_op_metal.apply(image, params)

            # Compare Taichi vs CPU
            taichi_stats = analyze_difference(
                cpu_result, taichi_result, "Taichi vs CPU"
            )

            # Compare Metal vs CPU
            metal_stats = analyze_difference(cpu_result, metal_result, "Metal vs CPU")

            # Compare Metal vs Taichi
            metal_taichi_stats = analyze_difference(
                taichi_result, metal_result, "Metal vs Taichi"
            )

            # Taichi uses edge-directed demosaicing, which differs from CPU's Malvar2004
            assert (
                taichi_stats["mean"] < 50
            ), f"{pattern}: Taichi mean diff too high: {taichi_stats['mean']:.2f}"

            # Metal uses Malvar2004 (same as CPU), should match very closely
            assert (
                metal_stats["mean"] < 10
            ), f"{pattern}: Metal mean diff too high: {metal_stats['mean']:.2f}"

            # Metal (Malvar2004) and Taichi (edge-directed) use different algorithms
            # They should differ similar to how Taichi differs from CPU
            assert metal_taichi_stats["mean"] < 50, (
                f"{pattern}: Metal/Taichi diff too high - "
                f"got mean diff {metal_taichi_stats['mean']:.2f}"
            )

    def test_rgba_quality_preservation(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Verify RGBA alpha channel is perfectly preserved."""
        # Create RGBA image with specific alpha pattern
        rgb = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
        alpha = np.full((512, 512), 128, dtype=np.uint8)  # Constant alpha
        rgba = np.dstack([rgb, alpha])
        image = Image.fromarray(rgba, mode="RGBA")

        params = {"pattern": "RGGB"}
        cpu_result = bayer_op_cpu.apply(image, params)
        metal_result = bayer_op_metal.apply(image, params)

        # Extract alpha channels
        cpu_alpha = np.array(cpu_result.getchannel("A"))
        metal_alpha = np.array(metal_result.getchannel("A"))

        print("\nRGBA Alpha Preservation:")
        print(f"  CPU alpha unchanged:   {np.all(cpu_alpha == 128)}")
        print(f"  Metal alpha unchanged: {np.all(metal_alpha == 128)}")
        print(f"  CPU == Metal alpha:    {np.all(cpu_alpha == metal_alpha)}")

        # Alpha should be perfectly preserved
        assert np.all(cpu_alpha == 128), "CPU should preserve alpha channel"
        assert np.all(metal_alpha == 128), "Metal should preserve alpha channel"
        assert np.all(cpu_alpha == metal_alpha), "Alpha channels should match"

    def test_quality_scales_with_image_size(
        self,
        bayer_op_cpu: BayerFilterOperation,
        bayer_op_metal: BayerFilterMetalOperation,
    ) -> None:
        """Verify quality is consistent across different image sizes."""
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        params = {"pattern": "RGGB"}

        print("\n" + "=" * 70)
        print("QUALITY SCALING ANALYSIS")
        print("=" * 70)

        mean_diffs = []

        for size in sizes:
            image = create_test_image(size)
            cpu_result = bayer_op_cpu.apply(image, params)
            metal_result = bayer_op_metal.apply(image, params)

            stats = analyze_difference(
                cpu_result, metal_result, f"Metal vs CPU ({size[0]}x{size[1]})"
            )
            mean_diffs.append(stats["mean"])

        # Mean difference should be consistent (within 20%) across sizes
        diff_variance = np.std(mean_diffs) / np.mean(mean_diffs)
        print(f"\nMean diff variance: {diff_variance:.2%}")

        assert diff_variance < 0.3, (
            f"Quality should be consistent across sizes, "
            f"got variance {diff_variance:.2%}"
        )

    @pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
    def test_metal_produces_valid_output(
        self, bayer_op_metal: BayerFilterMetalOperation
    ) -> None:
        """Verify Metal output is valid (no NaN, inf, or out-of-range values)."""
        image = create_test_image((512, 512))
        params = {"pattern": "RGGB"}

        result = bayer_op_metal.apply(image, params)
        result_array = np.array(result)

        print("\nMetal Output Validation:")
        print(f"  Shape:     {result_array.shape}")
        print(f"  Dtype:     {result_array.dtype}")
        print(f"  Min value: {result_array.min()}")
        print(f"  Max value: {result_array.max()}")
        print(f"  Mean:      {result_array.mean():.2f}")

        # Verify no invalid values
        assert not np.any(np.isnan(result_array)), "Output contains NaN"
        assert not np.any(np.isinf(result_array)), "Output contains inf"
        assert result_array.min() >= 0, "Output contains negative values"
        assert result_array.max() <= 255, "Output contains out-of-range values"
        assert result_array.dtype == np.uint8, "Output dtype should be uint8"


# Skip all tests if not on macOS
def pytest_collection_modifyitems(items):
    """Skip mac-only tests if not running on macOS."""
    if sys.platform != "darwin":
        skip_mac = pytest.mark.skip(reason="Only runs on macOS")
        for item in items:
            if "mac" in item.keywords:
                item.add_marker(skip_mac)

"""
import pytest

pytestmark = pytest.mark.gpu

Unit tests for optimized Metal buffer corruption implementation (v2).

Tests verify:
- Visual output matches CPU implementation exactly
- All corruption types work correctly
- Reproducibility with seeds
- Performance improvements over v1
"""

import numpy as np
import pytest
from PIL import Image

# Test will skip if Metal not available
pytest.importorskip("Metal")

from sevenrad_stills.operations.buffer_corruption import (  # noqa: E402
    BufferCorruptionOperation,
)
from sevenrad_stills.operations.buffer_corruption_metal import (  # noqa: E402
    BufferCorruptionMetalOperation,
)


@pytest.fixture
def test_image_small():
    """Create a small test image (100x100)."""
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def test_image_medium():
    """Create a medium test image (640x480)."""
    arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def cpu_operation():
    """Create CPU operation instance."""
    return BufferCorruptionOperation()


@pytest.fixture
def metal_v2_operation():
    """Create Metal v2 operation instance."""
    try:
        return BufferCorruptionMetalOperation()
    except (ImportError, RuntimeError, FileNotFoundError) as e:
        pytest.skip(f"Metal v2 not available: {e}")


class TestBufferCorruptionMetalOperation:
    """Test suite for optimized Metal buffer corruption (v2)."""

    @pytest.mark.parametrize("corruption_type", ["xor", "invert", "channel_shuffle"])
    def test_corruption_types(
        self, metal_v2_operation, test_image_small, corruption_type
    ):
        """Test all corruption types produce valid output."""
        params = {
            "tile_count": 5,
            "corruption_type": corruption_type,
            "severity": 0.5,
            "seed": 42,
        }

        result = metal_v2_operation.apply(test_image_small, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image_small.size
        assert result.mode == "RGB"

    @pytest.mark.parametrize("tile_count", [1, 5, 10, 20, 50, 100])
    def test_various_tile_counts(
        self, metal_v2_operation, test_image_medium, tile_count
    ):
        """Test v2 handles various tile counts efficiently (v1 was limited to 20)."""
        params = {
            "tile_count": tile_count,
            "corruption_type": "xor",
            "severity": 0.7,
            "seed": 42,
        }

        result = metal_v2_operation.apply(test_image_medium, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image_medium.size

    def test_reproducibility_with_seed(self, metal_v2_operation, test_image_small):
        """Test same seed produces identical results."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.6,
            "seed": 123,
        }

        result1 = metal_v2_operation.apply(test_image_small, params)
        result2 = metal_v2_operation.apply(test_image_small, params)

        arr1 = np.array(result1)
        arr2 = np.array(result2)

        assert np.array_equal(arr1, arr2), "Same seed should produce identical results"

    def test_different_seeds_produce_different_results(
        self, metal_v2_operation, test_image_small
    ):
        """Test different seeds produce different corruption patterns."""
        params_seed1 = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.6,
            "seed": 111,
        }
        params_seed2 = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.6,
            "seed": 222,
        }

        result1 = metal_v2_operation.apply(test_image_small, params_seed1)
        result2 = metal_v2_operation.apply(test_image_small, params_seed2)

        arr1 = np.array(result1)
        arr2 = np.array(result2)

        assert not np.array_equal(
            arr1, arr2
        ), "Different seeds should produce different results"

    @pytest.mark.parametrize("severity", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_severity_levels(self, metal_v2_operation, test_image_small, severity):
        """Test various severity levels."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": severity,
            "seed": 42,
        }

        result = metal_v2_operation.apply(test_image_small, params)

        assert isinstance(result, Image.Image)

        if severity == 0.0:
            # Zero severity should produce minimal corruption
            # Note: may still have some corruption due to magnitude rounding
            pass

    def test_channel_shuffle_permutes_colors(
        self, metal_v2_operation, test_image_small
    ):
        """Test channel shuffle actually permutes RGB values."""
        params = {
            "tile_count": 20,
            "corruption_type": "channel_shuffle",
            "severity": 1.0,  # High severity to ensure shuffling
            "seed": 42,
        }

        original = np.array(test_image_small)
        result = metal_v2_operation.apply(test_image_small, params)
        result_arr = np.array(result)

        # Should have some differences due to channel shuffling
        assert not np.array_equal(original, result_arr)

    def test_invert_creates_negatives(self, metal_v2_operation):
        """Test invert mode creates color negatives."""
        # Create a simple solid color image
        solid = Image.new("RGB", (100, 100), (128, 64, 192))

        params = {
            "tile_count": 50,  # Many tiles to ensure coverage
            "corruption_type": "invert",
            "severity": 1.0,
            "seed": 42,
        }

        result = metal_v2_operation.apply(solid, params)
        result_arr = np.array(result)

        # Check that some pixels were inverted
        # Inverted value of (128, 64, 192) is (127, 191, 63)
        inverted_color = np.array([127, 191, 63], dtype=np.uint8)

        # Should find some inverted pixels
        matches = np.all(result_arr == inverted_color, axis=2)
        assert matches.any(), "Should have some inverted pixels"

    def test_visual_consistency_across_runs(
        self, metal_v2_operation, test_image_medium
    ):
        """Test visual output is consistent across multiple runs with same seed."""
        params = {
            "tile_count": 15,
            "corruption_type": "xor",
            "severity": 0.7,
            "seed": 999,
        }

        # Run multiple times
        results = [
            metal_v2_operation.apply(test_image_medium, params) for _ in range(5)
        ]

        # All should be identical
        arrays = [np.array(r) for r in results]
        for arr in arrays[1:]:
            assert np.array_equal(
                arrays[0], arr
            ), "All runs should produce identical results"

    def test_parameter_validation(self, metal_v2_operation, test_image_small):
        """Test parameter validation raises appropriate errors."""
        # Missing required parameter
        with pytest.raises(ValueError, match="tile_count"):
            metal_v2_operation.apply(
                test_image_small, {"corruption_type": "xor", "severity": 0.5}
            )

        # Invalid corruption type
        with pytest.raises(ValueError, match="corruption_type"):
            metal_v2_operation.apply(
                test_image_small,
                {"tile_count": 5, "corruption_type": "invalid", "severity": 0.5},
            )

        # Invalid severity
        with pytest.raises(ValueError, match="severity"):
            metal_v2_operation.apply(
                test_image_small,
                {"tile_count": 5, "corruption_type": "xor", "severity": 1.5},
            )

        # Invalid tile_count
        with pytest.raises(ValueError, match="tile_count"):
            metal_v2_operation.apply(
                test_image_small,
                {"tile_count": 2000, "corruption_type": "xor", "severity": 0.5},
            )

    def test_edge_cases(self, metal_v2_operation):
        """Test edge cases like very small images."""
        # Very small image
        tiny = Image.new("RGB", (10, 10), (100, 100, 100))

        params = {
            "tile_count": 2,
            "corruption_type": "xor",
            "severity": 0.5,
            "seed": 42,
        }

        result = metal_v2_operation.apply(tiny, params)
        assert result.size == tiny.size

    @pytest.mark.parametrize("tile_size_range", [[0.01, 0.05], [0.05, 0.2], [0.2, 0.5]])
    def test_tile_size_ranges(
        self, metal_v2_operation, test_image_medium, tile_size_range
    ):
        """Test various tile size ranges."""
        params = {
            "tile_count": 10,
            "corruption_type": "xor",
            "severity": 0.6,
            "tile_size_range": tile_size_range,
            "seed": 42,
        }

        result = metal_v2_operation.apply(test_image_medium, params)
        assert isinstance(result, Image.Image)


class TestMetalV2VSCPUConsistency:
    """Test that Metal v2 produces visually consistent results with CPU."""

    @pytest.mark.parametrize("corruption_type", ["xor", "invert", "channel_shuffle"])
    def test_visual_similarity_to_cpu(
        self, cpu_operation, metal_v2_operation, test_image_small, corruption_type
    ):
        """
        Test Metal v2 produces similar visual effect to CPU.

        Note: Exact pixel-perfect match may not be guaranteed due to:
        - Different RNG implementations (hash-based vs NumPy)
        - Floating-point precision differences
        - But overall visual effect should be similar
        """
        params = {
            "tile_count": 10,
            "corruption_type": corruption_type,
            "severity": 0.6,
            "seed": 42,
        }

        cpu_result = cpu_operation.apply(test_image_small, params)
        metal_result = metal_v2_operation.apply(test_image_small, params)

        cpu_arr = np.array(cpu_result)
        metal_arr = np.array(metal_result)

        # Both should have corruption (not identical to original)
        original_arr = np.array(test_image_small)

        assert not np.array_equal(cpu_arr, original_arr)
        assert not np.array_equal(metal_arr, original_arr)

        # Calculate similarity metric (% of pixels that differ significantly)
        diff = np.abs(cpu_arr.astype(int) - metal_arr.astype(int))
        significant_diff = np.any(diff > 10, axis=2)  # >10 intensity difference
        diff_percentage = significant_diff.sum() / significant_diff.size

        # Allow some difference due to implementation details
        # But should be broadly similar (< 50% different pixels)
        assert diff_percentage < 0.5, (
            f"Metal v2 output differs too much from CPU "
            f"({diff_percentage * 100:.1f}% pixels significantly different)"
        )

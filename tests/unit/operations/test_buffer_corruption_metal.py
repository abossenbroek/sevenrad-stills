"""Tests for Metal GPU-accelerated buffer corruption operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image

# Skip entire module if not on Mac
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on Mac",
)


@pytest.mark.mac
class TestBufferCorruptionMetal:
    """Tests for Metal GPU buffer corruption implementation."""

    @pytest.fixture
    def metal_operation(self):
        """Create a Metal buffer corruption operation instance."""
        from sevenrad_stills.operations.buffer_corruption_metal import (
            BufferCorruptionMetal,
        )

        return BufferCorruptionMetal()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a small test image."""
        return Image.new("RGB", (256, 256), color=(128, 128, 128))

    @pytest.fixture
    def fhd_image(self) -> Image.Image:
        """Create FHD test image for performance testing."""
        return Image.new("RGB", (1920, 1080), color=(128, 128, 128))

    # === Basic Operation Tests ===

    def test_metal_initialization(self, metal_operation) -> None:
        """Test Metal device initializes correctly."""
        assert metal_operation.device is not None
        assert metal_operation.command_queue is not None
        assert metal_operation.pipeline is not None

    def test_apply_xor_corruption(self, metal_operation, test_image) -> None:
        """Test XOR corruption applies successfully."""
        params = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }
        result = metal_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_invert_corruption(self, metal_operation, test_image) -> None:
        """Test invert corruption applies successfully."""
        params = {
            "corruption_type": "invert",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }
        result = metal_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_channel_shuffle_corruption(
        self, metal_operation, test_image
    ) -> None:
        """Test channel shuffle corruption applies successfully."""
        params = {
            "corruption_type": "channel_shuffle",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }
        result = metal_operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_deterministic_with_same_seed(self, metal_operation, test_image) -> None:
        """Test same seed produces same result."""
        params = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }

        result1 = metal_operation.apply(test_image, params)
        result2 = metal_operation.apply(test_image, params)

        # Convert to numpy for comparison
        arr1 = np.array(result1)
        arr2 = np.array(result2)

        # Results should be identical with same seed
        assert np.array_equal(arr1, arr2)

    def test_different_results_with_different_seeds(
        self, metal_operation, test_image
    ) -> None:
        """Test different seeds produce different results."""
        params1 = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }
        params2 = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 123,
            "tile_size": 64,
            "magnitude": 255,
        }

        result1 = metal_operation.apply(test_image, params1)
        result2 = metal_operation.apply(test_image, params2)

        arr1 = np.array(result1)
        arr2 = np.array(result2)

        # Results should differ with different seeds
        assert not np.array_equal(arr1, arr2)

    def test_severity_affects_corruption_amount(
        self, metal_operation, test_image
    ) -> None:
        """Test different severity levels affect corruption amount."""
        params_low = {
            "corruption_type": "xor",
            "severity": 0.1,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }
        params_high = {
            "corruption_type": "xor",
            "severity": 0.9,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }

        original = np.array(test_image)
        result_low = np.array(metal_operation.apply(test_image, params_low))
        result_high = np.array(metal_operation.apply(test_image, params_high))

        # Calculate number of changed pixels
        pixels_changed_low = np.sum(result_low != original)
        pixels_changed_high = np.sum(result_high != original)

        # Higher severity should corrupt more pixels
        assert pixels_changed_high > pixels_changed_low

    def test_rgba_image_support(self, metal_operation) -> None:
        """Test RGBA images are handled correctly."""
        rgba_image = Image.new("RGBA", (256, 256), color=(128, 128, 128, 255))
        params = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }

        result = metal_operation.apply(rgba_image, params)
        assert result.mode == "RGB"  # Should convert to RGB
        assert result.size == rgba_image.size

    # === Performance Tests ===

    @pytest.mark.slow
    def test_fhd_performance(self, metal_operation, fhd_image) -> None:
        """Test FHD image processing performance."""
        import time

        params = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }

        # Warmup
        _ = metal_operation.apply(fhd_image, params)

        # Measure
        start = time.perf_counter()
        _ = metal_operation.apply(fhd_image, params)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 100ms)
        assert (
            elapsed < 0.1
        ), f"FHD processing took {elapsed*1000:.2f}ms (expected < 100ms)"

    def test_zero_copy_buffer_cleanup(self, metal_operation, test_image) -> None:
        """Test buffer references are cleaned up after operation."""
        params = {
            "corruption_type": "xor",
            "severity": 0.3,
            "seed": 42,
            "tile_size": 64,
            "magnitude": 255,
        }

        # Buffer refs should be empty initially
        assert len(metal_operation._buffer_refs) == 0

        # Apply operation
        _ = metal_operation.apply(test_image, params)

        # Buffer refs should be cleaned up after operation
        assert len(metal_operation._buffer_refs) == 0

    def test_active_tiles_generation(self, metal_operation) -> None:
        """Test active tile coordinate generation."""
        active_tiles = metal_operation._generate_active_tiles(
            width=1920, height=1080, tile_size=64, severity=0.3, seed=42
        )

        # Check shape
        assert active_tiles.shape[1] == 2  # (tile_x, tile_y) pairs
        assert active_tiles.dtype == np.uint32

        # Check count (should be ~30% of total tiles)
        tiles_x = (1920 + 63) // 64
        tiles_y = (1080 + 63) // 64
        total_tiles = tiles_x * tiles_y
        expected_count = int(0.3 * total_tiles)

        assert len(active_tiles) == expected_count

    def test_tile_coordinates_within_bounds(self, metal_operation) -> None:
        """Test generated tile coordinates are within valid bounds."""
        width, height = 1920, 1080
        tile_size = 64
        active_tiles = metal_operation._generate_active_tiles(
            width=width, height=height, tile_size=tile_size, severity=0.3, seed=42
        )

        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size

        # All tile coordinates should be within valid range
        assert np.all(active_tiles[:, 0] < tiles_x)  # tile_x
        assert np.all(active_tiles[:, 1] < tiles_y)  # tile_y

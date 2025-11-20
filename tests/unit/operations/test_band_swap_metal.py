"""Tests for Metal-accelerated band swap operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.band_swap import BandSwapOperation
from sevenrad_stills.operations.band_swap_metal import BandSwapMetalOperation

pytestmark = pytest.mark.gpu


@pytest.fixture
def band_swap_op_metal() -> BandSwapMetalOperation:
    """Create a Metal band swap operation instance."""
    return BandSwapMetalOperation()


@pytest.fixture
def band_swap_op_cpu() -> BandSwapOperation:
    """Create a CPU band swap operation instance for comparison."""
    return BandSwapOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a test image with distinct RGB values."""
    # Create an image where R=100, G=150, B=200 for easy verification
    return Image.new("RGB", (100, 100), color=(100, 150, 200))


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a test RGBA image with distinct values."""
    return Image.new("RGBA", (100, 100), color=(100, 150, 200, 255))


class TestBandSwapMetalValidation:
    """Test parameter validation for Metal band swap operation."""

    def test_operation_name(self, band_swap_op_metal: BandSwapMetalOperation) -> None:
        """Test operation has correct name."""
        assert band_swap_op_metal.name == "band_swap_metal"

    def test_valid_params(self, band_swap_op_metal: BandSwapMetalOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "permutation": "GRB"}
        band_swap_op_metal.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "permutation": "BGR",
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        band_swap_op_metal.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"permutation": "GRB"}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            band_swap_op_metal.validate_params(params)

    def test_missing_permutation_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test missing permutation parameter raises error."""
        params = {"tile_count": 5}
        with pytest.raises(ValueError, match="requires 'permutation' parameter"):
            band_swap_op_metal.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            band_swap_op_metal.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            band_swap_op_metal.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 100, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            band_swap_op_metal.validate_params(params)

    def test_invalid_permutation_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid permutation raises error."""
        params = {"tile_count": 5, "permutation": "XYZ"}
        with pytest.raises(ValueError, match="Permutation must be one of"):
            band_swap_op_metal.validate_params(params)

    def test_invalid_tile_size_range_type_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_size_range type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": 0.1}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            band_swap_op_metal.validate_params(params)

    def test_invalid_tile_size_range_length_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_size_range length raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            band_swap_op_metal.validate_params(params)

    def test_invalid_tile_size_range_values_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_size_range values raise error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [-0.1, 0.5]}
        with pytest.raises(ValueError, match="tile_size_range values must be between"):
            band_swap_op_metal.validate_params(params)

    def test_tile_size_range_min_greater_than_max_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test tile_size_range with min > max raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.5, 0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range min must be less than or equal to max"
        ):
            band_swap_op_metal.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            band_swap_op_metal.validate_params(params)


class TestBandSwapMetalApply:
    """Test applying Metal band swap to images."""

    def test_non_rgb_image_raises_error(
        self, band_swap_op_metal: BandSwapMetalOperation
    ) -> None:
        """Test that non-RGB images raise error."""
        grayscale = Image.new("L", (100, 100), color=128)
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        with pytest.raises(ValueError, match="Band swap requires RGB or RGBA image"):
            band_swap_op_metal.apply(grayscale, params)

    def test_apply_with_seed_deterministic(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that same seed produces same output."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}

        result1 = band_swap_op_metal.apply(test_image_rgb, params)
        result2 = band_swap_op_metal.apply(test_image_rgb, params)

        # Same seed should produce identical results
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_apply_different_seeds_different_output(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different seeds produce different outputs."""
        params1 = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        params2 = {"tile_count": 5, "permutation": "GRB", "seed": 123}

        result1 = band_swap_op_metal.apply(test_image_rgb, params1)
        result2 = band_swap_op_metal.apply(test_image_rgb, params2)

        # Different seeds should produce different results
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_different_permutations(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different permutations produce different results."""
        seed = 42
        tile_count = 10

        results = {}
        for perm in ["GRB", "BGR", "BRG", "GBR", "RBG"]:
            params = {"tile_count": tile_count, "permutation": perm, "seed": seed}
            results[perm] = np.array(band_swap_op_metal.apply(test_image_rgb, params))

        # All permutations should produce different results
        perms = list(results.keys())
        for i, perm1 in enumerate(perms):
            for perm2 in perms[i + 1 :]:
                assert not np.array_equal(results[perm1], results[perm2]), (
                    f"Permutations {perm1} and {perm2} should produce "
                    f"different results"
                )

    def test_apply_preserves_alpha_channel(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test that alpha channel is preserved."""
        params = {"tile_count": 5, "permutation": "BGR", "seed": 42}

        result = band_swap_op_metal.apply(test_image_rgba, params)

        # Verify mode is preserved
        assert result.mode == "RGBA"

        # Verify alpha channel is unchanged
        original_alpha = np.array(test_image_rgba)[..., 3]
        result_alpha = np.array(result)[..., 3]
        np.testing.assert_array_equal(result_alpha, original_alpha)

    def test_apply_multiple_tiles(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that multiple tiles are applied."""
        params = {"tile_count": 10, "permutation": "BGR", "seed": 42}

        result = band_swap_op_metal.apply(test_image_rgb, params)

        # Image should be modified
        assert not np.array_equal(np.array(result), np.array(test_image_rgb))

    def test_apply_changes_image(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that the operation modifies the image."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}

        result = band_swap_op_metal.apply(test_image_rgb, params)

        # Result should be different from original
        assert not np.array_equal(np.array(result), np.array(test_image_rgb))

    def test_dimensions_preserved(
        self, band_swap_op_metal: BandSwapMetalOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that image dimensions and mode are preserved."""
        params = {"tile_count": 5, "permutation": "BGR", "seed": 42}

        result = band_swap_op_metal.apply(test_image_rgb, params)

        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces same results as CPU version."""

    def test_rgb_consistency_with_seed(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce identical results with same seed."""
        params = {"tile_count": 10, "permutation": "BGR", "seed": 42}

        result_metal = band_swap_op_metal.apply(test_image_rgb, params)
        result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)

        # Results should be identical (deterministic integer operations)
        np.testing.assert_array_equal(
            metal_array,
            cpu_array,
            err_msg="Metal and CPU results should be identical",
        )

    def test_rgba_consistency_with_seed(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test Metal and CPU produce identical results for RGBA."""
        params = {"tile_count": 10, "permutation": "GRB", "seed": 42}

        result_metal = band_swap_op_metal.apply(test_image_rgba, params)
        result_cpu = band_swap_op_cpu.apply(test_image_rgba, params)

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)

        np.testing.assert_array_equal(
            metal_array,
            cpu_array,
            err_msg="Metal and CPU results should be identical for RGBA",
        )

    def test_various_permutations_consistency(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency across all valid permutations."""
        seed = 123
        tile_count = 5

        for perm in ["GRB", "BGR", "BRG", "GBR", "RBG"]:
            params = {"tile_count": tile_count, "permutation": perm, "seed": seed}

            result_metal = band_swap_op_metal.apply(test_image_rgb, params)
            result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

            metal_array = np.array(result_metal)
            cpu_array = np.array(result_cpu)

            np.testing.assert_array_equal(
                metal_array,
                cpu_array,
                err_msg=f"Metal and CPU should match for permutation {perm}",
            )

    def test_various_tile_counts_consistency(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with different tile counts."""
        for tile_count in [1, 5, 20]:
            params = {"tile_count": tile_count, "permutation": "BGR", "seed": 42}

            result_metal = band_swap_op_metal.apply(test_image_rgb, params)
            result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

            metal_array = np.array(result_metal)
            cpu_array = np.array(result_cpu)

            np.testing.assert_array_equal(
                metal_array,
                cpu_array,
                err_msg=f"Metal and CPU should match for tile_count={tile_count}",
            )

    def test_various_tile_sizes_consistency(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with different tile size ranges."""
        tile_size_ranges = [
            [0.01, 0.05],  # Very small tiles
            [0.1, 0.3],  # Medium tiles
            [0.3, 0.8],  # Large tiles
        ]

        for size_range in tile_size_ranges:
            params = {
                "tile_count": 5,
                "permutation": "GRB",
                "tile_size_range": size_range,
                "seed": 42,
            }

            result_metal = band_swap_op_metal.apply(test_image_rgb, params)
            result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

            metal_array = np.array(result_metal)
            cpu_array = np.array(result_cpu)

            np.testing.assert_array_equal(
                metal_array,
                cpu_array,
                err_msg=f"Metal and CPU should match for size_range={size_range}",
            )

    def test_edge_case_single_tile(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with single tile."""
        params = {"tile_count": 1, "permutation": "BGR", "seed": 42}

        result_metal = band_swap_op_metal.apply(test_image_rgb, params)
        result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)

        np.testing.assert_array_equal(
            metal_array,
            cpu_array,
            err_msg="Metal and CPU should match for single tile",
        )

    def test_edge_case_full_image_tile(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency when tile covers entire image."""
        params = {
            "tile_count": 1,
            "permutation": "GRB",
            "tile_size_range": [0.99, 1.0],
            "seed": 42,
        }

        result_metal = band_swap_op_metal.apply(test_image_rgb, params)
        result_cpu = band_swap_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)

        np.testing.assert_array_equal(
            metal_array,
            cpu_array,
            err_msg="Metal and CPU should match for full-image tile",
        )

    def test_large_image_consistency(
        self,
        band_swap_op_metal: BandSwapMetalOperation,
        band_swap_op_cpu: BandSwapOperation,
    ) -> None:
        """Test consistency on larger images."""
        large_image = Image.new("RGB", (512, 512), color=(100, 150, 200))
        params = {"tile_count": 20, "permutation": "BGR", "seed": 999}

        result_metal = band_swap_op_metal.apply(large_image, params)
        result_cpu = band_swap_op_cpu.apply(large_image, params)

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)

        np.testing.assert_array_equal(
            metal_array,
            cpu_array,
            err_msg="Metal and CPU should match on large images",
        )

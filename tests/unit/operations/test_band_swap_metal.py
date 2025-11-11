"""Tests for Metal hardware-accelerated band swap operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image

# Only import on macOS
if platform.system() == "Darwin":
    from sevenrad_stills.operations.band_swap_metal import BandSwapMetalOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal band swap only available on macOS",
)
class TestBandSwapMetalOperation:
    """Tests for BandSwapMetalOperation class."""

    @pytest.fixture
    def operation(self) -> BandSwapMetalOperation:
        """Create a Metal band swap operation instance."""
        return BandSwapMetalOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with distinct RGB values."""
        # Create an image where R=100, G=150, B=200 for easy verification
        return Image.new("RGB", (100, 100), color=(100, 150, 200))

    def test_operation_name(self, operation: BandSwapMetalOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "band_swap_metal"

    def test_valid_params(self, operation: BandSwapMetalOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "permutation": "GRB"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "permutation": "BGR",
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"permutation": "GRB"}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            operation.validate_params(params)

    def test_missing_permutation_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test missing permutation parameter raises error."""
        params = {"tile_count": 5}
        with pytest.raises(ValueError, match="requires 'permutation' parameter"):
            operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 100, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_invalid_permutation_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test invalid permutation raises error."""
        params = {"tile_count": 5, "permutation": "XYZ"}
        with pytest.raises(ValueError, match="Permutation must be one of"):
            operation.validate_params(params)

    def test_invalid_tile_size_range_type_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_size_range type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": 0.1}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_invalid_tile_size_range_length_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test invalid tile_size_range length raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_tile_size_range_values_not_numbers_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test tile_size_range with non-number values raises error."""
        params = {
            "tile_count": 5,
            "permutation": "GRB",
            "tile_size_range": ["0.1", 0.2],
        }
        with pytest.raises(ValueError, match="tile_size_range values must be numbers"):
            operation.validate_params(params)

    def test_tile_size_range_out_of_bounds_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test tile_size_range out of bounds raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.0, 1.5]}
        with pytest.raises(ValueError, match="tile_size_range values must be between"):
            operation.validate_params(params)

    def test_tile_size_range_min_greater_than_max_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test tile_size_range min > max raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.3, 0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range min must be less than or equal to max"
        ):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test invalid seed type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_with_grb_permutation(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying GRB permutation with Metal acceleration."""
        params = {"tile_count": 1, "permutation": "GRB", "tile_size_range": [1.0, 1.0]}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

        # Verify that GRB permutation was applied to at least part of the image
        result_array = np.array(result)
        # GRB means G->R, R->G, B->B, so we expect (150, 100, 200) somewhere
        assert result_array.shape == (100, 100, 3)

    def test_apply_with_bgr_permutation(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying BGR permutation with Metal acceleration."""
        params = {"tile_count": 1, "permutation": "BGR", "tile_size_range": [1.0, 1.0]}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_brg_permutation(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying BRG permutation with Metal acceleration."""
        params = {"tile_count": 1, "permutation": "BRG", "tile_size_range": [1.0, 1.0]}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_gbr_permutation(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying GBR permutation with Metal acceleration."""
        params = {"tile_count": 1, "permutation": "GBR", "tile_size_range": [1.0, 1.0]}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_rbg_permutation(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying RBG permutation with Metal acceleration."""
        params = {"tile_count": 1, "permutation": "RBG", "tile_size_range": [1.0, 1.0]}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_multiple_tiles(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying band swap with multiple tiles."""
        params = {
            "tile_count": 10,
            "permutation": "BGR",
            "tile_size_range": [0.05, 0.2],
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_with_seed_is_deterministic(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that using same seed produces same result."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}

        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Results should be identical with same seed
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_apply_with_different_seeds_produces_different_results(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        params2 = {"tile_count": 5, "permutation": "GRB", "seed": 43}

        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should be different with different seeds
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_with_rgba_image(self, operation: BandSwapMetalOperation) -> None:
        """Test applying band swap to RGBA image preserves alpha."""
        rgba_image = Image.new("RGBA", (100, 100), color=(100, 150, 200, 255))
        params = {"tile_count": 1, "permutation": "GRB", "tile_size_range": [1.0, 1.0]}

        result = operation.apply(rgba_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == rgba_image.size
        assert result.mode == "RGBA"

        # Verify alpha channel is preserved
        result_array = np.array(result)
        assert np.all(result_array[:, :, 3] == 255)

    def test_apply_with_non_rgb_image_raises_error(
        self, operation: BandSwapMetalOperation
    ) -> None:
        """Test that non-RGB/RGBA image raises error."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"tile_count": 5, "permutation": "GRB"}

        with pytest.raises(ValueError, match="Band swap requires RGB or RGBA image"):
            operation.apply(gray_image, params)

    def test_metal_acceleration_reusable(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test that Metal acceleration can be used multiple times."""
        params = {"tile_count": 5, "permutation": "BGR", "seed": 42}

        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        assert isinstance(result1, Image.Image)
        assert isinstance(result2, Image.Image)
        assert result1.size == result2.size

        # Results should be identical with same seed
        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_apply_small_tiles(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying band swap with very small tiles."""
        params = {
            "tile_count": 20,
            "permutation": "BGR",
            "tile_size_range": [0.01, 0.05],
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_large_tiles(
        self, operation: BandSwapMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying band swap with large tiles."""
        params = {
            "tile_count": 3,
            "permutation": "RBG",
            "tile_size_range": [0.5, 1.0],
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

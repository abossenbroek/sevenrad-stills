"""Tests for band swap operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.band_swap import BandSwapOperation


class TestBandSwapOperation:
    """Tests for BandSwapOperation class."""

    @pytest.fixture
    def operation(self) -> BandSwapOperation:
        """Create a band swap operation instance."""
        return BandSwapOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with distinct RGB values."""
        # Create an image where R=100, G=150, B=200 for easy verification
        return Image.new("RGB", (100, 100), color=(100, 150, 200))

    def test_operation_name(self, operation: BandSwapOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "band_swap"

    def test_valid_params(self, operation: BandSwapOperation) -> None:
        """Test valid parameter validation."""
        params = {"tile_count": 5, "permutation": "GRB"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(self, operation: BandSwapOperation) -> None:
        """Test valid parameters with all options."""
        params = {
            "tile_count": 10,
            "permutation": "BGR",
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_tile_count_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test missing tile_count parameter raises error."""
        params = {"permutation": "GRB"}
        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            operation.validate_params(params)

    def test_missing_permutation_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test missing permutation parameter raises error."""
        params = {"tile_count": 5}
        with pytest.raises(ValueError, match="requires 'permutation' parameter"):
            operation.validate_params(params)

    def test_invalid_tile_count_type_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test invalid tile_count type raises error."""
        params = {"tile_count": "5", "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer"):
            operation.validate_params(params)

    def test_tile_count_too_low_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test tile_count below minimum raises error."""
        params = {"tile_count": 0, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_tile_count_too_high_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test tile_count above maximum raises error."""
        params = {"tile_count": 100, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_invalid_permutation_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test invalid permutation raises error."""
        params = {"tile_count": 5, "permutation": "XYZ"}
        with pytest.raises(ValueError, match="Permutation must be one of"):
            operation.validate_params(params)

    def test_invalid_tile_size_range_type_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test invalid tile_size_range type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": 0.1}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_invalid_tile_size_range_length_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test invalid tile_size_range length raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range must be a list/tuple of two numbers"
        ):
            operation.validate_params(params)

    def test_invalid_tile_size_range_values_type_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test invalid tile_size_range value types raise error."""
        params = {
            "tile_count": 5,
            "permutation": "GRB",
            "tile_size_range": ["0.1", 0.3],
        }
        with pytest.raises(ValueError, match="tile_size_range values must be numbers"):
            operation.validate_params(params)

    def test_tile_size_range_out_of_bounds_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test tile_size_range out of bounds raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.0, 1.5]}
        with pytest.raises(ValueError, match="tile_size_range values must be between"):
            operation.validate_params(params)

    def test_tile_size_range_inverted_raises_error(
        self, operation: BandSwapOperation
    ) -> None:
        """Test inverted tile_size_range raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "tile_size_range": [0.3, 0.1]}
        with pytest.raises(
            ValueError, match="tile_size_range min must be less than or equal"
        ):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(self, operation: BandSwapOperation) -> None:
        """Test invalid seed type raises error."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": "42"}
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: BandSwapOperation, test_image: Image.Image
    ) -> None:
        """Test applying band swap."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: BandSwapOperation, test_image: Image.Image
    ) -> None:
        """Test that band swap actually changes the image."""
        params = {"tile_count": 10, "permutation": "GRB", "seed": 42}
        result = operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_grb_permutation_correctness(self, operation: BandSwapOperation) -> None:
        """Test GRB permutation correctness on simple image."""
        # Create a 3x1 image with [R=100, G=150, B=200]
        test_img = Image.new("RGB", (3, 1), color=(100, 150, 200))

        # Apply GRB swap to entire image (tile_size_range=[1.0, 1.0])
        params = {
            "tile_count": 1,
            "permutation": "GRB",
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        # GRB means: new_R=old_G, new_G=old_R, new_B=old_B
        # So [100, 150, 200] → [150, 100, 200]
        expected = np.array([[[150, 100, 200], [150, 100, 200], [150, 100, 200]]])
        np.testing.assert_array_equal(result_array, expected)

    def test_bgr_permutation_correctness(self, operation: BandSwapOperation) -> None:
        """Test BGR permutation correctness on simple image."""
        test_img = Image.new("RGB", (3, 1), color=(100, 150, 200))

        params = {
            "tile_count": 1,
            "permutation": "BGR",
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        # BGR means: new_R=old_B, new_G=old_G, new_B=old_R
        # So [100, 150, 200] → [200, 150, 100]
        expected = np.array([[[200, 150, 100], [200, 150, 100], [200, 150, 100]]])
        np.testing.assert_array_equal(result_array, expected)

    def test_rbg_permutation_correctness(self, operation: BandSwapOperation) -> None:
        """Test RBG permutation correctness on simple image."""
        test_img = Image.new("RGB", (3, 1), color=(100, 150, 200))

        params = {
            "tile_count": 1,
            "permutation": "RBG",
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }
        result = operation.apply(test_img, params)

        result_array = np.array(result)
        # RBG means: new_R=old_R, new_G=old_B, new_B=old_G
        # So [100, 150, 200] → [100, 200, 150]
        expected = np.array([[[100, 200, 150], [100, 200, 150], [100, 200, 150]]])
        np.testing.assert_array_equal(result_array, expected)

    def test_reproducibility_with_seed(
        self, operation: BandSwapOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: BandSwapOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        params2 = {"tile_count": 5, "permutation": "GRB", "seed": 100}
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ (tiles in different positions)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(self, operation: BandSwapOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(100, 150, 200, 200))
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_raises_error(self, operation: BandSwapOperation) -> None:
        """Test that grayscale images raise an error."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"tile_count": 5, "permutation": "GRB", "seed": 42}

        with pytest.raises(ValueError, match="Band swap requires RGB or RGBA"):
            operation.apply(gray_image, params)

    def test_zero_tile_count_edge_case(
        self,
        operation: BandSwapOperation,
        test_image: Image.Image,  # noqa: ARG002
    ) -> None:
        """Test that tile_count validation prevents zero."""
        params = {"tile_count": 0, "permutation": "GRB"}
        with pytest.raises(ValueError, match="tile_count must be an integer between"):
            operation.validate_params(params)

    def test_all_permutations_are_valid(self, operation: BandSwapOperation) -> None:
        """Test that all documented permutations are valid."""
        valid_perms = ["GRB", "BGR", "BRG", "GBR", "RBG"]
        for perm in valid_perms:
            params = {"tile_count": 1, "permutation": perm}
            operation.validate_params(params)  # Should not raise

    def test_tile_coverage(
        self, operation: BandSwapOperation, test_image: Image.Image
    ) -> None:
        """Test that with many tiles, significant portion is affected."""
        params = {"tile_count": 50, "permutation": "GRB", "seed": 42}
        result = operation.apply(test_image, params)

        result_array = np.array(result)
        original_array = np.array(test_image)

        # Count pixels that changed
        changed_pixels = np.any(result_array != original_array, axis=2).sum()
        total_pixels = result_array.shape[0] * result_array.shape[1]

        # With 50 tiles, expect at least 10% coverage
        coverage = changed_pixels / total_pixels
        assert coverage > 0.1

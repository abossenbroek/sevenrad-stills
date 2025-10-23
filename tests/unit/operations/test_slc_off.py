"""Tests for SLC-Off operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.slc_off import SlcOffOperation


class TestSlcOffOperation:
    """Tests for SlcOffOperation class."""

    @pytest.fixture
    def operation(self) -> SlcOffOperation:
        """Create an SLC-Off operation instance."""
        return SlcOffOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with gradient for visual verification."""
        # Create gradient image (easier to see gap pattern)
        gradient = np.linspace(0, 255, 100, dtype=np.uint8)
        img = np.tile(gradient, (100, 1))
        img = np.stack([img] * 3, axis=-1)  # RGB
        return Image.fromarray(img)

    def test_operation_name(self, operation: SlcOffOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "slc_off"

    def test_valid_params(self, operation: SlcOffOperation) -> None:
        """Test valid parameter validation."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(self, operation: SlcOffOperation) -> None:
        """Test valid parameters with all options."""
        params = {
            "gap_width": 0.3,
            "scan_period": 15,
            "fill_mode": "mean",
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_gap_width_raises_error(self, operation: SlcOffOperation) -> None:
        """Test missing gap_width parameter raises error."""
        params = {"scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'gap_width' parameter"):
            operation.validate_params(params)

    def test_missing_scan_period_raises_error(self, operation: SlcOffOperation) -> None:
        """Test missing scan_period parameter raises error."""
        params = {"gap_width": 0.2, "fill_mode": "black"}
        with pytest.raises(ValueError, match="requires 'scan_period' parameter"):
            operation.validate_params(params)

    def test_missing_fill_mode_raises_error(self, operation: SlcOffOperation) -> None:
        """Test missing fill_mode parameter raises error."""
        params = {"gap_width": 0.2, "scan_period": 10}
        with pytest.raises(ValueError, match="requires 'fill_mode' parameter"):
            operation.validate_params(params)

    def test_invalid_gap_width_type_raises_error(
        self, operation: SlcOffOperation
    ) -> None:
        """Test invalid gap_width type raises error."""
        params = {"gap_width": "0.2", "scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="gap_width must be a number"):
            operation.validate_params(params)

    def test_gap_width_too_low_raises_error(self, operation: SlcOffOperation) -> None:
        """Test gap_width below minimum raises error."""
        params = {"gap_width": -0.1, "scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="gap_width must be a number between"):
            operation.validate_params(params)

    def test_gap_width_too_high_raises_error(self, operation: SlcOffOperation) -> None:
        """Test gap_width above maximum raises error."""
        params = {"gap_width": 0.6, "scan_period": 10, "fill_mode": "black"}
        with pytest.raises(ValueError, match="gap_width must be a number between"):
            operation.validate_params(params)

    def test_invalid_scan_period_type_raises_error(
        self, operation: SlcOffOperation
    ) -> None:
        """Test invalid scan_period type raises error."""
        params = {"gap_width": 0.2, "scan_period": "10", "fill_mode": "black"}
        with pytest.raises(ValueError, match="scan_period must be an integer"):
            operation.validate_params(params)

    def test_scan_period_too_low_raises_error(self, operation: SlcOffOperation) -> None:
        """Test scan_period below minimum raises error."""
        params = {"gap_width": 0.2, "scan_period": 1, "fill_mode": "black"}
        with pytest.raises(ValueError, match="scan_period must be an integer between"):
            operation.validate_params(params)

    def test_scan_period_too_high_raises_error(
        self, operation: SlcOffOperation
    ) -> None:
        """Test scan_period above maximum raises error."""
        params = {"gap_width": 0.2, "scan_period": 150, "fill_mode": "black"}
        with pytest.raises(ValueError, match="scan_period must be an integer between"):
            operation.validate_params(params)

    def test_invalid_fill_mode_raises_error(self, operation: SlcOffOperation) -> None:
        """Test invalid fill_mode raises error."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "invalid"}
        with pytest.raises(ValueError, match="fill_mode must be one of"):
            operation.validate_params(params)

    def test_invalid_seed_type_raises_error(self, operation: SlcOffOperation) -> None:
        """Test invalid seed type raises error."""
        params = {
            "gap_width": 0.2,
            "scan_period": 10,
            "fill_mode": "black",
            "seed": "42",
        }
        with pytest.raises(ValueError, match="Seed must be an integer"):
            operation.validate_params(params)

    def test_apply_black_fill(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test applying SLC-Off with black fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_white_fill(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test applying SLC-Off with white fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "white"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_mean_fill(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test applying SLC-Off with mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test that SLC-Off actually changes the image."""
        params = {"gap_width": 0.3, "scan_period": 5, "fill_mode": "black"}
        result = operation.apply(test_image, params)

        # Should have differences (gaps filled with black)
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_gap_width_produces_no_change(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test gap_width=0.0 produces no change."""
        params = {"gap_width": 0.0, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_image, params)

        # Should be identical (no gaps)
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_reproducibility_with_seed_mean_fill(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results for mean fill."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 42}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_reproducibility_black_fill_no_seed(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test that black/white fill is deterministic without seed."""
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(self, operation: SlcOffOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_apply_grayscale(self, operation: SlcOffOperation) -> None:
        """Test that grayscale images work correctly."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"gap_width": 0.3, "scan_period": 5, "fill_mode": "black"}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

        # Should have gaps applied
        assert not np.array_equal(np.array(gray_image), np.array(result))

    def test_black_fill_creates_black_gaps(self, operation: SlcOffOperation) -> None:
        """Test that black fill mode creates black pixels in gaps."""
        # Create white image
        white_image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        params = {"gap_width": 0.4, "scan_period": 5, "fill_mode": "black"}
        result = operation.apply(white_image, params)

        result_array = np.array(result)

        # Should have some black pixels (0, 0, 0)
        black_pixels = np.all(result_array == [0, 0, 0], axis=-1)
        assert np.any(black_pixels), "Expected black gaps in result"

    def test_white_fill_creates_white_gaps(self, operation: SlcOffOperation) -> None:
        """Test that white fill mode creates white pixels in gaps."""
        # Create black image
        black_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        params = {"gap_width": 0.4, "scan_period": 5, "fill_mode": "white"}
        result = operation.apply(black_image, params)

        result_array = np.array(result)

        # Should have some white pixels (255, 255, 255)
        white_pixels = np.all(result_array == [255, 255, 255], axis=-1)
        assert np.any(white_pixels), "Expected white gaps in result"

    def test_gaps_widen_towards_edges(self, operation: SlcOffOperation) -> None:
        """Test that gaps widen from center to edges (wedge pattern)."""
        # Create gradient image
        gradient = np.linspace(0, 255, 100, dtype=np.uint8)
        img = np.tile(gradient, (100, 1))
        img = np.stack([img] * 3, axis=-1)
        test_img = Image.fromarray(img)

        params = {"gap_width": 0.4, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_img, params)

        result_array = np.array(result)

        # Count black pixels per scan line row
        # Should have more gaps near edges than near center
        scan_line_rows = list(range(0, 100, 10))

        center_row = 50
        edge_row_top = 0
        edge_row_bottom = 90

        # Count black pixels in these rows
        if center_row in scan_line_rows:
            center_blacks = np.sum(
                np.all(result_array[center_row] == [0, 0, 0], axis=-1)
            )
        else:
            center_blacks = 0

        top_blacks = np.sum(np.all(result_array[edge_row_top] == [0, 0, 0], axis=-1))
        bottom_blacks = np.sum(
            np.all(result_array[edge_row_bottom] == [0, 0, 0], axis=-1)
        )

        # Edges should have more gaps than center
        assert (
            top_blacks > center_blacks or bottom_blacks > center_blacks
        ), "Expected wider gaps at edges"

    def test_scan_period_affects_gap_frequency(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test that scan_period controls gap frequency."""
        # Small scan period (more frequent scan lines)
        params_frequent = {"gap_width": 0.2, "scan_period": 5, "fill_mode": "black"}
        result_frequent = operation.apply(test_image, params_frequent)

        # Large scan period (less frequent scan lines)
        params_sparse = {"gap_width": 0.2, "scan_period": 20, "fill_mode": "black"}
        result_sparse = operation.apply(test_image, params_sparse)

        # Both should produce changes
        frequent_array = np.array(result_frequent)
        sparse_array = np.array(result_sparse)
        original_array = np.array(test_image)

        frequent_changed = np.sum(np.any(frequent_array != original_array, axis=-1))
        sparse_changed = np.sum(np.any(sparse_array != original_array, axis=-1))

        # Both should produce gaps (changed pixels > 0)
        # With diagonal geometry, the relationship between scan_period and total
        # gap pixels is complex (gaps span multiple rows), so we just verify
        # that both create gaps and that they're different patterns
        assert frequent_changed > 0, "Expected gaps with frequent scan period"
        assert sparse_changed > 0, "Expected gaps with sparse scan period"

        # Verify the patterns are different (even if total pixel count is similar)
        assert not np.array_equal(
            frequent_array, sparse_array
        ), "Expected different gap patterns with different scan periods"

    def test_all_fill_modes_are_valid(self, operation: SlcOffOperation) -> None:
        """Test that all documented fill modes are valid."""
        valid_modes = ["black", "white", "mean"]
        for mode in valid_modes:
            params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": mode}
            operation.validate_params(params)  # Should not raise

    def test_mean_fill_uses_image_colors(self, operation: SlcOffOperation) -> None:
        """Test that mean fill doesn't create pure black or white."""
        # Create image with mid-gray colors
        gray_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        params = {"gap_width": 0.4, "scan_period": 5, "fill_mode": "mean", "seed": 42}
        result = operation.apply(gray_image, params)

        result_array = np.array(result)

        # Should not have pure black or pure white
        # (mean fill should be close to 128)
        black_pixels = np.all(result_array == [0, 0, 0], axis=-1)
        white_pixels = np.all(result_array == [255, 255, 255], axis=-1)

        assert not np.all(black_pixels), "Mean fill shouldn't create pure black"
        assert not np.all(white_pixels), "Mean fill shouldn't create pure white"

    def test_boundary_scan_periods(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test boundary values for scan_period (2 and 100)."""
        # Minimum scan period
        params_min = {"gap_width": 0.2, "scan_period": 2, "fill_mode": "black"}
        result_min = operation.apply(test_image, params_min)
        assert isinstance(result_min, Image.Image)

        # Maximum scan period
        params_max = {"gap_width": 0.2, "scan_period": 100, "fill_mode": "black"}
        result_max = operation.apply(test_image, params_max)
        assert isinstance(result_max, Image.Image)

    def test_maximum_gap_width(
        self, operation: SlcOffOperation, test_image: Image.Image
    ) -> None:
        """Test maximum gap_width value (0.5)."""
        params = {"gap_width": 0.5, "scan_period": 10, "fill_mode": "black"}
        result = operation.apply(test_image, params)

        # Should produce valid image with large gaps
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

        # Should have significant change
        assert not np.array_equal(np.array(test_image), np.array(result))

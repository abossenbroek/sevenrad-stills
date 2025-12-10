"""Tests for SlcOffTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.slc_off_taichi import SlcOffTaichiOperation


class TestSlcOffTaichiOperationInit:
    """Test SlcOffTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = SlcOffTaichiOperation()

        assert op.name == "slc_off_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that SLC-Off does not support in-place execution."""
        op = SlcOffTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = SlcOffTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that SLC-Off requires no temporary fields."""
        op = SlcOffTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params_black(self) -> None:
        """Test that valid parameters pass validation."""
        op = SlcOffTaichiOperation()

        # Should not raise
        op.validate_params({"gap_width": 0.22, "scan_period": 14, "fill_mode": "black"})

    def test_valid_params_white(self) -> None:
        """Test white fill mode."""
        op = SlcOffTaichiOperation()

        op.validate_params({"gap_width": 0.1, "scan_period": 10, "fill_mode": "white"})

    def test_valid_params_mean(self) -> None:
        """Test mean fill mode."""
        op = SlcOffTaichiOperation()

        op.validate_params(
            {"gap_width": 0.15, "scan_period": 8, "fill_mode": "mean", "seed": 42}
        )

    def test_valid_params_edge_values(self) -> None:
        """Test edge values for gap_width and scan_period."""
        op = SlcOffTaichiOperation()

        # Min values
        op.validate_params({"gap_width": 0.0, "scan_period": 2, "fill_mode": "black"})

        # Max values
        op.validate_params({"gap_width": 0.5, "scan_period": 100, "fill_mode": "white"})

    def test_missing_gap_width(self) -> None:
        """Test that missing gap_width raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="requires 'gap_width' parameter"):
            op.validate_params({"scan_period": 14, "fill_mode": "black"})

    def test_invalid_gap_width_range(self) -> None:
        """Test that gap_width out of range raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="must be a number between 0.0 and 0.5"):
            op.validate_params(
                {"gap_width": -0.1, "scan_period": 14, "fill_mode": "black"}
            )

        with pytest.raises(ValueError, match="must be a number between 0.0 and 0.5"):
            op.validate_params(
                {"gap_width": 0.6, "scan_period": 14, "fill_mode": "black"}
            )

    def test_invalid_gap_width_type(self) -> None:
        """Test that non-numeric gap_width raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="must be a number between"):
            op.validate_params(
                {"gap_width": "large", "scan_period": 14, "fill_mode": "black"}
            )

    def test_missing_scan_period(self) -> None:
        """Test that missing scan_period raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="requires 'scan_period' parameter"):
            op.validate_params({"gap_width": 0.22, "fill_mode": "black"})

    def test_invalid_scan_period_range(self) -> None:
        """Test that scan_period out of range raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer between 2 and 100"):
            op.validate_params(
                {"gap_width": 0.22, "scan_period": 1, "fill_mode": "black"}
            )

        with pytest.raises(ValueError, match="must be an integer between 2 and 100"):
            op.validate_params(
                {"gap_width": 0.22, "scan_period": 101, "fill_mode": "black"}
            )

    def test_invalid_scan_period_type(self) -> None:
        """Test that non-integer scan_period raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer between"):
            op.validate_params(
                {"gap_width": 0.22, "scan_period": 14.5, "fill_mode": "black"}
            )

    def test_missing_fill_mode(self) -> None:
        """Test that missing fill_mode raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="requires 'fill_mode' parameter"):
            op.validate_params({"gap_width": 0.22, "scan_period": 14})

    def test_invalid_fill_mode(self) -> None:
        """Test that invalid fill_mode raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(
            ValueError, match="must be one of: 'black', 'white', 'mean'"
        ):
            op.validate_params(
                {"gap_width": 0.22, "scan_period": 14, "fill_mode": "gray"}
            )

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = SlcOffTaichiOperation()

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params(
                {
                    "gap_width": 0.22,
                    "scan_period": 14,
                    "fill_mode": "black",
                    "seed": "random",
                }
            )


class TestComputeGapMask:
    """Test gap mask computation."""

    def test_zero_gap_width(self) -> None:
        """Test that zero gap width produces no gaps."""
        op = SlcOffTaichiOperation()

        mask = op._compute_gap_mask(
            height=100, width=100, gap_width=0.0, scan_period=10, seed=0
        )

        # All zeros (no gaps)
        assert np.all(mask == 0.0)

    def test_mask_shape(self) -> None:
        """Test that mask has correct shape."""
        op = SlcOffTaichiOperation()

        mask = op._compute_gap_mask(
            height=50, width=80, gap_width=0.2, scan_period=10, seed=0
        )

        assert mask.shape == (50, 80)

    def test_mask_dtype(self) -> None:
        """Test that mask is float32."""
        op = SlcOffTaichiOperation()

        mask = op._compute_gap_mask(
            height=100, width=100, gap_width=0.2, scan_period=10, seed=0
        )

        assert mask.dtype == np.float32

    def test_gaps_widen_from_center(self) -> None:
        """Test that gaps increase in width from center to edges."""
        op = SlcOffTaichiOperation()

        mask = op._compute_gap_mask(
            height=100, width=100, gap_width=0.3, scan_period=10, seed=0
        )

        # Count gaps in center row (should be minimal)
        center_y = 50
        center_gaps = np.sum(mask[center_y] > 0.5)

        # Count gaps near edge (should be larger)
        edge_y = 10
        edge_gaps = np.sum(mask[edge_y] > 0.5)

        # Edge should have more gaps than center
        assert edge_gaps >= center_gaps

    def test_scan_period_affects_gap_spacing(self) -> None:
        """Test that scan_period controls gap frequency."""
        op = SlcOffTaichiOperation()

        # Smaller scan period = more frequent gaps
        mask_small = op._compute_gap_mask(
            height=100, width=100, gap_width=0.2, scan_period=5, seed=0
        )

        # Larger scan period = less frequent gaps
        mask_large = op._compute_gap_mask(
            height=100, width=100, gap_width=0.2, scan_period=20, seed=0
        )

        gaps_small = np.sum(mask_small > 0.5)
        gaps_large = np.sum(mask_large > 0.5)

        # Smaller period should produce more gap pixels
        assert gaps_small >= gaps_large

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same mask."""
        op = SlcOffTaichiOperation()

        mask1 = op._compute_gap_mask(
            height=100, width=100, gap_width=0.2, scan_period=10, seed=42
        )
        mask2 = op._compute_gap_mask(
            height=100, width=100, gap_width=0.2, scan_period=10, seed=42
        )

        np.testing.assert_array_equal(mask1, mask2)


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_black_fill(self) -> None:
        """Test that black fill mode creates black gaps."""
        op = SlcOffTaichiOperation()

        # Create test image (all white)
        image = np.ones((100, 100, 3), dtype=np.float32)

        params = {"gap_width": 0.3, "scan_period": 10, "fill_mode": "black", "seed": 0}
        result = op.reference_numpy(image, params)

        # Check that some pixels are black (gaps)
        black_pixels = np.all(result == 0.0, axis=2)
        assert np.any(black_pixels)

        # Check that some pixels remain white (non-gaps)
        white_pixels = np.all(result == 1.0, axis=2)
        assert np.any(white_pixels)

    def test_white_fill(self) -> None:
        """Test that white fill mode creates white gaps."""
        op = SlcOffTaichiOperation()

        # Create test image (all black)
        image = np.zeros((100, 100, 3), dtype=np.float32)

        params = {"gap_width": 0.3, "scan_period": 10, "fill_mode": "white", "seed": 0}
        result = op.reference_numpy(image, params)

        # Check that some pixels are white (gaps)
        white_pixels = np.all(result == 1.0, axis=2)
        assert np.any(white_pixels)

    def test_mean_fill(self) -> None:
        """Test that mean fill mode uses image statistics."""
        op = SlcOffTaichiOperation()

        # Create test image with known mean (0.6, 0.3, 0.1)
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, :, 0] = 0.6
        image[:, :, 1] = 0.3
        image[:, :, 2] = 0.1

        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 0}
        result = op.reference_numpy(image, params)

        # Get gap pixels
        mask = op._compute_gap_mask(100, 100, 0.2, 10, 0)
        gap_pixels = result[mask > 0.5]

        if len(gap_pixels) > 0:
            # Gap pixels should be close to row means (with variation)
            # Just check they're in reasonable range
            assert np.all(gap_pixels >= 0.0)
            assert np.all(gap_pixels <= 1.0)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = SlcOffTaichiOperation()

        for shape in [(50, 50, 3), (100, 80, 3), (75, 125, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            params = {
                "gap_width": 0.2,
                "scan_period": 10,
                "fill_mode": "black",
                "seed": 0,
            }
            result = op.reference_numpy(image, params)
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is in valid [0, 1] range."""
        op = SlcOffTaichiOperation()

        image = np.random.rand(80, 80, 3).astype(np.float32)
        params = {"gap_width": 0.25, "scan_period": 12, "fill_mode": "mean", "seed": 42}
        result = op.reference_numpy(image, params)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = SlcOffTaichiOperation()

        image = np.random.rand(60, 60, 3).astype(np.float32)
        params = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "black", "seed": 0}
        result = op.reference_numpy(image, params)

        assert result.dtype == np.float32

    def test_zero_gap_width_preserves_image(self) -> None:
        """Test that zero gap width returns original image."""
        op = SlcOffTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)
        params = {"gap_width": 0.0, "scan_period": 10, "fill_mode": "black", "seed": 0}
        result = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result, image)


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = SlcOffTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        # Mock Taichi field for mask
        mock_mask_field = Mock()
        mock_mask_field.to_numpy.return_value = np.zeros((64, 64), dtype=np.float32)

        mock_ti = MagicMock()
        mock_ti.field.return_value = mock_mask_field

        with (
            patch(
                "sevenrad_stills.operations.slc_off_taichi._slc_off_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.slc_off_taichi.ti", mock_ti),
        ):
            params = {
                "gap_width": 0.22,
                "scan_period": 14,
                "fill_mode": "black",
                "seed": 0,
            }
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params=params,
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            # mask is call_args[2]
            assert call_args[3] == 0.0  # fill_r (black)
            assert call_args[4] == 0.0  # fill_g (black)
            assert call_args[5] == 0.0  # fill_b (black)
            assert call_args[6] == 0  # batch
            assert call_args[7] == 64  # height
            assert call_args[8] == 64  # width

    def test_apply_to_field_white_fill(self) -> None:
        """Test white fill color parameters."""
        op = SlcOffTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_mask_field = Mock()
        mock_mask_field.to_numpy.return_value = np.zeros((32, 32), dtype=np.float32)

        mock_ti = MagicMock()
        mock_ti.field.return_value = mock_mask_field

        with (
            patch(
                "sevenrad_stills.operations.slc_off_taichi._slc_off_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.slc_off_taichi.ti", mock_ti),
        ):
            params = {
                "gap_width": 0.15,
                "scan_period": 10,
                "fill_mode": "white",
                "seed": 42,
            }
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params=params,
                height=32,
                width=32,
            )

            call_args = mock_kernel.call_args[0]
            assert call_args[3] == 1.0  # fill_r (white)
            assert call_args[4] == 1.0  # fill_g (white)
            assert call_args[5] == 1.0  # fill_b (white)

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = SlcOffTaichiOperation()

        with (
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", False),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={
                    "gap_width": 0.2,
                    "scan_period": 10,
                    "fill_mode": "black",
                    "seed": 0,
                },
                height=64,
                width=64,
            )

    def test_mask_caching(self) -> None:
        """Test that mask is cached for identical calls."""
        op = SlcOffTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_mask_field = Mock()
        mock_mask_field.to_numpy.return_value = np.zeros((64, 64), dtype=np.float32)

        mock_ti = MagicMock()
        mock_ti.field.return_value = mock_mask_field

        with (
            patch("sevenrad_stills.operations.slc_off_taichi._slc_off_kernel"),
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.slc_off_taichi.ti", mock_ti),
        ):
            params = {
                "gap_width": 0.2,
                "scan_period": 10,
                "fill_mode": "black",
                "seed": 0,
            }

            # First call - should create field
            op.apply_to_field(source, dest, {}, params, 64, 64)
            first_call_count = mock_ti.field.call_count

            # Second call with same params - should reuse
            op.apply_to_field(source, dest, {}, params, 64, 64)
            second_call_count = mock_ti.field.call_count

            # Field should only be created once
            assert first_call_count == second_call_count


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = SlcOffTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field_vector = MagicMock()
        mock_field_scalar = MagicMock()
        mock_ti.Vector.field.return_value = mock_field_vector
        mock_ti.field.return_value = mock_field_scalar

        with (
            patch("sevenrad_stills.operations.slc_off_taichi._slc_off_kernel"),
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.slc_off_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = SlcOffTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field_vector = MagicMock()
        mock_field_scalar = MagicMock()
        mock_ti.Vector.field.return_value = mock_field_vector
        mock_ti.field.return_value = mock_field_scalar

        with (
            patch(
                "sevenrad_stills.operations.slc_off_taichi._slc_off_kernel",
                side_effect=count_calls,
            ),
            patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.slc_off_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = SlcOffTaichiOperation()

        with patch("sevenrad_stills.operations.slc_off_taichi.TAICHI_AVAILABLE", False):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_consistent_mask_application(self) -> None:
        """Test that mask is consistently applied."""
        op = SlcOffTaichiOperation()

        # Create consistent colored image
        image = np.full((80, 80, 3), 0.5, dtype=np.float32)

        params = {"gap_width": 0.25, "scan_period": 10, "fill_mode": "black", "seed": 0}

        # Apply twice - should be identical
        result1 = op.reference_numpy(image, params)
        result2 = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results_mean_fill(self) -> None:
        """Test that different seeds affect mean fill variation."""
        op = SlcOffTaichiOperation()

        image = np.random.rand(80, 80, 3).astype(np.float32)

        params1 = {"gap_width": 0.2, "scan_period": 10, "fill_mode": "mean", "seed": 0}
        params2 = {
            "gap_width": 0.2,
            "scan_period": 10,
            "fill_mode": "mean",
            "seed": 123,
        }

        result1 = op.reference_numpy(image, params1)
        result2 = op.reference_numpy(image, params2)

        # Results should differ due to random variation in mean fill
        # But only in gap regions
        mask = op._compute_gap_mask(80, 80, 0.2, 10, 0)
        gap_mask = mask > 0.5

        if np.any(gap_mask):
            # At least some gap pixels should differ
            gap_diff = np.abs(result1[gap_mask] - result2[gap_mask])
            assert np.any(gap_diff > 0.001)

    def test_large_gap_width(self) -> None:
        """Test maximum gap width (0.5) produces significant gaps."""
        op = SlcOffTaichiOperation()

        image = np.ones((100, 100, 3), dtype=np.float32)

        params = {"gap_width": 0.5, "scan_period": 10, "fill_mode": "black", "seed": 0}
        result = op.reference_numpy(image, params)

        # Count black pixels (gaps)
        black_pixels = np.all(result == 0.0, axis=2)
        gap_percentage = np.sum(black_pixels) / (100 * 100)

        # Should have significant gaps
        assert gap_percentage > 0.05  # At least 5% gaps

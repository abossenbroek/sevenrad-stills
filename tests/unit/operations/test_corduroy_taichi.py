"""Tests for CorduroyTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.corduroy_taichi import CorduroyTaichiOperation


class TestCorduroyTaichiOperationInit:
    """Test CorduroyTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = CorduroyTaichiOperation()

        assert op.name == "corduroy_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that corduroy supports in-place execution."""
        op = CorduroyTaichiOperation()

        assert op.supports_inplace is True

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = CorduroyTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that corduroy requires no temporary fields."""
        op = CorduroyTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params(self) -> None:
        """Test that valid parameters pass validation."""
        op = CorduroyTaichiOperation()

        # Should not raise
        op.validate_params(
            {
                "orientation": "vertical",
                "strength": 0.5,
                "density": 0.3,
                "seed": 42,
            }
        )

        op.validate_params(
            {
                "orientation": "horizontal",
                "strength": 0.0,
                "density": 1.0,
                "seed": 0,
            }
        )

    def test_missing_orientation(self) -> None:
        """Test that missing orientation raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="requires 'orientation' parameter"):
            op.validate_params(
                {
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                }
            )

    def test_invalid_orientation(self) -> None:
        """Test that invalid orientation raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="must be 'vertical' or 'horizontal'"):
            op.validate_params(
                {
                    "orientation": "diagonal",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                }
            )

    def test_missing_strength(self) -> None:
        """Test that missing strength raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="requires 'strength' parameter"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "density": 0.3,
                    "seed": 42,
                }
            )

    def test_invalid_strength_type(self) -> None:
        """Test that non-numeric strength raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="Strength must be a number"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": "high",
                    "density": 0.3,
                    "seed": 42,
                }
            )

    def test_strength_out_of_range(self) -> None:
        """Test that strength outside [0, 1] raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="Strength must be between"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 1.5,
                    "density": 0.3,
                    "seed": 42,
                }
            )

        with pytest.raises(ValueError, match="Strength must be between"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": -0.1,
                    "density": 0.3,
                    "seed": 42,
                }
            )

    def test_missing_density(self) -> None:
        """Test that missing density raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="requires 'density' parameter"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "seed": 42,
                }
            )

    def test_invalid_density_type(self) -> None:
        """Test that non-numeric density raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="Density must be a number"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": "high",
                    "seed": 42,
                }
            )

    def test_density_out_of_range(self) -> None:
        """Test that density outside [0, 1] raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="Density must be between"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 1.5,
                    "seed": 42,
                }
            )

        with pytest.raises(ValueError, match="Density must be between"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": -0.1,
                    "seed": 42,
                }
            )

    def test_missing_seed(self) -> None:
        """Test that missing seed raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="requires 'seed' parameter"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 0.3,
                }
            )

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = CorduroyTaichiOperation()

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params(
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42.5,
                }
            )


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_density_preserves_image(self) -> None:
        """Test that density=0 leaves image unchanged."""
        op = CorduroyTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 0.0,
                "seed": 42,
            },
        )

        np.testing.assert_allclose(result, image)

    def test_vertical_orientation(self) -> None:
        """Test that vertical orientation creates vertical stripes."""
        op = CorduroyTaichiOperation()

        # Create solid color image
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        # Each column should have uniform value (within column)
        # But columns should differ
        for col in range(10):
            col_values = result[:, col, 0]
            # All pixels in same column should be equal
            assert np.allclose(col_values, col_values[0])

        # Check that not all columns are identical (with high density/strength)
        first_col = result[:, 0, 0]
        all_same = all(np.allclose(result[:, col, 0], first_col) for col in range(10))
        assert not all_same, "Expected variation across columns"

    def test_horizontal_orientation(self) -> None:
        """Test that horizontal orientation creates horizontal stripes."""
        op = CorduroyTaichiOperation()

        # Create solid color image
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "horizontal",
                "strength": 1.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        # Each row should have uniform value (within row)
        # But rows should differ
        for row in range(10):
            row_values = result[row, :, 0]
            # All pixels in same row should be equal
            assert np.allclose(row_values, row_values[0])

        # Check that not all rows are identical (with high density/strength)
        first_row = result[0, :, 0]
        all_same = all(np.allclose(result[row, :, 0], first_row) for row in range(10))
        assert not all_same, "Expected variation across rows"

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same result."""
        op = CorduroyTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)
        params = {
            "orientation": "vertical",
            "strength": 0.8,
            "density": 0.5,
            "seed": 123,
        }

        result1 = op.reference_numpy(image, params)
        result2 = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_differ(self) -> None:
        """Test that different seeds produce different results."""
        op = CorduroyTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        result1 = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 0.8,
                "density": 0.5,
                "seed": 42,
            },
        )

        result2 = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 0.8,
                "density": 0.5,
                "seed": 123,
            },
        )

        # Results should differ
        assert not np.allclose(result1, result2)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = CorduroyTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(
                image,
                {
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                },
            )
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = CorduroyTaichiOperation()

        # Create image near boundaries
        image = np.random.rand(20, 20, 3).astype(np.float32)
        image[::2] = 0.95  # High values
        image[1::2] = 0.05  # Low values

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = CorduroyTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 0.5,
                "density": 0.3,
                "seed": 42,
            },
        )

        assert result.dtype == np.float32

    def test_grayscale_image(self) -> None:
        """Test that grayscale images are handled correctly."""
        op = CorduroyTaichiOperation()

        # Create 2D grayscale image
        image = np.random.rand(10, 10).astype(np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 0.5,
                "density": 0.5,
                "seed": 42,
            },
        )

        assert result.shape == image.shape
        assert result.dtype == np.float32


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = CorduroyTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        # Mock ti.field
        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_field.fill = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.corduroy_taichi._corduroy_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.corduroy_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                },
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            # multipliers field is call_args[2]
            assert call_args[3] == 1  # is_vertical for "vertical"
            assert call_args[4] == 0  # batch
            assert call_args[5] == 64  # height
            assert call_args[6] == 64  # width

    def test_apply_to_field_horizontal_orientation(self) -> None:
        """Test that horizontal orientation sets is_vertical=0."""
        op = CorduroyTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_field.fill = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.corduroy_taichi._corduroy_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.corduroy_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={
                    "orientation": "horizontal",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                },
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[3] == 0  # is_vertical for "horizontal"

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = CorduroyTaichiOperation()

        with (
            patch("sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", False),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={
                    "orientation": "vertical",
                    "strength": 0.5,
                    "density": 0.3,
                    "seed": 42,
                },
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = CorduroyTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field
        mock_field.fill = MagicMock()

        with (
            patch("sevenrad_stills.operations.corduroy_taichi._corduroy_kernel"),
            patch("sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.corduroy_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = CorduroyTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field
        mock_field.fill = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.corduroy_taichi._corduroy_kernel",
                side_effect=count_calls,
            ),
            patch("sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.corduroy_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = CorduroyTaichiOperation()

        with patch(
            "sevenrad_stills.operations.corduroy_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_zero_strength(self) -> None:
        """Test that strength=0 means no variation (all multipliers near 1.0)."""
        op = CorduroyTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 0.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        # With strength=0, multipliers should be very close to 1.0
        # So result should be very close to input
        np.testing.assert_allclose(result, image, rtol=1e-3, atol=1e-3)

    def test_partial_density(self) -> None:
        """Test that partial density affects only some lines."""
        op = CorduroyTaichiOperation()

        # Create uniform image
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 0.3,  # Only 30% of columns affected
                "seed": 42,
            },
        )

        # Some columns should be unchanged (multiplier=1.0, value=0.5)
        # Count how many columns are approximately unchanged
        unchanged_cols = 0
        for col in range(10):
            if np.allclose(result[:, col, 0], 0.5, atol=0.01):
                unchanged_cols += 1

        # Should have some unchanged columns with 30% density
        assert unchanged_cols > 0

    def test_extreme_multipliers_clipped(self) -> None:
        """Test that extreme multipliers don't cause overflow."""
        op = CorduroyTaichiOperation()

        # Create image with values at boundaries
        image = np.array(
            [
                [[1.0, 0.0, 0.5]],
            ],
            dtype=np.float32,
        )

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        # All values should remain in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_black_and_white_scaling(self) -> None:
        """Test behavior on black and white pixels."""
        op = CorduroyTaichiOperation()

        # Black and white image
        image = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        result = op.reference_numpy(
            image,
            {
                "orientation": "vertical",
                "strength": 1.0,
                "density": 1.0,
                "seed": 42,
            },
        )

        # Black should stay black or get darker (multiplier >= 0.8)
        # White might get dimmed
        # Both should stay in valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Black scaled by positive multiplier stays black or near-black
        black_pixel = result[0, 0]
        assert np.all(black_pixel <= 0.2)  # Should stay very dark

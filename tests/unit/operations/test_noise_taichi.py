"""Tests for NoiseTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.noise_taichi import NoiseTaichiOperation


class TestNoiseTaichiOperationInit:
    """Test NoiseTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = NoiseTaichiOperation()

        assert op.name == "noise_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that noise supports in-place execution."""
        op = NoiseTaichiOperation()

        assert op.supports_inplace is True

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = NoiseTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that noise requires no temporary fields."""
        op = NoiseTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params_gaussian(self) -> None:
        """Test that valid gaussian params pass validation."""
        op = NoiseTaichiOperation()

        # Should not raise
        op.validate_params({"mode": "gaussian", "amount": 0.5})
        op.validate_params({"mode": "gaussian", "amount": 0.0})
        op.validate_params({"mode": "gaussian", "amount": 1.0})
        op.validate_params({"mode": "gaussian", "amount": 0.5, "seed": 42})

    def test_valid_params_row(self) -> None:
        """Test that valid row params pass validation."""
        op = NoiseTaichiOperation()

        op.validate_params({"mode": "row", "amount": 0.5})
        op.validate_params({"mode": "row", "amount": 0.2, "seed": 123})

    def test_valid_params_column(self) -> None:
        """Test that valid column params pass validation."""
        op = NoiseTaichiOperation()

        op.validate_params({"mode": "column", "amount": 0.3})
        op.validate_params({"mode": "column", "amount": 0.7, "seed": 999})

    def test_missing_mode(self) -> None:
        """Test that missing mode raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="requires 'mode' parameter"):
            op.validate_params({"amount": 0.5})

    def test_invalid_mode(self) -> None:
        """Test that invalid mode raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="must be 'gaussian', 'row', or 'column'"):
            op.validate_params({"mode": "perlin", "amount": 0.5})

        with pytest.raises(ValueError, match="must be 'gaussian', 'row', or 'column'"):
            op.validate_params({"mode": "salt_pepper", "amount": 0.5})

    def test_missing_amount(self) -> None:
        """Test that missing amount raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="requires 'amount' parameter"):
            op.validate_params({"mode": "gaussian"})

    def test_invalid_amount_type(self) -> None:
        """Test that non-numeric amount raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"mode": "gaussian", "amount": "high"})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"mode": "gaussian", "amount": None})

    def test_amount_out_of_range(self) -> None:
        """Test that amount outside [0, 1] raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="must be in"):
            op.validate_params({"mode": "gaussian", "amount": -0.1})

        with pytest.raises(ValueError, match="must be in"):
            op.validate_params({"mode": "gaussian", "amount": 1.5})

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = NoiseTaichiOperation()

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params({"mode": "gaussian", "amount": 0.5, "seed": 3.14})

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params({"mode": "gaussian", "amount": 0.5, "seed": "random"})


class TestReferenceNumpyGaussian:
    """Test NumPy reference implementation for Gaussian mode."""

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = NoiseTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(
                image, {"mode": "gaussian", "amount": 0.1, "seed": 42}
            )
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = NoiseTaichiOperation()

        # Create image near boundaries
        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        result = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.5, "seed": 42}
        )

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = NoiseTaichiOperation()

        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.1, "seed": 42}
        )

        assert result.dtype == np.float32

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same noise."""
        op = NoiseTaichiOperation()

        image = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        params = {"mode": "gaussian", "amount": 0.2, "seed": 123}

        result1 = op.reference_numpy(image, params)
        result2 = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_produces_different_noise(self) -> None:
        """Test that different seeds produce different noise."""
        op = NoiseTaichiOperation()

        image = np.ones((8, 8, 3), dtype=np.float32) * 0.5

        result1 = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.2, "seed": 123}
        )
        result2 = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.2, "seed": 456}
        )

        # Results should be different (statistically very unlikely to be same)
        assert not np.array_equal(result1, result2)

    def test_zero_amount_preserves_image(self) -> None:
        """Test that amount=0 produces no noise."""
        op = NoiseTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.0, "seed": 42}
        )

        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_per_pixel_variation(self) -> None:
        """Test that Gaussian noise varies per pixel."""
        op = NoiseTaichiOperation()

        # Uniform gray image
        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        result = op.reference_numpy(
            image, {"mode": "gaussian", "amount": 0.1, "seed": 42}
        )

        # Noise should create variation between pixels
        variance = np.var(result)
        assert variance > 0.0


class TestReferenceNumpyRow:
    """Test NumPy reference implementation for row mode."""

    def test_row_uniformity(self) -> None:
        """Test that noise is uniform within each row."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 20, 3), dtype=np.float32) * 0.5
        result = op.reference_numpy(image, {"mode": "row", "amount": 0.2, "seed": 42})

        # Each row should have same noise across all columns
        for i in range(10):
            row_pixels = result[i, :, :]
            # All pixels in row should be identical
            for j in range(1, 20):
                np.testing.assert_allclose(
                    row_pixels[j], row_pixels[0], rtol=1e-5, atol=1e-5
                )

    def test_row_variation_between_rows(self) -> None:
        """Test that noise varies between different rows."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        result = op.reference_numpy(image, {"mode": "row", "amount": 0.2, "seed": 42})

        # Different rows should have different noise
        # (statistically very unlikely all rows are identical)
        rows_equal = np.all(result[0, 0, :] == result[1, 0, :])
        assert not rows_equal

    def test_output_range_row(self) -> None:
        """Test that row mode output is clipped to [0, 1]."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.9
        result = op.reference_numpy(image, {"mode": "row", "amount": 0.5, "seed": 42})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestReferenceNumpyColumn:
    """Test NumPy reference implementation for column mode."""

    def test_column_uniformity(self) -> None:
        """Test that noise is uniform within each column."""
        op = NoiseTaichiOperation()

        image = np.ones((20, 10, 3), dtype=np.float32) * 0.5
        result = op.reference_numpy(
            image, {"mode": "column", "amount": 0.2, "seed": 42}
        )

        # Each column should have same noise across all rows
        for j in range(10):
            col_pixels = result[:, j, :]
            # All pixels in column should be identical
            for i in range(1, 20):
                np.testing.assert_allclose(
                    col_pixels[i], col_pixels[0], rtol=1e-5, atol=1e-5
                )

    def test_column_variation_between_columns(self) -> None:
        """Test that noise varies between different columns."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        result = op.reference_numpy(
            image, {"mode": "column", "amount": 0.2, "seed": 42}
        )

        # Different columns should have different noise
        # (statistically very unlikely all columns are identical)
        cols_equal = np.all(result[0, 0, :] == result[0, 1, :])
        assert not cols_equal

    def test_output_range_column(self) -> None:
        """Test that column mode output is clipped to [0, 1]."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.1
        result = op.reference_numpy(
            image, {"mode": "column", "amount": 0.5, "seed": 42}
        )

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_gaussian_calls_kernel(self) -> None:
        """Test that apply_to_field invokes gaussian kernel."""
        op = NoiseTaichiOperation()

        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_gaussian_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"mode": "gaussian", "amount": 0.3, "seed": 42},
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 0.3  # amount
            assert call_args[3] == 42  # seed
            assert call_args[4] == 0  # batch
            assert call_args[5] == 64  # height
            assert call_args[6] == 64  # width

    def test_apply_row_calls_kernel(self) -> None:
        """Test that apply_to_field invokes row kernel."""
        op = NoiseTaichiOperation()

        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_row_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"mode": "row", "amount": 0.2, "seed": 123},
                height=32,
                width=32,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 0.2  # amount
            assert call_args[3] == 123  # seed
            assert call_args[4] == 0  # batch
            assert call_args[5] == 32  # height
            assert call_args[6] == 32  # width

    def test_apply_column_calls_kernel(self) -> None:
        """Test that apply_to_field invokes column kernel."""
        op = NoiseTaichiOperation()

        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_column_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"mode": "column", "amount": 0.4, "seed": 999},
                height=128,
                width=256,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 0.4  # amount
            assert call_args[3] == 999  # seed
            assert call_args[4] == 0  # batch
            assert call_args[5] == 128  # height
            assert call_args[6] == 256  # width

    def test_apply_default_seed(self) -> None:
        """Test that apply_to_field uses default seed=0 when not provided."""
        op = NoiseTaichiOperation()

        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_gaussian_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"mode": "gaussian", "amount": 0.1},
                height=64,
                width=64,
            )

            call_args = mock_kernel.call_args[0]
            assert call_args[3] == 0  # seed defaults to 0

    def test_apply_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = NoiseTaichiOperation()

        with (
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", False),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"mode": "gaussian", "amount": 0.5},
                height=64,
                width=64,
            )

    def test_apply_invalid_mode_raises(self) -> None:
        """Test that apply_to_field raises for invalid mode."""
        op = NoiseTaichiOperation()

        with (
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", MagicMock()),
            pytest.raises(ValueError, match="Unknown noise mode"),
        ):
            # This shouldn't happen if validate_params is called, but test defensively
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"mode": "invalid", "amount": 0.5},
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = NoiseTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch("sevenrad_stills.operations.noise_taichi._noise_gaussian_kernel"),
            patch("sevenrad_stills.operations.noise_taichi._noise_row_kernel"),
            patch("sevenrad_stills.operations.noise_taichi._noise_column_kernel"),
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_compiles_all_kernels(self) -> None:
        """Test that warmup triggers all three kernel compilations."""
        op = NoiseTaichiOperation()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_gaussian_kernel"
            ) as mock_gaussian,
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_row_kernel"
            ) as mock_row,
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_column_kernel"
            ) as mock_column,
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", mock_ti),
        ):
            op.warmup()

            # All three kernels should be called
            mock_gaussian.assert_called_once()
            mock_row.assert_called_once()
            mock_column.assert_called_once()

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = NoiseTaichiOperation()

        call_counts = {"gaussian": 0, "row": 0, "column": 0}

        def count_gaussian(*_args: object, **_kwargs: object) -> None:
            call_counts["gaussian"] += 1

        def count_row(*_args: object, **_kwargs: object) -> None:
            call_counts["row"] += 1

        def count_column(*_args: object, **_kwargs: object) -> None:
            call_counts["column"] += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_gaussian_kernel",
                side_effect=count_gaussian,
            ),
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_row_kernel",
                side_effect=count_row,
            ),
            patch(
                "sevenrad_stills.operations.noise_taichi._noise_column_kernel",
                side_effect=count_column,
            ),
            patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.noise_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        # Each kernel should only be called once
        assert call_counts["gaussian"] == 1
        assert call_counts["row"] == 1
        assert call_counts["column"] == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = NoiseTaichiOperation()

        with patch("sevenrad_stills.operations.noise_taichi.TAICHI_AVAILABLE", False):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_image(self) -> None:
        """Test noise on 1x1 image."""
        op = NoiseTaichiOperation()

        image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        for mode in ["gaussian", "row", "column"]:
            result = op.reference_numpy(
                image, {"mode": mode, "amount": 0.1, "seed": 42}
            )
            assert result.shape == (1, 1, 3)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_boundary_values(self) -> None:
        """Test noise on images with boundary values."""
        op = NoiseTaichiOperation()

        # All black
        black = np.zeros((5, 5, 3), dtype=np.float32)
        result_black = op.reference_numpy(
            black, {"mode": "gaussian", "amount": 0.1, "seed": 42}
        )
        assert np.all(result_black >= 0.0)

        # All white
        white = np.ones((5, 5, 3), dtype=np.float32)
        result_white = op.reference_numpy(
            white, {"mode": "gaussian", "amount": 0.1, "seed": 42}
        )
        assert np.all(result_white <= 1.0)

    def test_max_amount(self) -> None:
        """Test noise with maximum amount=1.0."""
        op = NoiseTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        for mode in ["gaussian", "row", "column"]:
            result = op.reference_numpy(
                image, {"mode": mode, "amount": 1.0, "seed": 42}
            )
            # Even with max noise, should be clipped to [0, 1]
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

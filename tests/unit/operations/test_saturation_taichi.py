"""Tests for SaturationTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.saturation_taichi import SaturationTaichiOperation


class TestSaturationTaichiOperationInit:
    """Test SaturationTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = SaturationTaichiOperation()

        assert op.name == "saturation_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that saturation supports in-place execution."""
        op = SaturationTaichiOperation()

        assert op.supports_inplace is True

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = SaturationTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that saturation requires no temporary fields."""
        op = SaturationTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_factor(self) -> None:
        """Test that valid factor passes validation."""
        op = SaturationTaichiOperation()

        # Should not raise
        op.validate_params({"factor": 1.0})
        op.validate_params({"factor": 0.0})
        op.validate_params({"factor": 2.5})
        op.validate_params({"factor": 0})  # int is ok

    def test_missing_factor(self) -> None:
        """Test that missing factor raises ValueError."""
        op = SaturationTaichiOperation()

        with pytest.raises(ValueError, match="requires 'factor' parameter"):
            op.validate_params({})

    def test_invalid_factor_type(self) -> None:
        """Test that non-numeric factor raises ValueError."""
        op = SaturationTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"factor": "high"})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"factor": None})

    def test_negative_factor(self) -> None:
        """Test that negative factor raises ValueError."""
        op = SaturationTaichiOperation()

        with pytest.raises(ValueError, match="must be >= 0.0"):
            op.validate_params({"factor": -0.5})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_identity_factor(self) -> None:
        """Test that factor=1.0 preserves image."""
        op = SaturationTaichiOperation()

        # Create test image with varying saturation
        image = np.array(
            [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], [[0.5, 0.0, 1.0], [0.5, 0.5, 0.5]]],
            dtype=np.float32,
        )

        result = op.reference_numpy(image, {"factor": 1.0})

        # With factor=1.0, output should match input
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_grayscale_factor(self) -> None:
        """Test that factor=0.0 produces grayscale."""
        op = SaturationTaichiOperation()

        # Pure red
        image = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

        result = op.reference_numpy(image, {"factor": 0.0})

        # Grayscale: all channels should equal value (max of RGB)
        # For pure red (1,0,0), value=1, with saturation=0, RGB should be (1,1,1)
        # Actually for saturation=0: result = value (grayscale)
        assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]

    def test_increased_saturation(self) -> None:
        """Test that factor>1.0 increases saturation."""
        op = SaturationTaichiOperation()

        # Desaturated color (gray-ish red)
        image = np.array([[[0.8, 0.4, 0.4]]], dtype=np.float32)

        result = op.reference_numpy(image, {"factor": 2.0})

        # Red channel should stay high, green/blue should decrease
        assert result[0, 0, 0] >= image[0, 0, 0]  # Red stays or increases
        # Saturation increased means more difference between max and min

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = SaturationTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(image, {"factor": 1.5})
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = SaturationTaichiOperation()

        # Create image that could overflow
        image = np.array([[[0.9, 0.1, 0.1]]], dtype=np.float32)

        result = op.reference_numpy(image, {"factor": 10.0})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = SaturationTaichiOperation()

        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = op.reference_numpy(image, {"factor": 1.5})

        assert result.dtype == np.float32


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = SaturationTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.saturation_taichi._saturation_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.saturation_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.saturation_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"factor": 1.5},
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 1.5  # factor
            assert call_args[3] == 0  # batch
            assert call_args[4] == 64  # height
            assert call_args[5] == 64  # width

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = SaturationTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.saturation_taichi.TAICHI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"factor": 1.5},
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = SaturationTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch("sevenrad_stills.operations.saturation_taichi._saturation_kernel"),
            patch(
                "sevenrad_stills.operations.saturation_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.saturation_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = SaturationTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.saturation_taichi._saturation_kernel",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.saturation_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.saturation_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = SaturationTaichiOperation()

        with patch(
            "sevenrad_stills.operations.saturation_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_pure_colors(self) -> None:
        """Test saturation adjustment on pure colors."""
        op = SaturationTaichiOperation()

        # Pure red, green, blue
        pure_colors = np.array(
            [
                [[1.0, 0.0, 0.0]],  # Red
                [[0.0, 1.0, 0.0]],  # Green
                [[0.0, 0.0, 1.0]],  # Blue
            ],
            dtype=np.float32,
        )

        # Factor 1.0 should preserve
        result = op.reference_numpy(pure_colors, {"factor": 1.0})
        np.testing.assert_allclose(result, pure_colors, rtol=1e-5)

    def test_grayscale_input(self) -> None:
        """Test that grayscale input remains grayscale."""
        op = SaturationTaichiOperation()

        # Grayscale image
        gray = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        # Any factor should preserve grayscale
        for factor in [0.0, 1.0, 2.0]:
            result = op.reference_numpy(gray, {"factor": factor})
            # All channels should be equal
            assert np.allclose(result[0, 0, 0], result[0, 0, 1], rtol=1e-5)
            assert np.allclose(result[0, 0, 1], result[0, 0, 2], rtol=1e-5)

    def test_black_and_white(self) -> None:
        """Test that black and white are preserved."""
        op = SaturationTaichiOperation()

        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)

        for factor in [0.0, 1.0, 2.0]:
            black_result = op.reference_numpy(black, {"factor": factor})
            white_result = op.reference_numpy(white, {"factor": factor})

            np.testing.assert_allclose(black_result, black, atol=1e-5)
            np.testing.assert_allclose(white_result, white, atol=1e-5)

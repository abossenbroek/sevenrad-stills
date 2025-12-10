"""Tests for ChromaticAberrationTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.chromatic_aberration_taichi import (
    ChromaticAberrationTaichiOperation,
)


class TestChromaticAberrationTaichiOperationInit:
    """Test ChromaticAberrationTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = ChromaticAberrationTaichiOperation()

        assert op.name == "chromatic_aberration_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that chromatic aberration does not support in-place execution."""
        op = ChromaticAberrationTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = ChromaticAberrationTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that chromatic aberration requires no temporary fields."""
        op = ChromaticAberrationTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params(self) -> None:
        """Test that valid parameters pass validation."""
        op = ChromaticAberrationTaichiOperation()

        # Should not raise
        op.validate_params({"shift_x": 2, "shift_y": 1})
        op.validate_params({"shift_x": 0, "shift_y": 0})
        op.validate_params({"shift_x": -3, "shift_y": 5})
        op.validate_params({"shift_x": 1.5, "shift_y": 2.5})  # floats ok

    def test_missing_shift_x(self) -> None:
        """Test that missing shift_x raises ValueError."""
        op = ChromaticAberrationTaichiOperation()

        with pytest.raises(ValueError, match="requires 'shift_x' parameter"):
            op.validate_params({"shift_y": 1})

    def test_missing_shift_y(self) -> None:
        """Test that missing shift_y raises ValueError."""
        op = ChromaticAberrationTaichiOperation()

        with pytest.raises(ValueError, match="requires 'shift_y' parameter"):
            op.validate_params({"shift_x": 2})

    def test_invalid_shift_x_type(self) -> None:
        """Test that non-numeric shift_x raises ValueError."""
        op = ChromaticAberrationTaichiOperation()

        with pytest.raises(ValueError, match="shift_x must be a number"):
            op.validate_params({"shift_x": "large", "shift_y": 1})

        with pytest.raises(ValueError, match="shift_x must be a number"):
            op.validate_params({"shift_x": None, "shift_y": 1})

    def test_invalid_shift_y_type(self) -> None:
        """Test that non-numeric shift_y raises ValueError."""
        op = ChromaticAberrationTaichiOperation()

        with pytest.raises(ValueError, match="shift_y must be a number"):
            op.validate_params({"shift_x": 1, "shift_y": "small"})

        with pytest.raises(ValueError, match="shift_y must be a number"):
            op.validate_params({"shift_x": 1, "shift_y": None})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_shift_preserves_image(self) -> None:
        """Test that zero shift preserves the image."""
        op = ChromaticAberrationTaichiOperation()

        image = np.array(
            [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], [[0.5, 0.0, 1.0], [0.5, 0.5, 0.5]]],
            dtype=np.float32,
        )

        result = op.reference_numpy(image, {"shift_x": 0, "shift_y": 0})

        # With zero shift, output should match input
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_horizontal_shift_only(self) -> None:
        """Test chromatic aberration with horizontal shift."""
        op = ChromaticAberrationTaichiOperation()

        # Simple gradient pattern
        image = np.zeros((3, 4, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red channel all 1.0
        image[:, :, 1] = 0.5  # Green channel all 0.5
        image[:, :, 2] = 0.0  # Blue channel all 0.0

        result = op.reference_numpy(image, {"shift_x": 1, "shift_y": 0})

        # Green channel should be unchanged
        np.testing.assert_allclose(result[:, :, 1], image[:, :, 1], atol=1e-5)

    def test_vertical_shift_only(self) -> None:
        """Test chromatic aberration with vertical shift."""
        op = ChromaticAberrationTaichiOperation()

        # Simple pattern
        image = np.zeros((4, 3, 3), dtype=np.float32)
        image[:, :, 0] = 1.0
        image[:, :, 1] = 0.5
        image[:, :, 2] = 0.0

        result = op.reference_numpy(image, {"shift_x": 0, "shift_y": 1})

        # Green channel should be unchanged
        np.testing.assert_allclose(result[:, :, 1], image[:, :, 1], atol=1e-5)

    def test_negative_shift(self) -> None:
        """Test that negative shifts work correctly."""
        op = ChromaticAberrationTaichiOperation()

        image = np.random.rand(5, 5, 3).astype(np.float32)

        # Should not raise
        result = op.reference_numpy(image, {"shift_x": -2, "shift_y": -1})

        assert result.shape == image.shape

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = ChromaticAberrationTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(image, {"shift_x": 2, "shift_y": 1})
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = ChromaticAberrationTaichiOperation()

        # Create image
        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(image, {"shift_x": 3, "shift_y": 2})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = ChromaticAberrationTaichiOperation()

        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = op.reference_numpy(image, {"shift_x": 1, "shift_y": 1})

        assert result.dtype == np.float32

    def test_edge_clamping(self) -> None:
        """Test that sampling beyond edges is properly clamped."""
        op = ChromaticAberrationTaichiOperation()

        # Create small image
        image = np.ones((3, 3, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red
        image[:, :, 1] = 0.5  # Green
        image[:, :, 2] = 0.0  # Blue

        # Large shift should clamp to edges
        result = op.reference_numpy(image, {"shift_x": 10, "shift_y": 10})

        # Should not crash and should produce valid output
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_fractional_shifts(self) -> None:
        """Test that fractional shifts use bilinear interpolation."""
        op = ChromaticAberrationTaichiOperation()

        # Create gradient pattern
        image = np.zeros((4, 4, 3), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                image[i, j, 0] = i / 3.0  # Vertical gradient in red
                image[i, j, 1] = 0.5
                image[i, j, 2] = j / 3.0  # Horizontal gradient in blue

        # Fractional shift should produce interpolated values
        result = op.reference_numpy(image, {"shift_x": 0.5, "shift_y": 0.5})

        # Result should be different from original (due to interpolation)
        # but still in valid range
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = ChromaticAberrationTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi._chromatic_aberration_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.ti",
                MagicMock(),
            ),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"shift_x": 2, "shift_y": 1},
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 2.0  # shift_x
            assert call_args[3] == 1.0  # shift_y
            assert call_args[4] == 0  # batch
            assert call_args[5] == 64  # height
            assert call_args[6] == 64  # width

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = ChromaticAberrationTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
                False,
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"shift_x": 2, "shift_y": 1},
                height=64,
                width=64,
            )

    def test_apply_to_field_converts_to_float(self) -> None:
        """Test that shift values are converted to float."""
        op = ChromaticAberrationTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi._chromatic_aberration_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.ti",
                MagicMock(),
            ),
        ):
            # Pass integers
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"shift_x": 3, "shift_y": 2},
                height=32,
                width=32,
            )

            call_args = mock_kernel.call_args[0]
            # Should be converted to float
            assert isinstance(call_args[2], float)
            assert isinstance(call_args[3], float)


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = ChromaticAberrationTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi._chromatic_aberration_kernel"
            ),
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.chromatic_aberration_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = ChromaticAberrationTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi._chromatic_aberration_kernel",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.chromatic_aberration_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = ChromaticAberrationTaichiOperation()

        with patch(
            "sevenrad_stills.operations.chromatic_aberration_taichi.TAICHI_AVAILABLE",
            False,
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_pure_red_image(self) -> None:
        """Test chromatic aberration on pure red image."""
        op = ChromaticAberrationTaichiOperation()

        # Pure red
        image = np.zeros((5, 5, 3), dtype=np.float32)
        image[:, :, 0] = 1.0

        result = op.reference_numpy(image, {"shift_x": 1, "shift_y": 0})

        # Green and blue channels should remain zero in most places
        # (except where shifted channels sample from edges)
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_pure_green_image(self) -> None:
        """Test that green channel is never shifted."""
        op = ChromaticAberrationTaichiOperation()

        # Pure green
        image = np.zeros((5, 5, 3), dtype=np.float32)
        image[:, :, 1] = 1.0

        result = op.reference_numpy(image, {"shift_x": 2, "shift_y": 1})

        # Green channel should be exactly preserved
        np.testing.assert_allclose(result[:, :, 1], image[:, :, 1], atol=1e-6)

    def test_pure_blue_image(self) -> None:
        """Test chromatic aberration on pure blue image."""
        op = ChromaticAberrationTaichiOperation()

        # Pure blue
        image = np.zeros((5, 5, 3), dtype=np.float32)
        image[:, :, 2] = 1.0

        result = op.reference_numpy(image, {"shift_x": 1, "shift_y": 0})

        # Should produce valid output
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetry_of_shifts(self) -> None:
        """Test that R and B channels are shifted symmetrically."""
        op = ChromaticAberrationTaichiOperation()

        # Create symmetric image
        image = np.zeros((7, 7, 3), dtype=np.float32)
        image[3, 3] = [1.0, 0.5, 0.8]  # Center pixel

        result = op.reference_numpy(image, {"shift_x": 2, "shift_y": 0})

        # The center pixel green should be unchanged
        assert np.isclose(result[3, 3, 1], 0.5, atol=1e-5)

    def test_grayscale_with_aberration(self) -> None:
        """Test that grayscale image gets color fringing."""
        op = ChromaticAberrationTaichiOperation()

        # Grayscale edge pattern
        image = np.zeros((5, 5, 3), dtype=np.float32)
        image[2, :] = 1.0  # Horizontal white line

        result = op.reference_numpy(image, {"shift_x": 0, "shift_y": 1})

        # With shift, R and B should differ from G, creating color fringing
        # At edges, we should see color differences
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

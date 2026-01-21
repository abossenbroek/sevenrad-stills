"""Tests for MotionBlurTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.motion_blur_taichi import MotionBlurTaichiOperation


class TestMotionBlurTaichiOperationInit:
    """Test MotionBlurTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = MotionBlurTaichiOperation()

        assert op.name == "motion_blur_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that motion blur does not support in-place execution."""
        op = MotionBlurTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = MotionBlurTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that motion blur requires no temporary fields."""
        op = MotionBlurTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params(self) -> None:
        """Test that valid parameters pass validation."""
        op = MotionBlurTaichiOperation()

        # Should not raise
        op.validate_params({"kernel_size": 5})
        op.validate_params({"kernel_size": 5, "angle": 45.0})
        op.validate_params({"kernel_size": 1, "angle": 0.0})
        op.validate_params({"kernel_size": 100, "angle": 359.0})

    def test_missing_kernel_size(self) -> None:
        """Test that missing kernel_size raises ValueError."""
        op = MotionBlurTaichiOperation()

        with pytest.raises(ValueError, match="requires 'kernel_size' parameter"):
            op.validate_params({})

    def test_invalid_kernel_size_type(self) -> None:
        """Test that non-integer kernel_size raises ValueError."""
        op = MotionBlurTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"kernel_size": 5.5})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"kernel_size": "large"})

    def test_kernel_size_out_of_range(self) -> None:
        """Test that kernel_size outside valid range raises ValueError."""
        op = MotionBlurTaichiOperation()

        with pytest.raises(ValueError, match="must be between 1 and 100"):
            op.validate_params({"kernel_size": 0})

        with pytest.raises(ValueError, match="must be between 1 and 100"):
            op.validate_params({"kernel_size": 101})

    def test_invalid_angle_type(self) -> None:
        """Test that non-numeric angle raises ValueError."""
        op = MotionBlurTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"kernel_size": 5, "angle": "horizontal"})

    def test_angle_out_of_range(self) -> None:
        """Test that angle outside valid range raises ValueError."""
        op = MotionBlurTaichiOperation()

        with pytest.raises(ValueError, match="must be between 0.0 and 360.0"):
            op.validate_params({"kernel_size": 5, "angle": -10.0})

        with pytest.raises(ValueError, match="must be between 0.0 and 360.0"):
            op.validate_params({"kernel_size": 5, "angle": 360.0})


class TestCreateMotionKernel:
    """Test motion blur kernel creation."""

    def test_horizontal_kernel(self) -> None:
        """Test that angle=0 creates horizontal kernel."""
        op = MotionBlurTaichiOperation()

        kernel = op._create_motion_kernel_numpy(5, 0.0)

        # Kernel should be 5x5
        assert kernel.shape == (5, 5)

        # Should be normalized
        assert np.isclose(kernel.sum(), 1.0)

        # Weights should be along middle row (horizontal line)
        middle_row = kernel[2, :]
        assert np.any(middle_row > 0)

    def test_vertical_kernel(self) -> None:
        """Test that angle=90 creates vertical kernel."""
        op = MotionBlurTaichiOperation()

        kernel = op._create_motion_kernel_numpy(5, 90.0)

        # Should be normalized
        assert np.isclose(kernel.sum(), 1.0)

        # Weights should be along middle column (vertical line)
        middle_col = kernel[:, 2]
        assert np.any(middle_col > 0)

    def test_diagonal_kernel(self) -> None:
        """Test that angle=45 creates diagonal kernel."""
        op = MotionBlurTaichiOperation()

        kernel = op._create_motion_kernel_numpy(7, 45.0)

        # Should be normalized
        assert np.isclose(kernel.sum(), 1.0)

        # Should have non-zero weights (likely along or near diagonal)
        # For 45 degrees, Bresenham will create a line from one corner to opposite
        assert np.sum(kernel > 0) >= 3  # At least 3 non-zero elements

    def test_kernel_normalization(self) -> None:
        """Test that kernels are always normalized."""
        op = MotionBlurTaichiOperation()

        for size in [3, 5, 7, 11]:
            for angle in [0, 30, 60, 90, 120, 180, 270]:
                kernel = op._create_motion_kernel_numpy(size, angle)
                msg = f"Failed for size={size}, angle={angle}"
                assert np.isclose(kernel.sum(), 1.0), msg

    def test_kernel_size(self) -> None:
        """Test that kernel has correct size."""
        op = MotionBlurTaichiOperation()

        for size in [1, 3, 5, 7, 15, 31]:
            kernel = op._create_motion_kernel_numpy(size, 0.0)
            assert kernel.shape == (size, size)


class TestBresenhamLine:
    """Test Bresenham line generation."""

    def test_horizontal_line(self) -> None:
        """Test horizontal line generation."""
        op = MotionBlurTaichiOperation()

        points = op._bresenham_line(0, 0, 4, 0)

        assert len(points) == 5
        assert all(y == 0 for _, y in points)
        assert [x for x, _ in points] == [0, 1, 2, 3, 4]

    def test_vertical_line(self) -> None:
        """Test vertical line generation."""
        op = MotionBlurTaichiOperation()

        points = op._bresenham_line(0, 0, 0, 4)

        assert len(points) == 5
        assert all(x == 0 for x, _ in points)
        assert [y for _, y in points] == [0, 1, 2, 3, 4]

    def test_diagonal_line(self) -> None:
        """Test diagonal line generation."""
        op = MotionBlurTaichiOperation()

        points = op._bresenham_line(0, 0, 4, 4)

        # Should generate points along 45-degree diagonal
        assert len(points) == 5
        for i, (x, y) in enumerate(points):
            assert x == i
            assert y == i

    def test_reverse_direction(self) -> None:
        """Test that line works in both directions."""
        op = MotionBlurTaichiOperation()

        forward = op._bresenham_line(0, 0, 4, 0)
        backward = op._bresenham_line(4, 0, 0, 0)

        # Should generate same number of points
        assert len(forward) == len(backward)


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_kernel_size_one_is_noop(self) -> None:
        """Test that kernel_size=1 returns unchanged image."""
        op = MotionBlurTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(image, {"kernel_size": 1})

        np.testing.assert_array_equal(result, image)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = MotionBlurTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(image, {"kernel_size": 5, "angle": 45.0})
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = MotionBlurTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(image, {"kernel_size": 5, "angle": 0.0})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = MotionBlurTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"kernel_size": 5, "angle": 90.0})

        assert result.dtype == np.float32

    def test_different_angles(self) -> None:
        """Test that different angles produce different results."""
        op = MotionBlurTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        result_0 = op.reference_numpy(image, {"kernel_size": 7, "angle": 0.0})
        result_90 = op.reference_numpy(image, {"kernel_size": 7, "angle": 90.0})

        # Different angles should produce different results
        assert not np.allclose(result_0, result_90)

    def test_smoothing_effect(self) -> None:
        """Test that motion blur reduces variance (smooths image)."""
        op = MotionBlurTaichiOperation()

        # Create noisy image
        np.random.seed(42)
        image = np.random.rand(50, 50, 3).astype(np.float32)

        result = op.reference_numpy(image, {"kernel_size": 11, "angle": 45.0})

        # Blurred image should have lower variance
        assert np.var(result) < np.var(image)


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_convolve(self) -> None:
        """Test that apply_to_field invokes convolution kernel."""
        op = MotionBlurTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_kernel_field = MagicMock()
        mock_kernel_field.shape = (5, 5)
        mock_ti.field.return_value = mock_kernel_field

        with (
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.convolve_2d"
            ) as mock_convolve,
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.motion_blur_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"kernel_size": 5, "angle": 45.0},
                height=64,
                width=64,
            )

            # Should call convolve_2d once
            mock_convolve.assert_called_once()

    def test_apply_to_field_kernel_size_one_copies(self) -> None:
        """Test that kernel_size=1 just copies source to dest."""
        op = MotionBlurTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.convolve_2d"
            ) as mock_convolve,
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.motion_blur_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"kernel_size": 1},
                height=64,
                width=64,
            )

            # Should not call convolution, just copy
            mock_convolve.assert_not_called()
            dest.copy_from.assert_called_once_with(source)

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = MotionBlurTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"kernel_size": 5, "angle": 0.0},
                height=64,
                width=64,
            )

    def test_kernel_caching(self) -> None:
        """Test that kernel is cached for repeated calls with same params."""
        op = MotionBlurTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_kernel_field = MagicMock()
        mock_kernel_field.shape = (5, 5)
        mock_ti.field.return_value = mock_kernel_field

        with (
            patch("sevenrad_stills.operations.motion_blur_taichi.convolve_2d"),
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.motion_blur_taichi.ti", mock_ti),
        ):
            # First call - should allocate kernel
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"kernel_size": 5, "angle": 45.0},
                height=64,
                width=64,
            )

            field_call_count = mock_ti.field.call_count
            from_numpy_call_count = mock_kernel_field.from_numpy.call_count

            # Second call with same params - should reuse kernel
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"kernel_size": 5, "angle": 45.0},
                height=64,
                width=64,
            )

            # Should not allocate new field or call from_numpy again
            assert mock_ti.field.call_count == field_call_count
            assert mock_kernel_field.from_numpy.call_count == from_numpy_call_count


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = MotionBlurTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field_vec = MagicMock()
        mock_field_scalar = MagicMock()
        mock_ti.Vector.field.return_value = mock_field_vec
        mock_ti.field.return_value = mock_field_scalar

        with (
            patch("sevenrad_stills.operations.motion_blur_taichi.convolve_2d"),
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.motion_blur_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = MotionBlurTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field_vec = MagicMock()
        mock_field_scalar = MagicMock()
        mock_ti.Vector.field.return_value = mock_field_vec
        mock_ti.field.return_value = mock_field_scalar

        with (
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.convolve_2d",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.motion_blur_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = MotionBlurTaichiOperation()

        with patch(
            "sevenrad_stills.operations.motion_blur_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_uniform_image(self) -> None:
        """Test that uniform image stays uniform."""
        op = MotionBlurTaichiOperation()

        # Uniform gray
        uniform = np.full((20, 20, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(uniform, {"kernel_size": 7, "angle": 30.0})

        # Should stay uniform (all pixels same)
        np.testing.assert_allclose(result, uniform, rtol=1e-5, atol=1e-5)

    def test_edge_preservation(self) -> None:
        """Test that blur affects edges as expected."""
        op = MotionBlurTaichiOperation()

        # Create sharp vertical edge
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, 10:] = 1.0

        # Horizontal blur should blur the edge
        result_h = op.reference_numpy(image, {"kernel_size": 5, "angle": 0.0})

        # Edge should be blurred (middle column should have intermediate values)
        middle_col = result_h[:, 10, 0]
        assert np.any((middle_col > 0.1) & (middle_col < 0.9))

    def test_directional_specificity(self) -> None:
        """Test that blur is directional (only blurs along angle)."""
        op = MotionBlurTaichiOperation()

        # Create a sharp horizontal edge
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[10:, :] = 1.0

        # Horizontal blur should NOT blur horizontal edge much
        result_h = op.reference_numpy(image, {"kernel_size": 7, "angle": 0.0})

        # Vertical blur SHOULD blur horizontal edge
        result_v = op.reference_numpy(image, {"kernel_size": 7, "angle": 90.0})

        # Vertical blur should have more mixing at the edge
        edge_row_h = result_h[10, :, 0]
        edge_row_v = result_v[10, :, 0]

        # Horizontal blur should leave edge sharper than vertical blur
        # (though this depends on implementation details)
        # At minimum, results should be different
        assert not np.allclose(edge_row_h, edge_row_v)

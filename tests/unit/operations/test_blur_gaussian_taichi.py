"""Tests for BlurGaussianTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.blur_gaussian_taichi import (
    BlurGaussianTaichiOperation,
)


class TestBlurGaussianTaichiOperationInit:
    """Test BlurGaussianTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = BlurGaussianTaichiOperation()

        assert op.name == "blur_gaussian_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that Gaussian blur does not support in-place execution."""
        op = BlurGaussianTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = BlurGaussianTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements(self) -> None:
        """Test that Gaussian blur requires one intermediate temporary field."""
        op = BlurGaussianTaichiOperation()

        temp_reqs = op.temp_field_requirements
        assert len(temp_reqs) == 1
        assert temp_reqs[0].name == "intermediate"
        assert temp_reqs[0].shape_factor == (1.0, 1.0, 4)
        assert temp_reqs[0].dtype == "f32"


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_sigma(self) -> None:
        """Test that valid sigma passes validation."""
        op = BlurGaussianTaichiOperation()

        # Should not raise
        op.validate_params({"sigma": 0.0})
        op.validate_params({"sigma": 1.0})
        op.validate_params({"sigma": 5.5})
        op.validate_params({"sigma": 0})  # int is ok

    def test_missing_sigma(self) -> None:
        """Test that missing sigma raises ValueError."""
        op = BlurGaussianTaichiOperation()

        with pytest.raises(ValueError, match="requires 'sigma' parameter"):
            op.validate_params({})

    def test_invalid_sigma_type(self) -> None:
        """Test that non-numeric sigma raises ValueError."""
        op = BlurGaussianTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"sigma": "medium"})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"sigma": None})

    def test_negative_sigma(self) -> None:
        """Test that negative sigma raises ValueError."""
        op = BlurGaussianTaichiOperation()

        with pytest.raises(ValueError, match="must be >= 0"):
            op.validate_params({"sigma": -1.0})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_sigma(self) -> None:
        """Test that sigma=0 preserves image."""
        op = BlurGaussianTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"sigma": 0.0})

        # With sigma=0, output should match input
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_small_sigma_preserves_image(self) -> None:
        """Test that very small sigma (< 0.01) preserves image."""
        op = BlurGaussianTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"sigma": 0.005})

        # Small sigma should be skipped
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_blur_reduces_sharpness(self) -> None:
        """Test that blur reduces high-frequency content."""
        op = BlurGaussianTaichiOperation()

        # Create checkerboard pattern (high frequency)
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[::2, ::2] = 1.0
        image[1::2, 1::2] = 1.0

        result = op.reference_numpy(image, {"sigma": 1.0})

        # After blur, edges should be smoother (values between 0 and 1)
        # Center pixels should no longer be exactly 0 or 1
        assert np.any((result > 0.1) & (result < 0.9))

    def test_blur_preserves_uniform_regions(self) -> None:
        """Test that blur preserves uniform color regions."""
        op = BlurGaussianTaichiOperation()

        # Uniform color
        image = np.full((10, 10, 3), 0.7, dtype=np.float32)
        result = op.reference_numpy(image, {"sigma": 2.0})

        # Uniform region should stay uniform
        np.testing.assert_allclose(result, image, rtol=1e-4, atol=1e-4)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = BlurGaussianTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(image, {"sigma": 1.5})
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = BlurGaussianTaichiOperation()

        # Create image at boundaries
        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = op.reference_numpy(image, {"sigma": 3.0})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = BlurGaussianTaichiOperation()

        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = op.reference_numpy(image, {"sigma": 1.0})

        assert result.dtype == np.float32

    def test_larger_sigma_more_blur(self) -> None:
        """Test that larger sigma produces more blur."""
        op = BlurGaussianTaichiOperation()

        # Sharp edge
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, 10:] = 1.0

        result_small = op.reference_numpy(image, {"sigma": 0.5})
        result_large = op.reference_numpy(image, {"sigma": 3.0})

        # At the edge (column 10), larger sigma should have more blending
        # Check middle row
        mid_row = 10
        edge_col = 10

        # Gradient at edge should be gentler with larger sigma
        grad_small = np.abs(
            result_small[mid_row, edge_col, 0] - result_small[mid_row, edge_col - 1, 0]
        )
        grad_large = np.abs(
            result_large[mid_row, edge_col, 0] - result_large[mid_row, edge_col - 1, 0]
        )

        assert grad_large < grad_small


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernels(self) -> None:
        """Test that apply_to_field invokes convolution kernels."""
        op = BlurGaussianTaichiOperation()

        # Mock source, dest, and temp fields
        source = Mock()
        dest = Mock()
        temp_field = Mock()

        mock_ti = MagicMock()
        mock_kernel_field = MagicMock()
        mock_ti.field.return_value = mock_kernel_field

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_horizontal"
            ) as mock_h,
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_vertical"
            ) as mock_v,
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={"intermediate": temp_field},
                params={"sigma": 2.0},
                height=64,
                width=64,
            )

            # Should call horizontal then vertical
            assert mock_h.call_count == 1
            assert mock_v.call_count == 1

            # Horizontal: source -> temp
            h_call_args = mock_h.call_args[0]
            assert h_call_args[0] is source
            assert h_call_args[1] is temp_field

            # Vertical: temp -> dest
            v_call_args = mock_v.call_args[0]
            assert v_call_args[0] is temp_field
            assert v_call_args[1] is dest

    def test_apply_to_field_small_sigma_copies(self) -> None:
        """Test that very small sigma triggers copy instead of blur."""
        op = BlurGaussianTaichiOperation()

        source = Mock()
        dest = Mock()
        temp_field = Mock()

        mock_ti = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_horizontal"
            ) as mock_h,
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_vertical"
            ) as mock_v,
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={"intermediate": temp_field},
                params={"sigma": 0.005},
                height=64,
                width=64,
            )

            # Should not call convolution kernels
            assert mock_h.call_count == 0
            assert mock_v.call_count == 0

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = BlurGaussianTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                False,
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={"intermediate": Mock()},
                params={"sigma": 1.0},
                height=64,
                width=64,
            )

    def test_apply_to_field_missing_temp_field(self) -> None:
        """Test that missing intermediate field raises ValueError."""
        op = BlurGaussianTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.ti", MagicMock()),
            pytest.raises(ValueError, match="Missing 'intermediate' temporary field"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},  # Missing intermediate
                params={"sigma": 1.0},
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = BlurGaussianTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_vector_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_ti.Vector.field.return_value = mock_vector_field

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_horizontal"
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.convolve_vertical"),
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = BlurGaussianTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_vector_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_ti.Vector.field.return_value = mock_vector_field

        with (
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.convolve_horizontal",
                side_effect=count_calls,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.convolve_vertical"),
            patch(
                "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_gaussian_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = BlurGaussianTaichiOperation()

        with patch(
            "sevenrad_stills.operations.blur_gaussian_taichi.TAICHI_AVAILABLE",
            False,
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_symmetric_blur(self) -> None:
        """Test that blur is symmetric (horizontal and vertical)."""
        op = BlurGaussianTaichiOperation()

        # Create vertical edge
        image_v = np.zeros((20, 20, 3), dtype=np.float32)
        image_v[:, 10:] = 1.0

        # Create horizontal edge
        image_h = np.zeros((20, 20, 3), dtype=np.float32)
        image_h[10:, :] = 1.0

        result_v = op.reference_numpy(image_v, {"sigma": 2.0})
        result_h = op.reference_numpy(image_h, {"sigma": 2.0})

        # Vertical and horizontal blur should behave symmetrically
        # Check gradient magnitudes along the edge
        grad_v = np.abs(np.diff(result_v[:, 10, 0]))
        grad_h = np.abs(np.diff(result_h[10, :, 0]))

        # Gradients should have similar statistics
        assert np.abs(np.mean(grad_v) - np.mean(grad_h)) < 0.1

    def test_all_channels_blurred_equally(self) -> None:
        """Test that all RGB channels are blurred equally."""
        op = BlurGaussianTaichiOperation()

        # Create image with different values per channel
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red
        image[:, :, 1] = 0.5  # Green
        image[:, :, 2] = 0.0  # Blue

        result = op.reference_numpy(image, {"sigma": 2.0})

        # Since the input is uniform per channel, output should also be uniform
        assert np.allclose(result[:, :, 0], 1.0, rtol=1e-4)
        assert np.allclose(result[:, :, 1], 0.5, rtol=1e-4)
        assert np.allclose(result[:, :, 2], 0.0, rtol=1e-4)

    def test_single_bright_pixel_spreads(self) -> None:
        """Test that a single bright pixel spreads with blur."""
        op = BlurGaussianTaichiOperation()

        # Black image with single white pixel in center
        image = np.zeros((21, 21, 3), dtype=np.float32)
        image[10, 10] = 1.0

        result = op.reference_numpy(image, {"sigma": 2.0})

        # Center should still be brightest
        assert result[10, 10, 0] == np.max(result[:, :, 0])

        # Neighboring pixels should have non-zero values (spread from center)
        assert result[10, 9, 0] > 0.01
        assert result[10, 11, 0] > 0.01
        assert result[9, 10, 0] > 0.01
        assert result[11, 10, 0] > 0.01

        # Corners should have lower values than immediate neighbors
        assert result[9, 9, 0] < result[10, 9, 0]

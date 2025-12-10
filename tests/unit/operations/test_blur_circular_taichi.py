"""Tests for BlurCircularTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.blur_circular_taichi import BlurCircularTaichiOperation


class TestBlurCircularTaichiOperationInit:
    """Test BlurCircularTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = BlurCircularTaichiOperation()

        assert op.name == "blur_circular_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that circular blur does not support in-place execution."""
        op = BlurCircularTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = BlurCircularTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that circular blur requires no temporary fields."""
        op = BlurCircularTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_radius(self) -> None:
        """Test that valid radius passes validation."""
        op = BlurCircularTaichiOperation()

        # Should not raise
        op.validate_params({"radius": 0})
        op.validate_params({"radius": 1})
        op.validate_params({"radius": 10})
        op.validate_params({"radius": 50})

    def test_missing_radius(self) -> None:
        """Test that missing radius raises ValueError."""
        op = BlurCircularTaichiOperation()

        with pytest.raises(ValueError, match="requires 'radius' parameter"):
            op.validate_params({})

    def test_invalid_radius_type(self) -> None:
        """Test that non-integer radius raises ValueError."""
        op = BlurCircularTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"radius": 5.5})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"radius": "large"})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"radius": None})

    def test_negative_radius(self) -> None:
        """Test that negative radius raises ValueError."""
        op = BlurCircularTaichiOperation()

        with pytest.raises(ValueError, match="must be non-negative"):
            op.validate_params({"radius": -1})

        with pytest.raises(ValueError, match="must be non-negative"):
            op.validate_params({"radius": -10})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_radius_identity(self) -> None:
        """Test that radius=0 preserves image."""
        op = BlurCircularTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"radius": 0})

        # With radius=0, output should match input exactly
        np.testing.assert_array_equal(result, image)

    def test_small_radius_blur(self) -> None:
        """Test that small radius produces blur."""
        op = BlurCircularTaichiOperation()

        # Create sharp edge
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[4:6, 4:6, :] = 1.0  # White square in center

        result = op.reference_numpy(image, {"radius": 2})

        # Center should be brighter than background (blur spreads white square)
        assert result[5, 5, 0] > 0.1  # Significantly brighter than zero
        assert result[5, 5, 0] < 1.0  # But not as bright as original due to averaging

        # Edges should be blurred
        # Adjacent pixels should have intermediate values
        assert 0.0 < result[3, 5, 0] < 1.0
        assert 0.0 < result[6, 5, 0] < 1.0

    def test_uniform_image_unchanged(self) -> None:
        """Test that uniform image remains uniform."""
        op = BlurCircularTaichiOperation()

        # Uniform gray image
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(image, {"radius": 5})

        # Blur of uniform image should be uniform
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = BlurCircularTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(image, {"radius": 3})
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = BlurCircularTaichiOperation()

        # Create image with values at boundaries
        image = np.random.rand(20, 20, 3).astype(np.float32)

        result = op.reference_numpy(image, {"radius": 5})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = BlurCircularTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"radius": 3})

        assert result.dtype == np.float32

    def test_different_radii(self) -> None:
        """Test that larger radius produces more blur."""
        op = BlurCircularTaichiOperation()

        # Create sharp feature
        image = np.zeros((30, 30, 3), dtype=np.float32)
        image[14:16, 14:16, :] = 1.0

        result_r2 = op.reference_numpy(image, {"radius": 2})
        result_r5 = op.reference_numpy(image, {"radius": 5})

        # Center pixel should be lower with larger radius (more averaging)
        center_r2 = result_r2[15, 15, 0]
        center_r5 = result_r5[15, 15, 0]

        # With larger radius, more zeros averaged in, so center value decreases
        assert center_r5 < center_r2

        # Far pixel should have more blur spread with larger radius
        far_r2 = result_r2[20, 15, 0]
        far_r5 = result_r5[20, 15, 0]

        # Larger radius spreads blur further
        assert far_r5 >= far_r2


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the convolution kernel."""
        op = BlurCircularTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        # Mock Taichi modules
        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_field.from_numpy = MagicMock()

        with (
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.convolve_2d"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", mock_ti),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"radius": 3},
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            # kernel field is call_args[2]
            assert call_args[3] == 3  # radius_h
            assert call_args[4] == 3  # radius_w
            assert call_args[5] == 0  # batch
            assert call_args[6] == 64  # height
            assert call_args[7] == 64  # width

    def test_apply_to_field_zero_radius_copies(self) -> None:
        """Test that radius=0 copies source to dest."""
        op = BlurCircularTaichiOperation()

        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.convolve_2d"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"radius": 0},
                height=64,
                width=64,
            )

            # Should not call convolution kernel
            mock_kernel.assert_not_called()
            # Should call copy_from instead
            dest.copy_from.assert_called_once_with(source)

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = BlurCircularTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                False,
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"radius": 3},
                height=64,
                width=64,
            )

    def test_kernel_reuse(self) -> None:
        """Test that kernel field is reused for same radius."""
        op = BlurCircularTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_field.from_numpy = MagicMock()

        with (
            patch("sevenrad_stills.operations.blur_circular_taichi.convolve_2d"),
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", mock_ti),
        ):
            # First call with radius=5
            op.apply_to_field(source, dest, {}, {"radius": 5}, 64, 64)
            first_kernel = op._kernel_field

            # Second call with same radius=5
            op.apply_to_field(source, dest, {}, {"radius": 5}, 64, 64)
            second_kernel = op._kernel_field

            # Kernel should be reused
            assert first_kernel is second_kernel

    def test_kernel_recreation(self) -> None:
        """Test that kernel field is recreated for different radius."""
        op = BlurCircularTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field_5 = MagicMock()
        mock_field_7 = MagicMock()
        mock_ti.field.side_effect = [mock_field_5, mock_field_7]

        with (
            patch("sevenrad_stills.operations.blur_circular_taichi.convolve_2d"),
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", mock_ti),
        ):
            # First call with radius=5
            op.apply_to_field(source, dest, {}, {"radius": 5}, 64, 64)
            first_kernel = op._kernel_field

            # Second call with different radius=7
            op.apply_to_field(source, dest, {}, {"radius": 7}, 64, 64)
            second_kernel = op._kernel_field

            # Kernel should be different
            assert first_kernel is not second_kernel


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = BlurCircularTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_scalar_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_scalar_field

        with (
            patch("sevenrad_stills.operations.blur_circular_taichi.convolve_2d"),
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = BlurCircularTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_scalar_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_scalar_field

        with (
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.convolve_2d",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.blur_circular_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = BlurCircularTaichiOperation()

        with patch(
            "sevenrad_stills.operations.blur_circular_taichi.TAICHI_AVAILABLE",
            False,
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_circular_symmetry(self) -> None:
        """Test that circular blur maintains circular symmetry."""
        op = BlurCircularTaichiOperation()

        # Create centered point source with larger bright area
        image = np.zeros((31, 31, 3), dtype=np.float32)
        image[14:17, 14:17, :] = 1.0  # 3x3 white square

        result = op.reference_numpy(image, {"radius": 3})

        # Check radial symmetry - points at same distance should have similar values
        center = result[15, 15, 0]
        assert center > 0.1  # Center should be brighter than background

        # Points at distance 3 in cardinal directions
        top = result[12, 15, 0]
        bottom = result[18, 15, 0]
        left = result[15, 12, 0]
        right = result[15, 18, 0]

        # All should be approximately equal (radial symmetry)
        np.testing.assert_allclose([top, bottom, left, right], top, rtol=0.2)

    def test_energy_conservation(self) -> None:
        """Test that blur conserves total energy (approximately)."""
        op = BlurCircularTaichiOperation()

        # Create image with known total
        image = np.random.rand(50, 50, 3).astype(np.float32)
        original_sum = image.sum()

        result = op.reference_numpy(image, {"radius": 5})

        # Total should be approximately conserved (within boundary effects)
        # Allow 5% tolerance due to boundary handling
        np.testing.assert_allclose(result.sum(), original_sum, rtol=0.05)

    def test_no_ringing_artifacts(self) -> None:
        """Test that blur doesn't introduce values outside input range."""
        op = BlurCircularTaichiOperation()

        # Image with values in [0.2, 0.8]
        image = np.random.uniform(0.2, 0.8, (20, 20, 3)).astype(np.float32)

        result = op.reference_numpy(image, {"radius": 3})

        # Output should stay within input range (no overshooting)
        assert np.all(result >= 0.2)
        assert np.all(result <= 0.8)

    def test_large_radius(self) -> None:
        """Test that large radius works correctly."""
        op = BlurCircularTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)

        # Should not raise
        result = op.reference_numpy(image, {"radius": 20})

        assert result.shape == image.shape
        assert result.dtype == np.float32

    def test_black_and_white_preserved(self) -> None:
        """Test that pure black and white images are preserved."""
        op = BlurCircularTaichiOperation()

        black = np.zeros((10, 10, 3), dtype=np.float32)
        white = np.ones((10, 10, 3), dtype=np.float32)

        black_result = op.reference_numpy(black, {"radius": 5})
        white_result = op.reference_numpy(white, {"radius": 5})

        np.testing.assert_allclose(black_result, black, atol=1e-5)
        np.testing.assert_allclose(white_result, white, atol=1e-5)

"""Tests for BufferCorruptionTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.buffer_corruption_taichi import (
    BufferCorruptionTaichiOperation,
)


class TestBufferCorruptionTaichiOperationInit:
    """Test BufferCorruptionTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = BufferCorruptionTaichiOperation()

        assert op.name == "buffer_corruption_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that buffer corruption does not support in-place execution."""
        op = BufferCorruptionTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = BufferCorruptionTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that buffer corruption requires no temporary fields."""
        op = BufferCorruptionTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params_xor(self) -> None:
        """Test that valid xor parameters pass validation."""
        op = BufferCorruptionTaichiOperation()

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.7,
            "tile_size_range": [0.05, 0.2],
            "seed": 42,
        }

        # Should not raise
        op.validate_params(params)

    def test_valid_params_invert(self) -> None:
        """Test that valid invert parameters pass validation."""
        op = BufferCorruptionTaichiOperation()

        params = {
            "corruption_type": "invert",
            "tile_count": 10,
            "severity": 0.5,
        }

        # Should not raise
        op.validate_params(params)

    def test_valid_params_shuffle(self) -> None:
        """Test that valid shuffle parameters pass validation."""
        op = BufferCorruptionTaichiOperation()

        params = {
            "corruption_type": "shuffle",
            "tile_count": 3,
            "severity": 0.9,
            "tile_size_range": [0.1, 0.3],
        }

        # Should not raise
        op.validate_params(params)

    def test_missing_corruption_type(self) -> None:
        """Test that missing corruption_type raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="requires 'corruption_type' parameter"):
            op.validate_params({"tile_count": 5, "severity": 0.5})

    def test_invalid_corruption_type(self) -> None:
        """Test that invalid corruption_type raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="corruption_type must be one of"):
            op.validate_params(
                {
                    "corruption_type": "invalid",
                    "tile_count": 5,
                    "severity": 0.5,
                }
            )

    def test_missing_tile_count(self) -> None:
        """Test that missing tile_count raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            op.validate_params({"corruption_type": "xor", "severity": 0.5})

    def test_invalid_tile_count_type(self) -> None:
        """Test that non-integer tile_count raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="tile_count must be an integer"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5.5,
                    "severity": 0.5,
                }
            )

    def test_invalid_tile_count_range(self) -> None:
        """Test that tile_count out of range raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="tile_count must be between"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 0,
                    "severity": 0.5,
                }
            )

        with pytest.raises(ValueError, match="tile_count must be between"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 21,
                    "severity": 0.5,
                }
            )

    def test_missing_severity(self) -> None:
        """Test that missing severity raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="requires 'severity' parameter"):
            op.validate_params({"corruption_type": "xor", "tile_count": 5})

    def test_invalid_severity_type(self) -> None:
        """Test that non-numeric severity raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="severity must be a number"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": "high",
                }
            )

    def test_invalid_severity_range(self) -> None:
        """Test that severity out of range raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="severity must be between"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": -0.1,
                }
            )

        with pytest.raises(ValueError, match="severity must be between"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 1.1,
                }
            )

    def test_invalid_tile_size_range_format(self) -> None:
        """Test that invalid tile_size_range format raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="must be a list/tuple of two numbers"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 0.5,
                    "tile_size_range": [0.1],
                }
            )

    def test_invalid_tile_size_range_values(self) -> None:
        """Test that invalid tile_size_range values raise ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="tile_size_range values must be numbers"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 0.5,
                    "tile_size_range": ["0.1", 0.2],
                }
            )

    def test_invalid_tile_size_range_bounds(self) -> None:
        """Test that tile_size_range out of bounds raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="tile_size_range values must be between"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 0.5,
                    "tile_size_range": [0.0, 0.2],
                }
            )

    def test_invalid_tile_size_range_order(self) -> None:
        """Test that min > max in tile_size_range raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="min must be less than or equal to max"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 0.5,
                    "tile_size_range": [0.3, 0.1],
                }
            )

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = BufferCorruptionTaichiOperation()

        with pytest.raises(ValueError, match="seed must be an integer"):
            op.validate_params(
                {
                    "corruption_type": "xor",
                    "tile_count": 5,
                    "severity": 0.5,
                    "seed": 42.5,
                }
            )


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_xor_mode_zero_severity(self) -> None:
        """Test that XOR mode with severity=0 preserves image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.0,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_invert_mode_zero_severity(self) -> None:
        """Test that invert mode with severity=0 preserves image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        params = {
            "corruption_type": "invert",
            "tile_count": 5,
            "severity": 0.0,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_shuffle_mode_zero_severity(self) -> None:
        """Test that shuffle mode with severity=0 mostly preserves image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        params = {
            "corruption_type": "shuffle",
            "tile_count": 5,
            "severity": 0.0,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # With severity=0, probability of shuffle is 0, so image should be preserved
        np.testing.assert_allclose(result, image, rtol=1e-5, atol=1e-5)

    def test_xor_mode_changes_image(self) -> None:
        """Test that XOR mode with nonzero severity changes image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 10,
            "severity": 0.8,
            "tile_size_range": [0.2, 0.5],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # Image should be modified
        assert not np.allclose(result, image)

    def test_invert_mode_changes_image(self) -> None:
        """Test that invert mode with nonzero severity changes image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        params = {
            "corruption_type": "invert",
            "tile_count": 10,
            "severity": 0.8,
            "tile_size_range": [0.2, 0.5],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # Image should be modified
        assert not np.allclose(result, image)

    def test_shuffle_mode_changes_image(self) -> None:
        """Test that shuffle mode with high severity changes image."""
        op = BufferCorruptionTaichiOperation()

        # Create image with distinct channels
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red
        image[:, :, 1] = 0.5  # Green
        image[:, :, 2] = 0.0  # Blue

        params = {
            "corruption_type": "shuffle",
            "tile_count": 10,
            "severity": 1.0,  # Always shuffle
            "tile_size_range": [0.2, 0.5],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # Image should be modified (some tiles shuffled)
        assert not np.allclose(result, image)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = BufferCorruptionTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            params = {
                "corruption_type": "xor",
                "tile_count": 5,
                "severity": 0.5,
                "seed": 42,
            }
            result = op.reference_numpy(image, params)
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        for corruption_type in ["xor", "invert", "shuffle"]:
            params = {
                "corruption_type": corruption_type,
                "tile_count": 5,
                "severity": 1.0,
                "seed": 42,
            }
            result = op.reference_numpy(image, params)

            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.5,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.dtype == np.float32

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same results."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.7,
            "seed": 42,
        }

        result1 = op.reference_numpy(image, params)
        result2 = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_produces_different_results(self) -> None:
        """Test that different seeds produce different results."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        params1 = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.7,
            "seed": 42,
        }

        params2 = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.7,
            "seed": 123,
        }

        result1 = op.reference_numpy(image, params1)
        result2 = op.reference_numpy(image, params2)

        assert not np.allclose(result1, result2)

    def test_invert_mode_full_severity(self) -> None:
        """Test that invert mode with severity=1.0 fully inverts tiles."""
        op = BufferCorruptionTaichiOperation()

        # Create uniform image
        image = np.full((10, 10, 3), 0.3, dtype=np.float32)

        params = {
            "corruption_type": "invert",
            "tile_count": 1,
            "severity": 1.0,
            "tile_size_range": [0.8, 0.9],  # Large tile
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # Most pixels should be inverted (0.7)
        # Since tile is large, most of image should be affected
        inverted_count = np.sum(np.isclose(result[:, :, 0], 0.7, atol=0.01))
        original_count = np.sum(np.isclose(result[:, :, 0], 0.3, atol=0.01))

        # Inverted pixels should be more common
        assert inverted_count > original_count


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_xor_calls_kernel(self) -> None:
        """Test that apply_to_field invokes XOR kernel."""
        op = BufferCorruptionTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_ti.i32 = int
        mock_ti.f32 = float

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_xor"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.buffer_corruption_taichi.ti", mock_ti),
        ):
            params = {
                "corruption_type": "xor",
                "tile_count": 5,
                "severity": 0.7,
                "seed": 42,
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

    def test_apply_to_field_invert_calls_kernel(self) -> None:
        """Test that apply_to_field invokes invert kernel."""
        op = BufferCorruptionTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_ti.i32 = int
        mock_ti.f32 = float

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_invert"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.buffer_corruption_taichi.ti", mock_ti),
        ):
            params = {
                "corruption_type": "invert",
                "tile_count": 3,
                "severity": 0.5,
                "seed": 42,
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

    def test_apply_to_field_shuffle_calls_kernel(self) -> None:
        """Test that apply_to_field invokes shuffle kernel."""
        op = BufferCorruptionTaichiOperation()

        source = Mock()
        dest = Mock()

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.field.return_value = mock_field
        mock_ti.i32 = int
        mock_ti.f32 = float

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_shuffle"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.buffer_corruption_taichi.ti", mock_ti),
        ):
            params = {
                "corruption_type": "shuffle",
                "tile_count": 5,
                "severity": 0.8,
                "seed": 42,
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

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = BufferCorruptionTaichiOperation()

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.5,
        }

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                False,
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params=params,
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = BufferCorruptionTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_vector_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field
        mock_ti.i32 = int
        mock_ti.f32 = float

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_xor"
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_invert"
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_shuffle"
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.buffer_corruption_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = BufferCorruptionTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_vector_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field
        mock_ti.i32 = int
        mock_ti.f32 = float

        with (
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_xor",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_invert"
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi._buffer_corruption_kernel_shuffle"
            ),
            patch(
                "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.buffer_corruption_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        # Should only call once (3 kernels compiled in first warmup)
        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = BufferCorruptionTaichiOperation()

        with patch(
            "sevenrad_stills.operations.buffer_corruption_taichi.TAICHI_AVAILABLE",
            False,
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestEdgeCases:
    """Test edge cases and corner conditions."""

    def test_single_tile(self) -> None:
        """Test corruption with single tile."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 1,
            "severity": 0.5,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_max_tiles(self) -> None:
        """Test corruption with maximum number of tiles."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 20,
            "severity": 0.5,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_small_image(self) -> None:
        """Test corruption on very small image."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(5, 5, 3).astype(np.float32)

        params = {
            "corruption_type": "invert",
            "tile_count": 2,
            "severity": 0.8,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_large_tiles(self) -> None:
        """Test corruption with large tiles."""
        op = BufferCorruptionTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)

        params = {
            "corruption_type": "shuffle",
            "tile_count": 2,
            "severity": 1.0,
            "tile_size_range": [0.5, 0.8],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_all_white_image(self) -> None:
        """Test corruption on all-white image."""
        op = BufferCorruptionTaichiOperation()

        image = np.ones((10, 10, 3), dtype=np.float32)

        params = {
            "corruption_type": "xor",
            "tile_count": 5,
            "severity": 0.5,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_all_black_image(self) -> None:
        """Test corruption on all-black image."""
        op = BufferCorruptionTaichiOperation()

        image = np.zeros((10, 10, 3), dtype=np.float32)

        params = {
            "corruption_type": "invert",
            "tile_count": 5,
            "severity": 0.5,
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

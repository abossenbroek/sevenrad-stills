"""Tests for BandSwapTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.band_swap_taichi import BandSwapTaichiOperation


class TestBandSwapTaichiOperationInit:
    """Test BandSwapTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = BandSwapTaichiOperation()

        assert op.name == "band_swap_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that band swap does not support in-place execution."""
        op = BandSwapTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = BandSwapTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that band swap requires no temporary fields."""
        op = BandSwapTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_params_minimal(self) -> None:
        """Test that minimal valid params pass validation."""
        op = BandSwapTaichiOperation()

        # Should not raise
        op.validate_params({"tile_count": 5, "permutation": "BGR"})
        op.validate_params({"tile_count": 1, "permutation": "GRB"})
        op.validate_params({"tile_count": 50, "permutation": "RBG"})

    def test_valid_params_with_optional(self) -> None:
        """Test that params with optional fields pass validation."""
        op = BandSwapTaichiOperation()

        # Should not raise
        op.validate_params(
            {
                "tile_count": 5,
                "permutation": "BGR",
                "tile_size_range": [0.1, 0.3],
                "seed": 42,
            }
        )

    def test_missing_tile_count(self) -> None:
        """Test that missing tile_count raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="requires 'tile_count' parameter"):
            op.validate_params({"permutation": "BGR"})

    def test_missing_permutation(self) -> None:
        """Test that missing permutation raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="requires 'permutation' parameter"):
            op.validate_params({"tile_count": 5})

    def test_invalid_tile_count_type(self) -> None:
        """Test that non-integer tile_count raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"tile_count": 5.5, "permutation": "BGR"})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"tile_count": "5", "permutation": "BGR"})

    def test_invalid_tile_count_range(self) -> None:
        """Test that out-of-range tile_count raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer between"):
            op.validate_params({"tile_count": 0, "permutation": "BGR"})

        with pytest.raises(ValueError, match="must be an integer between"):
            op.validate_params({"tile_count": 51, "permutation": "BGR"})

        with pytest.raises(ValueError, match="must be an integer between"):
            op.validate_params({"tile_count": -1, "permutation": "BGR"})

    def test_invalid_permutation(self) -> None:
        """Test that invalid permutation raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="Permutation must be one of"):
            op.validate_params({"tile_count": 5, "permutation": "RGB"})

        with pytest.raises(ValueError, match="Permutation must be one of"):
            op.validate_params({"tile_count": 5, "permutation": "XYZ"})

        with pytest.raises(ValueError, match="Permutation must be one of"):
            op.validate_params({"tile_count": 5, "permutation": ""})

    def test_invalid_tile_size_range_structure(self) -> None:
        """Test that malformed tile_size_range raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="must be a list/tuple of two numbers"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": [0.1]}
            )

        with pytest.raises(ValueError, match="must be a list/tuple of two numbers"):
            op.validate_params(
                {
                    "tile_count": 5,
                    "permutation": "BGR",
                    "tile_size_range": [0.1, 0.2, 0.3],
                }
            )

        with pytest.raises(ValueError, match="must be a list/tuple of two numbers"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": 0.1}
            )

    def test_invalid_tile_size_range_types(self) -> None:
        """Test that non-numeric tile_size_range values raise ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="values must be numbers"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": ["0.1", 0.2]}
            )

    def test_invalid_tile_size_range_bounds(self) -> None:
        """Test that out-of-range tile_size_range values raise ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="must be between"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": [0.0, 0.2]}
            )

        with pytest.raises(ValueError, match="must be between"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": [0.1, 1.5]}
            )

    def test_invalid_tile_size_range_order(self) -> None:
        """Test that inverted tile_size_range raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="min must be <= max"):
            op.validate_params(
                {"tile_count": 5, "permutation": "BGR", "tile_size_range": [0.3, 0.1]}
            )

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params({"tile_count": 5, "permutation": "BGR", "seed": "42"})

        with pytest.raises(ValueError, match="Seed must be an integer"):
            op.validate_params({"tile_count": 5, "permutation": "BGR", "seed": 42.5})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_tiles(self) -> None:
        """Test that zero tile_count raises during validation."""
        op = BandSwapTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer between"):
            op.validate_params({"tile_count": 0, "permutation": "BGR"})

    def test_single_tile_entire_image(self) -> None:
        """Test that single tile covering entire image swaps all channels."""
        op = BandSwapTaichiOperation()

        # Create test image with distinct channels
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red = 1.0
        image[:, :, 1] = 0.5  # Green = 0.5
        image[:, :, 2] = 0.0  # Blue = 0.0

        params = {
            "tile_count": 1,
            "permutation": "BGR",  # Swap to B,G,R
            "tile_size_range": [1.0, 1.0],  # Full image
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # With BGR permutation: R←B, G←G, B←R
        # Original: R=1.0, G=0.5, B=0.0
        # Result: R=0.0, G=0.5, B=1.0
        assert np.allclose(result[:, :, 0], 0.0)  # Red channel from Blue
        assert np.allclose(result[:, :, 1], 0.5)  # Green unchanged
        assert np.allclose(result[:, :, 2], 1.0)  # Blue channel from Red

    def test_grb_permutation(self) -> None:
        """Test GRB permutation swaps correctly."""
        op = BandSwapTaichiOperation()

        # Create test image
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :, 0] = 0.8  # Red
        image[:, :, 1] = 0.5  # Green
        image[:, :, 2] = 0.2  # Blue

        params = {
            "tile_count": 1,
            "permutation": "GRB",  # R←G, G←R, B←B
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # GRB means: output[R] = input[G], output[G] = input[R], output[B] = input[B]
        assert np.allclose(result[:, :, 0], 0.5)  # Red from Green
        assert np.allclose(result[:, :, 1], 0.8)  # Green from Red
        assert np.allclose(result[:, :, 2], 0.2)  # Blue unchanged

    def test_rbg_permutation(self) -> None:
        """Test RBG permutation swaps blue and green."""
        op = BandSwapTaichiOperation()

        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red
        image[:, :, 1] = 0.6  # Green
        image[:, :, 2] = 0.3  # Blue

        params = {
            "tile_count": 1,
            "permutation": "RBG",  # R←R, G←B, B←G
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        assert np.allclose(result[:, :, 0], 1.0)  # Red unchanged
        assert np.allclose(result[:, :, 1], 0.3)  # Green from Blue
        assert np.allclose(result[:, :, 2], 0.6)  # Blue from Green

    def test_deterministic_with_seed(self) -> None:
        """Test that same seed produces same result."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(20, 20, 3).astype(np.float32)
        params = {"tile_count": 5, "permutation": "BGR", "seed": 12345}

        result1 = op.reference_numpy(image, params)
        result2 = op.reference_numpy(image, params)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_different_result(self) -> None:
        """Test that different seeds produce different results."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)

        result1 = op.reference_numpy(
            image, {"tile_count": 10, "permutation": "BGR", "seed": 1}
        )
        result2 = op.reference_numpy(
            image, {"tile_count": 10, "permutation": "BGR", "seed": 2}
        )

        # Should be different (with high probability)
        assert not np.array_equal(result1, result2)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = BandSwapTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(
                image, {"tile_count": 3, "permutation": "BGR", "seed": 42}
            )
            assert result.shape == shape

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"tile_count": 2, "permutation": "BGR", "seed": 42}
        )

        assert result.dtype == np.float32

    def test_partial_tiles(self) -> None:
        """Test that partial tiles only affect their region."""
        op = BandSwapTaichiOperation()

        # Create image with distinct pattern
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, :, 0] = 1.0  # Red everywhere

        params = {
            "tile_count": 1,
            "permutation": "BGR",
            "tile_size_range": [0.25, 0.25],  # Small tile
            "seed": 100,
        }

        result = op.reference_numpy(image, params)

        # Some pixels should be unchanged (red=1.0), some swapped
        unchanged_pixels = np.sum(result[:, :, 0] == 1.0)
        changed_pixels = np.sum(result[:, :, 0] != 1.0)

        # Should have both unchanged and changed pixels
        assert unchanged_pixels > 0
        assert changed_pixels > 0

    def test_multiple_tiles(self) -> None:
        """Test that multiple tiles can affect image."""
        op = BandSwapTaichiOperation()

        image = np.ones((30, 30, 3), dtype=np.float32)
        image[:, :, 0] = 1.0
        image[:, :, 1] = 0.5
        image[:, :, 2] = 0.0

        params = {
            "tile_count": 10,
            "permutation": "BGR",
            "tile_size_range": [0.1, 0.3],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)

        # With 10 tiles, we expect some significant changes
        # Check that at least some pixels have been swapped
        swapped = np.sum(result[:, :, 0] != 1.0)
        assert swapped > 0


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = BandSwapTaichiOperation()

        source = Mock()
        dest = Mock()

        # Mock tile field
        mock_tile_field = Mock()
        mock_ti = MagicMock()
        mock_ti.field.return_value = mock_tile_field

        with (
            patch(
                "sevenrad_stills.operations.band_swap_taichi._band_swap_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.band_swap_taichi.ti", mock_ti),
        ):
            params = {"tile_count": 3, "permutation": "BGR", "seed": 42}
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
            # call_args[2] is tile_field
            assert call_args[3] == 2  # perm_r (BGR → B=2)
            assert call_args[4] == 1  # perm_g (BGR → G=1)
            assert call_args[5] == 0  # perm_b (BGR → R=0)
            assert call_args[6] == 0  # batch
            assert call_args[7] == 64  # height
            assert call_args[8] == 64  # width
            assert call_args[9] == 3  # tile_count

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = BandSwapTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"tile_count": 5, "permutation": "BGR"},
                height=64,
                width=64,
            )

    def test_apply_to_field_with_custom_tile_size(self) -> None:
        """Test that custom tile_size_range is used."""
        op = BandSwapTaichiOperation()

        mock_tile_field = Mock()
        mock_ti = MagicMock()
        mock_ti.field.return_value = mock_tile_field

        with (
            patch(
                "sevenrad_stills.operations.band_swap_taichi._band_swap_kernel"
            ) as mock_kernel,
            patch("sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.band_swap_taichi.ti", mock_ti),
        ):
            params = {
                "tile_count": 2,
                "permutation": "GRB",
                "tile_size_range": [0.3, 0.5],
                "seed": 100,
            }
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params=params,
                height=100,
                width=100,
            )

            # Kernel should be called
            mock_kernel.assert_called_once()


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = BandSwapTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field

        with (
            patch("sevenrad_stills.operations.band_swap_taichi._band_swap_kernel"),
            patch("sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.band_swap_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = BandSwapTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.band_swap_taichi._band_swap_kernel",
                side_effect=count_calls,
            ),
            patch("sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", True),
            patch("sevenrad_stills.operations.band_swap_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = BandSwapTaichiOperation()

        with patch(
            "sevenrad_stills.operations.band_swap_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestPermutations:
    """Test all valid permutations produce correct mappings."""

    def test_all_permutations(self) -> None:
        """Test that all valid permutations work correctly."""
        op = BandSwapTaichiOperation()

        # Create image with distinct channels
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[:, :, 0] = 0.9  # Red
        image[:, :, 1] = 0.6  # Green
        image[:, :, 2] = 0.3  # Blue

        permutations_expected = {
            "GRB": (0.6, 0.9, 0.3),  # R←G, G←R, B←B
            "BGR": (0.3, 0.6, 0.9),  # R←B, G←G, B←R
            "BRG": (0.3, 0.9, 0.6),  # R←B, G←R, B←G
            "GBR": (0.6, 0.3, 0.9),  # R←G, G←B, B←R
            "RBG": (0.9, 0.3, 0.6),  # R←R, G←B, B←G
        }

        for perm, expected in permutations_expected.items():
            params = {
                "tile_count": 1,
                "permutation": perm,
                "tile_size_range": [1.0, 1.0],
                "seed": 42,
            }

            result = op.reference_numpy(image, params)

            assert np.allclose(result[:, :, 0], expected[0]), f"Failed for {perm} R"
            assert np.allclose(result[:, :, 1], expected[1]), f"Failed for {perm} G"
            assert np.allclose(result[:, :, 2], expected[2]), f"Failed for {perm} B"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_tile_size(self) -> None:
        """Test with minimum tile size."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(100, 100, 3).astype(np.float32)
        params = {
            "tile_count": 5,
            "permutation": "BGR",
            "tile_size_range": [0.01, 0.01],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)
        assert result.shape == image.shape

    def test_maximum_tile_size(self) -> None:
        """Test with maximum tile size."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)
        params = {
            "tile_count": 1,
            "permutation": "BGR",
            "tile_size_range": [1.0, 1.0],
            "seed": 42,
        }

        result = op.reference_numpy(image, params)
        assert result.shape == image.shape

    def test_maximum_tile_count(self) -> None:
        """Test with maximum number of tiles."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(50, 50, 3).astype(np.float32)
        params = {
            "tile_count": 50,
            "permutation": "BGR",
            "tile_size_range": [0.05, 0.1],
            "seed": 42,
        }

        # Should not raise
        result = op.reference_numpy(image, params)
        assert result.shape == image.shape

    def test_small_image(self) -> None:
        """Test with very small image."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(2, 2, 3).astype(np.float32)
        params = {
            "tile_count": 1,
            "permutation": "BGR",
            "seed": 42,
        }

        result = op.reference_numpy(image, params)
        assert result.shape == (2, 2, 3)

    def test_rectangular_image(self) -> None:
        """Test with non-square image."""
        op = BandSwapTaichiOperation()

        image = np.random.rand(10, 50, 3).astype(np.float32)
        params = {
            "tile_count": 5,
            "permutation": "GRB",
            "seed": 42,
        }

        result = op.reference_numpy(image, params)
        assert result.shape == (10, 50, 3)

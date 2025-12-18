"""Tests for SaltPepperTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.salt_pepper_taichi import SaltPepperTaichiOperation


class TestSaltPepperTaichiOperationInit:
    """Test SaltPepperTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = SaltPepperTaichiOperation()

        assert op.name == "salt_pepper_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that salt and pepper supports in-place execution."""
        op = SaltPepperTaichiOperation()

        assert op.supports_inplace is True

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = SaltPepperTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements_empty(self) -> None:
        """Test that salt and pepper requires no temporary fields."""
        op = SaltPepperTaichiOperation()

        assert op.temp_field_requirements == []


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_amount(self) -> None:
        """Test that valid amount with salt_vs_pepper passes validation."""
        op = SaltPepperTaichiOperation()

        # Should not raise
        op.validate_params({"amount": 0.0, "salt_vs_pepper": 0.5})
        op.validate_params({"amount": 0.5, "salt_vs_pepper": 0.5})
        op.validate_params({"amount": 1.0, "salt_vs_pepper": 0.5})
        op.validate_params({"amount": 0, "salt_vs_pepper": 0.5})  # int is ok

    def test_valid_amount_with_seed(self) -> None:
        """Test that valid amount with seed passes validation."""
        op = SaltPepperTaichiOperation()

        # Should not raise
        op.validate_params({"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42})
        op.validate_params({"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 0})
        op.validate_params({"amount": 0.9, "salt_vs_pepper": 0.5, "seed": -1})

    def test_backward_compat_density(self) -> None:
        """Test that density parameter still works for backward compatibility."""
        op = SaltPepperTaichiOperation()

        # Should not raise - density is accepted for backward compatibility
        op.validate_params({"density": 0.5, "salt_vs_pepper": 0.5})
        op.validate_params({"density": 0.1, "salt_vs_pepper": 0.5, "seed": 42})

    def test_missing_amount(self) -> None:
        """Test that missing amount/density raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="requires 'amount' parameter"):
            op.validate_params({"salt_vs_pepper": 0.5})

        with pytest.raises(ValueError, match="requires 'amount' parameter"):
            op.validate_params({"seed": 42})

    def test_missing_salt_vs_pepper(self) -> None:
        """Test that missing salt_vs_pepper raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="requires 'salt_vs_pepper' parameter"):
            op.validate_params({"amount": 0.5})

        with pytest.raises(ValueError, match="requires 'salt_vs_pepper' parameter"):
            op.validate_params({"amount": 0.5, "seed": 42})

    def test_invalid_amount_type(self) -> None:
        """Test that non-numeric amount raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"amount": "high", "salt_vs_pepper": 0.5})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"amount": None, "salt_vs_pepper": 0.5})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"amount": [0.5], "salt_vs_pepper": 0.5})

    def test_amount_out_of_range(self) -> None:
        """Test that amount outside [0, 1] raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            op.validate_params({"amount": -0.1, "salt_vs_pepper": 0.5})

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            op.validate_params({"amount": 1.1, "salt_vs_pepper": 0.5})

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            op.validate_params({"amount": 2.0, "salt_vs_pepper": 0.5})

    def test_invalid_salt_vs_pepper_type(self) -> None:
        """Test that non-numeric salt_vs_pepper raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": "high"})

        with pytest.raises(ValueError, match="must be a number"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": None})

    def test_salt_vs_pepper_out_of_range(self) -> None:
        """Test that salt_vs_pepper outside [0, 1] raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": -0.1})

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": 1.1})

    def test_invalid_seed_type(self) -> None:
        """Test that non-integer seed raises ValueError."""
        op = SaltPepperTaichiOperation()

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 3.14})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": 0.5, "seed": "42"})

        with pytest.raises(ValueError, match="must be an integer"):
            op.validate_params({"amount": 0.5, "salt_vs_pepper": 0.5, "seed": None})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_zero_amount(self) -> None:
        """Test that amount=0.0 preserves image unchanged."""
        op = SaltPepperTaichiOperation()

        # Create test image with varying colors
        image = np.array(
            [[[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], [[0.5, 0.0, 1.0], [0.5, 0.5, 0.5]]],
            dtype=np.float32,
        )

        result = op.reference_numpy(
            image, {"amount": 0.0, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # With amount=0.0, output should match input exactly
        np.testing.assert_array_equal(result, image)

    def test_full_amount_produces_salt_or_pepper(self) -> None:
        """Test that amount=1.0 affects all pixels."""
        op = SaltPepperTaichiOperation()

        # Create test image
        image = np.array(
            [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]],
            dtype=np.float32,
        )

        result = op.reference_numpy(
            image, {"amount": 1.0, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # All pixels should be either white (1,1,1) or black (0,0,0)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                pixel = result[i, j]
                is_salt = np.allclose(pixel, [1.0, 1.0, 1.0])
                is_pepper = np.allclose(pixel, [0.0, 0.0, 0.0])
                assert is_salt or is_pepper

    def test_partial_amount_affects_some_pixels(self) -> None:
        """Test that 0 < amount < 1 affects some but not all pixels."""
        op = SaltPepperTaichiOperation()

        # Create uniform gray image
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        result = op.reference_numpy(
            image, {"amount": 0.3, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # Count pixels that changed
        changed_pixels = 0
        unchanged_pixels = 0

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.allclose(result[i, j], image[i, j]):
                    unchanged_pixels += 1
                else:
                    changed_pixels += 1

        # With amount=0.3, we expect roughly 30% changed
        # (allowing some variance due to randomness)
        assert changed_pixels > 0
        assert unchanged_pixels > 0

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces identical results."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)

        result1 = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5, "seed": 123}
        )
        result2 = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5, "seed": 123}
        )

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different noise patterns."""
        op = SaltPepperTaichiOperation()

        image = np.full((20, 20, 3), 0.5, dtype=np.float32)

        result1 = op.reference_numpy(
            image, {"amount": 0.3, "salt_vs_pepper": 0.5, "seed": 1}
        )
        result2 = op.reference_numpy(
            image, {"amount": 0.3, "salt_vs_pepper": 0.5, "seed": 2}
        )

        # Results should differ
        assert not np.array_equal(result1, result2)

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = SaltPepperTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            result = op.reference_numpy(
                image, {"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42}
            )
            assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is in [0, 1]."""
        op = SaltPepperTaichiOperation()

        # Create image with various values
        image = np.random.rand(10, 10, 3).astype(np.float32)

        result = op.reference_numpy(
            image, {"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 42}
        )

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(4, 4, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42}
        )

        assert result.dtype == np.float32

    def test_default_seed(self) -> None:
        """Test that missing seed defaults to 0."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(5, 5, 3).astype(np.float32)

        # These should produce identical results
        result_no_seed = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5}
        )
        result_seed_zero = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5, "seed": 0}
        )

        np.testing.assert_array_equal(result_no_seed, result_seed_zero)

    def test_salt_vs_pepper_ratio(self) -> None:
        """Test that salt_vs_pepper parameter controls the ratio."""
        op = SaltPepperTaichiOperation()

        # Create uniform gray image
        image = np.full((50, 50, 3), 0.5, dtype=np.float32)

        # Test with all salt (salt_vs_pepper=1.0)
        result_all_salt = op.reference_numpy(
            image, {"amount": 1.0, "salt_vs_pepper": 1.0, "seed": 42}
        )
        # All pixels should be white
        for i in range(result_all_salt.shape[0]):
            for j in range(result_all_salt.shape[1]):
                assert np.allclose(result_all_salt[i, j], [1.0, 1.0, 1.0])

        # Test with all pepper (salt_vs_pepper=0.0)
        result_all_pepper = op.reference_numpy(
            image, {"amount": 1.0, "salt_vs_pepper": 0.0, "seed": 42}
        )
        # All pixels should be black
        for i in range(result_all_pepper.shape[0]):
            for j in range(result_all_pepper.shape[1]):
                assert np.allclose(result_all_pepper[i, j], [0.0, 0.0, 0.0])


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernel(self) -> None:
        """Test that apply_to_field invokes the kernel."""
        op = SaltPepperTaichiOperation()

        # Mock source and dest fields
        source = Mock()
        dest = Mock()

        with (
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi._salt_pepper_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.salt_pepper_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={},
                params={"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42},
                height=64,
                width=64,
            )

            mock_kernel.assert_called_once()
            call_args = mock_kernel.call_args[0]
            assert call_args[0] is source
            assert call_args[1] is dest
            assert call_args[2] == 0.1  # amount
            assert call_args[3] == 0.5  # salt_vs_pepper
            assert call_args[4] == 42  # seed
            assert call_args[5] == 0  # batch
            assert call_args[6] == 64  # height
            assert call_args[7] == 64  # width

    def test_apply_to_field_default_seed(self) -> None:
        """Test that apply_to_field uses default seed=0 when not provided."""
        op = SaltPepperTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi._salt_pepper_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.salt_pepper_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"amount": 0.1, "salt_vs_pepper": 0.5},
                height=32,
                width=32,
            )

            call_args = mock_kernel.call_args[0]
            assert call_args[4] == 0  # seed should default to 0

    def test_apply_to_field_backward_compat_density(self) -> None:
        """Test that apply_to_field accepts density for backward compatibility."""
        op = SaltPepperTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi._salt_pepper_kernel"
            ) as mock_kernel,
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.salt_pepper_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"density": 0.1, "salt_vs_pepper": 0.5, "seed": 42},
                height=32,
                width=32,
            )

            call_args = mock_kernel.call_args[0]
            assert call_args[2] == 0.1  # amount (from density)

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = SaltPepperTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},
                params={"amount": 0.1, "salt_vs_pepper": 0.5, "seed": 42},
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = SaltPepperTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch("sevenrad_stills.operations.salt_pepper_taichi._salt_pepper_kernel"),
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.salt_pepper_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = SaltPepperTaichiOperation()

        call_count = 0

        def count_calls(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1

        mock_ti = MagicMock()
        mock_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_field

        with (
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi._salt_pepper_kernel",
                side_effect=count_calls,
            ),
            patch(
                "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.salt_pepper_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = SaltPepperTaichiOperation()

        with patch(
            "sevenrad_stills.operations.salt_pepper_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_only_produces_black_or_white(self) -> None:
        """Test that affected pixels are exactly black or white."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # Check each pixel
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                pixel = result[i, j]
                # Pixel is either unchanged, white, or black
                is_unchanged = np.allclose(pixel, image[i, j])
                is_white = np.allclose(pixel, [1.0, 1.0, 1.0])
                is_black = np.allclose(pixel, [0.0, 0.0, 0.0])
                assert is_unchanged or is_white or is_black

    def test_pure_white_can_become_black(self) -> None:
        """Test that pure white pixels can be changed to black."""
        op = SaltPepperTaichiOperation()

        # All white image
        white = np.ones((5, 5, 3), dtype=np.float32)

        result = op.reference_numpy(
            white, {"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # Some pixels should be black
        has_black = False
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.allclose(result[i, j], [0.0, 0.0, 0.0]):
                    has_black = True
                    break

        assert has_black

    def test_pure_black_can_become_white(self) -> None:
        """Test that pure black pixels can be changed to white."""
        op = SaltPepperTaichiOperation()

        # All black image
        black = np.zeros((5, 5, 3), dtype=np.float32)

        result = op.reference_numpy(
            black, {"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # Some pixels should be white
        has_white = False
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.allclose(result[i, j], [1.0, 1.0, 1.0]):
                    has_white = True
                    break

        assert has_white

    def test_affects_all_rgb_channels_together(self) -> None:
        """Test that all RGB channels of a pixel change together."""
        op = SaltPepperTaichiOperation()

        # Create image with distinct channel values
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(
            image, {"amount": 0.3, "salt_vs_pepper": 0.5, "seed": 42}
        )

        # Check that if a pixel changed, all channels are equal
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if not np.allclose(result[i, j], image[i, j]):
                    # Pixel changed - all channels should be equal
                    assert result[i, j, 0] == result[i, j, 1] == result[i, j, 2]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_pixel_image(self) -> None:
        """Test operation on 1x1 image."""
        op = SaltPepperTaichiOperation()

        image = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)

        # Should not raise
        result = op.reference_numpy(
            image, {"amount": 0.5, "salt_vs_pepper": 0.5, "seed": 42}
        )
        assert result.shape == (1, 1, 3)

    def test_very_small_amount(self) -> None:
        """Test with very small amount value."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(100, 100, 3).astype(np.float32)

        # Should not raise
        result = op.reference_numpy(
            image, {"amount": 0.001, "salt_vs_pepper": 0.5, "seed": 42}
        )
        assert result.shape == image.shape

    def test_negative_seed(self) -> None:
        """Test that negative seed works correctly."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(5, 5, 3).astype(np.float32)

        # Should not raise
        result = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5, "seed": -999}
        )
        assert result.shape == image.shape

    def test_large_seed(self) -> None:
        """Test that large seed works correctly."""
        op = SaltPepperTaichiOperation()

        image = np.random.rand(5, 5, 3).astype(np.float32)

        # Should not raise
        result = op.reference_numpy(
            image, {"amount": 0.2, "salt_vs_pepper": 0.5, "seed": 999999}
        )
        assert result.shape == image.shape

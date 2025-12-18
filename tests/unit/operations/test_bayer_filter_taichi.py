"""Tests for BayerFilterTaichiOperation."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.operations.bayer_filter_taichi import BayerFilterTaichiOperation


class TestBayerFilterTaichiOperationInit:
    """Test BayerFilterTaichiOperation initialization."""

    def test_initialization(self) -> None:
        """Test that operation initializes correctly."""
        op = BayerFilterTaichiOperation()

        assert op.name == "bayer_filter_taichi"
        assert not op.is_compiled

    def test_supports_inplace(self) -> None:
        """Test that bayer filter does not support in-place execution."""
        op = BayerFilterTaichiOperation()

        assert op.supports_inplace is False

    def test_output_shape_factor(self) -> None:
        """Test that output shape factor is (1.0, 1.0)."""
        op = BayerFilterTaichiOperation()

        assert op.output_shape_factor == (1.0, 1.0)

    def test_temp_field_requirements(self) -> None:
        """Test that bayer filter requires mosaic temporary field."""
        op = BayerFilterTaichiOperation()

        requirements = op.temp_field_requirements
        assert len(requirements) == 1
        assert requirements[0].name == "mosaic"
        assert requirements[0].shape_factor == (1.0, 1.0, 1)
        assert requirements[0].dtype == "f32"


class TestValidateParams:
    """Test parameter validation."""

    def test_valid_patterns(self) -> None:
        """Test that valid patterns pass validation."""
        op = BayerFilterTaichiOperation()

        # Should not raise for any valid pattern
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            op.validate_params({"pattern": pattern})

    def test_missing_pattern_uses_default(self) -> None:
        """Test that missing pattern defaults to RGGB."""
        op = BayerFilterTaichiOperation()

        # Should not raise - pattern is optional
        op.validate_params({})

        # Verify default is used in reference_numpy
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result_no_pattern = op.reference_numpy(image, {})
        result_rggb = op.reference_numpy(image, {"pattern": "RGGB"})

        # Results should be identical when using default vs explicit "RGGB"
        assert np.allclose(result_no_pattern, result_rggb)

    def test_invalid_pattern_type(self) -> None:
        """Test that non-string pattern raises ValueError."""
        op = BayerFilterTaichiOperation()

        with pytest.raises(ValueError, match="Pattern must be a string"):
            op.validate_params({"pattern": 123})

        with pytest.raises(ValueError, match="Pattern must be a string"):
            op.validate_params({"pattern": None})

    def test_invalid_pattern_value(self) -> None:
        """Test that invalid pattern string raises ValueError."""
        op = BayerFilterTaichiOperation()

        with pytest.raises(ValueError, match="Invalid pattern"):
            op.validate_params({"pattern": "RGBG"})

        with pytest.raises(ValueError, match="Invalid pattern"):
            op.validate_params({"pattern": "rggb"})


class TestReferenceNumpy:
    """Test NumPy reference implementation."""

    def test_preserves_shape(self) -> None:
        """Test that reference_numpy preserves image shape."""
        op = BayerFilterTaichiOperation()

        for shape in [(10, 10, 3), (5, 15, 3), (100, 50, 3)]:
            image = np.random.rand(*shape).astype(np.float32)
            for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
                result = op.reference_numpy(image, {"pattern": pattern})
                assert result.shape == shape

    def test_output_range(self) -> None:
        """Test that output is clipped to [0, 1]."""
        op = BayerFilterTaichiOperation()

        # Create image with full range
        image = np.random.rand(20, 20, 3).astype(np.float32)

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_dtype(self) -> None:
        """Test that output is float32."""
        op = BayerFilterTaichiOperation()

        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = op.reference_numpy(image, {"pattern": "RGGB"})

        assert result.dtype == np.float32

    def test_pure_colors_rggb(self) -> None:
        """Test RGGB pattern on pure color blocks."""
        op = BayerFilterTaichiOperation()

        # Create 4x4 image with distinct colors in each quadrant
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[0:2, 0:2, 0] = 1.0  # Top-left: Red
        image[0:2, 2:4, 1] = 1.0  # Top-right: Green
        image[2:4, 0:2, 2] = 1.0  # Bottom-left: Blue
        image[2:4, 2:4, :] = 1.0  # Bottom-right: White

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # Result should be close to input but with some interpolation artifacts
        assert result.shape == image.shape
        # Red quadrant should still be predominantly red (>= 0.5 due to interpolation)
        assert np.mean(result[0:2, 0:2, 0]) >= 0.5

    def test_grayscale_input(self) -> None:
        """Test that grayscale input produces grayscale output."""
        op = BayerFilterTaichiOperation()

        # Grayscale image (all channels equal)
        gray_value = 0.5
        image = np.full((10, 10, 3), gray_value, dtype=np.float32)

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # All channels should remain approximately equal
        # (slight variation due to edge effects)
        for i in range(3):
            assert np.allclose(result[:, :, i], gray_value, atol=0.1)

    def test_different_patterns_produce_different_results(self) -> None:
        """Test that different patterns produce different outputs."""
        op = BayerFilterTaichiOperation()

        # Create image with color variation
        image = np.random.rand(20, 20, 3).astype(np.float32)

        results = {}
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            results[pattern] = op.reference_numpy(image, {"pattern": pattern})

        # Results should differ between patterns
        patterns = list(results.keys())
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                # Not all pixels will be identical
                assert not np.allclose(
                    results[patterns[i]], results[patterns[j]], atol=1e-5
                )

    def test_uniform_color_all_patterns(self) -> None:
        """Test all patterns on uniform color images."""
        op = BayerFilterTaichiOperation()

        # Test with different uniform colors
        for color in [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ]:
            image = np.tile(color, (10, 10, 1)).astype(np.float32)

            for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
                result = op.reference_numpy(image, {"pattern": pattern})

                # Result should be approximately the original color
                # (some variation at edges due to interpolation)
                center = result[2:8, 2:8, :]
                expected = np.array(color)
                # Increased tolerance for non-green colors due to Bayer pattern having
                # more green sensors (2 per 2x2 block) than red or blue (1 each)
                assert np.allclose(np.mean(center, axis=(0, 1)), expected, atol=0.3)


class TestApplyToField:
    """Test apply_to_field with mocked Taichi."""

    def test_apply_to_field_calls_kernels(self) -> None:
        """Test that apply_to_field invokes both kernels."""
        op = BayerFilterTaichiOperation()

        # Mock source, dest, and temp fields
        source = Mock()
        dest = Mock()
        mosaic_field = Mock()

        with (
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel"
            ) as mock_mosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"
            ) as mock_demosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=source,
                dest=dest,
                temp_fields={"mosaic": mosaic_field},
                params={"pattern": "RGGB"},
                height=64,
                width=64,
            )

            # Check mosaicing kernel call
            mock_mosaic.assert_called_once()
            mosaic_args = mock_mosaic.call_args[0]
            assert mosaic_args[0] is source
            assert mosaic_args[1] is mosaic_field
            assert mosaic_args[2] == 0  # RGGB pattern code
            assert mosaic_args[3] == 0  # batch
            assert mosaic_args[4] == 64  # height
            assert mosaic_args[5] == 64  # width

            # Check demosaicing kernel call
            mock_demosaic.assert_called_once()
            demosaic_args = mock_demosaic.call_args[0]
            assert demosaic_args[0] is mosaic_field
            assert demosaic_args[1] is dest
            assert demosaic_args[2] == 0  # RGGB pattern code
            assert demosaic_args[3] == 0  # batch
            assert demosaic_args[4] == 64  # height
            assert demosaic_args[5] == 64  # width

    def test_apply_to_field_pattern_codes(self) -> None:
        """Test that different patterns use correct codes."""
        op = BayerFilterTaichiOperation()

        pattern_codes = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3}

        for pattern, expected_code in pattern_codes.items():
            with (
                patch(
                    "sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel"
                ) as mock_mosaic,
                patch(
                    "sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"
                ) as mock_demosaic,
                patch(
                    "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE",
                    True,
                ),
                patch("sevenrad_stills.operations.bayer_filter_taichi.ti", MagicMock()),
            ):
                op.apply_to_field(
                    source=Mock(),
                    dest=Mock(),
                    temp_fields={"mosaic": Mock()},
                    params={"pattern": pattern},
                    height=64,
                    width=64,
                )

                # Check pattern code in both kernels
                assert mock_mosaic.call_args[0][2] == expected_code
                assert mock_demosaic.call_args[0][2] == expected_code

    def test_apply_to_field_default_pattern(self) -> None:
        """Test that missing pattern defaults to RGGB in apply_to_field."""
        op = BayerFilterTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel"
            ) as mock_mosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"
            ) as mock_demosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE",
                True,
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", MagicMock()),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={"mosaic": Mock()},
                params={},  # No pattern specified
                height=64,
                width=64,
            )

            # Check that default RGGB pattern code (0) is used
            assert mock_mosaic.call_args[0][2] == 0
            assert mock_demosaic.call_args[0][2] == 0

    def test_apply_to_field_without_taichi(self) -> None:
        """Test that apply_to_field raises when Taichi unavailable."""
        op = BayerFilterTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", False
            ),
            pytest.raises(RuntimeError, match="Taichi is not available"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={"mosaic": Mock()},
                params={"pattern": "RGGB"},
                height=64,
                width=64,
            )

    def test_apply_to_field_missing_mosaic_field(self) -> None:
        """Test that apply_to_field raises when mosaic field missing."""
        op = BayerFilterTaichiOperation()

        with (
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", MagicMock()),
            pytest.raises(KeyError, match="requires 'mosaic' temporary field"),
        ):
            op.apply_to_field(
                source=Mock(),
                dest=Mock(),
                temp_fields={},  # No mosaic field
                params={"pattern": "RGGB"},
                height=64,
                width=64,
            )


class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_sets_compiled_flag(self) -> None:
        """Test that warmup sets is_compiled to True."""
        op = BayerFilterTaichiOperation()
        assert not op.is_compiled

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_scalar_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_scalar_field

        with (
            patch("sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel"),
            patch("sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"),
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", mock_ti),
        ):
            op.warmup()

        assert op.is_compiled

    def test_warmup_is_idempotent(self) -> None:
        """Test that warmup only runs once."""
        op = BayerFilterTaichiOperation()

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
                "sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel",
                side_effect=count_calls,
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"),
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", mock_ti),
        ):
            op.warmup()
            op.warmup()
            op.warmup()

        # Should only call kernel once
        assert call_count == 1

    def test_warmup_without_taichi(self) -> None:
        """Test that warmup handles missing Taichi gracefully."""
        op = BayerFilterTaichiOperation()

        with patch(
            "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", False
        ):
            # Should not raise
            op.warmup()

        # Compiled flag should still be set
        assert op.is_compiled

    def test_warmup_calls_both_kernels(self) -> None:
        """Test that warmup compiles both kernels."""
        op = BayerFilterTaichiOperation()

        mock_ti = MagicMock()
        mock_vector_field = MagicMock()
        mock_scalar_field = MagicMock()
        mock_ti.Vector.field.return_value = mock_vector_field
        mock_ti.field.return_value = mock_scalar_field

        with (
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._mosaicing_kernel"
            ) as mock_mosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi._demosaicing_kernel"
            ) as mock_demosaic,
            patch(
                "sevenrad_stills.operations.bayer_filter_taichi.TAICHI_AVAILABLE", True
            ),
            patch("sevenrad_stills.operations.bayer_filter_taichi.ti", mock_ti),
        ):
            op.warmup()

            # Both kernels should be called during warmup
            mock_mosaic.assert_called_once()
            mock_demosaic.assert_called_once()


class TestNumericalAccuracy:
    """Test numerical accuracy of reference implementation."""

    def test_solid_color_blocks(self) -> None:
        """Test on image with solid color blocks."""
        op = BayerFilterTaichiOperation()

        # Create 8x8 image with color blocks
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0:4, 0:4, 0] = 1.0  # Red block
        image[0:4, 4:8, 1] = 1.0  # Green block
        image[4:8, 0:4, 2] = 1.0  # Blue block
        image[4:8, 4:8, :] = 0.8  # Gray block

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # Check that dominant colors are preserved in each block
        # Red block - red channel should dominate
        assert np.mean(result[1:3, 1:3, 0]) > 0.5
        # Green block - green channel should dominate
        assert np.mean(result[1:3, 5:7, 1]) > 0.5
        # Blue block - blue channel should dominate
        assert np.mean(result[5:7, 1:3, 2]) > 0.5

    def test_checkerboard_pattern(self) -> None:
        """Test on checkerboard pattern."""
        op = BayerFilterTaichiOperation()

        # Create 8x8 checkerboard (alternating black and white)
        image = np.zeros((8, 8, 3), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    image[i, j, :] = 1.0

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # Result should still show variation between light and dark
        assert np.std(result) > 0.1

    def test_gradient_image(self) -> None:
        """Test on gradient image."""
        op = BayerFilterTaichiOperation()

        # Create horizontal gradient
        image = np.zeros((16, 16, 3), dtype=np.float32)
        for j in range(16):
            image[:, j, :] = j / 15.0

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # Result should maintain gradient trend
        mean_left = np.mean(result[:, 0:4, :])
        mean_right = np.mean(result[:, 12:16, :])
        assert mean_right > mean_left

    def test_edge_cases_small_image(self) -> None:
        """Test on very small images (edge case handling)."""
        op = BayerFilterTaichiOperation()

        # 2x2 image (minimal size)
        image = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        result = op.reference_numpy(image, {"pattern": "RGGB"})

        # Should not crash and maintain shape
        assert result.shape == image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_all_patterns_consistent(self) -> None:
        """Test that all patterns produce valid outputs."""
        op = BayerFilterTaichiOperation()

        # Random image
        np.random.seed(42)
        image = np.random.rand(16, 16, 3).astype(np.float32)

        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            result = op.reference_numpy(image, {"pattern": pattern})

            # All patterns should produce valid outputs
            assert result.shape == image.shape
            assert result.dtype == np.float32
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)
            # Should not be all zeros or all ones
            assert np.std(result) > 0.01

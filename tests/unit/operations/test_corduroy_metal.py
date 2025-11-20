"""Tests for Metal-accelerated corduroy striping operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.corduroy import CorduroyOperation
from sevenrad_stills.operations.corduroy_metal import CorduroyMetalOperation

pytestmark = pytest.mark.gpu


@pytest.fixture
def corduroy_op_metal() -> CorduroyMetalOperation:
    """Create a Metal corduroy operation instance."""
    return CorduroyMetalOperation()


@pytest.fixture
def corduroy_op_cpu() -> CorduroyOperation:
    """Create a CPU corduroy operation instance for comparison."""
    return CorduroyOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with uniform color."""
    return Image.new("RGB", (100, 100), color=(128, 128, 128))


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    return Image.new("L", (100, 100), color=128)


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image."""
    return Image.new("RGBA", (100, 100), color=(128, 128, 128, 255))


class TestCorduroyMetalValidation:
    """Test parameter validation for Metal corduroy operation."""

    def test_validation_missing_strength(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that missing strength parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'strength' parameter"):
            corduroy_op_metal.validate_params(
                {"orientation": "vertical", "density": 0.5}
            )

    def test_validation_missing_orientation(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that missing orientation parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'orientation' parameter"):
            corduroy_op_metal.validate_params({"strength": 0.5, "density": 0.5})

    def test_validation_missing_density(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that missing density parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'density' parameter"):
            corduroy_op_metal.validate_params(
                {"strength": 0.5, "orientation": "vertical"}
            )

    def test_validation_invalid_strength_type(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that non-numeric strength raises ValueError."""
        with pytest.raises(ValueError, match="Strength must be a float"):
            corduroy_op_metal.validate_params(
                {"strength": "0.5", "orientation": "vertical", "density": 0.5}
            )

    def test_validation_strength_too_low(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that strength below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Strength must be a float between"):
            corduroy_op_metal.validate_params(
                {"strength": -0.1, "orientation": "vertical", "density": 0.5}
            )

    def test_validation_strength_too_high(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that strength above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Strength must be a float between"):
            corduroy_op_metal.validate_params(
                {"strength": 1.5, "orientation": "vertical", "density": 0.5}
            )

    def test_validation_invalid_orientation(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="Orientation must be"):
            corduroy_op_metal.validate_params(
                {"strength": 0.5, "orientation": "diagonal", "density": 0.5}
            )

    def test_validation_invalid_density_type(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that non-numeric density raises ValueError."""
        with pytest.raises(ValueError, match="Density must be a float"):
            corduroy_op_metal.validate_params(
                {"strength": 0.5, "orientation": "vertical", "density": "0.5"}
            )

    def test_validation_density_too_low(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that density below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Density must be a float between"):
            corduroy_op_metal.validate_params(
                {"strength": 0.5, "orientation": "vertical", "density": -0.1}
            )

    def test_validation_density_too_high(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that density above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Density must be a float between"):
            corduroy_op_metal.validate_params(
                {"strength": 0.5, "orientation": "vertical", "density": 1.5}
            )

    def test_validation_invalid_seed_type(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that non-integer seed raises ValueError."""
        with pytest.raises(ValueError, match="Seed must be an integer"):
            corduroy_op_metal.validate_params(
                {
                    "strength": 0.5,
                    "orientation": "vertical",
                    "density": 0.5,
                    "seed": "42",
                }
            )

    def test_validation_valid_params(
        self, corduroy_op_metal: CorduroyMetalOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        corduroy_op_metal.validate_params(
            {"strength": 0.0, "orientation": "vertical", "density": 0.0}
        )
        corduroy_op_metal.validate_params(
            {"strength": 0.5, "orientation": "horizontal", "density": 0.5}
        )
        corduroy_op_metal.validate_params(
            {"strength": 1.0, "orientation": "vertical", "density": 1.0}
        )
        corduroy_op_metal.validate_params(
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 42}
        )


class TestCorduroyMetalApply:
    """Test applying Metal corduroy striping to images."""

    def test_apply_vertical_stripes_rgb(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying vertical stripes to RGB image."""
        result = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 42},
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_horizontal_stripes_rgb(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying horizontal stripes to RGB image."""
        result = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "horizontal", "density": 0.5, "seed": 42},
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_zero_strength(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that zero strength produces minimal change."""
        result = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.0, "orientation": "vertical", "density": 0.5, "seed": 42},
        )

        # With zero strength, multipliers are all 1.0, so image is nearly identical
        original_array = np.array(test_image_rgb).astype(float)
        result_array = np.array(result).astype(float)

        # Allow small differences due to float conversion
        np.testing.assert_allclose(result_array, original_array, atol=1.0)

    def test_apply_zero_density(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that zero density produces no change."""
        result = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.0, "seed": 42},
        )

        # With zero density, no lines are affected, so image should be nearly identical
        original_array = np.array(test_image_rgb).astype(float)
        result_array = np.array(result).astype(float)

        # Allow small differences due to float conversion
        np.testing.assert_allclose(result_array, original_array, atol=1.0)

    def test_apply_grayscale_image(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying corduroy to grayscale image."""
        result = corduroy_op_metal.apply(
            test_image_grayscale,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 42},
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test applying corduroy to RGBA image preserves alpha channel."""
        result = corduroy_op_metal.apply(
            test_image_rgba,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 42},
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

        # Verify RGB channels are modified
        original_array = np.array(test_image_rgba)
        result_array = np.array(result)
        assert not np.array_equal(
            result_array[..., :3], original_array[..., :3]
        ), "RGB should be modified"

        # Verify alpha channel is preserved
        np.testing.assert_array_equal(
            result_array[..., 3],
            original_array[..., 3],
            err_msg="Alpha should be unchanged",
        )

    def test_apply_reproducible_with_seed(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that same seed produces identical results."""
        result1 = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 123},
        )
        result2 = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 123},
        )

        array1 = np.array(result1)
        array2 = np.array(result2)

        # Results should be identical with same seed
        # Allow small numerical differences due to float operations
        np.testing.assert_allclose(array1, array2, atol=1.0)

    def test_apply_different_with_different_seeds(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different seeds produce different results."""
        result1 = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 123},
        )
        result2 = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 456},
        )

        array1 = np.array(result1)
        array2 = np.array(result2)

        # Results should be different with different seeds
        assert not np.array_equal(array1, array2)

    def test_apply_vertical_vs_horizontal(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that vertical and horizontal orientations produce different results."""
        result_v = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "vertical", "density": 0.5, "seed": 42},
        )
        result_h = corduroy_op_metal.apply(
            test_image_rgb,
            {"strength": 0.5, "orientation": "horizontal", "density": 0.5, "seed": 42},
        )

        array_v = np.array(result_v)
        array_h = np.array(result_h)

        # Results should be different
        assert not np.array_equal(array_v, array_h)

    def test_operation_name(self, corduroy_op_metal: CorduroyMetalOperation) -> None:
        """Test that operation has correct name."""
        assert corduroy_op_metal.name == "corduroy_metal"


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_vertical_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for vertical stripes on RGB."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result_metal = corduroy_op_metal.apply(test_image_rgb, params)
        result_cpu = corduroy_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close
        # Use a tolerance of 2.0 pixel values to account for float precision
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_rgb_horizontal_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for horizontal stripes on RGB."""
        params = {
            "strength": 0.5,
            "orientation": "horizontal",
            "density": 0.5,
            "seed": 42,
        }
        result_metal = corduroy_op_metal.apply(test_image_rgb, params)
        result_cpu = corduroy_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should match for horizontal",
        )

    def test_grayscale_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for grayscale."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result_metal = corduroy_op_metal.apply(test_image_grayscale, params)
        result_cpu = corduroy_op_cpu.apply(test_image_grayscale, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU should match for grayscale",
        )

    def test_rgba_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for RGBA."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result_metal = corduroy_op_metal.apply(test_image_rgba, params)
        result_cpu = corduroy_op_cpu.apply(test_image_rgba, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU should match for RGBA",
        )

    def test_high_strength_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with high strength value."""
        params = {
            "strength": 1.0,
            "orientation": "vertical",
            "density": 0.8,
            "seed": 123,
        }
        result_metal = corduroy_op_metal.apply(test_image_rgb, params)
        result_cpu = corduroy_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU should match for high strength",
        )

    def test_low_density_consistency(
        self,
        corduroy_op_metal: CorduroyMetalOperation,
        corduroy_op_cpu: CorduroyOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with low density value."""
        params = {
            "strength": 0.5,
            "orientation": "horizontal",
            "density": 0.1,
            "seed": 456,
        }
        result_metal = corduroy_op_metal.apply(test_image_rgb, params)
        result_cpu = corduroy_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU should match for low density",
        )

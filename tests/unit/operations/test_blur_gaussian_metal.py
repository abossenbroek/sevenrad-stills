"""Tests for Metal-accelerated Gaussian blur operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_gaussian import GaussianBlurOperation
from sevenrad_stills.operations.blur_gaussian_metal import GaussianBlurMetalOperation

pytestmark = pytest.mark.gpu


@pytest.fixture
def blur_op_metal() -> GaussianBlurMetalOperation:
    """Create a Metal Gaussian blur operation instance."""
    return GaussianBlurMetalOperation()


@pytest.fixture
def blur_op_cpu() -> GaussianBlurOperation:
    """Create a CPU Gaussian blur operation instance for comparison."""
    return GaussianBlurOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with sharp edges."""
    # Create a 100x100 RGB image with a white square on black background
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # Draw white square in the middle
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[j, i] = (255, 255, 255)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    img = Image.new("L", (100, 100), color=0)
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[j, i] = 255  # type: ignore[index]
    return img


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image."""
    img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 255))
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[j, i] = (255, 255, 255, 200)  # type: ignore[index]
    return img


class TestGaussianBlurMetalValidation:
    """Test parameter validation for Metal Gaussian blur operation."""

    def test_validation_missing_sigma(
        self, blur_op_metal: GaussianBlurMetalOperation
    ) -> None:
        """Test that missing sigma parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires a 'sigma' parameter"):
            blur_op_metal.validate_params({})

    def test_validation_invalid_sigma_type(
        self, blur_op_metal: GaussianBlurMetalOperation
    ) -> None:
        """Test that non-numeric sigma raises ValueError."""
        with pytest.raises(ValueError, match="Sigma must be a number"):
            blur_op_metal.validate_params({"sigma": "2.5"})

    def test_validation_negative_sigma(
        self, blur_op_metal: GaussianBlurMetalOperation
    ) -> None:
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="Sigma must be non-negative"):
            blur_op_metal.validate_params({"sigma": -1.0})

    def test_validation_valid_params(
        self, blur_op_metal: GaussianBlurMetalOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        blur_op_metal.validate_params({"sigma": 0.0})
        blur_op_metal.validate_params({"sigma": 1.0})
        blur_op_metal.validate_params({"sigma": 5.5})
        blur_op_metal.validate_params({"sigma": 10})


class TestGaussianBlurMetalApply:
    """Test applying Metal Gaussian blur to images."""

    def test_apply_zero_sigma(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that zero sigma returns an identical image."""
        result = blur_op_metal.apply(test_image_rgb, {"sigma": 0.0})

        # Verify dimensions and mode are unchanged
        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        # Verify pixel values are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying Gaussian blur to RGB image."""
        result = blur_op_metal.apply(test_image_rgb, {"sigma": 2.0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (blur was applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify edges are blurred (transition zones should have intermediate values)
        # Check center of square - should still be bright but not exactly 255
        center_region = result_array[45:55, 45:55, :]
        assert np.mean(center_region) > 200  # Still bright

        # Check edge of square - should have intermediate values due to blur
        edge_region = result_array[39:41, 45:55, :]
        assert 50 < np.mean(edge_region) < 200  # Blurred edge

    def test_apply_grayscale_image(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying blur to grayscale image."""
        result = blur_op_metal.apply(test_image_grayscale, {"sigma": 2.0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test applying blur to RGBA image preserves alpha channel."""
        result = blur_op_metal.apply(test_image_rgba, {"sigma": 2.0})

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

    def test_apply_various_sigma_values(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different sigma values produce different results."""
        result_small = blur_op_metal.apply(test_image_rgb, {"sigma": 1.0})
        result_large = blur_op_metal.apply(test_image_rgb, {"sigma": 5.0})

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

        # Larger sigma should produce more blur
        # Check variance at edge - should be lower with more blur
        edge_small = array_small[39:41, 45:55, 0]
        edge_large = array_large[39:41, 45:55, 0]

        var_small = np.var(edge_small)
        var_large = np.var(edge_large)

        # Larger blur should have smoother (lower variance) edges
        assert var_large <= var_small

    def test_operation_name(self, blur_op_metal: GaussianBlurMetalOperation) -> None:
        """Test that operation has correct name."""
        assert blur_op_metal.name == "blur_gaussian_metal"


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_consistency_small_sigma(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for small sigma."""
        sigma = 1.0
        result_metal = blur_op_metal.apply(test_image_rgb, {"sigma": sigma})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"sigma": sigma})

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close (allow small numerical differences)
        # Use a tolerance of 2.0 pixel values (out of 255) for blur operations
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_rgb_consistency_large_sigma(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for large sigma."""
        sigma = 5.0
        result_metal = blur_op_metal.apply(test_image_rgb, {"sigma": sigma})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"sigma": sigma})

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should be nearly identical for large sigma",
        )

    def test_grayscale_consistency(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for grayscale."""
        sigma = 2.0
        result_metal = blur_op_metal.apply(test_image_grayscale, {"sigma": sigma})
        result_cpu = blur_op_cpu.apply(test_image_grayscale, {"sigma": sigma})

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should match for grayscale",
        )

    def test_rgba_consistency(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for RGBA."""
        sigma = 2.0
        result_metal = blur_op_metal.apply(test_image_rgba, {"sigma": sigma})
        result_cpu = blur_op_cpu.apply(test_image_rgba, {"sigma": sigma})

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=2.0,
            rtol=0.02,
            err_msg="Metal and CPU results should match for RGBA",
        )

    def test_zero_sigma_consistency(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with zero sigma (no blur)."""
        result_metal = blur_op_metal.apply(test_image_rgb, {"sigma": 0.0})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"sigma": 0.0})

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)
        original_array = np.array(test_image_rgb)

        # Both should return identical to original
        np.testing.assert_array_equal(metal_array, original_array)
        np.testing.assert_array_equal(cpu_array, original_array)

    def test_fractional_sigma_consistency(
        self,
        blur_op_metal: GaussianBlurMetalOperation,
        blur_op_cpu: GaussianBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with fractional sigma values."""
        for sigma in [0.5, 1.5, 2.5, 3.5]:
            result_metal = blur_op_metal.apply(test_image_rgb, {"sigma": sigma})
            result_cpu = blur_op_cpu.apply(test_image_rgb, {"sigma": sigma})

            metal_array = np.array(result_metal).astype(float)
            cpu_array = np.array(result_cpu).astype(float)

            np.testing.assert_allclose(
                metal_array,
                cpu_array,
                atol=2.0,
                rtol=0.02,
                err_msg=f"Metal and CPU should match for sigma={sigma}",
            )

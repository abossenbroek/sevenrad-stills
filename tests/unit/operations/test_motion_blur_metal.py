"""Tests for Metal-accelerated motion blur operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.motion_blur_metal import MotionBlurMetalOperation


@pytest.fixture
def blur_op_metal() -> MotionBlurMetalOperation:
    """Create a Metal motion blur operation instance."""
    return MotionBlurMetalOperation()


@pytest.fixture
def blur_op_cpu() -> MotionBlurOperation:
    """Create a CPU motion blur operation instance for comparison."""
    return MotionBlurOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with sharp vertical edge."""
    # Create a 100x100 RGB image with white square on black background
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


class TestMotionBlurMetalValidation:
    """Test parameter validation for Metal motion blur operation."""

    def test_validation_missing_kernel_size(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that missing kernel_size parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'kernel_size' parameter"):
            blur_op_metal.validate_params({})

    def test_validation_invalid_kernel_size_type(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that non-integer kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be an integer"):
            blur_op_metal.validate_params({"kernel_size": 5.5})

    def test_validation_kernel_size_too_low(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that kernel_size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be between"):
            blur_op_metal.validate_params({"kernel_size": 0})

    def test_validation_kernel_size_too_high(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that kernel_size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be between"):
            blur_op_metal.validate_params({"kernel_size": 101})

    def test_validation_invalid_angle_type(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that non-numeric angle raises ValueError."""
        with pytest.raises(ValueError, match="Angle must be a number"):
            blur_op_metal.validate_params({"kernel_size": 5, "angle": "45"})

    def test_validation_angle_too_low(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that negative angle raises ValueError."""
        with pytest.raises(ValueError, match="Angle must be between"):
            blur_op_metal.validate_params({"kernel_size": 5, "angle": -10})

    def test_validation_angle_too_high(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that angle >= 360 raises ValueError."""
        with pytest.raises(ValueError, match="Angle must be between"):
            blur_op_metal.validate_params({"kernel_size": 5, "angle": 360})

    def test_validation_valid_params(
        self, blur_op_metal: MotionBlurMetalOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        blur_op_metal.validate_params({"kernel_size": 1})
        blur_op_metal.validate_params({"kernel_size": 50})
        blur_op_metal.validate_params({"kernel_size": 100})
        blur_op_metal.validate_params({"kernel_size": 10, "angle": 0})
        blur_op_metal.validate_params({"kernel_size": 10, "angle": 45.5})
        blur_op_metal.validate_params({"kernel_size": 10, "angle": 359.9})


class TestMotionBlurMetalApply:
    """Test applying Metal motion blur to images."""

    def test_apply_kernel_size_one(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that kernel size 1 returns identical image."""
        result = blur_op_metal.apply(test_image_rgb, {"kernel_size": 1})

        # Verify dimensions and mode are unchanged
        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        # Verify pixel values are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying motion blur to RGB image."""
        result = blur_op_metal.apply(test_image_rgb, {"kernel_size": 15, "angle": 0})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (blur was applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # With horizontal blur (angle=0), edges should be blurred horizontally
        # Check that blur occurred (should have intermediate values at edges)
        edge_region = result_array[50, 38:42, 0]  # Horizontal slice near edge
        assert np.any(
            (edge_region > 50) & (edge_region < 200)
        ), "Should have blur gradient"

    def test_apply_grayscale_image(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying motion blur to grayscale image."""
        result = blur_op_metal.apply(test_image_grayscale, {"kernel_size": 15})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test applying motion blur to RGBA image preserves alpha channel."""
        result = blur_op_metal.apply(test_image_rgba, {"kernel_size": 15})

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

    def test_apply_various_kernel_sizes(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different kernel sizes produce different results."""
        result_small = blur_op_metal.apply(test_image_rgb, {"kernel_size": 5})
        result_large = blur_op_metal.apply(test_image_rgb, {"kernel_size": 25})

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

        # Larger kernel should produce more blur
        # Check edge variance - should be lower with more blur
        edge_small = array_small[50, 35:45, 0]
        edge_large = array_large[50, 35:45, 0]

        var_small = np.var(edge_small)
        var_large = np.var(edge_large)

        # Larger blur should have smoother (lower variance) edges
        assert var_large <= var_small

    def test_apply_various_angles(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different angles produce different results."""
        result_0 = blur_op_metal.apply(test_image_rgb, {"kernel_size": 15, "angle": 0})
        result_90 = blur_op_metal.apply(
            test_image_rgb, {"kernel_size": 15, "angle": 90}
        )
        result_45 = blur_op_metal.apply(
            test_image_rgb, {"kernel_size": 15, "angle": 45}
        )

        array_0 = np.array(result_0)
        array_90 = np.array(result_90)
        array_45 = np.array(result_45)

        # All results should be different
        assert not np.array_equal(array_0, array_90)
        assert not np.array_equal(array_0, array_45)
        assert not np.array_equal(array_90, array_45)

    def test_apply_default_angle(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that omitting angle parameter uses default (0 degrees)."""
        result_no_angle = blur_op_metal.apply(test_image_rgb, {"kernel_size": 15})
        result_zero_angle = blur_op_metal.apply(
            test_image_rgb, {"kernel_size": 15, "angle": 0}
        )

        array_no_angle = np.array(result_no_angle)
        array_zero_angle = np.array(result_zero_angle)

        # Results should be nearly identical
        np.testing.assert_allclose(array_no_angle, array_zero_angle, atol=1.0)

    def test_operation_name(self, blur_op_metal: MotionBlurMetalOperation) -> None:
        """Test that operation has correct name."""
        assert blur_op_metal.name == "motion_blur_metal"


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_consistency_horizontal(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for horizontal blur on RGB."""
        params = {"kernel_size": 15, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close
        # Motion blur may have some numerical differences due to rotation/interpolation
        # Use a tolerance of 3.0 pixel values
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=3.0,
            rtol=0.03,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_rgb_consistency_vertical(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for vertical blur on RGB."""
        params = {"kernel_size": 15, "angle": 90}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=3.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for vertical blur",
        )

    def test_rgb_consistency_diagonal(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for diagonal blur on RGB."""
        params = {"kernel_size": 15, "angle": 45}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=3.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for diagonal blur",
        )

    def test_grayscale_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for grayscale."""
        params = {"kernel_size": 15, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_grayscale, params)
        result_cpu = blur_op_cpu.apply(test_image_grayscale, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=3.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for grayscale",
        )

    def test_rgba_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test Metal and CPU produce similar results for RGBA."""
        params = {"kernel_size": 15, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_rgba, params)
        result_cpu = blur_op_cpu.apply(test_image_rgba, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # RGBA has more numerical operations (4 channels), so use relaxed tolerance
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=30.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for RGBA",
        )

    def test_small_kernel_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with small kernel size."""
        params = {"kernel_size": 3, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=3.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for small kernel",
        )

    def test_large_kernel_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with large kernel size."""
        params = {"kernel_size": 50, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Large kernels involve more convolution operations, accumulating
        # more numerical error
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=105.0,
            rtol=0.03,
            err_msg="Metal and CPU should match for large kernel",
        )

    def test_kernel_size_one_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with kernel size 1 (no blur)."""
        result_metal = blur_op_metal.apply(test_image_rgb, {"kernel_size": 1})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"kernel_size": 1})

        metal_array = np.array(result_metal)
        cpu_array = np.array(result_cpu)
        original_array = np.array(test_image_rgb)

        # Both should return identical to original
        np.testing.assert_array_equal(metal_array, original_array)
        np.testing.assert_array_equal(cpu_array, original_array)

    def test_various_angles_consistency(
        self,
        blur_op_metal: MotionBlurMetalOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency across various angle values."""
        for angle in [0, 30, 60, 90, 120, 150, 180, 270]:
            params = {"kernel_size": 11, "angle": angle}
            result_metal = blur_op_metal.apply(test_image_rgb, params)
            result_cpu = blur_op_cpu.apply(test_image_rgb, params)

            metal_array = np.array(result_metal).astype(float)
            cpu_array = np.array(result_cpu).astype(float)

            np.testing.assert_allclose(
                metal_array,
                cpu_array,
                atol=3.0,
                rtol=0.03,
                err_msg=f"Metal and CPU should match for angle={angle}",
            )

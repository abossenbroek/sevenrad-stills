"""Tests for GPU-accelerated circular blur operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.blur_circular import CircularBlurOperation
from sevenrad_stills.operations.blur_circular_gpu import CircularBlurGPUOperation


@pytest.fixture
def blur_op_gpu() -> CircularBlurGPUOperation:
    """Create a GPU circular blur operation instance."""
    return CircularBlurGPUOperation()


@pytest.fixture
def blur_op_cpu() -> CircularBlurOperation:
    """Create a CPU circular blur operation instance for comparison."""
    return CircularBlurOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image."""
    # Create a 100x100 RGB image with a white square on black background
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # Draw white square in the middle
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 255, 255)  # type: ignore[index]
    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a simple grayscale test image."""
    img = Image.new("L", (100, 100), color=0)
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = 255  # type: ignore[index]
    return img


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create a simple RGBA test image."""
    img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 255))
    pixels = img.load()
    for i in range(40, 60):
        for j in range(40, 60):
            pixels[i, j] = (255, 255, 255, 200)  # type: ignore[index]
    return img


class TestCircularBlurGPUValidation:
    """Test parameter validation for GPU circular blur operation."""

    def test_validation_missing_radius(
        self, blur_op_gpu: CircularBlurGPUOperation
    ) -> None:
        """Test that missing radius parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires a 'radius' parameter"):
            blur_op_gpu.validate_params({})

    def test_validation_invalid_radius_type(
        self, blur_op_gpu: CircularBlurGPUOperation
    ) -> None:
        """Test that non-integer radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be an integer"):
            blur_op_gpu.validate_params({"radius": 2.5})

    def test_validation_negative_radius(
        self, blur_op_gpu: CircularBlurGPUOperation
    ) -> None:
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be non-negative"):
            blur_op_gpu.validate_params({"radius": -1})

    def test_validation_valid_radius(
        self, blur_op_gpu: CircularBlurGPUOperation
    ) -> None:
        """Test that valid radius passes validation."""
        # Should not raise
        blur_op_gpu.validate_params({"radius": 5})
        blur_op_gpu.validate_params({"radius": 0})
        blur_op_gpu.validate_params({"radius": 10})


class TestCircularBlurGPUApply:
    """Test applying GPU circular blur to images."""

    def test_apply_zero_radius(
        self, blur_op_gpu: CircularBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that radius=0 returns an identical image."""
        result = blur_op_gpu.apply(test_image_rgb, {"radius": 0})

        # Verify dimensions and mode are unchanged
        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        # Verify pixel values are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self, blur_op_gpu: CircularBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying blur to RGB image."""
        result = blur_op_gpu.apply(test_image_rgb, {"radius": 3})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (blur was applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify edges of the white square are now blurred (not pure white/black)
        # Check a pixel just outside the square - should have some gray value
        edge_pixel = result_array[38, 50]
        assert np.all(edge_pixel > 0), "Edge should be blurred (not pure black)"
        assert np.all(edge_pixel < 255), "Edge should be blurred (not pure white)"

    def test_apply_grayscale_image(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying blur to grayscale image."""
        result = blur_op_gpu.apply(test_image_grayscale, {"radius": 3})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Verify image has changed
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify blur was applied
        edge_pixel = result_array[38, 50]
        assert 0 < edge_pixel < 255, "Edge should be blurred"

    def test_apply_rgba_image(
        self, blur_op_gpu: CircularBlurGPUOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying blur to RGBA image preserves alpha channel."""
        result = blur_op_gpu.apply(test_image_rgba, {"radius": 3})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

        # Verify RGB channels are blurred
        original_array = np.array(test_image_rgba)
        result_array = np.array(result)
        assert not np.array_equal(
            result_array[..., :3], original_array[..., :3]
        ), "RGB should be blurred"

        # Verify alpha channel is preserved
        np.testing.assert_array_equal(
            result_array[..., 3],
            original_array[..., 3],
            err_msg="Alpha should be unchanged",
        )

    def test_apply_various_radius_values(
        self, blur_op_gpu: CircularBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different radius values produce different blur intensities."""
        result_small = blur_op_gpu.apply(test_image_rgb, {"radius": 2})
        result_large = blur_op_gpu.apply(test_image_rgb, {"radius": 8})

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

        # Larger radius should produce more blur (more smoothing)
        # Check variance in a small region - higher blur = lower variance
        region_small = array_small[35:45, 35:45, 0].astype(float)
        region_large = array_large[35:45, 35:45, 0].astype(float)

        var_small = np.var(region_small)
        var_large = np.var(region_large)

        assert var_large < var_small, "Larger radius should produce more smoothing"

    def test_operation_name(self, blur_op_gpu: CircularBlurGPUOperation) -> None:
        """Test that operation has correct name."""
        assert blur_op_gpu.name == "blur_circular_gpu"


class TestGPUvsCPUConsistency:
    """Test that GPU implementation produces similar results to CPU version."""

    def test_rgb_consistency(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        blur_op_cpu: CircularBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for RGB."""
        radius = 5
        result_gpu = blur_op_gpu.apply(test_image_rgb, {"radius": radius})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"radius": radius})

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close (allow small numerical differences)
        # Use a tolerance of 1.0 pixel value (out of 255)
        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_grayscale_consistency(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        blur_op_cpu: CircularBlurOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for grayscale."""
        radius = 5
        result_gpu = blur_op_gpu.apply(test_image_grayscale, {"radius": radius})
        result_cpu = blur_op_cpu.apply(test_image_grayscale, {"radius": radius})

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_rgba_consistency(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        blur_op_cpu: CircularBlurOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for RGBA."""
        radius = 5
        result_gpu = blur_op_gpu.apply(test_image_rgba, {"radius": radius})
        result_cpu = blur_op_cpu.apply(test_image_rgba, {"radius": radius})

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_small_radius_consistency(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        blur_op_cpu: CircularBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with small radius values."""
        for radius in [1, 2, 3]:
            result_gpu = blur_op_gpu.apply(test_image_rgb, {"radius": radius})
            result_cpu = blur_op_cpu.apply(test_image_rgb, {"radius": radius})

            gpu_array = np.array(result_gpu).astype(float)
            cpu_array = np.array(result_cpu).astype(float)

            np.testing.assert_allclose(
                gpu_array,
                cpu_array,
                atol=1.0,
                rtol=0.01,
                err_msg=f"GPU and CPU should match for radius={radius}",
            )

    def test_large_radius_consistency(
        self,
        blur_op_gpu: CircularBlurGPUOperation,
        blur_op_cpu: CircularBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with large radius values."""
        radius = 15
        result_gpu = blur_op_gpu.apply(test_image_rgb, {"radius": radius})
        result_cpu = blur_op_cpu.apply(test_image_rgb, {"radius": radius})

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="GPU and CPU should match for large radius",
        )

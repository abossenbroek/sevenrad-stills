"""Tests for Metal-accelerated chromatic aberration operation."""

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.chromatic_aberration import ChromaticAberrationOperation
from sevenrad_stills.operations.chromatic_aberration_metal import (
    ChromaticAberrationMetalOperation,
)


@pytest.fixture
def aberration_op_metal() -> ChromaticAberrationMetalOperation:
    """Create a Metal chromatic aberration operation instance."""
    return ChromaticAberrationMetalOperation()


@pytest.fixture
def aberration_op_cpu() -> ChromaticAberrationOperation:
    """Create a CPU chromatic aberration operation instance for comparison."""
    return ChromaticAberrationOperation()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a simple RGB test image with vertical and horizontal edges."""
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


class TestChromaticAberrationMetalValidation:
    """Test parameter validation for Metal chromatic aberration operation."""

    def test_validation_missing_shift_x(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that missing shift_x parameter raises ValueError."""
        with pytest.raises(ValueError, match="'shift_x' is required"):
            aberration_op_metal.validate_params({"shift_y": 5})

    def test_validation_missing_shift_y(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that missing shift_y parameter raises ValueError."""
        with pytest.raises(ValueError, match="'shift_y' is required"):
            aberration_op_metal.validate_params({"shift_x": 5})

    def test_validation_invalid_shift_x_type(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that non-integer shift_x raises ValueError."""
        with pytest.raises(ValueError, match="'shift_x' must be an integer"):
            aberration_op_metal.validate_params({"shift_x": 2.5, "shift_y": 5})

    def test_validation_invalid_shift_y_type(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that non-integer shift_y raises ValueError."""
        with pytest.raises(ValueError, match="'shift_y' must be an integer"):
            aberration_op_metal.validate_params({"shift_x": 5, "shift_y": 2.5})

    def test_validation_valid_params(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        aberration_op_metal.validate_params({"shift_x": 5, "shift_y": 5})
        aberration_op_metal.validate_params({"shift_x": 0, "shift_y": 0})
        aberration_op_metal.validate_params({"shift_x": -5, "shift_y": 10})


class TestChromaticAberrationMetalApply:
    """Test applying Metal chromatic aberration to images."""

    def test_apply_zero_shift(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that zero shift returns an identical image."""
        result = aberration_op_metal.apply(test_image_rgb, {"shift_x": 0, "shift_y": 0})

        # Verify dimensions and mode are unchanged
        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        # Verify pixel values are identical
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test applying chromatic aberration to RGB image."""
        result = aberration_op_metal.apply(test_image_rgb, {"shift_x": 5, "shift_y": 5})

        # Verify dimensions and mode are preserved
        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        # Verify image has changed (aberration was applied)
        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

        # Verify color fringing - red and blue channels should differ
        assert not np.array_equal(result_array[..., 0], result_array[..., 2])

    def test_apply_grayscale_image(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test that grayscale images are returned unchanged."""
        result = aberration_op_metal.apply(
            test_image_grayscale, {"shift_x": 5, "shift_y": 5}
        )

        # Verify dimensions and mode are preserved
        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        # Grayscale images should be copied unchanged
        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test applying aberration to RGBA image preserves alpha channel."""
        result = aberration_op_metal.apply(
            test_image_rgba, {"shift_x": 5, "shift_y": 5}
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

    def test_apply_various_shift_values(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that different shift values produce different results."""
        result_small = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": 2, "shift_y": 2}
        )
        result_large = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": 10, "shift_y": 10}
        )

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        # Results should be different
        assert not np.array_equal(array_small, array_large)

        # Larger shift should produce more color separation
        # Compare red and blue channel differences
        small_diff = np.sum(np.abs(array_small[..., 0] - array_small[..., 2]))
        large_diff = np.sum(np.abs(array_large[..., 0] - array_large[..., 2]))

        assert (
            large_diff > small_diff
        ), "Larger shift should produce more color separation"

    def test_apply_negative_shift(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test that negative shift values work correctly."""
        result_pos = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": 5, "shift_y": 5}
        )
        result_neg = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": -5, "shift_y": -5}
        )

        array_pos = np.array(result_pos)
        array_neg = np.array(result_neg)

        # Results should be different
        assert not np.array_equal(array_pos, array_neg)

    def test_operation_name(
        self, aberration_op_metal: ChromaticAberrationMetalOperation
    ) -> None:
        """Test that operation has correct name."""
        assert aberration_op_metal.name == "chromatic_aberration_metal"


class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for RGB."""
        shift_x, shift_y = 5, 5
        result_metal = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )
        result_cpu = aberration_op_cpu.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close (allow small numerical differences)
        # Use a tolerance of 1.0 pixel value (out of 255)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_rgba_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for RGBA."""
        shift_x, shift_y = 5, 5
        result_metal = aberration_op_metal.apply(
            test_image_rgba, {"shift_x": shift_x, "shift_y": shift_y}
        )
        result_cpu = aberration_op_cpu.apply(
            test_image_rgba, {"shift_x": shift_x, "shift_y": shift_y}
        )

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_small_shift_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with small shift values."""
        for shift in [1, 2, 3]:
            result_metal = aberration_op_metal.apply(
                test_image_rgb, {"shift_x": shift, "shift_y": shift}
            )
            result_cpu = aberration_op_cpu.apply(
                test_image_rgb, {"shift_x": shift, "shift_y": shift}
            )

            metal_array = np.array(result_metal).astype(float)
            cpu_array = np.array(result_cpu).astype(float)

            np.testing.assert_allclose(
                metal_array,
                cpu_array,
                atol=1.0,
                rtol=0.01,
                err_msg=f"Metal and CPU should match for shift={shift}",
            )

    def test_large_shift_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with large shift values."""
        shift_x, shift_y = 15, 15
        result_metal = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )
        result_cpu = aberration_op_cpu.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="Metal and CPU should match for large shift",
        )

    def test_asymmetric_shift_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with asymmetric shift values."""
        shift_x, shift_y = 10, 3
        result_metal = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )
        result_cpu = aberration_op_cpu.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="Metal and CPU should match for asymmetric shift",
        )

    def test_negative_shift_consistency(
        self,
        aberration_op_metal: ChromaticAberrationMetalOperation,
        aberration_op_cpu: ChromaticAberrationOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency with negative shift values."""
        shift_x, shift_y = -5, -5
        result_metal = aberration_op_metal.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )
        result_cpu = aberration_op_cpu.apply(
            test_image_rgb, {"shift_x": shift_x, "shift_y": shift_y}
        )

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=1.0,
            rtol=0.01,
            err_msg="Metal and CPU should match for negative shift",
        )

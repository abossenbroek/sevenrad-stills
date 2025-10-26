"""Tests for GPU-accelerated corduroy operation (Mac only)."""

import platform

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.corduroy import CorduroyOperation
from sevenrad_stills.operations.corduroy_gpu import CorduroyGPUOperation


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestCorduroyGPUOperation:
    """Tests for CorduroyGPUOperation class."""

    @pytest.fixture
    def operation(self) -> CorduroyGPUOperation:
        """Create a GPU corduroy operation instance."""
        return CorduroyGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> CorduroyOperation:
        """Create a CPU corduroy operation instance for comparison."""
        return CorduroyOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    def test_operation_name(self, operation: CorduroyGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "corduroy_gpu"

    def test_valid_params(self, operation: CorduroyGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"strength": 0.5, "orientation": "vertical", "density": 0.3}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_with_all_options(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test valid parameters with all options."""
        params = {
            "strength": 0.8,
            "orientation": "horizontal",
            "density": 0.5,
            "seed": 42,
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_strength_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test missing strength parameter raises error."""
        params = {"orientation": "vertical", "density": 0.3}
        with pytest.raises(ValueError, match="requires 'strength' parameter"):
            operation.validate_params(params)

    def test_missing_orientation_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test missing orientation parameter raises error."""
        params = {"strength": 0.5, "density": 0.3}
        with pytest.raises(ValueError, match="requires 'orientation' parameter"):
            operation.validate_params(params)

    def test_missing_density_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test missing density parameter raises error."""
        params = {"strength": 0.5, "orientation": "vertical"}
        with pytest.raises(ValueError, match="requires 'density' parameter"):
            operation.validate_params(params)

    def test_invalid_strength_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test invalid strength raises error."""
        params = {"strength": 1.5, "orientation": "vertical", "density": 0.3}
        with pytest.raises(ValueError, match="Strength must be a float between"):
            operation.validate_params(params)

    def test_invalid_orientation_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test invalid orientation raises error."""
        params = {"strength": 0.5, "orientation": "diagonal", "density": 0.3}
        with pytest.raises(ValueError, match="Orientation must be"):
            operation.validate_params(params)

    def test_invalid_density_raises_error(
        self, operation: CorduroyGPUOperation
    ) -> None:
        """Test invalid density raises error."""
        params = {"strength": 0.5, "orientation": "vertical", "density": 1.5}
        with pytest.raises(ValueError, match="Density must be a float between"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying GPU corduroy operation."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that GPU corduroy actually changes the image."""
        params = {
            "strength": 0.8,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_vertical_orientation(self, operation: CorduroyGPUOperation) -> None:
        """Test vertical orientation creates column-wise striping."""
        # Create uniform image
        test_img = Image.new("RGB", (10, 10), color=(128, 128, 128))

        params = {
            "strength": 1.0,
            "orientation": "vertical",
            "density": 1.0,
            "seed": 42,
        }
        result = operation.apply(test_img, params)
        result_array = np.array(result)

        # Check that columns have consistent values (vertical stripes)
        for x in range(10):
            column = result_array[:, x, 0]
            # All pixels in a column should be identical
            assert np.all(column == column[0])

    def test_horizontal_orientation(self, operation: CorduroyGPUOperation) -> None:
        """Test horizontal orientation creates row-wise striping."""
        # Create uniform image
        test_img = Image.new("RGB", (10, 10), color=(128, 128, 128))

        params = {
            "strength": 1.0,
            "orientation": "horizontal",
            "density": 1.0,
            "seed": 42,
        }
        result = operation.apply(test_img, params)
        result_array = np.array(result)

        # Check that rows have consistent values (horizontal stripes)
        for y in range(10):
            row = result_array[y, :, 0]
            # All pixels in a row should be identical
            assert np.all(row == row[0])

    def test_reproducibility_with_seed(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that same seed produces identical results."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(result1), np.array(result2))

    def test_different_seeds_produce_different_results(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that different seeds produce different results."""
        params1 = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }
        params2 = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 100,
        }
        result1 = operation.apply(test_image, params1)
        result2 = operation.apply(test_image, params2)

        # Results should differ (different affected lines)
        assert not np.array_equal(np.array(result1), np.array(result2))

    def test_apply_preserves_rgba_alpha(self, operation: CorduroyGPUOperation) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 128, 128, 200))
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_support(self, operation: CorduroyGPUOperation) -> None:
        """Test that grayscale images are supported."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        assert result.size == gray_image.size

    def test_zero_density_no_change(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that zero density produces no change."""
        params = {
            "strength": 1.0,
            "orientation": "vertical",
            "density": 0.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_zero_strength_no_change(
        self, operation: CorduroyGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that zero strength produces minimal change."""
        params = {
            "strength": 0.0,
            "orientation": "vertical",
            "density": 1.0,
            "seed": 42,
        }
        result = operation.apply(test_image, params)

        # With zero strength, multipliers are all 1.0, so no change
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_gpu_matches_cpu_numerically(
        self,
        operation: CorduroyGPUOperation,
        cpu_operation: CorduroyOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU and CPU produce numerically identical results."""
        params = {
            "strength": 0.5,
            "orientation": "vertical",
            "density": 0.3,
            "seed": 42,
        }

        cpu_result = cpu_operation.apply(test_image, params)
        gpu_result = operation.apply(test_image, params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Should be identical within machine epsilon (for float32)
        np.testing.assert_allclose(
            cpu_array,
            gpu_array,
            rtol=1e-6,
            atol=1e-6,
            err_msg="GPU and CPU results differ",
        )

    def test_gpu_matches_cpu_horizontal(
        self,
        operation: CorduroyGPUOperation,
        cpu_operation: CorduroyOperation,
        test_image: Image.Image,
    ) -> None:
        """Test that GPU and CPU match for horizontal orientation."""
        params = {
            "strength": 0.7,
            "orientation": "horizontal",
            "density": 0.4,
            "seed": 42,
        }

        cpu_result = cpu_operation.apply(test_image, params)
        gpu_result = operation.apply(test_image, params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Should be identical within machine epsilon
        np.testing.assert_allclose(
            cpu_array,
            gpu_array,
            rtol=1e-6,
            atol=1e-6,
            err_msg="GPU and CPU results differ",
        )

    def test_gpu_matches_cpu_grayscale(
        self, operation: CorduroyGPUOperation, cpu_operation: CorduroyOperation
    ) -> None:
        """Test that GPU and CPU match for grayscale images."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {
            "strength": 0.6,
            "orientation": "vertical",
            "density": 0.5,
            "seed": 42,
        }

        cpu_result = cpu_operation.apply(gray_image, params)
        gpu_result = operation.apply(gray_image, params)

        cpu_array = np.array(cpu_result).astype(np.float32)
        gpu_array = np.array(gpu_result).astype(np.float32)

        # Should be identical within machine epsilon
        np.testing.assert_allclose(
            cpu_array,
            gpu_array,
            rtol=1e-6,
            atol=1e-6,
            err_msg="GPU and CPU results differ",
        )

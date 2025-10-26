"""Tests for GPU-accelerated chromatic aberration operation (Mac only)."""

import platform
import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.chromatic_aberration import (
    ChromaticAberrationOperation,
)
from sevenrad_stills.operations.chromatic_aberration_gpu import (
    ChromaticAberrationGPUOperation,
)


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="GPU tests only run on Mac (Metal backend)",
)
class TestChromaticAberrationGPUOperation:
    """Tests for ChromaticAberrationGPUOperation class."""

    @pytest.fixture
    def operation(self) -> ChromaticAberrationGPUOperation:
        """Create a GPU chromatic aberration operation instance."""
        return ChromaticAberrationGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> ChromaticAberrationOperation:
        """Create a CPU chromatic aberration operation instance for comparison."""
        return ChromaticAberrationOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image with distinct RGB values."""
        # Create an image where R=100, G=150, B=200 for easy verification
        return Image.new("RGB", (100, 100), color=(100, 150, 200))

    @pytest.fixture
    def large_test_image(self) -> Image.Image:
        """Create a large test image for performance testing."""
        return Image.new("RGB", (2048, 2048), color=(100, 150, 200))

    def test_operation_name(self, operation: ChromaticAberrationGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "chromatic_aberration_gpu"

    def test_valid_params(self, operation: ChromaticAberrationGPUOperation) -> None:
        """Test valid parameter validation."""
        params = {"shift_x": 5, "shift_y": 3}
        operation.validate_params(params)  # Should not raise

    def test_missing_shift_x_raises_error(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test missing shift_x parameter raises error."""
        params = {"shift_y": 3}
        with pytest.raises(ValueError, match="Parameter 'shift_x' is required"):
            operation.validate_params(params)

    def test_missing_shift_y_raises_error(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test missing shift_y parameter raises error."""
        params = {"shift_x": 5}
        with pytest.raises(ValueError, match="Parameter 'shift_y' is required"):
            operation.validate_params(params)

    def test_invalid_shift_x_type_raises_error(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test invalid shift_x type raises error."""
        params = {"shift_x": "5", "shift_y": 3}
        with pytest.raises(ValueError, match="Parameter 'shift_x' must be an integer"):
            operation.validate_params(params)

    def test_invalid_shift_y_type_raises_error(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test invalid shift_y type raises error."""
        params = {"shift_x": 5, "shift_y": 3.5}
        with pytest.raises(ValueError, match="Parameter 'shift_y' must be an integer"):
            operation.validate_params(params)

    def test_apply_basic(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying GPU chromatic aberration."""
        params = {"shift_x": 5, "shift_y": 3}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_produces_change(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that GPU chromatic aberration actually changes the image."""
        params = {"shift_x": 5, "shift_y": 3}
        result = operation.apply(test_image, params)

        # Should have differences
        assert not np.array_equal(np.array(test_image), np.array(result))

    def test_zero_shift_preserves_image(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that zero shift preserves the image."""
        params = {"shift_x": 0, "shift_y": 0}
        result = operation.apply(test_image, params)

        # Should be identical
        np.testing.assert_array_equal(np.array(test_image), np.array(result))

    def test_green_channel_unchanged(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that green channel is never shifted (reference channel)."""
        params = {"shift_x": 5, "shift_y": 3}
        result = operation.apply(test_image, params)

        original_array = np.array(test_image)
        result_array = np.array(result)

        # Green channel (index 1) should be unchanged
        np.testing.assert_array_equal(original_array[:, :, 1], result_array[:, :, 1])

    def test_red_blue_shift_opposite_directions(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test that red and blue channels shift in opposite directions."""
        # Create a simple gradient image
        img_array = np.zeros((50, 50, 3), dtype=np.uint8)
        # Set unique patterns for each channel
        img_array[:, :, 0] = 100  # Red
        img_array[:, :, 1] = 150  # Green
        img_array[:, :, 2] = 200  # Blue

        # Create a bright spot in the center for each channel
        img_array[24:26, 24:26, 0] = 255
        img_array[24:26, 24:26, 2] = 255

        test_img = Image.fromarray(img_array)

        params = {"shift_x": 5, "shift_y": 0}
        result = operation.apply(test_img, params)
        result_array = np.array(result)

        # Red channel should shift right (positive x)
        # The bright spot at center should now be at x+5
        assert result_array[24, 29, 0] == 255  # Red shifted right

        # Blue channel should shift left (negative x)
        # The bright spot at center should now be at x-5
        assert result_array[24, 19, 2] == 255  # Blue shifted left

    def test_preserves_rgba_alpha(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test that RGBA images preserve alpha channel."""
        rgba_image = Image.new("RGBA", (100, 100), color=(100, 150, 200, 200))
        params = {"shift_x": 5, "shift_y": 3}
        result = operation.apply(rgba_image, params)

        assert result.mode == "RGBA"

        # Check that alpha channel is preserved
        original_array = np.array(rgba_image)
        result_array = np.array(result)

        np.testing.assert_array_equal(original_array[:, :, 3], result_array[:, :, 3])

    def test_grayscale_returns_copy(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test that grayscale images return unmodified copy."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"shift_x": 5, "shift_y": 3}
        result = operation.apply(gray_image, params)

        assert result.mode == "L"
        np.testing.assert_array_equal(np.array(gray_image), np.array(result))

    def test_negative_shift(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that negative shifts work correctly."""
        params = {"shift_x": -5, "shift_y": -3}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_large_shift(
        self, operation: ChromaticAberrationGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that large shifts work correctly."""
        params = {"shift_x": 50, "shift_y": 50}
        result = operation.apply(test_image, params)

        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_consistency_with_cpu_version_zero_shift(
        self,
        operation: ChromaticAberrationGPUOperation,
        cpu_operation: ChromaticAberrationOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU versions produce identical results with zero shift."""
        params = {"shift_x": 0, "shift_y": 0}

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(gpu_result), np.array(cpu_result))

    def test_consistency_with_cpu_version_positive_shift(
        self,
        operation: ChromaticAberrationGPUOperation,
        cpu_operation: ChromaticAberrationOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU versions produce identical results with positive shift."""
        params = {"shift_x": 5, "shift_y": 3}

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(gpu_result), np.array(cpu_result))

    def test_consistency_with_cpu_version_negative_shift(
        self,
        operation: ChromaticAberrationGPUOperation,
        cpu_operation: ChromaticAberrationOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU versions produce identical results with negative shift."""
        params = {"shift_x": -5, "shift_y": -3}

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(gpu_result), np.array(cpu_result))

    def test_consistency_with_cpu_version_mixed_shift(
        self,
        operation: ChromaticAberrationGPUOperation,
        cpu_operation: ChromaticAberrationOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU versions produce identical results with mixed signs."""
        params = {"shift_x": 5, "shift_y": -3}

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        np.testing.assert_array_equal(np.array(gpu_result), np.array(cpu_result))

    def test_gpu_performance_better_than_cpu(
        self,
        operation: ChromaticAberrationGPUOperation,
        cpu_operation: ChromaticAberrationOperation,
        large_test_image: Image.Image,
    ) -> None:
        """Test that GPU version is faster than CPU version on large images."""
        params = {"shift_x": 10, "shift_y": 5}

        # Warm up GPU (first call may include initialization overhead)
        _ = operation.apply(large_test_image, params)

        # Benchmark GPU version
        gpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = operation.apply(large_test_image, params)
            end = time.perf_counter()
            gpu_times.append(end - start)
        gpu_avg = np.mean(gpu_times)

        # Benchmark CPU version
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = cpu_operation.apply(large_test_image, params)
            end = time.perf_counter()
            cpu_times.append(end - start)
        cpu_avg = np.mean(cpu_times)

        # GPU must be faster than CPU
        speedup = cpu_avg / gpu_avg
        print(f"\nPerformance comparison (2048x2048 image):")
        print(f"  CPU average: {cpu_avg*1000:.2f}ms")
        print(f"  GPU average: {gpu_avg*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert GPU is faster (must have at least 1.1x speedup to ensure clear win)
        assert (
            gpu_avg < cpu_avg
        ), f"GPU ({gpu_avg*1000:.2f}ms) must be faster than CPU ({cpu_avg*1000:.2f}ms)"
        assert (
            speedup >= 1.1
        ), f"GPU speedup ({speedup:.2f}x) must be at least 1.1x over CPU"

    def test_boundary_handling(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test that boundary pixels are handled correctly (constant mode)."""
        # Create a small image with unique values
        img_array = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)
        img_array = np.repeat(img_array, 3, axis=2)  # Make RGB
        test_img = Image.fromarray(img_array)

        params = {"shift_x": 2, "shift_y": 0}
        result = operation.apply(test_img, params)
        result_array = np.array(result)

        # Red channel shifted right by 2 - left edge should be 0 (constant boundary)
        assert result_array[0, 0, 0] == 0
        assert result_array[0, 1, 0] == 0

        # Blue channel shifted left by 2 - right edge should be 0 (constant boundary)
        assert result_array[0, 4, 2] == 0
        assert result_array[0, 3, 2] == 0

    def test_various_image_sizes(
        self, operation: ChromaticAberrationGPUOperation
    ) -> None:
        """Test GPU operation works with various image sizes."""
        sizes = [(50, 50), (100, 200), (200, 100), (1024, 768), (768, 1024)]
        params = {"shift_x": 5, "shift_y": 3}

        for width, height in sizes:
            test_img = Image.new("RGB", (width, height), color=(100, 150, 200))
            result = operation.apply(test_img, params)

            assert result.size == (width, height)
            assert result.mode == "RGB"

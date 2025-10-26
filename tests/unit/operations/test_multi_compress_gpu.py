"""Tests for GPU-accelerated multi-compression operation."""

import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.multi_compress_gpu import MultiCompressGPUOperation


class TestMultiCompressGPUOperation:
    """Tests for MultiCompressGPUOperation class."""

    @pytest.fixture
    def operation(self) -> MultiCompressGPUOperation:
        """Create a GPU multi-compress operation instance."""
        return MultiCompressGPUOperation()

    @pytest.fixture
    def cpu_operation(self) -> MultiCompressOperation:
        """Create a CPU multi-compress operation instance for comparison."""
        return MultiCompressOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    @pytest.fixture
    def large_test_image(self) -> Image.Image:
        """Create a larger test image for performance testing."""
        return Image.new("RGB", (1920, 1080), color=(128, 64, 192))

    def test_operation_name(self, operation: MultiCompressGPUOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "multi_compress_gpu"

    def test_valid_params_fixed_decay(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test valid parameters with fixed decay."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_linear_decay(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test valid parameters with linear decay."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
        operation.validate_params(params)  # Should not raise

    def test_valid_params_exponential_decay(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test valid parameters with exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 15,
            "decay": "exponential",
        }
        operation.validate_params(params)  # Should not raise

    def test_missing_iterations_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test missing iterations parameter raises error."""
        params = {"quality_start": 50}
        with pytest.raises(ValueError, match="requires 'iterations' parameter"):
            operation.validate_params(params)

    def test_invalid_iterations_type_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test invalid iterations type raises error."""
        params = {"iterations": "5", "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be an integer"):
            operation.validate_params(params)

    def test_iterations_too_low_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test iterations below minimum raises error."""
        params = {"iterations": 0, "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be between"):
            operation.validate_params(params)

    def test_iterations_too_high_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test iterations above maximum raises error."""
        params = {"iterations": 51, "quality_start": 50}
        with pytest.raises(ValueError, match="Iterations must be between"):
            operation.validate_params(params)

    def test_missing_quality_start_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test missing quality_start parameter raises error."""
        params = {"iterations": 5}
        with pytest.raises(ValueError, match="requires 'quality_start' parameter"):
            operation.validate_params(params)

    def test_invalid_quality_start_type_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test invalid quality_start type raises error."""
        params = {"iterations": 5, "quality_start": "50"}
        with pytest.raises(ValueError, match="Quality start must be an integer"):
            operation.validate_params(params)

    def test_quality_start_out_of_range_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test quality_start out of range raises error."""
        params = {"iterations": 5, "quality_start": 101}
        with pytest.raises(ValueError, match="Quality start must be between"):
            operation.validate_params(params)

    def test_invalid_decay_type_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test invalid decay type raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "logarithmic"}
        with pytest.raises(ValueError, match="Decay must be"):
            operation.validate_params(params)

    def test_linear_decay_missing_quality_end_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test linear decay without quality_end raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "linear"}
        with pytest.raises(ValueError, match="requires 'quality_end' parameter"):
            operation.validate_params(params)

    def test_exponential_decay_missing_quality_end_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test exponential decay without quality_end raises error."""
        params = {"iterations": 5, "quality_start": 50, "decay": "exponential"}
        with pytest.raises(ValueError, match="requires 'quality_end' parameter"):
            operation.validate_params(params)

    def test_quality_end_not_less_than_start_raises_error(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test quality_end >= quality_start raises error."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 60,
            "decay": "linear",
        }
        with pytest.raises(ValueError, match=r"Quality end.*must be less than"):
            operation.validate_params(params)

    def test_apply_single_iteration(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying single compression iteration."""
        params = {"iterations": 1, "quality_start": 50}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_fixed_decay(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with fixed quality."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_linear_decay(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with linear decay."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_exponential_decay(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 10,
            "decay": "exponential",
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_with_subsampling(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test multi-compression with subsampling."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
            "subsampling": 2,
        }
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_converts_rgba_to_rgb(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_preserves_grayscale(
        self, operation: MultiCompressGPUOperation
    ) -> None:
        """Test that grayscale images remain grayscale."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(gray_image, params)
        assert result.mode == "L"

    def test_deterministic_output(
        self, operation: MultiCompressGPUOperation, test_image: Image.Image
    ) -> None:
        """Test that multi-compression produces deterministic output."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }
        result1 = operation.apply(test_image, params)
        result2 = operation.apply(test_image, params)

        # Images should be the same size and mode
        assert result1.size == result2.size
        assert result1.mode == result2.mode

        # Pixel values should be very similar (JPEG can have minor variations)
        pixels1 = list(result1.getdata())
        pixels2 = list(result2.getdata())
        assert len(pixels1) == len(pixels2)

        # Check that most pixels match (allow for minor JPEG variations)
        matching_pixels = sum(
            1 for p1, p2 in zip(pixels1, pixels2, strict=False) if p1 == p2
        )
        match_ratio = matching_pixels / len(pixels1)
        assert match_ratio > 0.99  # At least 99% of pixels should match

    def test_numerical_equivalence_with_cpu(
        self,
        operation: MultiCompressGPUOperation,
        cpu_operation: MultiCompressOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU versions produce identical results."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Convert to numpy arrays for comparison
        gpu_array = np.array(gpu_result)
        cpu_array = np.array(cpu_result)

        # Results should be identical (machine epsilon)
        # JPEG encoding is deterministic, so results should be byte-for-byte identical
        np.testing.assert_array_equal(gpu_array, cpu_array)

    def test_numerical_equivalence_exponential_decay(
        self,
        operation: MultiCompressGPUOperation,
        cpu_operation: MultiCompressOperation,
        test_image: Image.Image,
    ) -> None:
        """Test GPU and CPU produce identical results for exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 10,
            "decay": "exponential",
        }

        gpu_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Convert to numpy arrays for comparison
        gpu_array = np.array(gpu_result)
        cpu_array = np.array(cpu_result)

        # Results should be identical
        np.testing.assert_array_equal(gpu_array, cpu_array)

    @pytest.mark.mac
    def test_performance_comparison_small_image(
        self,
        operation: MultiCompressGPUOperation,
        cpu_operation: MultiCompressOperation,
        test_image: Image.Image,
    ) -> None:
        """
        Test performance comparison between CPU and GPU for small images.

        NOTE: GPU version is expected to be similar or slightly slower than CPU
        due to Huffman encoding being CPU-bound and lack of GPU-acceleratable
        preprocessing operations.
        """
        params = {"iterations": 10, "quality_start": 50}

        # Warm-up
        _ = cpu_operation.apply(test_image, params)
        _ = operation.apply(test_image, params)

        # CPU benchmark
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = cpu_operation.apply(test_image, params)
            cpu_times.append(time.perf_counter() - start)

        # GPU benchmark
        gpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = operation.apply(test_image, params)
            gpu_times.append(time.perf_counter() - start)

        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)

        # Document performance characteristics
        print(f"\nSmall image (100x100) performance:")  # noqa: T201
        print(f"  CPU average: {cpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  GPU average: {gpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  Ratio (GPU/CPU): {gpu_avg/cpu_avg:.2f}x")  # noqa: T201

        # GPU version should be within reasonable range of CPU
        # (may be slightly slower due to being CPU-bound)
        # Allow 3x slower as acceptable due to Huffman bottleneck
        assert gpu_avg < cpu_avg * 3.0, (
            f"GPU version significantly slower than expected: "
            f"{gpu_avg/cpu_avg:.2f}x slower"
        )

    @pytest.mark.mac
    def test_performance_comparison_large_image(
        self,
        operation: MultiCompressGPUOperation,
        cpu_operation: MultiCompressOperation,
        large_test_image: Image.Image,
    ) -> None:
        """
        Test performance comparison between CPU and GPU for large images.

        NOTE: GPU version is expected to be similar to CPU due to being
        CPU-bound by serial Huffman encoding. This test documents the
        performance characteristics.
        """
        params = {"iterations": 5, "quality_start": 50}

        # Warm-up
        _ = cpu_operation.apply(large_test_image, params)
        _ = operation.apply(large_test_image, params)

        # CPU benchmark
        cpu_times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = cpu_operation.apply(large_test_image, params)
            cpu_times.append(time.perf_counter() - start)

        # GPU benchmark
        gpu_times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = operation.apply(large_test_image, params)
            gpu_times.append(time.perf_counter() - start)

        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)

        # Document performance characteristics
        print(f"\nLarge image (1920x1080) performance:")  # noqa: T201
        print(f"  CPU average: {cpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  GPU average: {gpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  Ratio (GPU/CPU): {gpu_avg/cpu_avg:.2f}x")  # noqa: T201

        # For large images, GPU should still be within reasonable range
        # (Huffman encoding dominates, so no speedup expected)
        assert gpu_avg < cpu_avg * 3.0, (
            f"GPU version significantly slower than expected: "
            f"{gpu_avg/cpu_avg:.2f}x slower"
        )

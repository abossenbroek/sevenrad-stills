"""Tests for Metal-accelerated multi-compression operation."""

import sys
import time

import numpy as np
import pytest
from PIL import Image

# Only import Metal operation on macOS
if sys.platform == "darwin":
    from sevenrad_stills.operations.multi_compress_metal import (
        MultiCompressMetalOperation,
    )

from sevenrad_stills.operations.multi_compress import MultiCompressOperation
from sevenrad_stills.operations.multi_compress_gpu import MultiCompressGPUOperation

pytestmark = pytest.mark.mac


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal only available on macOS")
class TestMultiCompressMetalOperation:
    """Tests for MultiCompressMetalOperation class."""

    @pytest.fixture
    def operation(self) -> MultiCompressMetalOperation:
        """Create a Metal multi-compress operation instance."""
        return MultiCompressMetalOperation()

    @pytest.fixture
    def cpu_operation(self) -> MultiCompressOperation:
        """Create a CPU multi-compress operation instance for comparison."""
        return MultiCompressOperation()

    @pytest.fixture
    def gpu_operation(self) -> MultiCompressGPUOperation:
        """Create a GPU multi-compress operation instance for comparison."""
        return MultiCompressGPUOperation()

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (100, 100), color=(128, 64, 192))

    @pytest.fixture
    def large_test_image(self) -> Image.Image:
        """Create a larger test image for performance testing."""
        return Image.new("RGB", (1920, 1080), color=(128, 64, 192))

    def test_operation_name(self, operation: MultiCompressMetalOperation) -> None:
        """Test operation has correct name."""
        assert operation.name == "multi_compress_metal"

    def test_valid_params_fixed_decay(
        self, operation: MultiCompressMetalOperation
    ) -> None:
        """Test valid parameters with fixed decay."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        operation.validate_params(params)  # Should not raise

    def test_valid_params_linear_decay(
        self, operation: MultiCompressMetalOperation
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
        self, operation: MultiCompressMetalOperation
    ) -> None:
        """Test valid parameters with exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 15,
            "decay": "exponential",
        }
        operation.validate_params(params)  # Should not raise

    def test_apply_single_iteration(
        self, operation: MultiCompressMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying single compression iteration."""
        params = {"iterations": 1, "quality_start": 50}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_apply_fixed_decay(
        self, operation: MultiCompressMetalOperation, test_image: Image.Image
    ) -> None:
        """Test applying multi-compression with fixed quality."""
        params = {"iterations": 5, "quality_start": 50, "decay": "fixed"}
        result = operation.apply(test_image, params)
        assert isinstance(result, Image.Image)
        assert result.size == test_image.size

    def test_apply_linear_decay(
        self, operation: MultiCompressMetalOperation, test_image: Image.Image
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
        self, operation: MultiCompressMetalOperation, test_image: Image.Image
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

    def test_apply_converts_rgba_to_rgb(
        self, operation: MultiCompressMetalOperation
    ) -> None:
        """Test that RGBA images are converted to RGB."""
        rgba_image = Image.new("RGBA", (100, 100), color=(128, 64, 192, 255))
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(rgba_image, params)
        assert result.mode == "RGB"

    def test_apply_preserves_grayscale(
        self, operation: MultiCompressMetalOperation
    ) -> None:
        """Test that grayscale images remain grayscale."""
        gray_image = Image.new("L", (100, 100), color=128)
        params = {"iterations": 3, "quality_start": 50}
        result = operation.apply(gray_image, params)
        assert result.mode == "L"

    def test_deterministic_output(
        self, operation: MultiCompressMetalOperation, test_image: Image.Image
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

        # Pixel values should be identical (JPEG is deterministic)
        pixels1 = list(result1.getdata())
        pixels2 = list(result2.getdata())
        assert len(pixels1) == len(pixels2)

        # Check that pixels match exactly
        matching_pixels = sum(
            1 for p1, p2 in zip(pixels1, pixels2, strict=False) if p1 == p2
        )
        match_ratio = matching_pixels / len(pixels1)
        assert match_ratio > 0.99  # At least 99% of pixels should match

    def test_numerical_equivalence_with_cpu(
        self,
        operation: MultiCompressMetalOperation,
        cpu_operation: MultiCompressOperation,
        test_image: Image.Image,
    ) -> None:
        """Test Metal and CPU versions produce identical results."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }

        metal_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Convert to numpy arrays for comparison
        metal_array = np.array(metal_result)
        cpu_array = np.array(cpu_result)

        # Results should be identical (machine epsilon)
        # JPEG encoding is deterministic, so results should be byte-for-byte identical
        np.testing.assert_array_equal(metal_array, cpu_array)

    def test_numerical_equivalence_with_gpu(
        self,
        operation: MultiCompressMetalOperation,
        gpu_operation: MultiCompressGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """Test Metal and GPU versions produce identical results."""
        params = {
            "iterations": 5,
            "quality_start": 50,
            "quality_end": 10,
            "decay": "linear",
        }

        metal_result = operation.apply(test_image, params)
        gpu_result = gpu_operation.apply(test_image, params)

        # Convert to numpy arrays for comparison
        metal_array = np.array(metal_result)
        gpu_array = np.array(gpu_result)

        # Results should be identical
        np.testing.assert_array_equal(metal_array, gpu_array)

    def test_numerical_equivalence_exponential_decay(
        self,
        operation: MultiCompressMetalOperation,
        cpu_operation: MultiCompressOperation,
        test_image: Image.Image,
    ) -> None:
        """Test Metal and CPU produce identical results for exponential decay."""
        params = {
            "iterations": 10,
            "quality_start": 80,
            "quality_end": 10,
            "decay": "exponential",
        }

        metal_result = operation.apply(test_image, params)
        cpu_result = cpu_operation.apply(test_image, params)

        # Convert to numpy arrays for comparison
        metal_array = np.array(metal_result)
        cpu_array = np.array(cpu_result)

        # Results should be identical
        np.testing.assert_array_equal(metal_array, cpu_array)

    def test_performance_comparison_small_image(
        self,
        operation: MultiCompressMetalOperation,
        cpu_operation: MultiCompressOperation,
        gpu_operation: MultiCompressGPUOperation,
        test_image: Image.Image,
    ) -> None:
        """
        Test performance comparison between CPU, GPU, and Metal for small images.

        NOTE: Metal version is expected to be similar to CPU/GPU due to being
        CPU-bound by serial Huffman encoding. This test documents the
        performance hierarchy: CPU ≈ GPU ≈ Metal (all CPU-bound).
        """
        params = {"iterations": 10, "quality_start": 50}

        # Warm-up
        _ = cpu_operation.apply(test_image, params)
        _ = gpu_operation.apply(test_image, params)
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
            _ = gpu_operation.apply(test_image, params)
            gpu_times.append(time.perf_counter() - start)

        # Metal benchmark
        metal_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = operation.apply(test_image, params)
            metal_times.append(time.perf_counter() - start)

        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)
        metal_avg = np.mean(metal_times)

        # Document performance characteristics
        print(f"\nSmall image (100x100) performance:")  # noqa: T201
        print(f"  CPU average:   {cpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  GPU average:   {gpu_avg*1000:.2f}ms (ratio: {gpu_avg/cpu_avg:.2f}x)")  # noqa: T201
        print(  # noqa: T201
            f"  Metal average: {metal_avg*1000:.2f}ms (ratio: {metal_avg/cpu_avg:.2f}x)"
        )

        # All versions should be within reasonable range (CPU-bound)
        assert metal_avg < cpu_avg * 3.0, (
            f"Metal version significantly slower than expected: "
            f"{metal_avg/cpu_avg:.2f}x slower"
        )

    def test_performance_comparison_large_image(
        self,
        operation: MultiCompressMetalOperation,
        cpu_operation: MultiCompressOperation,
        gpu_operation: MultiCompressGPUOperation,
        large_test_image: Image.Image,
    ) -> None:
        """
        Test performance comparison for large images.

        Documents performance hierarchy: CPU ≈ GPU ≈ Metal (all CPU-bound by
        Huffman encoding). This validates our architectural decision to use
        CPU-based JPEG encoding for all variants.
        """
        params = {"iterations": 5, "quality_start": 50}

        # Warm-up
        _ = cpu_operation.apply(large_test_image, params)
        _ = gpu_operation.apply(large_test_image, params)
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
            _ = gpu_operation.apply(large_test_image, params)
            gpu_times.append(time.perf_counter() - start)

        # Metal benchmark
        metal_times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = operation.apply(large_test_image, params)
            metal_times.append(time.perf_counter() - start)

        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)
        metal_avg = np.mean(metal_times)

        # Document performance characteristics
        print(f"\nLarge image (1920x1080) performance:")  # noqa: T201
        print(f"  CPU average:   {cpu_avg*1000:.2f}ms")  # noqa: T201
        print(f"  GPU average:   {gpu_avg*1000:.2f}ms (ratio: {gpu_avg/cpu_avg:.2f}x)")  # noqa: T201
        print(  # noqa: T201
            f"  Metal average: {metal_avg*1000:.2f}ms (ratio: {metal_avg/cpu_avg:.2f}x)"
        )

        # All versions should be within reasonable range
        assert metal_avg < cpu_avg * 3.0, (
            f"Metal version significantly slower than expected: "
            f"{metal_avg/cpu_avg:.2f}x slower"
        )

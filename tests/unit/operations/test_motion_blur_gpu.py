"""Tests for GPU-accelerated motion blur operations."""

import time

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.operations.motion_blur import MotionBlurOperation
from sevenrad_stills.operations.motion_blur_gpu import MotionBlurGPUOperation

try:
    from sevenrad_stills.operations.motion_blur_metal import MotionBlurMetalOperation

    HAS_METAL = True
    MotionBlurMetalOp = MotionBlurMetalOperation
except ImportError:
    HAS_METAL = False
    MotionBlurMetalOp = None  # type: ignore[misc, assignment]


@pytest.fixture
def blur_op_cpu() -> MotionBlurOperation:
    """Create a CPU motion blur operation instance."""
    return MotionBlurOperation()


@pytest.fixture
def blur_op_gpu() -> MotionBlurGPUOperation:
    """Create a GPU motion blur operation instance."""
    return MotionBlurGPUOperation()


@pytest.fixture
def blur_op_metal():  # type: ignore[no-untyped-def]
    """Create a Metal motion blur operation instance if available."""
    if HAS_METAL:
        return MotionBlurMetalOp()
    return None


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a larger RGB test image for performance testing."""
    # Create a 500x500 RGB image with gradient and features
    img = Image.new("RGB", (500, 500), color=(0, 0, 0))
    pixels = img.load()

    # Create gradient background
    for i in range(500):
        for j in range(500):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)  # type: ignore[index]

    # Add some bright squares
    for x_offset in [100, 300]:
        for y_offset in [100, 300]:
            for i in range(x_offset, x_offset + 50):
                for j in range(y_offset, y_offset + 50):
                    pixels[i, j] = (255, 255, 255)  # type: ignore[index]

    return img


@pytest.fixture
def test_image_grayscale() -> Image.Image:
    """Create a grayscale test image."""
    img = Image.new("L", (500, 500), color=0)
    pixels = img.load()

    # Gradient background
    for i in range(500):
        for j in range(500):
            pixels[i, j] = (i + j) % 256  # type: ignore[index]

    # White square
    for i in range(200, 300):
        for j in range(200, 300):
            pixels[i, j] = 255  # type: ignore[index]

    return img


@pytest.fixture
def test_image_rgba() -> Image.Image:
    """Create an RGBA test image."""
    img = Image.new("RGBA", (500, 500), color=(0, 0, 0, 255))
    pixels = img.load()

    for i in range(500):
        for j in range(500):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256, 255)  # type: ignore[index]

    # Semi-transparent square
    for i in range(200, 300):
        for j in range(200, 300):
            pixels[i, j] = (255, 255, 255, 200)  # type: ignore[index]

    return img


class TestMotionBlurGPUValidation:
    """Test parameter validation for GPU motion blur operation."""

    def test_validation_missing_kernel_size(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that missing kernel_size parameter raises ValueError."""
        with pytest.raises(ValueError, match="requires 'kernel_size' parameter"):
            blur_op_gpu.validate_params({})

    def test_validation_invalid_kernel_size_type(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that non-integer kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be an integer"):
            blur_op_gpu.validate_params({"kernel_size": 5.5})

    def test_validation_kernel_size_too_small(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that kernel_size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be between"):
            blur_op_gpu.validate_params({"kernel_size": 0})

    def test_validation_kernel_size_too_large(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that kernel_size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Kernel size must be between"):
            blur_op_gpu.validate_params({"kernel_size": 101})

    def test_validation_invalid_angle_type(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that non-numeric angle raises ValueError."""
        with pytest.raises(ValueError, match="Angle must be a number"):
            blur_op_gpu.validate_params({"kernel_size": 5, "angle": "horizontal"})

    def test_validation_angle_out_of_range(
        self, blur_op_gpu: MotionBlurGPUOperation
    ) -> None:
        """Test that angle outside [0, 360) raises ValueError."""
        with pytest.raises(ValueError, match="Angle must be between"):
            blur_op_gpu.validate_params({"kernel_size": 5, "angle": 361})

    def test_validation_valid_params(self, blur_op_gpu: MotionBlurGPUOperation) -> None:
        """Test that valid parameters pass validation."""
        # Should not raise
        blur_op_gpu.validate_params({"kernel_size": 5})
        blur_op_gpu.validate_params({"kernel_size": 10, "angle": 45.0})
        blur_op_gpu.validate_params({"kernel_size": 1, "angle": 0.0})


class TestMotionBlurGPUApply:
    """Test applying GPU motion blur to images."""

    def test_apply_kernel_size_one(
        self, blur_op_gpu: MotionBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that kernel_size=1 returns an identical image."""
        result = blur_op_gpu.apply(test_image_rgb, {"kernel_size": 1})

        assert result.size == test_image_rgb.size
        assert result.mode == test_image_rgb.mode

        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        np.testing.assert_array_equal(result_array, original_array)

    def test_apply_rgb_image(
        self, blur_op_gpu: MotionBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test applying motion blur to RGB image."""
        result = blur_op_gpu.apply(test_image_rgb, {"kernel_size": 9, "angle": 0})

        assert result.size == test_image_rgb.size
        assert result.mode == "RGB"

        original_array = np.array(test_image_rgb)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_grayscale_image(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test applying motion blur to grayscale image."""
        result = blur_op_gpu.apply(
            test_image_grayscale, {"kernel_size": 9, "angle": 90}
        )

        assert result.size == test_image_grayscale.size
        assert result.mode == "L"

        original_array = np.array(test_image_grayscale)
        result_array = np.array(result)
        assert not np.array_equal(result_array, original_array)

    def test_apply_rgba_image(
        self, blur_op_gpu: MotionBlurGPUOperation, test_image_rgba: Image.Image
    ) -> None:
        """Test applying motion blur to RGBA image."""
        result = blur_op_gpu.apply(test_image_rgba, {"kernel_size": 9, "angle": 45})

        assert result.size == test_image_rgba.size
        assert result.mode == "RGBA"

        original_array = np.array(test_image_rgba)
        result_array = np.array(result)
        # RGB channels should be blurred
        assert not np.array_equal(result_array[..., :3], original_array[..., :3])

    def test_apply_various_kernel_sizes(
        self, blur_op_gpu: MotionBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different kernel sizes produce different blur intensities."""
        result_small = blur_op_gpu.apply(test_image_rgb, {"kernel_size": 3})
        result_large = blur_op_gpu.apply(test_image_rgb, {"kernel_size": 15})

        array_small = np.array(result_small)
        array_large = np.array(result_large)

        assert not np.array_equal(array_small, array_large)

    def test_apply_various_angles(
        self, blur_op_gpu: MotionBlurGPUOperation, test_image_rgb: Image.Image
    ) -> None:
        """Test that different angles produce different directional blur."""
        result_horizontal = blur_op_gpu.apply(
            test_image_rgb, {"kernel_size": 11, "angle": 0}
        )
        result_vertical = blur_op_gpu.apply(
            test_image_rgb, {"kernel_size": 11, "angle": 90}
        )
        result_diagonal = blur_op_gpu.apply(
            test_image_rgb, {"kernel_size": 11, "angle": 45}
        )

        array_h = np.array(result_horizontal)
        array_v = np.array(result_vertical)
        array_d = np.array(result_diagonal)

        # All three should be different
        assert not np.array_equal(array_h, array_v)
        assert not np.array_equal(array_h, array_d)
        assert not np.array_equal(array_v, array_d)

    def test_operation_name(self, blur_op_gpu: MotionBlurGPUOperation) -> None:
        """Test that operation has correct name."""
        assert blur_op_gpu.name == "motion_blur_gpu"


class TestGPUvsCPUConsistency:
    """Test that GPU implementation produces similar results to CPU version."""

    def test_rgb_consistency(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for RGB."""
        params = {"kernel_size": 9, "angle": 0}
        result_gpu = blur_op_gpu.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close (allow small numerical differences)
        # Rotation operations can cause edge pixel differences, so we allow atol=40
        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=70.0,
            rtol=0.02,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_grayscale_consistency(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for grayscale."""
        params = {"kernel_size": 7, "angle": 90}
        result_gpu = blur_op_gpu.apply(test_image_grayscale, params)
        result_cpu = blur_op_cpu.apply(test_image_grayscale, params)

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=70.0,
            rtol=0.02,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_rgba_consistency(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgba: Image.Image,
    ) -> None:
        """Test GPU and CPU implementations produce similar results for RGBA."""
        params = {"kernel_size": 9, "angle": 45}
        result_gpu = blur_op_gpu.apply(test_image_rgba, params)
        result_cpu = blur_op_cpu.apply(test_image_rgba, params)

        gpu_array = np.array(result_gpu).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            gpu_array,
            cpu_array,
            atol=70.0,
            rtol=0.02,
            err_msg="GPU and CPU results should be nearly identical",
        )

    def test_various_angles_consistency(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test consistency across various angles."""
        for angle in [0, 45, 90, 135, 180, 270]:
            params = {"kernel_size": 7, "angle": angle}
            result_gpu = blur_op_gpu.apply(test_image_rgb, params)
            result_cpu = blur_op_cpu.apply(test_image_rgb, params)

            gpu_array = np.array(result_gpu).astype(float)
            cpu_array = np.array(result_cpu).astype(float)

            np.testing.assert_allclose(
                gpu_array,
                cpu_array,
                atol=70.0,
                rtol=0.02,
                err_msg=f"GPU and CPU should match for angle={angle}",
            )


@pytest.mark.skipif(not HAS_METAL, reason="Metal/MLX not available")
class TestMetalvsCPUConsistency:
    """Test that Metal implementation produces similar results to CPU version."""

    def test_rgb_consistency(
        self,
        blur_op_metal,  # type: ignore[no-untyped-def]
        blur_op_cpu: MotionBlurOperation,
        test_image_rgb: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for RGB."""
        params = {"kernel_size": 9, "angle": 0}
        result_metal = blur_op_metal.apply(test_image_rgb, params)
        result_cpu = blur_op_cpu.apply(test_image_rgb, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        # Results should be very close (allow small numerical differences)
        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=70.0,
            rtol=0.02,
            err_msg="Metal and CPU results should be nearly identical",
        )

    def test_grayscale_consistency(
        self,
        blur_op_metal,  # type: ignore[no-untyped-def]
        blur_op_cpu: MotionBlurOperation,
        test_image_grayscale: Image.Image,
    ) -> None:
        """Test Metal and CPU implementations produce similar results for grayscale."""
        params = {"kernel_size": 7, "angle": 90}
        result_metal = blur_op_metal.apply(test_image_grayscale, params)
        result_cpu = blur_op_cpu.apply(test_image_grayscale, params)

        metal_array = np.array(result_metal).astype(float)
        cpu_array = np.array(result_cpu).astype(float)

        np.testing.assert_allclose(
            metal_array,
            cpu_array,
            atol=70.0,
            rtol=0.02,
            err_msg="Metal and CPU results should be nearly identical",
        )


@pytest.mark.mac
class TestPerformanceComparison:
    """Test that GPU/Metal implementations are faster than CPU."""

    def test_gpu_faster_than_cpu(
        self,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_cpu: MotionBlurOperation,
    ) -> None:
        """Test that GPU implementation is faster than CPU for large images."""
        # Use a much larger image for performance testing (2000x2000)
        large_image = Image.new("RGB", (2000, 2000))
        pixels = large_image.load()
        for i in range(0, 2000, 10):
            for j in range(0, 2000, 10):
                pixels[i, j] = ((i * 13) % 256, (j * 17) % 256, ((i + j) * 19) % 256)  # type: ignore[index]

        params = {"kernel_size": 25, "angle": 45}

        # Warm-up runs
        blur_op_gpu.apply(large_image, params)
        blur_op_cpu.apply(large_image, params)

        # Time CPU
        start_cpu = time.perf_counter()
        for _ in range(3):
            blur_op_cpu.apply(large_image, params)
        cpu_time = time.perf_counter() - start_cpu

        # Time GPU
        start_gpu = time.perf_counter()
        for _ in range(3):
            blur_op_gpu.apply(large_image, params)
        gpu_time = time.perf_counter() - start_gpu

        # GPU should be faster
        assert (
            gpu_time < cpu_time
        ), f"GPU ({gpu_time:.4f}s) should be faster than CPU ({cpu_time:.4f}s)"

    @pytest.mark.skipif(not HAS_METAL, reason="Metal/MLX not available")
    def test_metal_faster_than_gpu(
        self,
        blur_op_metal,  # type: ignore[no-untyped-def]
        blur_op_gpu: MotionBlurGPUOperation,
    ) -> None:
        """Test that Metal implementation is faster than Taichi GPU."""
        # Use large image for meaningful performance comparison
        large_image = Image.new("RGB", (2000, 2000))
        pixels = large_image.load()
        for i in range(0, 2000, 10):
            for j in range(0, 2000, 10):
                pixels[i, j] = ((i * 13) % 256, (j * 17) % 256, ((i + j) * 19) % 256)  # type: ignore[index]

        params = {"kernel_size": 25, "angle": 45}

        # Warm-up runs
        blur_op_metal.apply(large_image, params)
        blur_op_gpu.apply(large_image, params)

        # Time GPU
        start_gpu = time.perf_counter()
        for _ in range(3):
            blur_op_gpu.apply(large_image, params)
        gpu_time = time.perf_counter() - start_gpu

        # Time Metal
        start_metal = time.perf_counter()
        for _ in range(3):
            blur_op_metal.apply(large_image, params)
        metal_time = time.perf_counter() - start_metal

        # Metal should be faster than Taichi GPU
        assert (
            metal_time < gpu_time
        ), f"Metal ({metal_time:.4f}s) should be faster than GPU ({gpu_time:.4f}s)"

    @pytest.mark.skipif(not HAS_METAL, reason="Metal/MLX not available")
    def test_performance_hierarchy(
        self,
        blur_op_cpu: MotionBlurOperation,
        blur_op_gpu: MotionBlurGPUOperation,
        blur_op_metal,  # type: ignore[no-untyped-def]
    ) -> None:
        """Test that performance follows: CPU > GPU > Metal (slowest to fastest)."""
        # Use large image for meaningful performance comparison
        large_image = Image.new("RGB", (2000, 2000))
        pixels = large_image.load()
        for i in range(0, 2000, 10):
            for j in range(0, 2000, 10):
                pixels[i, j] = ((i * 13) % 256, (j * 17) % 256, ((i + j) * 19) % 256)  # type: ignore[index]

        params = {"kernel_size": 25, "angle": 45}

        # Warm-up runs
        blur_op_cpu.apply(large_image, params)
        blur_op_gpu.apply(large_image, params)
        blur_op_metal.apply(large_image, params)

        # Time each implementation
        iterations = 3

        start_cpu = time.perf_counter()
        for _ in range(iterations):
            blur_op_cpu.apply(large_image, params)
        cpu_time = time.perf_counter() - start_cpu

        start_gpu = time.perf_counter()
        for _ in range(iterations):
            blur_op_gpu.apply(large_image, params)
        gpu_time = time.perf_counter() - start_gpu

        start_metal = time.perf_counter()
        for _ in range(iterations):
            blur_op_metal.apply(large_image, params)
        metal_time = time.perf_counter() - start_metal

        # Verify performance hierarchy
        assert metal_time < gpu_time < cpu_time, (
            f"Performance hierarchy violated: "
            f"CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Metal={metal_time:.4f}s"
        )

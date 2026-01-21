"""
Tests for shared Taichi kernel utilities.

Tests the reference NumPy implementations to ensure correctness.
GPU-specific tests are marked with @pytest.mark.gpu.
"""

import math

import numpy as np
import pytest


class TestPCGHash:
    """Tests for PCG hash-based random number generation."""

    def test_pcg_hash_deterministic(self) -> None:
        """Same input always produces same output."""
        from sevenrad_stills.operations.taichi_kernels.random import pcg_hash_numpy

        result1 = pcg_hash_numpy(12345)
        result2 = pcg_hash_numpy(12345)
        assert result1 == result2

    def test_pcg_hash_different_inputs(self) -> None:
        """Different inputs produce different outputs."""
        from sevenrad_stills.operations.taichi_kernels.random import pcg_hash_numpy

        result1 = pcg_hash_numpy(12345)
        result2 = pcg_hash_numpy(12346)
        assert result1 != result2

    def test_pcg_hash_wraps_32bit(self) -> None:
        """Output is 32-bit unsigned."""
        from sevenrad_stills.operations.taichi_kernels.random import pcg_hash_numpy

        result = pcg_hash_numpy(0xFFFFFFFF)
        assert 0 <= result <= 0xFFFFFFFF


class TestRandFloat:
    """Tests for deterministic random float generation."""

    def test_rand_float_in_range(self) -> None:
        """Output is in [0, 1)."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        for x in range(10):
            for y in range(10):
                for seed in [0, 42, 12345]:
                    result = rand_float_numpy(x, y, seed)
                    assert 0.0 <= result < 1.0

    def test_rand_float_deterministic(self) -> None:
        """Same inputs produce same output."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        result1 = rand_float_numpy(5, 10, 42)
        result2 = rand_float_numpy(5, 10, 42)
        assert result1 == result2

    def test_rand_float_varies_with_position(self) -> None:
        """Different positions produce different values."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        result1 = rand_float_numpy(0, 0, 42)
        result2 = rand_float_numpy(0, 1, 42)
        result3 = rand_float_numpy(1, 0, 42)
        assert result1 != result2
        assert result2 != result3

    def test_rand_float_varies_with_seed(self) -> None:
        """Different seeds produce different values."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_float_numpy

        result1 = rand_float_numpy(5, 5, 0)
        result2 = rand_float_numpy(5, 5, 1)
        assert result1 != result2


class TestRandGaussian:
    """Tests for Gaussian random number generation."""

    def test_rand_gaussian_mean_approximately_zero(self) -> None:
        """Mean of many samples should be close to 0."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_gaussian_numpy

        samples = []
        for x in range(50):
            for y in range(50):
                samples.append(rand_gaussian_numpy(x, y, 42, 1.0))

        mean = np.mean(samples)
        assert abs(mean) < 0.2  # Allow some variation

    def test_rand_gaussian_std_approximately_sigma(self) -> None:
        """Standard deviation should be close to sigma."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_gaussian_numpy

        sigma = 2.0
        samples = []
        for x in range(50):
            for y in range(50):
                samples.append(rand_gaussian_numpy(x, y, 42, sigma))

        std = np.std(samples)
        # Allow 30% tolerance
        assert abs(std - sigma) / sigma < 0.3

    def test_rand_gaussian_deterministic(self) -> None:
        """Same inputs produce same output."""
        from sevenrad_stills.operations.taichi_kernels.random import rand_gaussian_numpy

        result1 = rand_gaussian_numpy(5, 10, 42, 0.5)
        result2 = rand_gaussian_numpy(5, 10, 42, 0.5)
        assert result1 == result2


class TestClampCoords:
    """Tests for coordinate clamping."""

    def test_clamp_coords_in_bounds(self) -> None:
        """Coordinates in bounds are unchanged."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            clamp_coords_numpy,
        )

        result = clamp_coords_numpy(5, 10, 20, 30)
        assert result == (5, 10)

    def test_clamp_coords_negative(self) -> None:
        """Negative coordinates clamp to 0."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            clamp_coords_numpy,
        )

        result = clamp_coords_numpy(-5, -10, 20, 30)
        assert result == (0, 0)

    def test_clamp_coords_too_large(self) -> None:
        """Large coordinates clamp to size-1."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            clamp_coords_numpy,
        )

        result = clamp_coords_numpy(25, 35, 20, 30)
        assert result == (19, 29)


class TestReflectBoundary:
    """Tests for coordinate reflection at boundaries."""

    def test_reflect_in_bounds(self) -> None:
        """Coordinates in bounds are unchanged."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            reflect_boundary_numpy,
        )

        assert reflect_boundary_numpy(3, 10) == 3
        assert reflect_boundary_numpy(0, 10) == 0
        assert reflect_boundary_numpy(9, 10) == 9

    def test_reflect_negative(self) -> None:
        """Negative coordinates reflect to positive."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            reflect_boundary_numpy,
        )

        assert reflect_boundary_numpy(-1, 10) == 0
        assert reflect_boundary_numpy(-2, 10) == 1
        assert reflect_boundary_numpy(-3, 10) == 2

    def test_reflect_too_large(self) -> None:
        """Large coordinates reflect back."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            reflect_boundary_numpy,
        )

        # For size=10 (indices 0-9):
        # coord=10 is 1 past edge -> reflects to 9
        # coord=11 is 2 past edge -> reflects to 8
        # coord=12 is 3 past edge -> reflects to 7
        assert reflect_boundary_numpy(10, 10) == 9
        assert reflect_boundary_numpy(11, 10) == 8
        assert reflect_boundary_numpy(12, 10) == 7


class TestBilinearSample:
    """Tests for bilinear interpolation."""

    def test_bilinear_at_integer(self) -> None:
        """Sampling at integer coordinates returns exact pixel."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            bilinear_sample_numpy,
        )

        image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]).astype(
            np.float32
        )

        result = bilinear_sample_numpy(image, 0, 0)
        np.testing.assert_array_almost_equal(result, [1, 2, 3])

        result = bilinear_sample_numpy(image, 1, 1)
        np.testing.assert_array_almost_equal(result, [10, 11, 12])

    def test_bilinear_center(self) -> None:
        """Sampling at 0.5, 0.5 averages four corners."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            bilinear_sample_numpy,
        )

        # All corners have different values
        image = np.array([[[0, 0, 0], [4, 4, 4]], [[8, 8, 8], [12, 12, 12]]]).astype(
            np.float32
        )

        result = bilinear_sample_numpy(image, 0.5, 0.5)
        expected = (0 + 4 + 8 + 12) / 4
        np.testing.assert_array_almost_equal(result, [expected, expected, expected])

    def test_bilinear_edge_clamp(self) -> None:
        """Sampling beyond edge clamps to edge values."""
        from sevenrad_stills.operations.taichi_kernels.sampling import (
            bilinear_sample_numpy,
        )

        image = np.ones((3, 3, 3), dtype=np.float32)
        image[0, 0] = [1, 2, 3]

        # Sample way beyond - should still get valid result
        result = bilinear_sample_numpy(image, -10, -10)
        np.testing.assert_array_almost_equal(result, [1, 2, 3])


class TestGaussianKernel:
    """Tests for Gaussian kernel generation."""

    def test_gaussian_kernel_sums_to_one(self) -> None:
        """Kernel weights sum to 1.0."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            gaussian_kernel_1d,
        )

        for sigma in [0.5, 1.0, 2.0, 5.0]:
            kernel = gaussian_kernel_1d(sigma)
            assert abs(kernel.sum() - 1.0) < 1e-10

    def test_gaussian_kernel_symmetric(self) -> None:
        """Kernel is symmetric around center."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            gaussian_kernel_1d,
        )

        kernel = gaussian_kernel_1d(2.0)
        n = len(kernel)
        center = n // 2

        for i in range(center):
            assert abs(kernel[center - i] - kernel[center + i]) < 1e-10

    def test_gaussian_kernel_size(self) -> None:
        """Kernel has expected size."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            gaussian_kernel_1d,
        )

        kernel = gaussian_kernel_1d(1.0, radius=5)
        assert len(kernel) == 11  # 2*5 + 1


class TestCircularKernel:
    """Tests for circular kernel generation."""

    def test_circular_kernel_mask_shape(self) -> None:
        """Mask has correct shape."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            circular_kernel,
        )

        kernel, mask = circular_kernel(5)
        assert kernel.shape == (11, 11)
        assert mask.shape == (11, 11)

    def test_circular_kernel_center_included(self) -> None:
        """Center pixel is always included."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            circular_kernel,
        )

        for r in [1, 2, 5, 10]:
            kernel, mask = circular_kernel(r)
            assert mask[r, r] == 1.0

    def test_circular_kernel_normalized(self) -> None:
        """Kernel weights sum to 1.0."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            circular_kernel,
        )

        kernel, mask = circular_kernel(3)
        assert abs(kernel.sum() - 1.0) < 1e-10


class TestConvolve1D:
    """Tests for 1D convolution."""

    def test_convolve_horizontal_identity(self) -> None:
        """Identity kernel [0, 1, 0] doesn't change image."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_horizontal_numpy,
        )

        image = np.random.rand(10, 10, 3).astype(np.float32)
        kernel = np.array([0.0, 1.0, 0.0])

        result = convolve_horizontal_numpy(image, kernel)
        np.testing.assert_array_almost_equal(result, image)

    def test_convolve_vertical_identity(self) -> None:
        """Identity kernel [0, 1, 0] doesn't change image."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_vertical_numpy,
        )

        image = np.random.rand(10, 10, 3).astype(np.float32)
        kernel = np.array([0.0, 1.0, 0.0])

        result = convolve_vertical_numpy(image, kernel)
        np.testing.assert_array_almost_equal(result, image)

    def test_convolve_horizontal_box_blur(self) -> None:
        """Box blur kernel averages neighbors."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_horizontal_numpy,
        )

        # Simple gradient image
        image = np.zeros((3, 5, 1), dtype=np.float32)
        image[1, :, 0] = [0, 1, 2, 3, 4]

        kernel = np.array([1 / 3, 1 / 3, 1 / 3])
        result = convolve_horizontal_numpy(image, kernel)

        # Center pixel should average 1, 2, 3
        expected_center = 2.0
        assert abs(result[1, 2, 0] - expected_center) < 1e-5


class TestConvolve2D:
    """Tests for 2D convolution."""

    def test_convolve_2d_identity(self) -> None:
        """Identity kernel with center=1 doesn't change image."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_2d_numpy,
        )

        image = np.random.rand(10, 10, 3).astype(np.float32)
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

        result = convolve_2d_numpy(image, kernel)
        np.testing.assert_array_almost_equal(result, image)

    def test_convolve_2d_box_blur(self) -> None:
        """3x3 box blur averages 9 neighbors."""
        from sevenrad_stills.operations.taichi_kernels.convolution import (
            convolve_2d_numpy,
        )

        image = np.ones((5, 5, 1), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32) / 9

        result = convolve_2d_numpy(image, kernel)
        # All ones input, box blur should still be all ones
        np.testing.assert_array_almost_equal(result, image)


@pytest.mark.gpu
class TestTaichiKernels:
    """GPU tests for Taichi kernel utilities."""

    def test_rand_float_taichi_matches_numpy(self) -> None:
        """Taichi rand_float matches NumPy reference."""
        try:
            import taichi as ti
            from sevenrad_stills.operations.taichi_kernels.random import (
                rand_float,
                rand_float_numpy,
            )

            ti.init(arch=ti.cpu)

            @ti.kernel
            def test_kernel(x: ti.i32, y: ti.i32, seed: ti.i32) -> ti.f32:
                return rand_float(x, y, seed)

            for x in range(5):
                for y in range(5):
                    taichi_result = test_kernel(x, y, 42)
                    numpy_result = rand_float_numpy(x, y, 42)
                    assert abs(taichi_result - numpy_result) < 1e-6

        except ImportError:
            pytest.skip("Taichi not available")

    def test_bilinear_sample_taichi_matches_numpy(self) -> None:
        """Taichi bilinear sampling matches NumPy reference."""
        try:
            import taichi as ti
            from sevenrad_stills.operations.taichi_kernels.sampling import (
                bilinear_sample,
                bilinear_sample_numpy,
            )

            ti.init(arch=ti.cpu)

            # Create test image
            image = np.random.rand(4, 4, 4).astype(np.float32)

            # Create Taichi field
            field = ti.Vector.field(4, dtype=ti.f32, shape=(1, 4, 4))
            for i in range(4):
                for j in range(4):
                    field[0, i, j] = [
                        image[i, j, 0],
                        image[i, j, 1],
                        image[i, j, 2],
                        1.0,
                    ]

            @ti.kernel
            def sample_kernel(y: ti.f32, x: ti.f32) -> ti.math.vec4:
                return bilinear_sample(field, 0, y, x, 4, 4)

            # Test various positions
            for y in [0.5, 1.5, 2.5]:
                for x in [0.5, 1.5, 2.5]:
                    taichi_result = sample_kernel(y, x)
                    numpy_result = bilinear_sample_numpy(image, y, x)

                    for c in range(3):
                        assert abs(taichi_result[c] - numpy_result[c]) < 1e-5

        except ImportError:
            pytest.skip("Taichi not available")

"""
Tests for Taichi downscale operation.

This test suite verifies:
- Parameter validation
- Dimension changes (output_shape_factor)
- Correctness vs NumPy reference
- Upscale round-trip behavior
- Different sampling methods
"""

import numpy as np
import pytest
from sevenrad_stills.operations.downscale_taichi import (
    TAICHI_AVAILABLE,
    DownscaleTaichiOperation,
)

# Skip all tests if Taichi is not available
pytestmark = pytest.mark.skipif(not TAICHI_AVAILABLE, reason="Taichi not available")

if TAICHI_AVAILABLE:
    import taichi as ti


@pytest.fixture
def operation():
    """Create downscale operation instance."""
    return DownscaleTaichiOperation()


class TestDownscaleTaichiOperation:
    """Test suite for DownscaleTaichiOperation."""

    def test_initialization(self, operation):
        """Test operation initialization."""
        assert operation.name == "downscale_taichi"
        assert not operation.supports_inplace
        assert not operation.is_compiled

    def test_output_shape_factor_downscale(self, operation):
        """Test output_shape_factor with downscale only."""
        params = {"scale": 0.5, "upscale": False}
        factor = operation.output_shape_factor(params)
        assert factor == (0.5, 0.5)

        params = {"scale": 0.25, "upscale": False}
        factor = operation.output_shape_factor(params)
        assert factor == (0.25, 0.25)

    def test_output_shape_factor_upscale(self, operation):
        """Test output_shape_factor with upscale enabled."""
        params = {"scale": 0.5, "upscale": True}
        factor = operation.output_shape_factor(params)
        assert factor == (1.0, 1.0)

        params = {"scale": 0.1, "upscale": True}
        factor = operation.output_shape_factor(params)
        assert factor == (1.0, 1.0)

    def test_validate_params_success(self, operation):
        """Test successful parameter validation."""
        # Basic required params
        params = {"scale": 0.5}
        operation.validate_params(params)

        # With all optional params
        params = {"scale": 0.5, "upscale": True, "method": "nearest"}
        operation.validate_params(params)

        params = {"scale": 0.25, "upscale": False, "method": "bilinear"}
        operation.validate_params(params)

    def test_validate_params_missing_scale(self, operation):
        """Test validation fails when scale is missing."""
        params = {"upscale": True}
        with pytest.raises(ValueError, match="requires 'scale' parameter"):
            operation.validate_params(params)

    def test_validate_params_invalid_scale_type(self, operation):
        """Test validation fails with invalid scale type."""
        params = {"scale": "0.5"}
        with pytest.raises(ValueError, match="Scale must be a number"):
            operation.validate_params(params)

    def test_validate_params_scale_out_of_range(self, operation):
        """Test validation fails when scale is out of range."""
        # Too small
        params = {"scale": 0.005}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

        # Too large
        params = {"scale": 1.5}
        with pytest.raises(ValueError, match="Scale must be between"):
            operation.validate_params(params)

    def test_validate_params_invalid_upscale_type(self, operation):
        """Test validation fails with invalid upscale type."""
        params = {"scale": 0.5, "upscale": "yes"}
        with pytest.raises(ValueError, match="Upscale must be a boolean"):
            operation.validate_params(params)

    def test_validate_params_invalid_method(self, operation):
        """Test validation fails with invalid method."""
        params = {"scale": 0.5, "method": "bicubic"}
        with pytest.raises(ValueError, match="GPU supports: nearest, bilinear"):
            operation.validate_params(params)

        params = {"scale": 0.5, "method": 123}
        with pytest.raises(ValueError, match="Method must be a string"):
            operation.validate_params(params)

    def test_warmup_compiles_kernels(self, operation):
        """Test that warmup triggers kernel compilation."""
        assert not operation.is_compiled

        operation.warmup()
        assert operation.is_compiled

        # Second warmup should be no-op
        operation.warmup()
        assert operation.is_compiled

    def test_reference_numpy_downscale_only(self, operation):
        """Test NumPy reference implementation with downscale only."""
        # Create test image with distinct patterns
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0:4, 0:4] = [1.0, 0.0, 0.0]  # Red quadrant
        image[0:4, 4:8] = [0.0, 1.0, 0.0]  # Green quadrant
        image[4:8, 0:4] = [0.0, 0.0, 1.0]  # Blue quadrant
        image[4:8, 4:8] = [1.0, 1.0, 0.0]  # Yellow quadrant

        params = {"scale": 0.5, "upscale": False, "method": "nearest"}
        result = operation.reference_numpy(image, params)

        # Should be 4x4
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_reference_numpy_with_upscale(self, operation):
        """Test NumPy reference implementation with upscale."""
        image = np.random.rand(8, 8, 3).astype(np.float32)

        params = {"scale": 0.5, "upscale": True, "method": "nearest"}
        result = operation.reference_numpy(image, params)

        # Should be back to original size
        assert result.shape == (8, 8, 3)
        assert result.dtype == np.float32

    def test_apply_to_field_downscale_only(self, operation):
        """Test GPU field operation with downscale only."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        # Create 8x8 test pattern
        height, width = 8, 8
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Initialize with gradient pattern
        for i in range(height):
            for j in range(width):
                r = i / (height - 1)
                g = j / (width - 1)
                source[0, i, j] = [r, g, 0.5, 1.0]

        # Downscale to 4x4
        out_height, out_width = 4, 4
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, out_height, out_width))

        params = {"scale": 0.5, "upscale": False, "method": "bilinear"}
        operation.apply_to_field(source, dest, {}, params, height, width)

        # Convert to numpy for verification
        dest_np = dest.to_numpy()[0]

        # Check dimensions
        assert dest_np.shape == (4, 4, 4)

        # Check reasonable values
        assert np.all(dest_np[..., :3] >= 0.0)
        assert np.all(dest_np[..., :3] <= 1.0)
        assert np.allclose(dest_np[..., 3], 1.0)  # Alpha preserved

        # Check gradient is preserved (first pixel should be darker than last)
        assert dest_np[0, 0, 0] < dest_np[3, 3, 0]  # Red channel increases
        assert dest_np[0, 0, 1] < dest_np[3, 3, 1]  # Green channel increases

    def test_apply_to_field_with_upscale_requires_temp(self, operation):
        """Test GPU field operation with upscale requires temp field."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 8, 8
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        for i in range(height):
            for j in range(width):
                source[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        params = {"scale": 0.5, "upscale": True, "method": "bilinear"}

        # Should fail without temp field
        with pytest.raises(RuntimeError, match="Missing temporary field"):
            operation.apply_to_field(source, dest, {}, params, height, width)

    def test_apply_to_field_with_upscale_success(self, operation):
        """Test GPU field operation with upscale and proper temp field."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 8, 8
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Create temp field for intermediate downscaled image
        temp_height, temp_width = 4, 4
        temp_field = ti.Vector.field(
            4, dtype=ti.f32, shape=(1, temp_height, temp_width)
        )
        temp_fields = {f"downscale_{temp_height}x{temp_width}": temp_field}

        # Initialize with checkerboard pattern
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 0:
                    source[0, i, j] = [1.0, 1.0, 1.0, 1.0]
                else:
                    source[0, i, j] = [0.0, 0.0, 0.0, 1.0]

        params = {"scale": 0.5, "upscale": True, "method": "nearest"}
        operation.apply_to_field(source, dest, temp_fields, params, height, width)

        # Convert to numpy
        dest_np = dest.to_numpy()[0]

        # Check dimensions match original
        assert dest_np.shape == (8, 8, 4)

        # Check values are in range
        assert np.all(dest_np[..., :3] >= 0.0)
        assert np.all(dest_np[..., :3] <= 1.0)

    def test_apply_to_field_nearest_vs_bilinear(self, operation):
        """Test that nearest and bilinear produce different results."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 16, 16
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Create gradient
        for i in range(height):
            for j in range(width):
                r = i / (height - 1)
                g = j / (width - 1)
                source[0, i, j] = [r, g, 0.5, 1.0]

        out_height, out_width = 8, 8
        dest_nearest = ti.Vector.field(
            4, dtype=ti.f32, shape=(1, out_height, out_width)
        )
        dest_bilinear = ti.Vector.field(
            4, dtype=ti.f32, shape=(1, out_height, out_width)
        )

        params_nearest = {"scale": 0.5, "upscale": False, "method": "nearest"}
        params_bilinear = {"scale": 0.5, "upscale": False, "method": "bilinear"}

        operation.apply_to_field(
            source, dest_nearest, {}, params_nearest, height, width
        )
        operation.apply_to_field(
            source, dest_bilinear, {}, params_bilinear, height, width
        )

        nearest_np = dest_nearest.to_numpy()[0]
        bilinear_np = dest_bilinear.to_numpy()[0]

        # Results should be different (bilinear is smoother)
        assert not np.allclose(nearest_np, bilinear_np, atol=0.01)

    def test_correctness_vs_reference_downscale(self, operation):
        """Test GPU result matches NumPy reference for downscale."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        # Create test image
        height, width = 16, 16
        image_np = np.random.rand(height, width, 3).astype(np.float32)

        # NumPy reference
        params = {"scale": 0.5, "upscale": False, "method": "bilinear"}
        reference = operation.reference_numpy(image_np, params)

        # GPU computation
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))
        out_height, out_width = 8, 8
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, out_height, out_width))

        # Load data (add alpha channel)
        for i in range(height):
            for j in range(width):
                r, g, b = image_np[i, j]
                source[0, i, j] = [r, g, b, 1.0]

        operation.apply_to_field(source, dest, {}, params, height, width)

        # Extract result
        result = dest.to_numpy()[0, :, :, :3]

        # Should match closely (allowing for minor numerical differences)
        assert result.shape == reference.shape
        assert np.allclose(result, reference, atol=0.02)

    def test_correctness_vs_reference_with_upscale(self, operation):
        """Test GPU result matches NumPy reference with upscale."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        # Create test image
        height, width = 16, 16
        image_np = np.random.rand(height, width, 3).astype(np.float32)

        # NumPy reference
        params = {"scale": 0.5, "upscale": True, "method": "bilinear"}
        reference = operation.reference_numpy(image_np, params)

        # GPU computation
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Create temp field
        temp_height, temp_width = 8, 8
        temp_field = ti.Vector.field(
            4, dtype=ti.f32, shape=(1, temp_height, temp_width)
        )
        temp_fields = {f"downscale_{temp_height}x{temp_width}": temp_field}

        # Load data
        for i in range(height):
            for j in range(width):
                r, g, b = image_np[i, j]
                source[0, i, j] = [r, g, b, 1.0]

        operation.apply_to_field(source, dest, temp_fields, params, height, width)

        # Extract result
        result = dest.to_numpy()[0, :, :, :3]

        # Should match closely
        assert result.shape == reference.shape
        assert np.allclose(result, reference, atol=0.02)

    def test_extreme_downscale_minimum_size(self, operation):
        """Test extreme downscale produces minimum 1x1 output."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 100, 100
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        for i in range(height):
            for j in range(width):
                source[0, i, j] = [0.8, 0.6, 0.4, 1.0]

        # Extreme downscale
        params = {"scale": 0.01, "upscale": False, "method": "bilinear"}
        out_height = max(1, int(height * 0.01))
        out_width = max(1, int(width * 0.01))

        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, out_height, out_width))
        operation.apply_to_field(source, dest, {}, params, height, width)

        # Should produce valid output
        result = dest.to_numpy()[0]
        assert result.shape == (out_height, out_width, 4)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_edge_case_scale_one(self, operation):
        """Test scale=1.0 produces unchanged output."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 8, 8
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Create test pattern
        source_data = np.random.rand(height, width, 4).astype(np.float32)
        source_data[..., 3] = 1.0  # Set alpha

        for i in range(height):
            for j in range(width):
                source[0, i, j] = source_data[i, j].tolist()

        params = {"scale": 1.0, "upscale": False, "method": "bilinear"}
        operation.apply_to_field(source, dest, {}, params, height, width)

        result = dest.to_numpy()[0]

        # Should be nearly identical (minor sampling differences allowed)
        assert np.allclose(result, source_data, atol=0.01)

    def test_alpha_channel_preserved(self, operation):
        """Test that alpha channel is preserved through operation."""
        if not TAICHI_AVAILABLE:
            pytest.skip("Taichi not available")

        ti.init(arch=ti.cpu)

        height, width = 8, 8
        source = ti.Vector.field(4, dtype=ti.f32, shape=(1, height, width))

        # Initialize with constant alpha
        for i in range(height):
            for j in range(width):
                source[0, i, j] = [0.5, 0.5, 0.5, 1.0]

        out_height, out_width = 4, 4
        dest = ti.Vector.field(4, dtype=ti.f32, shape=(1, out_height, out_width))

        params = {"scale": 0.5, "upscale": False, "method": "bilinear"}
        operation.apply_to_field(source, dest, {}, params, height, width)

        result = dest.to_numpy()[0]

        # Alpha should be preserved
        assert np.allclose(result[..., 3], 1.0, atol=0.01)

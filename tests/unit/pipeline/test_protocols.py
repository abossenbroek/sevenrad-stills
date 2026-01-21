"""Tests for pipeline protocols and dataclasses."""

import numpy as np
import pytest
from sevenrad_stills.pipeline.protocols import BufferPair, TempFieldSpec


class TestTempFieldSpec:
    """Tests for TempFieldSpec dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating TempFieldSpec with all fields specified."""
        spec = TempFieldSpec(
            name="temp_buffer",
            shape_factor=(1.0, 1.0, 3),
            dtype="f32",
        )
        assert spec.name == "temp_buffer"
        assert spec.shape_factor == (1.0, 1.0, 3)
        assert spec.dtype == "f32"

    def test_creation_with_default_dtype(self) -> None:
        """Test creating TempFieldSpec with default dtype."""
        spec = TempFieldSpec(
            name="temp_buffer",
            shape_factor=(2.0, 2.0, 4),
        )
        assert spec.name == "temp_buffer"
        assert spec.shape_factor == (2.0, 2.0, 4)
        assert spec.dtype == "f32"

    def test_custom_dtype(self) -> None:
        """Test creating TempFieldSpec with custom dtype."""
        spec = TempFieldSpec(
            name="int_buffer",
            shape_factor=(1.0, 1.0, 1),
            dtype="i32",
        )
        assert spec.dtype == "i32"

    def test_fractional_shape_factors(self) -> None:
        """Test TempFieldSpec with fractional shape factors for downscaling."""
        spec = TempFieldSpec(
            name="downscaled",
            shape_factor=(0.5, 0.5, 3),
        )
        assert spec.shape_factor == (0.5, 0.5, 3)

    def test_shape_factor_channels(self) -> None:
        """Test various channel counts in shape_factor."""
        spec_rgb = TempFieldSpec("rgb", (1.0, 1.0, 3))
        spec_rgba = TempFieldSpec("rgba", (1.0, 1.0, 4))
        spec_gray = TempFieldSpec("gray", (1.0, 1.0, 1))

        assert spec_rgb.shape_factor[2] == 3
        assert spec_rgba.shape_factor[2] == 4
        assert spec_gray.shape_factor[2] == 1


class TestBufferPair:
    """Tests for BufferPair dataclass."""

    def test_creation(self) -> None:
        """Test creating BufferPair with mock buffers."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)

        assert pair.current_is_a is True
        assert np.array_equal(pair.a, buffer_a)
        assert np.array_equal(pair.b, buffer_b)

    def test_source_property_initial(self) -> None:
        """Test source property returns 'a' initially."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)

        assert np.array_equal(pair.source, buffer_a)

    def test_dest_property_initial(self) -> None:
        """Test dest property returns 'b' initially."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)

        assert np.array_equal(pair.dest, buffer_b)

    def test_swap_once(self) -> None:
        """Test swapping buffers once."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)
        pair.swap()

        assert pair.current_is_a is False
        assert np.array_equal(pair.source, buffer_b)
        assert np.array_equal(pair.dest, buffer_a)

    def test_swap_twice(self) -> None:
        """Test swapping buffers twice returns to initial state."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)
        pair.swap()
        pair.swap()

        assert pair.current_is_a is True
        assert np.array_equal(pair.source, buffer_a)
        assert np.array_equal(pair.dest, buffer_b)

    def test_multiple_swaps(self) -> None:
        """Test multiple sequential swaps alternate correctly."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b)

        # After each swap, source and dest should alternate
        for i in range(10):
            if i % 2 == 0:
                assert pair.current_is_a is True
                assert np.array_equal(pair.source, buffer_a)
                assert np.array_equal(pair.dest, buffer_b)
            else:
                assert pair.current_is_a is False
                assert np.array_equal(pair.source, buffer_b)
                assert np.array_equal(pair.dest, buffer_a)
            pair.swap()

    def test_create_with_current_is_b(self) -> None:
        """Test creating BufferPair starting with 'b' as current."""
        buffer_a = np.zeros((10, 10, 3))
        buffer_b = np.ones((10, 10, 3))

        pair = BufferPair(a=buffer_a, b=buffer_b, current_is_a=False)

        assert pair.current_is_a is False
        assert np.array_equal(pair.source, buffer_b)
        assert np.array_equal(pair.dest, buffer_a)

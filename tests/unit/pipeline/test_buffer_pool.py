"""
Unit tests for PipelineBufferPool.

Tests buffer pool management with mocked Taichi fields to avoid GPU dependencies.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from sevenrad_stills.pipeline.buffer_pool import BufferPair, PipelineBufferPool


@pytest.fixture
def mock_taichi():
    """Mock Taichi module for testing without GPU."""
    with patch("sevenrad_stills.pipeline.buffer_pool.ti") as mock_ti:
        # Create mock Vector.field that returns a mock field object
        mock_field = MagicMock()
        mock_field.shape = (1, 64, 64)
        mock_ti.Vector.field.return_value = mock_field
        mock_ti.f32 = "f32"  # Mock dtype
        yield mock_ti


@pytest.fixture
def buffer_pool(mock_taichi):
    """Create a buffer pool with mocked Taichi."""
    return PipelineBufferPool()


class TestBufferPoolInitialization:
    """Test buffer pool initialization."""

    def test_init_creates_empty_pool(self, buffer_pool):
        """Test that initialization creates an empty pool."""
        assert len(buffer_pool.get_allocated_shapes()) == 0
        assert buffer_pool.current_shape is None

    def test_init_sets_default_state(self, buffer_pool):
        """Test that initialization sets correct default state."""
        assert buffer_pool._current_shape is None
        assert isinstance(buffer_pool._pools, dict)


class TestEnsureShape:
    """Test ensure_shape functionality."""

    def test_ensure_shape_creates_new_entry(self, buffer_pool, mock_taichi):
        """Test that ensure_shape creates new buffer pair."""
        buffer_pool.ensure_shape(1, 64, 64)

        # Verify Vector.field was called twice (buffer_a and buffer_b)
        assert mock_taichi.Vector.field.call_count == 2

        # Verify correct parameters
        call_args = mock_taichi.Vector.field.call_args_list[0]
        assert call_args[0][0] == 4  # 4 channels for RGBA
        assert call_args[1]["dtype"] == "f32"
        assert call_args[1]["shape"] == (1, 64, 64)

        # Verify shape was added to pool
        assert (1, 64, 64) in buffer_pool.get_allocated_shapes()
        assert buffer_pool.current_shape == (1, 64, 64)

    def test_ensure_shape_reuses_existing_entry(self, buffer_pool, mock_taichi):
        """Test that ensure_shape reuses existing buffer pair."""
        # First allocation
        buffer_pool.ensure_shape(1, 64, 64)
        initial_call_count = mock_taichi.Vector.field.call_count

        # Second call with same shape
        buffer_pool.ensure_shape(1, 64, 64)

        # Should not create new fields
        assert mock_taichi.Vector.field.call_count == initial_call_count

        # Should still have only one shape
        assert len(buffer_pool.get_allocated_shapes()) == 1

    def test_ensure_shape_handles_multiple_shapes(self, buffer_pool, mock_taichi):
        """Test that ensure_shape handles multiple different shapes."""
        shapes = [(1, 64, 64), (1, 128, 128), (2, 64, 64)]

        for shape in shapes:
            buffer_pool.ensure_shape(*shape)

        # Verify all shapes are allocated
        allocated = buffer_pool.get_allocated_shapes()
        assert len(allocated) == 3
        for shape in shapes:
            assert shape in allocated

    def test_ensure_shape_updates_current_shape(self, buffer_pool):
        """Test that ensure_shape updates current_shape property."""
        buffer_pool.ensure_shape(1, 64, 64)
        assert buffer_pool.current_shape == (1, 64, 64)

        buffer_pool.ensure_shape(1, 128, 128)
        assert buffer_pool.current_shape == (1, 128, 128)

    def test_ensure_shape_without_taichi(self):
        """Test that ensure_shape raises error when Taichi is not available."""
        with patch("sevenrad_stills.pipeline.buffer_pool.ti", None):
            pool = PipelineBufferPool()
            with pytest.raises(RuntimeError, match="Taichi is not available"):
                pool.ensure_shape(1, 64, 64)


class TestGetPair:
    """Test get_pair functionality."""

    def test_get_pair_returns_correct_buffer_pair(self, buffer_pool):
        """Test that get_pair returns correct BufferPair."""
        buffer_pool.ensure_shape(1, 64, 64)
        pair = buffer_pool.get_pair(1, 64, 64)

        assert isinstance(pair, BufferPair)
        assert pair.a is not None
        assert pair.b is not None

    def test_get_pair_raises_error_for_unallocated_shape(self, buffer_pool):
        """Test that get_pair raises KeyError for unallocated shape."""
        with pytest.raises(KeyError, match="No buffer allocated"):
            buffer_pool.get_pair(1, 64, 64)

    def test_get_pair_returns_same_buffers_on_multiple_calls(self, buffer_pool):
        """Test that get_pair returns same buffers for same shape."""
        buffer_pool.ensure_shape(1, 64, 64)

        pair1 = buffer_pool.get_pair(1, 64, 64)
        pair2 = buffer_pool.get_pair(1, 64, 64)

        # Should return same buffer objects
        assert pair1.a is pair2.a
        assert pair1.b is pair2.b


class TestLoadImage:
    """Test load_image functionality."""

    def test_load_image_converts_rgb_to_rgba(self, buffer_pool):
        """Test that load_image adds alpha channel to RGB images."""
        buffer_pool.ensure_shape(1, 2, 2)
        pair = buffer_pool.get_pair(1, 2, 2)

        # Create mock buffer with setitem tracking - MagicMock supports item assignment
        mock_buffer = MagicMock()
        rgb_image = np.ones((2, 2, 3), dtype=np.uint8) * 128

        buffer_pool.load_image(rgb_image, mock_buffer, batch_idx=0)

        # Verify buffer was written to (4 pixels = 4 writes)
        assert mock_buffer.__setitem__.call_count == 4

    def test_load_image_normalizes_uint8(self, buffer_pool):
        """Test that load_image normalizes uint8 to float [0, 1]."""
        buffer_pool.ensure_shape(1, 2, 2)

        # Use MagicMock to support item assignment
        mock_buffer = MagicMock()
        # Create a proper 2x2x4 RGBA image
        rgba_image = np.array(
            [
                [[255, 0, 128, 255], [0, 255, 64, 255]],
                [[128, 128, 128, 255], [0, 0, 0, 255]],
            ],
            dtype=np.uint8,
        )

        buffer_pool.load_image(rgba_image, mock_buffer, batch_idx=0)

        # Check that normalized values were written
        # First pixel should be [1.0, 0.0, ~0.5, 1.0]
        call_args = mock_buffer.__setitem__.call_args_list[0]
        written_value = call_args[0][1]
        # Should be normalized float array
        assert isinstance(written_value, np.ndarray)
        assert written_value.dtype == np.float32

    def test_load_image_validates_dimensions(self, buffer_pool):
        """Test that load_image validates image dimensions."""
        buffer_pool.ensure_shape(1, 2, 2)
        mock_buffer = Mock()

        # 2D array (missing channel dimension)
        invalid_image = np.ones((2, 2), dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected 3D array"):
            buffer_pool.load_image(invalid_image, mock_buffer)

    def test_load_image_validates_channels(self, buffer_pool):
        """Test that load_image validates channel count."""
        buffer_pool.ensure_shape(1, 2, 2)
        mock_buffer = Mock()

        # 5-channel image (invalid)
        invalid_image = np.ones((2, 2, 5), dtype=np.uint8)

        with pytest.raises(ValueError, match="must have 3 or 4 channels"):
            buffer_pool.load_image(invalid_image, mock_buffer)

    def test_load_image_without_taichi(self, buffer_pool):
        """Test that load_image raises error when Taichi is not available."""
        with patch("sevenrad_stills.pipeline.buffer_pool.ti", None):
            mock_buffer = Mock()
            image = np.ones((2, 2, 3), dtype=np.uint8)

            with pytest.raises(RuntimeError, match="Taichi is not available"):
                buffer_pool.load_image(image, mock_buffer)


class TestExtractResult:
    """Test extract_result functionality."""

    def test_extract_result_converts_to_uint8(self, buffer_pool):
        """Test that extract_result converts float to uint8."""
        buffer_pool.ensure_shape(1, 2, 2)

        # Use MagicMock to support __getitem__
        mock_buffer = MagicMock()
        mock_buffer.shape = (1, 2, 2)

        # Mock individual field access
        def getitem_side_effect(key):
            mock_pixel = MagicMock()
            mock_pixel.to_numpy.return_value = np.array(
                [0.5, 0.5, 0.5, 1.0], dtype=np.float32
            )
            return mock_pixel

        mock_buffer.__getitem__.side_effect = getitem_side_effect

        result = buffer_pool.extract_result(mock_buffer, batch_idx=0)

        # Verify result shape and dtype
        assert result.shape == (2, 2, 3)  # RGB output
        assert result.dtype == np.uint8

    def test_extract_result_clips_values(self, buffer_pool):
        """Test that extract_result clips values to [0, 255]."""
        buffer_pool.ensure_shape(1, 2, 2)

        # Use MagicMock to support __getitem__
        mock_buffer = MagicMock()
        mock_buffer.shape = (1, 2, 2)

        # Mock pixels with out-of-range values
        def getitem_side_effect(key):
            mock_pixel = MagicMock()
            # Values outside [0, 1] range
            mock_pixel.to_numpy.return_value = np.array(
                [1.5, -0.5, 0.5, 1.0], dtype=np.float32
            )
            return mock_pixel

        mock_buffer.__getitem__.side_effect = getitem_side_effect

        result = buffer_pool.extract_result(mock_buffer, batch_idx=0)

        # All values should be in valid uint8 range
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_extract_result_discards_alpha(self, buffer_pool):
        """Test that extract_result returns only RGB channels."""
        buffer_pool.ensure_shape(1, 2, 2)

        # Use MagicMock to support __getitem__
        mock_buffer = MagicMock()
        mock_buffer.shape = (1, 2, 2)

        def getitem_side_effect(key):
            mock_pixel = MagicMock()
            mock_pixel.to_numpy.return_value = np.array(
                [0.5, 0.5, 0.5, 0.0], dtype=np.float32
            )  # Alpha = 0
            return mock_pixel

        mock_buffer.__getitem__.side_effect = getitem_side_effect

        result = buffer_pool.extract_result(mock_buffer, batch_idx=0)

        # Should only have 3 channels (RGB)
        assert result.shape[-1] == 3

    def test_extract_result_without_taichi(self, buffer_pool):
        """Test that extract_result raises error when Taichi is not available."""
        with patch("sevenrad_stills.pipeline.buffer_pool.ti", None):
            mock_buffer = Mock()

            with pytest.raises(RuntimeError, match="Taichi is not available"):
                buffer_pool.extract_result(mock_buffer)


class TestRelease:
    """Test release functionality."""

    def test_release_clears_all_buffers(self, buffer_pool):
        """Test that release clears all allocated buffers."""
        # Allocate multiple shapes
        buffer_pool.ensure_shape(1, 64, 64)
        buffer_pool.ensure_shape(1, 128, 128)
        assert len(buffer_pool.get_allocated_shapes()) == 2

        # Release all
        buffer_pool.release()

        assert len(buffer_pool.get_allocated_shapes()) == 0
        assert buffer_pool.current_shape is None

    def test_release_can_be_called_multiple_times(self, buffer_pool):
        """Test that release is idempotent."""
        buffer_pool.ensure_shape(1, 64, 64)
        buffer_pool.release()
        buffer_pool.release()  # Should not raise error

        assert len(buffer_pool.get_allocated_shapes()) == 0

    def test_release_allows_reallocation(self, buffer_pool):
        """Test that buffers can be reallocated after release."""
        buffer_pool.ensure_shape(1, 64, 64)
        buffer_pool.release()

        # Should be able to allocate again
        buffer_pool.ensure_shape(1, 64, 64)
        assert (1, 64, 64) in buffer_pool.get_allocated_shapes()


class TestGetAllocatedShapes:
    """Test get_allocated_shapes functionality."""

    def test_get_allocated_shapes_returns_empty_list_initially(self, buffer_pool):
        """Test that get_allocated_shapes returns empty list initially."""
        assert buffer_pool.get_allocated_shapes() == []

    def test_get_allocated_shapes_returns_all_shapes(self, buffer_pool):
        """Test that get_allocated_shapes returns all allocated shapes."""
        shapes = [(1, 64, 64), (1, 128, 128), (2, 64, 64)]

        for shape in shapes:
            buffer_pool.ensure_shape(*shape)

        allocated = buffer_pool.get_allocated_shapes()
        assert len(allocated) == 3
        for shape in shapes:
            assert shape in allocated


class TestCurrentShape:
    """Test current_shape property."""

    def test_current_shape_is_none_initially(self, buffer_pool):
        """Test that current_shape is None initially."""
        assert buffer_pool.current_shape is None

    def test_current_shape_tracks_most_recent(self, buffer_pool):
        """Test that current_shape tracks most recently accessed shape."""
        buffer_pool.ensure_shape(1, 64, 64)
        assert buffer_pool.current_shape == (1, 64, 64)

        buffer_pool.ensure_shape(1, 128, 128)
        assert buffer_pool.current_shape == (1, 128, 128)

    def test_current_shape_resets_on_release(self, buffer_pool):
        """Test that current_shape is reset on release."""
        buffer_pool.ensure_shape(1, 64, 64)
        assert buffer_pool.current_shape is not None

        buffer_pool.release()
        assert buffer_pool.current_shape is None


class TestBufferPairDataclass:
    """Test BufferPair dataclass."""

    def test_buffer_pair_creation(self):
        """Test that BufferPair can be created."""
        mock_a = Mock()
        mock_b = Mock()

        pair = BufferPair(a=mock_a, b=mock_b)

        assert pair.a is mock_a
        assert pair.b is mock_b

    def test_buffer_pair_is_dataclass(self):
        """Test that BufferPair is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(BufferPair)

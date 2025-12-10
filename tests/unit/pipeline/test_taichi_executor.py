"""
Unit tests for TaichiPipelineExecutor.

Tests executor functionality with mocked Taichi and buffer pool to avoid GPU
dependencies.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image
from sevenrad_stills.pipeline.models import ImageOperationStep
from sevenrad_stills.pipeline.taichi_executor import (
    GPUOperationError,
    TaichiArch,
    TaichiContext,
    TaichiPipelineExecutor,
)
from sevenrad_stills.utils.exceptions import PipelineError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset TaichiContext singleton between tests."""
    # Reset before test
    TaichiContext._instance = None
    TaichiContext._initialized = False
    yield
    # Reset after test
    TaichiContext._instance = None
    TaichiContext._initialized = False


@pytest.fixture
def mock_taichi():
    """Mock Taichi module for testing without GPU."""
    with patch("sevenrad_stills.pipeline.taichi_executor.ti") as mock_ti:
        # Mock architecture constants
        mock_ti.cpu = "cpu"
        mock_ti.cuda = "cuda"
        mock_ti.metal = "metal"
        mock_ti.vulkan = "vulkan"
        mock_ti.opengl = "opengl"
        mock_ti.f32 = "f32"

        # Mock Vector.field
        mock_field = MagicMock()
        mock_field.shape = (1, 64, 64)
        mock_ti.Vector.field.return_value = mock_field

        # Mock init and reset
        mock_ti.init = Mock()
        mock_ti.reset = Mock()
        mock_ti.sync = Mock()

        yield mock_ti


@pytest.fixture
def mock_operation_registry():
    """Mock operation registry for testing without real operations."""
    with patch(
        "sevenrad_stills.pipeline.taichi_executor.has_taichi_operation"
    ) as mock_has:
        with patch(
            "sevenrad_stills.pipeline.taichi_executor.get_taichi_operation"
        ) as mock_get:
            mock_has.return_value = True
            mock_op = MagicMock()
            mock_op.warmup = Mock()
            mock_op.validate_params = Mock()
            mock_op.apply_to_field = Mock()
            mock_get.return_value = mock_op
            yield mock_has, mock_get, mock_op


@pytest.fixture
def mock_buffer_pool():
    """Mock PipelineBufferPool for testing."""
    mock_pool = MagicMock()
    mock_pool.ensure_shape = Mock()
    mock_pool.get_pair = Mock()
    mock_pool.release = Mock()
    mock_pool.load_image = Mock()
    mock_pool.extract_result = Mock(return_value=np.zeros((64, 64, 3), dtype=np.uint8))
    mock_pool.get_allocated_shapes = Mock(return_value=[])

    # Make get_pair return a mock BufferPair
    mock_pair = MagicMock()
    mock_pair.a = MagicMock()
    mock_pair.b = MagicMock()
    mock_pool.get_pair.return_value = mock_pair

    # Patch both the class (for initialization) and the property access path
    with patch(
        "sevenrad_stills.pipeline.taichi_executor.PipelineBufferPool",
        return_value=mock_pool,
    ):
        yield mock_pool


@pytest.fixture
def sample_steps():
    """Create sample pipeline steps for testing."""
    return [
        ImageOperationStep(
            name="step1",
            operation="saturation",
            params={"factor": 1.5},
            repeat=1,
        ),
        ImageOperationStep(
            name="step2",
            operation="blur",
            params={"sigma": 2.0},
            repeat=1,
        ),
    ]


@pytest.fixture
def sample_image():
    """Create sample PIL Image for testing."""
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


class TestTaichiArch:
    """Test TaichiArch enum."""

    def test_arch_values(self):
        """Test that TaichiArch has correct values."""
        assert TaichiArch.CPU.value == "cpu"
        assert TaichiArch.CUDA.value == "cuda"
        assert TaichiArch.METAL.value == "metal"
        assert TaichiArch.VULKAN.value == "vulkan"
        assert TaichiArch.OPENGL.value == "opengl"


class TestTaichiContext:
    """Test TaichiContext singleton."""

    def test_singleton_pattern(self, mock_taichi):
        """Test that TaichiContext is a singleton."""
        context1 = TaichiContext()
        context2 = TaichiContext()

        assert context1 is context2

    def test_startup_initializes_taichi(self, mock_taichi):
        """Test that startup initializes Taichi runtime."""
        context = TaichiContext()
        context._initialized = False  # Reset for test

        context.startup(TaichiArch.METAL)

        mock_taichi.init.assert_called_once()
        call_kwargs = mock_taichi.init.call_args[1]
        assert call_kwargs["arch"] == "metal"
        assert call_kwargs["default_fp"] == "f32"

    def test_startup_is_idempotent(self, mock_taichi):
        """Test that startup can be called multiple times safely."""
        context = TaichiContext()
        context._initialized = False

        context.startup(TaichiArch.METAL)
        initial_call_count = mock_taichi.init.call_count

        context.startup(TaichiArch.METAL)

        # Should not call init again
        assert mock_taichi.init.call_count == initial_call_count

    def test_startup_without_taichi(self):
        """Test that startup raises error when Taichi is not available."""
        with patch("sevenrad_stills.pipeline.taichi_executor.ti", None):
            context = TaichiContext()
            context._initialized = False

            with pytest.raises(RuntimeError, match="Taichi is not available"):
                context.startup()

    def test_shutdown_releases_resources(self, mock_taichi, mock_buffer_pool):
        """Test that shutdown releases resources and resets Taichi."""
        context = TaichiContext()
        context._initialized = True

        context.shutdown()

        # Should reset Taichi
        mock_taichi.reset.assert_called_once()

    def test_shutdown_is_idempotent(self, mock_taichi):
        """Test that shutdown can be called multiple times safely."""
        context = TaichiContext()
        context._initialized = True

        context.shutdown()
        initial_call_count = mock_taichi.reset.call_count

        context.shutdown()

        # Should not call reset again
        assert mock_taichi.reset.call_count == initial_call_count

    def test_is_initialized_property(self, mock_taichi):
        """Test is_initialized property."""
        context = TaichiContext()
        context._initialized = False

        assert not context.is_initialized

        context.startup(TaichiArch.METAL)
        assert context.is_initialized

        context.shutdown()
        assert not context.is_initialized


class TestTaichiPipelineExecutorInit:
    """Test TaichiPipelineExecutor initialization."""

    def test_init_creates_executor(self, mock_taichi, mock_buffer_pool):
        """Test that executor is initialized correctly."""
        executor = TaichiPipelineExecutor(arch=TaichiArch.METAL, debug=True)

        assert executor._arch == TaichiArch.METAL
        assert executor._debug is True
        assert not executor._prepared

    def test_init_initializes_context(self, mock_taichi, mock_buffer_pool):
        """Test that executor initializes Taichi context."""
        with patch.object(TaichiContext, "_initialized", False):
            executor = TaichiPipelineExecutor(arch=TaichiArch.METAL)

            # Should have called startup
            mock_taichi.init.assert_called()

    def test_init_default_architecture(self, mock_taichi, mock_buffer_pool):
        """Test that default architecture is Metal."""
        executor = TaichiPipelineExecutor()

        assert executor._arch == TaichiArch.METAL


class TestPrepare:
    """Test prepare method."""

    def test_prepare_analyzes_and_allocates_buffers(
        self, mock_taichi, mock_buffer_pool, sample_steps
    ):
        """Test that prepare analyzes pipeline and allocates buffers."""
        executor = TaichiPipelineExecutor()

        executor.prepare(sample_steps, height=1080, width=1920, batch_size=1)

        # Should call ensure_shape for required dimensions
        assert mock_buffer_pool.ensure_shape.called
        assert executor._prepared

    def test_prepare_is_idempotent(self, mock_taichi, mock_buffer_pool, sample_steps):
        """Test that prepare can be called multiple times safely."""
        executor = TaichiPipelineExecutor()

        executor.prepare(sample_steps, height=1080, width=1920)
        initial_call_count = mock_buffer_pool.ensure_shape.call_count

        executor.prepare(sample_steps, height=1080, width=1920)

        # Should not allocate again
        assert mock_buffer_pool.ensure_shape.call_count == initial_call_count

    def test_prepare_handles_errors(self, mock_taichi, mock_buffer_pool, sample_steps):
        """Test that prepare raises PipelineError on failure."""
        mock_buffer_pool.ensure_shape.side_effect = Exception("Allocation failed")

        executor = TaichiPipelineExecutor()

        with pytest.raises(PipelineError, match="Failed to prepare pipeline"):
            executor.prepare(sample_steps, height=1080, width=1920)


class TestAnalyzeShapes:
    """Test _analyze_shapes method."""

    def test_analyze_shapes_returns_initial_shape(
        self, mock_taichi, mock_buffer_pool, sample_steps
    ):
        """Test that _analyze_shapes includes initial shape."""
        executor = TaichiPipelineExecutor()

        shapes = executor._analyze_shapes(sample_steps, 1, 1080, 1920)

        # Should at minimum include initial shape
        assert (1, 1080, 1920) in shapes
        assert isinstance(shapes, list)

    def test_analyze_shapes_returns_list(
        self, mock_taichi, mock_buffer_pool, sample_steps
    ):
        """Test that _analyze_shapes returns list of tuples."""
        executor = TaichiPipelineExecutor()

        shapes = executor._analyze_shapes(sample_steps, 1, 1080, 1920)

        assert isinstance(shapes, list)
        for shape in shapes:
            assert isinstance(shape, tuple)
            assert len(shape) == 3  # (batch, height, width)


class TestWarmup:
    """Test warmup method."""

    def test_warmup_calls_operation_warmup(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that warmup calls operation.warmup() for each Taichi operation."""
        mock_has, mock_get, mock_op = mock_operation_registry

        executor = TaichiPipelineExecutor()
        steps = [
            ImageOperationStep(
                name="test", operation="saturation", params={"factor": 1.0}, repeat=1
            )
        ]

        executor.warmup(steps)

        # Should call operation.warmup()
        mock_op.warmup.assert_called()

    def test_warmup_handles_no_taichi(self, mock_buffer_pool):
        """Test that warmup handles missing Taichi gracefully."""
        with patch("sevenrad_stills.pipeline.taichi_executor.ti", None):
            with patch(
                "sevenrad_stills.pipeline.taichi_executor.TAICHI_AVAILABLE", False
            ):
                # Need to also patch the context to not require Taichi for init
                with patch.object(TaichiContext, "_initialized", True):
                    with patch.object(TaichiContext, "startup"):
                        executor = TaichiPipelineExecutor()
                        steps = []

                        # Should not raise error
                        executor.warmup(steps)

    def test_warmup_handles_errors(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that warmup raises GPUOperationError on failure."""
        mock_has, mock_get, mock_op = mock_operation_registry
        mock_op.warmup.side_effect = Exception("Compilation failed")

        executor = TaichiPipelineExecutor()
        steps = [
            ImageOperationStep(
                name="test", operation="saturation", params={"factor": 1.0}, repeat=1
            )
        ]

        with pytest.raises(GPUOperationError, match="Pipeline warmup failed"):
            executor.warmup(steps)


class TestProcessImage:
    """Test process_image method."""

    def test_process_image_auto_prepares(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry, sample_image
    ):
        """Test that process_image auto-prepares if not prepared."""
        # Create steps with only saturation (has Taichi implementation)
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()
        assert not executor._prepared

        executor.process_image(sample_image, steps)

        # Should have prepared
        assert executor._prepared

    def test_process_image_loads_and_extracts(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry, sample_image
    ):
        """Test that process_image loads to GPU and extracts result."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        result = executor.process_image(sample_image, steps)

        # Should call load_image and extract_result
        assert mock_buffer_pool.load_image.called
        assert mock_buffer_pool.extract_result.called

        # Should return PIL Image
        assert isinstance(result, Image.Image)

    def test_process_image_handles_errors(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry, sample_image
    ):
        """Test that process_image raises GPUOperationError on failure."""
        mock_buffer_pool.load_image.side_effect = Exception("GPU transfer failed")

        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        with pytest.raises(GPUOperationError, match="Image processing failed"):
            executor.process_image(sample_image, steps)

    def test_process_image_uses_debug_sync(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry, sample_image
    ):
        """Test that debug mode calls ti.sync() after operations."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor(debug=True)

        executor.process_image(sample_image, steps)

        # Debug mode should call sync
        mock_taichi.sync.assert_called()
        assert executor._debug is True


class TestProcessBatch:
    """Test process_batch method."""

    def test_process_batch_validates_dimensions(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that process_batch validates all images have same dimensions."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        images = [
            Image.new("RGB", (64, 64)),
            Image.new("RGB", (128, 128)),  # Different size
        ]

        with pytest.raises(ValueError, match="same dimensions"):
            executor.process_batch(images, steps)

    def test_process_batch_handles_empty_list(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that process_batch handles empty image list."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        result = executor.process_batch([], steps)

        assert result == []

    def test_process_batch_processes_all_images(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that process_batch processes all images."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        images = [
            Image.new("RGB", (64, 64)),
            Image.new("RGB", (64, 64)),
            Image.new("RGB", (64, 64)),
        ]

        results = executor.process_batch(images, steps)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, Image.Image)

    def test_process_batch_auto_prepares(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that process_batch auto-prepares with correct batch size."""
        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()

        images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))]

        executor.process_batch(images, steps)

        # Should have prepared with batch size = 2
        assert executor._prepared

    def test_process_batch_handles_errors(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that process_batch raises GPUOperationError on failure."""
        mock_buffer_pool.load_image.side_effect = Exception("Batch processing failed")

        steps = [
            ImageOperationStep(
                name="step1",
                operation="saturation",
                params={"factor": 1.5},
                repeat=1,
            )
        ]

        executor = TaichiPipelineExecutor()
        images = [Image.new("RGB", (64, 64))]

        with pytest.raises(GPUOperationError, match="Batch processing failed"):
            executor.process_batch(images, steps)


class TestExecuteStep:
    """Test _execute_step method."""

    def test_execute_step_calls_operation(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that _execute_step calls the Taichi operation."""
        mock_has, mock_get, mock_op = mock_operation_registry

        executor = TaichiPipelineExecutor()
        step = ImageOperationStep(
            name="test_step",
            operation="saturation",
            params={"factor": 1.5},
            repeat=1,
        )

        source = MagicMock()
        dest = MagicMock()

        executor._execute_step(step, source, dest, height=64, width=64)

        # Should validate params
        mock_op.validate_params.assert_called_once_with({"factor": 1.5})

        # Should call apply_to_field
        mock_op.apply_to_field.assert_called_once()
        call_kwargs = mock_op.apply_to_field.call_args[1]
        assert call_kwargs["source"] is source
        assert call_kwargs["dest"] is dest
        assert call_kwargs["params"] == {"factor": 1.5}
        assert call_kwargs["height"] == 64
        assert call_kwargs["width"] == 64

    def test_execute_step_raises_for_missing_operation(
        self, mock_taichi, mock_buffer_pool
    ):
        """Test that _execute_step raises GPUOperationError for missing operation."""
        with patch(
            "sevenrad_stills.pipeline.taichi_executor.has_taichi_operation"
        ) as mock_has:
            mock_has.return_value = False

            executor = TaichiPipelineExecutor()
            step = ImageOperationStep(
                name="test_step",
                operation="nonexistent_op",
                params={},
                repeat=1,
            )

            with pytest.raises(GPUOperationError, match="has no Taichi implementation"):
                executor._execute_step(step, Mock(), Mock(), height=64, width=64)

    def test_execute_step_handles_repeat(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that _execute_step handles repeat parameter correctly."""
        mock_has, mock_get, mock_op = mock_operation_registry

        executor = TaichiPipelineExecutor()
        step = ImageOperationStep(
            name="test_step",
            operation="saturation",
            params={"factor": 1.5},
            repeat=3,  # Repeat 3 times
        )

        source = MagicMock()
        dest = MagicMock()

        executor._execute_step(step, source, dest, height=64, width=64)

        # Should call apply_to_field 3 times
        assert mock_op.apply_to_field.call_count == 3

    def test_execute_step_validates_params_before_execution(
        self, mock_taichi, mock_buffer_pool, mock_operation_registry
    ):
        """Test that _execute_step validates params before executing."""
        mock_has, mock_get, mock_op = mock_operation_registry
        mock_op.validate_params.side_effect = ValueError("Invalid factor")

        executor = TaichiPipelineExecutor()
        step = ImageOperationStep(
            name="test_step",
            operation="saturation",
            params={"factor": -1},  # Invalid param
            repeat=1,
        )

        with pytest.raises(ValueError, match="Invalid factor"):
            executor._execute_step(step, Mock(), Mock(), height=64, width=64)

        # Should validate but not execute
        mock_op.validate_params.assert_called_once()
        mock_op.apply_to_field.assert_not_called()


class TestCoalesceOperations:
    """Test _coalesce_operations method."""

    def test_coalesce_operations_returns_list(
        self, mock_taichi, mock_buffer_pool, sample_steps
    ):
        """Test that _coalesce_operations returns list of operation groups."""
        executor = TaichiPipelineExecutor()

        groups = executor._coalesce_operations(sample_steps)

        assert isinstance(groups, list)
        assert len(groups) > 0

    def test_coalesce_operations_placeholder_returns_single_group(
        self, mock_taichi, mock_buffer_pool, sample_steps
    ):
        """Test that placeholder implementation returns single group."""
        executor = TaichiPipelineExecutor()

        groups = executor._coalesce_operations(sample_steps)

        # Current placeholder returns all steps as single group
        assert len(groups) == 1
        assert groups[0] == sample_steps


class TestCleanup:
    """Test cleanup method."""

    def test_cleanup_resets_prepared_state(self, mock_taichi, mock_buffer_pool):
        """Test that cleanup resets prepared state."""
        executor = TaichiPipelineExecutor()
        executor._prepared = True

        executor.cleanup()

        assert not executor._prepared

    def test_cleanup_is_idempotent(self, mock_taichi, mock_buffer_pool):
        """Test that cleanup can be called multiple times safely."""
        executor = TaichiPipelineExecutor()
        executor._prepared = True

        executor.cleanup()
        executor.cleanup()  # Should not raise error

        assert not executor._prepared


class TestContextManager:
    """Test context manager support."""

    def test_context_manager_enter(self, mock_taichi, mock_buffer_pool):
        """Test that executor can be used as context manager."""
        with TaichiPipelineExecutor() as executor:
            assert isinstance(executor, TaichiPipelineExecutor)

    def test_context_manager_exit_calls_cleanup(self, mock_taichi, mock_buffer_pool):
        """Test that context manager exit calls cleanup."""
        with TaichiPipelineExecutor() as executor:
            executor._prepared = True

        # After exiting context, should be cleaned up
        assert not executor._prepared


class TestGPUOperationError:
    """Test GPUOperationError exception."""

    def test_gpu_operation_error_is_pipeline_error(self):
        """Test that GPUOperationError is a PipelineError."""
        error = GPUOperationError("Test error")

        assert isinstance(error, PipelineError)

    def test_gpu_operation_error_message(self):
        """Test that GPUOperationError carries message."""
        error = GPUOperationError("Test error message")

        assert str(error) == "Test error message"


class TestBufferPoolProperty:
    """Test buffer_pool property."""

    def test_buffer_pool_property_returns_pool(self, mock_taichi, mock_buffer_pool):
        """Test that buffer_pool property returns the buffer pool."""
        executor = TaichiPipelineExecutor()

        pool = executor.buffer_pool

        # Should return a buffer pool instance (the mock we injected)
        assert pool is mock_buffer_pool

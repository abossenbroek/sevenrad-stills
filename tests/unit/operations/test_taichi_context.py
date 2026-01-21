"""Tests for Taichi context lifecycle management."""

from unittest.mock import patch

import pytest
from sevenrad_stills.operations.taichi_context import TaichiArch, TaichiContext
from sevenrad_stills.utils.exceptions import TaichiInitializationError


class TestTaichiContext:
    """Tests for TaichiContext singleton and lifecycle management."""

    @pytest.fixture(autouse=True)
    def _reset_context(self) -> None:
        """Reset TaichiContext singleton state before each test."""
        # Reset singleton state
        TaichiContext._instance = None
        TaichiContext._initialized = False
        TaichiContext._current_arch = None

    def test_singleton_returns_same_instance(self) -> None:
        """Test that TaichiContext returns the same instance (singleton pattern)."""
        context1 = TaichiContext()
        context2 = TaichiContext()
        assert context1 is context2

    @pytest.mark.gpu
    def test_startup_initializes_taichi(self) -> None:
        """Test that startup() initializes Taichi successfully."""
        context = TaichiContext()
        assert not context.is_initialized
        assert context.current_arch is None

        context.startup(TaichiArch.CPU)

        assert context.is_initialized
        assert context.current_arch == TaichiArch.CPU

        # Clean up
        context.shutdown()

    @pytest.mark.gpu
    def test_shutdown_resets_state(self) -> None:
        """Test that shutdown() resets Taichi and context state."""
        context = TaichiContext()
        context.startup(TaichiArch.CPU)

        assert context.is_initialized
        assert context.current_arch == TaichiArch.CPU

        context.shutdown()

        assert not context.is_initialized
        assert context.current_arch is None

    @pytest.mark.gpu
    def test_double_startup_is_idempotent(self) -> None:
        """Test that calling startup() twice with same arch is safe (idempotent)."""
        context = TaichiContext()
        context.startup(TaichiArch.CPU)

        assert context.is_initialized
        assert context.current_arch == TaichiArch.CPU

        # Second startup with same arch should be safe
        context.startup(TaichiArch.CPU)

        assert context.is_initialized
        assert context.current_arch == TaichiArch.CPU

        # Clean up
        context.shutdown()

    @pytest.mark.gpu
    def test_double_shutdown_is_safe(self) -> None:
        """Test that calling shutdown() multiple times is safe."""
        context = TaichiContext()
        context.startup(TaichiArch.CPU)
        context.shutdown()

        assert not context.is_initialized

        # Second shutdown should be safe
        context.shutdown()

        assert not context.is_initialized

    def test_startup_with_different_arch_does_not_reinitialize(self) -> None:
        """Test startup with different arch when initialized doesn't change state."""
        with patch("taichi.init") as mock_ti_init, patch("taichi.cpu"):
            context = TaichiContext()
            context.startup(TaichiArch.CPU)

            assert context.current_arch == TaichiArch.CPU
            assert mock_ti_init.call_count == 1

            # Try to startup with different arch
            context.startup(TaichiArch.GPU)

            # Should still be CPU, not reinitialized
            assert context.current_arch == TaichiArch.CPU
            assert mock_ti_init.call_count == 1  # Not called again

            # Clean up
            context.shutdown()

    def test_initialization_failure_raises_error(self) -> None:
        """Test that Taichi initialization failure raises TaichiInitializationError."""
        with patch("taichi.init", side_effect=RuntimeError("GPU not available")):
            context = TaichiContext()

            with pytest.raises(
                TaichiInitializationError, match="Failed to initialize Taichi"
            ):
                context.startup(TaichiArch.GPU)

            # Context should remain uninitialized
            assert not context.is_initialized

    @pytest.mark.gpu
    def test_startup_with_metal_arch(self) -> None:
        """Test startup with METAL architecture."""
        context = TaichiContext()

        try:
            # May fail on non-macOS systems
            context.startup(TaichiArch.METAL)
            assert context.is_initialized
            assert context.current_arch == TaichiArch.METAL
        except TaichiInitializationError:
            # Expected on non-macOS or systems without Metal support
            pytest.skip("Metal not available on this system")
        finally:
            context.shutdown()

    @pytest.mark.gpu
    def test_startup_with_gpu_arch(self) -> None:
        """Test startup with GPU architecture."""
        context = TaichiContext()

        try:
            context.startup(TaichiArch.GPU)
            assert context.is_initialized
            assert context.current_arch == TaichiArch.GPU
        except TaichiInitializationError:
            # Expected on systems without CUDA/Vulkan
            pytest.skip("GPU not available on this system")
        finally:
            context.shutdown()

    def test_is_initialized_property(self) -> None:
        """Test is_initialized property reflects correct state."""
        with patch("taichi.init"), patch("taichi.cpu"), patch("taichi.reset"):
            context = TaichiContext()

            assert not context.is_initialized

            context.startup(TaichiArch.CPU)
            assert context.is_initialized

            context.shutdown()
            assert not context.is_initialized

    def test_current_arch_property(self) -> None:
        """Test current_arch property returns correct architecture."""
        with patch("taichi.init"), patch("taichi.cpu"), patch("taichi.reset"):
            context = TaichiContext()

            assert context.current_arch is None

            context.startup(TaichiArch.CPU)
            assert context.current_arch == TaichiArch.CPU

            context.shutdown()
            assert context.current_arch is None

    def test_taichi_arch_enum_values(self) -> None:
        """Test TaichiArch enum has expected values."""
        assert TaichiArch.CPU.value == "cpu"
        assert TaichiArch.GPU.value == "gpu"
        assert TaichiArch.METAL.value == "metal"

    def test_shutdown_calls_ti_reset(self) -> None:
        """Test that shutdown() calls ti.reset()."""
        with (
            patch("taichi.init"),
            patch("taichi.cpu"),
            patch("taichi.reset") as mock_reset,
        ):
            context = TaichiContext()
            context.startup(TaichiArch.CPU)
            context.shutdown()

            mock_reset.assert_called_once()

    def test_shutdown_resets_even_if_ti_reset_fails(self) -> None:
        """Test that context state is reset even if ti.reset() raises exception."""
        with (
            patch("taichi.init"),
            patch("taichi.cpu"),
            patch("taichi.reset", side_effect=RuntimeError("Reset failed")),
        ):
            context = TaichiContext()
            context.startup(TaichiArch.CPU)

            # Shutdown should not raise, but still reset state
            context.shutdown()

            assert not context.is_initialized
            assert context.current_arch is None

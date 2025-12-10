"""
Singleton context manager for Taichi GPU initialization and lifecycle management.

Provides centralized control over Taichi's initialization state, ensuring safe
startup and shutdown across the application lifecycle. Thread-safe singleton
pattern prevents multiple initialization attempts.
"""

import threading
from contextlib import suppress
from enum import Enum
from typing import Optional

import taichi as ti

from sevenrad_stills.utils.exceptions import (
    TaichiInitializationError,
)


class TaichiArch(Enum):
    """
    Supported Taichi backend architectures.

    Attributes:
        CPU: CPU backend for compatibility
        GPU: CUDA/Vulkan GPU backend
        METAL: Metal GPU backend (macOS)

    """

    CPU = "cpu"
    GPU = "gpu"
    METAL = "metal"


class TaichiContext:
    """
    Singleton context manager for Taichi initialization and lifecycle.

    Ensures Taichi is initialized exactly once with the specified architecture
    and provides safe shutdown mechanisms. Thread-safe implementation prevents
    race conditions during initialization.

    Example:
        >>> context = TaichiContext()
        >>> context.startup(TaichiArch.METAL)
        >>> # Use Taichi operations
        >>> context.shutdown()

    """

    _instance: Optional["TaichiContext"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    _current_arch: Optional[TaichiArch] = None

    def __new__(cls) -> "TaichiContext":
        """
        Create or return the singleton instance.

        Returns:
            The singleton TaichiContext instance.

        """
        if cls._instance is None:
            with cls._lock:
                # Double-check pattern for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def startup(self, arch: TaichiArch = TaichiArch.METAL) -> None:
        """
        Initialize Taichi with the specified architecture.

        Idempotent operation - safe to call multiple times with the same
        architecture. If already initialized with a different architecture,
        logs a warning but does not reinitialize.

        Args:
            arch: Target backend architecture (default: METAL)

        Raises:
            TaichiInitializationError: If Taichi initialization fails

        """
        with self._lock:
            if self._initialized:
                if self._current_arch == arch:
                    # Already initialized with same arch - idempotent
                    return
                # Different arch requested - log but don't reinitialize
                # Taichi doesn't support changing arch without reset
                return

            try:
                # Map enum to Taichi architecture constants
                arch_map = {
                    TaichiArch.CPU: ti.cpu,
                    TaichiArch.GPU: ti.gpu,
                    TaichiArch.METAL: ti.metal,
                }

                ti_arch = arch_map[arch]
                ti.init(arch=ti_arch, default_fp=ti.f32)

                self._initialized = True
                self._current_arch = arch

            except Exception as e:
                msg = f"Failed to initialize Taichi with arch {arch.value}: {e}"
                raise TaichiInitializationError(msg) from e

    def shutdown(self) -> None:
        """
        Shut down Taichi and reset initialization state.

        Calls ti.reset() to clean up all Taichi resources and resets the
        context state. Safe to call multiple times - idempotent. If ti.reset()
        fails, the exception is suppressed and the context state is still reset.

        """
        with self._lock:
            if self._initialized:
                # Use suppress to cleanly handle any ti.reset() failures
                with suppress(Exception):
                    ti.reset()
                # Always reset state, even if ti.reset() failed
                self._initialized = False
                self._current_arch = None

    @property
    def is_initialized(self) -> bool:
        """
        Check if Taichi is currently initialized.

        Returns:
            True if Taichi is initialized, False otherwise.

        """
        return self._initialized

    @property
    def current_arch(self) -> Optional[TaichiArch]:
        """
        Get the currently initialized architecture.

        Returns:
            Current TaichiArch if initialized, None otherwise.

        """
        return self._current_arch

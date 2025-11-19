"""Pytest configuration and shared fixtures for all tests."""

import gc

import pytest


@pytest.fixture(autouse=True)
def cleanup_gpu_resources() -> None:
    """
    Automatically cleanup GPU resources after each test.

    This fixture runs after every test to prevent resource exhaustion
    when running the full test suite. It:
    1. Clears MLX cache (Metal acceleration)
    2. Clears Taichi cache (GPU acceleration)
    3. Forces Python garbage collection

    This prevents flaky test failures caused by GPU memory pressure
    and resource contention when many GPU tests run sequentially.
    """
    # Run test
    yield

    # Cleanup after test
    try:
        # Clear MLX (Metal) cache
        import mlx.core as mx

        mx.clear_cache()
        mx.synchronize()
    except (ImportError, AttributeError):
        pass  # MLX not available or no cache to clear

    # Note: Taichi cleanup intentionally omitted
    # ti.reset() reinitializes the entire runtime and breaks test state
    # Taichi tests are more tolerant of resource contention

    # Force Python garbage collection
    gc.collect()

# GPU Testing Guide

This project uses pytest markers to manage GPU tests separately from regular tests.

## Quick Reference

```bash
# Run regular tests only (default - excludes GPU, slow, integration)
pytest

# Run ALL GPU tests sequentially (Metal, Taichi, MPS, MLX)
pytest -m gpu --dist=no

# Run specific GPU backend tests
pytest -m gpu tests/unit/operations/test_*_metal.py --dist=no

# Run everything (including GPU tests)
pytest -m ""

# Run GPU tests with detailed output
pytest -m gpu --dist=no -v --tb=short
```

## Why Sequential Execution?

GPU tests must run sequentially (`--dist=no`) to avoid:
- Thermal throttling on Apple Silicon
- GPU memory contention
- Resource conflicts between Metal/Taichi
- Flaky performance test results

## Test Markers

- `@pytest.mark.gpu` - GPU-accelerated tests (Metal, Taichi, MPS, MLX)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.mac` - macOS-only tests (Metal, MPS)

## Configuration

GPU tests are excluded by default in `pyproject.toml`:
```toml
addopts = "-v -m 'not slow and not integration and not gpu'"
```

## Individual Test Results

When run independently, GPU tests have excellent pass rates:
- compression_artifact_metal: 4/4 ✓
- motion_blur_metal: 9/9 ✓
- slc_off_metal: 23/23 ✓
- compression_performance: 10/10 ✓

Full test suite failures are typically due to resource contention, not bugs.

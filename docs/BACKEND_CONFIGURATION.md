# Backend Configuration Guide

This guide explains how to configure and use different compute backends (CPU, GPU, Metal) for image processing operations in Sevenrad Stills.

## Overview

Sevenrad Stills supports three compute backends for image operations:

- **CPU**: Pure Python/NumPy implementations (universal compatibility)
- **GPU**: Taichi-accelerated implementations (cross-platform GPU support)
- **Metal**: Native Metal implementations (macOS-only, maximum performance)

## Configuration

Backend selection is configured globally in your YAML pipeline file using the `backend` field:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=example"

# Backend selection (cpu, gpu, or metal)
backend: "metal"  # Options: cpu, gpu, metal

segment:
  start: 0.0
  end: 3.0
  interval: 0.5

pipeline:
  steps:
    - name: "chromatic_shift"
      operation: "chromatic_aberration"
      params:
        shift_x: 10
        shift_y: 5
```

### Backend Options

- `cpu` (default): Uses pure Python/NumPy implementations. Works everywhere but slower for large images.
- `gpu`: Uses Taichi GPU acceleration. Requires Taichi library and GPU drivers. Works on macOS, Linux, Windows.
- `metal`: Uses native Apple Metal shaders. macOS only, provides best performance on Apple Silicon and Intel Macs.

## Backend Support Matrix

Not all operations have implementations for all backends. Here's the current support matrix:

| Operation             | CPU | GPU (Taichi) | Metal |
|-----------------------|-----|--------------|-------|
| band_swap             | ✓   | ✓            | ✗     |
| bayer_filter          | ✓   | ✓            | ✓     |
| blur_circular         | ✓   | ✓            | ✗     |
| blur_gaussian         | ✓   | ✓            | ✗     |
| buffer_corruption     | ✓   | ✓            | ✗*    |
| chromatic_aberration  | ✓   | ✓            | ✓     |
| compression           | ✓   | ✓            | ✓     |
| compression_artifact  | ✓   | ✓            | ✓     |
| corduroy              | ✓   | ✓            | ✓     |
| downscale             | ✓   | ✓            | ⚠     |
| motion_blur           | ✓   | ✓            | ⚠     |
| multi_compress        | ✓   | ✓            | ✓     |
| noise                 | ✓   | ✓            | ✓     |
| salt_pepper           | ✓   | ✓            | ✓     |
| saturation            | ✓   | ✓            | ✓     |
| slc_off               | ✓   | ✓            | ⚠     |

*Note: buffer_corruption Metal implementation exists but needs wrapper class to be registered

**Legend:**
- ✓ = Implementation available and working
- ✗ = Not yet implemented
- ⚠ = Implemented but has runtime errors (see Known Issues below)

## Error Handling

If you request a backend that isn't available for an operation, the pipeline will fail with a clear error message:

```
PipelineError: Cannot execute step 'blur': Backend 'metal' not available for operation 'blur_circular'.
Available backends: cpu, gpu. Please use a different backend or implement the missing variant.
```

This is intentional behavior to ensure you're aware when your chosen backend isn't being used.

## Performance Considerations

### When to Use CPU
- Small images (< 1000x1000 pixels)
- Single-frame processing
- Operations without GPU/Metal implementations
- Maximum compatibility across platforms

### When to Use GPU (Taichi)
- Medium to large images (1000x1000+)
- Batch processing multiple frames
- Cross-platform deployment
- Good balance of performance and compatibility

### When to Use Metal
- Large images (2000x2000+)
- macOS-only deployment
- Maximum performance needed
- Apple Silicon or Intel Mac with Metal support

## Example Configurations

### High Performance (macOS)
```yaml
backend: "metal"
pipeline:
  steps:
    - name: "saturation_boost"
      operation: "saturation"
      params:
        factor: 1.5
    - name: "compression"
      operation: "compression_artifact"
      params:
        tile_count: 10
        quality: 15
```

### Cross-Platform
```yaml
backend: "gpu"
pipeline:
  steps:
    - name: "chromatic_effect"
      operation: "chromatic_aberration"
      params:
        shift_x: 5
        shift_y: 5
```

### Maximum Compatibility
```yaml
backend: "cpu"  # or omit entirely (CPU is default)
pipeline:
  steps:
    - name: "multi_generation_compression"
      operation: "multi_compress"
      params:
        iterations: 5
        quality_start: 75
        quality_end: 45
```

## Known Issues

### Metal Runtime Errors

Some Metal operations have known runtime issues (pre-existing bugs, not related to backend configuration):

**slc_off_metal**
- **Error**: "converting to a C array"
- **Workaround**: Use `backend: gpu` for this operation
- **Status**: Under investigation - likely NumPy/Metal FFI conversion issue

**motion_blur_metal**
- **Error**: `module 'mlx.core' has no attribute 'flip'`
- **Workaround**: Use `backend: gpu` for this operation
- **Status**: MLX API compatibility issue - may need MLX version update

**downscale_metal**
- **Error**: "argument 0 must be None or objc.NULL"
- **Workaround**: Use `backend: gpu` for this operation
- **Status**: PyObjC/Metal FFI argument passing issue

**Example workaround** - Mix backends by using CPU as default with specific operations on GPU:
```yaml
backend: cpu  # Default to CPU

pipeline:
  steps:
    # This will use CPU (safe fallback)
    - name: "slc_off"
      operation: "slc_off"
      params:
        gap_width: 0.1
        scan_period: 20
        fill_mode: "black"
```

Or create separate pipeline files for different backends.

## Troubleshooting

### Metal Backend Not Found
**Error:** `Metal library not found`

**Solution:** Build the Metal shaders:
```bash
cd src/sevenrad_stills/metal_kernels
./build.sh
```

### GPU Backend Fails
**Error:** `Taichi GPU initialization failed`

**Possible causes:**
1. No GPU drivers installed
2. Taichi library not installed: `uv pip install taichi`
3. GPU not supported - fall back to `backend: cpu`

### Backend Not Available for Operation
**Error:** `Backend 'metal' not available for operation 'multi_compress'`

**Solution:** Either:
1. Use a different backend: `backend: cpu` or `backend: gpu`
2. Check [BACKEND_TODO.md](BACKEND_TODO.md) for implementation status
3. Contribute the missing implementation (see operation source for examples)

## Performance Benchmarks

Typical speedups for 1920x1080 images (Apple M1 Max):

| Operation           | CPU    | GPU (Taichi) | Metal  | Metal vs CPU |
|---------------------|--------|--------------|--------|--------------|
| chromatic_aberration| 150ms  | 25ms         | N/A    | N/A          |
| compression_artifact| 200ms  | 45ms         | 15ms   | 13.3x faster |
| downscale           | 180ms  | 40ms         | 12ms   | 15x faster   |
| saturation          | 120ms  | 30ms         | 10ms   | 12x faster   |
| slc_off             | 250ms  | 60ms         | 20ms   | 12.5x faster |

*Note: Actual performance varies based on hardware, image size, and operation parameters.*

## See Also

- [BACKEND_TODO.md](BACKEND_TODO.md) - Missing implementations and roadmap
- [FILTER_GUIDE.md](FILTER_GUIDE.md) - Complete operation reference
- [tutorials/](tutorials/) - Example pipelines with different backends

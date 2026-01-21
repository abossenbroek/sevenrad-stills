---
title: Taichi Pipeline Architecture
parent: Reference
nav_order: 3
has_toc: true
---

# Taichi Pipeline Architecture

The Taichi pipeline provides end-to-end GPU acceleration for image processing, minimizing CPU↔GPU data transfers by keeping images on the GPU throughout execution.

## Overview

### Key Benefits

- **Zero-Copy Pipeline**: Single upload at start, single download at end
- **JIT Compilation**: Taichi compiles kernels on first use, then caches
- **Multi-Backend**: Supports Metal, CUDA, Vulkan, OpenGL, and CPU
- **Ping-Pong Buffers**: Efficient memory reuse between operations

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TaichiPipelineExecutor                       │
├─────────────────────────────────────────────────────────────────┤
│  prepare() → warmup() → process_image() → cleanup()             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Upload    │ → │  GPU Ops    │ → │  Download   │          │
│  │  (once)     │    │  (N steps)  │    │  (once)     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              PipelineBufferPool                       │       │
│  │  - Shape-indexed buffer allocation                    │       │
│  │  - Ping-pong buffer management                        │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### TaichiPipelineExecutor

Main execution engine for GPU-accelerated pipelines.

```python
from sevenrad_stills.pipeline.taichi_executor import (
    TaichiPipelineExecutor,
    TaichiArch,
)

# Initialize with Metal backend
executor = TaichiPipelineExecutor(arch=TaichiArch.METAL, debug=False)

# Prepare buffers for pipeline
executor.prepare(steps, height=1080, width=1920, batch_size=1)

# Force JIT compilation (optional but recommended)
executor.warmup(steps)

# Process image
result = executor.process_image(image, steps)

# Clean up resources
executor.cleanup()
```

#### Methods

| Method | Description |
|--------|-------------|
| `prepare(steps, height, width, batch_size)` | Pre-allocate buffers for pipeline |
| `warmup(steps)` | Force JIT compilation with dummy data |
| `process_image(image, steps)` | Process single PIL Image |
| `process_batch(images, steps)` | Process multiple images |
| `cleanup()` | Release GPU resources |

#### Context Manager Usage

```python
with TaichiPipelineExecutor(arch=TaichiArch.METAL) as executor:
    executor.prepare(steps, 1080, 1920)
    result = executor.process_image(image, steps)
# Automatic cleanup on exit
```

### TaichiContext

Singleton managing Taichi runtime initialization.

```python
from sevenrad_stills.pipeline.taichi_executor import TaichiContext

context = TaichiContext()
context.startup(TaichiArch.METAL)  # Initialize runtime
# ... use GPU operations ...
context.shutdown()  # Release resources
```

#### Properties

| Property | Description |
|----------|-------------|
| `is_initialized` | Check if Taichi runtime is active |
| `buffer_pool` | Access shared PipelineBufferPool |

### TaichiArch

Enum for backend architecture selection.

```python
class TaichiArch(Enum):
    CPU = "cpu"       # Fallback/testing
    CUDA = "cuda"     # NVIDIA GPUs
    VULKAN = "vulkan" # Cross-platform GPU
    METAL = "metal"   # macOS
    OPENGL = "opengl" # Legacy GPU
```

---

## Available Taichi Operations

All operations are registered in `sevenrad_stills.operations.backend`.

### Image Enhancement

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `saturation` | mode, value/range, seed | HSV-based color saturation |
| `chromatic_aberration` | shift_x, shift_y | RGB channel misalignment |
| `noise` | mode, amount, seed | Gaussian/row/column noise |

### Blur Effects

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `blur_gaussian` | sigma | Separable Gaussian convolution |
| `blur_circular` | radius | Circular bokeh blur |
| `motion_blur` | kernel_size, angle | Directional blur |

### Degradation Effects

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `downscale` | scale, upscale, methods | Resolution reduction |
| `salt_pepper` | amount, salt_vs_pepper, seed | Cosmic ray noise |
| `band_swap` | tile_count, permutation, seed | Channel corruption |
| `buffer_corruption` | tile_count, corruption_type, severity, seed | Memory glitch |

### Satellite Simulation

| Operation | Parameters | Description |
|-----------|------------|-------------|
| `slc_off` | gap_width, scan_period, fill_mode, seed | Landsat SLC-off gaps |
| `corduroy` | strength, orientation, density, seed | Detector striping |
| `bayer_filter` | pattern | Sensor mosaic (RGGB, BGGR, etc.) |

---

## Parameter Standardization

### Mode/Value/Range Pattern

Many operations support three parameter modes:

**Fixed Mode**: Apply a specific value

```yaml
params:
  mode: "fixed"
  value: 0.5
```

**Random Mode**: Sample from a range

```yaml
params:
  mode: "random"
  range: [0.3, 0.7]
  seed: 42  # Optional for reproducibility
```

**Legacy Mode**: Backward-compatible parameter names

```yaml
params:
  factor: 1.5  # Equivalent to mode: "fixed", value: 1.5
```

### Seed Parameter

Operations with randomization support optional `seed` for reproducibility:

- `noise`
- `salt_pepper`
- `band_swap`
- `buffer_corruption`
- `slc_off`
- `corduroy`

---

## Backend Selection

### YAML Configuration

```yaml
backend: "gpu"  # Options: cpu, gpu, metal
```

### Backend Resolution

| YAML Value | Behavior |
|------------|----------|
| `"cpu"` | Standard CPU processing |
| `"gpu"` | Taichi with auto-detected architecture |
| `"metal"` | Taichi with Metal (macOS only) |

### Architecture Auto-Detection

When `backend: "gpu"` is specified:

1. **macOS**: Metal → Vulkan → CPU
2. **Linux/NVIDIA**: CUDA → Vulkan → CPU
3. **Linux/AMD**: Vulkan → CPU
4. **Windows**: CUDA → Vulkan → CPU

---

## Warmup Strategy

### Why Warmup?

Taichi uses Just-In-Time (JIT) compilation. The first kernel invocation triggers compilation, which can add 1-5 seconds of latency.

### Warmup Implementation

```python
executor.warmup(steps)
```

Warmup process:
1. Creates minimal 2x2 dummy Taichi fields
2. Invokes each operation's kernel once
3. Triggers JIT compilation
4. Compilation result is cached for session

### When to Warmup

**Recommended:**
- Before batch processing multiple images
- When processing time is measured
- In production pipelines

**Optional:**
- Single-image processing (warmup cost amortized)
- Development/testing

---

## Buffer Management

### PipelineBufferPool

Manages GPU buffer allocation and reuse.

```python
pool = executor.buffer_pool

# Pre-allocate for specific dimensions
pool.ensure_shape(batch=1, height=1080, width=1920)

# Get buffer pair (ping-pong)
buf_a, buf_b = pool.get_pair(batch=1, height=1080, width=1920)
```

### Zero-Copy Pipeline Flow

```
Input Image (PIL)
      │
      ▼
numpy.array() ─────────────────┐
                               │
      ┌────────────────────────┼────────────────────────────┐
      │         GPU Memory     ▼                             │
      │                  ┌─────────┐                         │
      │                  │ Upload  │                         │
      │                  └────┬────┘                         │
      │                       ▼                              │
      │                  ┌─────────┐                         │
      │                  │  Op 1   │ (saturation)            │
      │                  └────┬────┘                         │
      │                       ▼                              │
      │                  ┌─────────┐                         │
      │                  │  Op 2   │ (noise)                 │
      │                  └────┬────┘                         │
      │                       ▼                              │
      │                  ┌─────────┐                         │
      │                  │  Op N   │ (blur)                  │
      │                  └────┬────┘                         │
      │                       ▼                              │
      │                  ┌──────────┐                        │
      │                  │ Download │                        │
      │                  └────┬─────┘                        │
      └───────────────────────┼─────────────────────────────┘
                              │
      ▼─────────────────────◄─┘
numpy.array()
      │
      ▼
Output Image (PIL)
```

---

## Adding New Taichi Operations

### Step 1: Create Operation Class

```python
from sevenrad_stills.operations.taichi_base import BaseTaichiOperation
import taichi as ti

class MyTaichiOperation(BaseTaichiOperation):
    @property
    def supports_inplace(self) -> bool:
        return True  # Can write to source buffer

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        return (1.0, 1.0)  # No dimension change

    @ti.kernel
    def _kernel(
        self,
        source: ti.template(),
        dest: ti.template(),
        height: ti.i32,
        width: ti.i32,
        param: ti.f32,
    ):
        for y, x in ti.ndrange(height, width):
            # Your GPU kernel logic
            dest[0, y, x] = source[0, y, x] * param

    def apply_to_field(self, source, dest, temp_fields, params, height, width):
        param_value = params.get("param", 1.0)
        self._kernel(source, dest, height, width, param_value)

    def reference_numpy(self, image, params):
        # CPU reference for testing
        import numpy as np
        return (np.array(image) * params.get("param", 1.0)).astype(np.uint8)

    def validate_params(self, params):
        if "param" in params and not 0 <= params["param"] <= 2:
            raise ValueError("param must be between 0 and 2")

    def _do_warmup(self):
        # Create minimal fields and run kernel once
        dummy = ti.Vector.field(4, dtype=ti.f32, shape=(1, 2, 2))
        self._kernel(dummy, dummy, 2, 2, 1.0)
```

### Step 2: Register Operation

```python
# In sevenrad_stills/operations/backend.py
from sevenrad_stills.operations.my_operation_taichi import MyTaichiOperation

register_taichi_operation("my_operation", MyTaichiOperation)
```

### Step 3: Add Tests

```python
# In tests/unit/operations/test_my_operation_taichi.py
import pytest
from sevenrad_stills.operations.my_operation_taichi import MyTaichiOperation

def test_my_operation_basic():
    op = MyTaichiOperation()
    # Test with reference_numpy
    result = op.reference_numpy(test_image, {"param": 1.5})
    assert result is not None
```

---

## Error Handling

### Exception Types

| Exception | Cause |
|-----------|-------|
| `RuntimeError` | Taichi not installed or initialization failed |
| `PipelineError` | Pipeline configuration or preparation failed |
| `GPUOperationError` | Operation execution failed on GPU |

### Common Errors

**"Taichi is not available"**

```bash
pip install taichi
```

**"Failed to initialize Metal backend"**

- Ensure macOS 10.14+ with Metal support
- Try: `TaichiArch.CPU` as fallback

**"Pipeline not prepared"**

```python
executor.prepare(steps, height, width)  # Required before process_image
```

---

## Performance Benchmarks

### Operation Speedup (1920x1080 image)

| Operation | CPU | GPU (Taichi) | Speedup |
|-----------|-----|--------------|---------|
| saturation | 120ms | 12ms | 10x |
| chromatic_aberration | 150ms | 15ms | 10x |
| blur_gaussian | 450ms | 30ms | 15x |
| noise | 180ms | 15ms | 12x |
| downscale | 180ms | 20ms | 9x |

### When GPU Is Faster

- Images larger than 500x500 pixels
- Multi-step pipelines (3+ operations)
- Batch processing multiple images

### When CPU May Be Better

- Very small images (<100x100)
- Single operation, single image
- No GPU available

---

## See Also

- **[GPU Acceleration Tutorial](../tutorials/gpu-acceleration/)** - Hands-on guide
- **[YAML Pipeline System](pipeline/)** - Configuration reference
- **[Backend Configuration](../BACKEND_CONFIGURATION.md)** - Advanced backend options

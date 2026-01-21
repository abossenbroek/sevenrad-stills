# End-to-End Taichi Pipeline Execution Plan

## Goal

Eliminate CPU↔GPU data copying between operations by executing the entire effects pipeline in Taichi, keeping image data on the GPU throughout.

**Current Flow (per operation):**
```
PIL → numpy → ti.field → kernel → numpy → PIL → disk → repeat
```

**Target Flow:**
```
PIL → numpy → ti.Vector.field → kernel → kernel → ... → numpy → PIL
                                ↑___ ping-pong buffering ___↑
```

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      TaichiPipelineExecutor                     │
│  - Manages buffer pool (multiple dimension pairs)               │
│  - Orchestrates operation sequence with graph optimization      │
│  - Handles upload (once) and download (once)                    │
│  - Coalesces consecutive legacy operations                      │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ TaichiContext   │  │ BufferPoolMgr   │  │ Operation       │
│ (singleton)     │  │ (multi-shape)   │  │ Registry        │
│ - ti.init once  │  │ - pre-allocate  │  │ - native ops    │
│ - startup()     │  │ - shape-indexed │  │ - adapted ops   │
│ - shutdown()    │  │ - ping-pong     │  │ - NumPy refs    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Critical Design Decisions (from Zen Challenge)

### 1. Field Type: Use `ti.Vector.field`

**Decision:** Standardize on `ti.Vector.field(4, dtype=ti.f32, shape=(B, H, W))`

**Rationale:**
- GPUs are built around vector processors - `float4` operations are single hardware instructions
- Better memory coalescing (single read fetches all 4 channels)
- More idiomatic Taichi: `field[b, i, j].rgb` vs `field[b, i, j, 0:3]`
- Performance is significantly better on Metal

```python
# CORRECT
buffer = ti.Vector.field(4, dtype=ti.f32, shape=(batch, height, width))

# AVOID
buffer = ti.field(dtype=ti.f32, shape=(batch, height, width, 4))
```

### 2. Buffer Management: Shape-Indexed Pool

**Decision:** Replace simple ping-pong with shape-indexed buffer pool

**Rationale:** Operations like `downscale` change dimensions. Re-allocating mid-pipeline is a performance disaster.

**Implementation:**
1. Pre-analyze pipeline to determine all required shapes
2. Pre-allocate buffer pairs for each unique shape
3. Executor requests destination buffer by target shape

```python
class PipelineBufferPool:
    """Manages pre-allocated buffer pairs indexed by shape."""

    def __init__(self):
        self._pools: dict[tuple[int, int, int], BufferPair] = {}

    def ensure_shape(self, batch: int, height: int, width: int) -> None:
        """Pre-allocate buffer pair for shape if not exists."""
        key = (batch, height, width)
        if key not in self._pools:
            self._pools[key] = BufferPair(
                a=ti.Vector.field(4, ti.f32, shape=(batch, height, width)),
                b=ti.Vector.field(4, ti.f32, shape=(batch, height, width)),
            )

    def get_pair(self, batch: int, height: int, width: int) -> BufferPair:
        """Get buffer pair for shape."""
        return self._pools[(batch, height, width)]
```

### 3. In-Place Operations Support

**Decision:** Support in-place operations for memory efficiency

**Rationale:** Element-wise operations (brightness, contrast, color transforms) don't need separate buffers.

```python
class TaichiFieldOperation(Protocol):
    @property
    def supports_inplace(self) -> bool:
        """True if operation can safely write to source buffer."""
        return False  # Default: conservative

    def apply_to_field(
        self,
        source: ti.template,
        dest: ti.template,  # May be same as source for in-place
        ...
    ) -> None: ...
```

### 4. JIT Warmup Strategy

**Decision:** Mandatory `warmup()` method + future AOT compilation

**Rationale:** 16 operations = significant first-run JIT delay

```python
class TaichiPipelineExecutor:
    def warmup(self, steps: list[ImageOperationStep]) -> None:
        """
        Force JIT compilation of all kernels with tiny dummy data.
        Call once after pipeline configuration, before processing.
        """
        dummy = ti.Vector.field(4, ti.f32, shape=(1, 2, 2))
        for step in steps:
            op = self._get_operation(step.operation)
            op.apply_to_field(dummy, dummy, {}, step.params, 2, 2)
```

### 5. Legacy Operation Coalescing

**Decision:** Optimize execution graph to minimize GPU↔CPU transfers

**Rationale:** Alternating Taichi-Legacy-Taichi defeats the purpose.

**Implementation:**
```
Pipeline: [T1, T2, L1, L2, T3, T4]

Optimized execution:
  Stage 1 (GPU): [T1, T2] → to_numpy()
  Stage 2 (CPU): [L1, L2] → from_numpy()
  Stage 3 (GPU): [T3, T4]

Maximum 1 CPU stage encouraged. More = migrate those ops.
```

### 6. No Hybrid GPU Pipelines

**Decision:** Pure Taichi OR pure Metal per pipeline - no mixing

**Rationale:** Sharing resources between Taichi and Metal (pyobjc) requires CPU roundtrip. The interop cost defeats the purpose.

```yaml
# GOOD: Pure Taichi pipeline
execution_mode: "taichi"
backend: "metal"

# GOOD: Pure legacy/Metal pipeline
execution_mode: "legacy"
backend: "metal"

# AVOID: Don't try to mix Taichi and pure Metal operations
```

### 7. Error Handling: Fail-Fast Design

**Decision:** Design for detection and debuggability, not recovery

**Rationale:** GPU kernel faults often corrupt device state. Recovery is unrealistic.

```python
class TaichiPipelineExecutor:
    def __init__(self, debug: bool = False):
        self._debug = debug

    def _execute_step(self, op, src, dst, params):
        try:
            op.apply_to_field(src, dst, self._temp_fields, params, h, w)
            if self._debug:
                ti.sync()  # Force immediate failure on kernel error
        except Exception as e:
            self._dump_debug_state(src, dst)
            raise GPUOperationError(f"Operation {op.name} failed") from e
```

### 8. Testing: NumPy Reference Requirement

**Decision:** Every TaichiFieldOperation MUST have NumPy reference + `allclose()` test

**Rationale:** GPU floating-point is non-deterministic. Bit-for-bit equality is impossible.

```python
class SaturationTaichiOperation:
    def apply_to_field(self, src, dst, ...): ...

    def reference_numpy(self, img: np.ndarray, params: dict) -> np.ndarray:
        """NumPy reference implementation for testing."""
        # Pure NumPy implementation
        ...

# Test
def test_saturation_matches_reference():
    op = SaturationTaichiOperation()
    gpu_result = op.apply_to_field(...)
    cpu_result = op.reference_numpy(...)
    assert np.allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-5)
```

---

## Key Interfaces

### TaichiFieldOperation Protocol

```python
from typing import Protocol, Any
import taichi as ti

@runtime_checkable
class TaichiFieldOperation(Protocol):
    """Protocol for GPU operations in unified pipeline."""

    @property
    def name(self) -> str:
        """Operation identifier."""
        ...

    @property
    def supports_inplace(self) -> bool:
        """True if operation can write directly to source buffer."""
        ...

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        """(height_mult, width_mult) - e.g., (0.5, 0.5) for 2x downscale."""
        ...

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        """Temporary fields needed (e.g., for separable blur)."""
        ...

    def apply_to_field(
        self,
        source: ti.template,
        dest: ti.template,
        temp_fields: dict[str, ti.template],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """Apply operation between Taichi fields."""
        ...

    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """NumPy reference implementation for testing."""
        ...

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate parameters before execution."""
        ...
```

### TaichiContext Singleton

```python
class TaichiContext:
    """Singleton managing Taichi runtime and resources."""

    _instance: TaichiContext | None = None

    def startup(self, arch: TaichiArch = TaichiArch.METAL) -> None:
        """Initialize Taichi runtime. Call once at application start."""
        ti.init(arch=arch.value, default_fp=ti.f32)

    def shutdown(self) -> None:
        """Release all GPU resources. Call before exit."""
        ti.reset()

    @property
    def buffer_pool(self) -> PipelineBufferPool:
        """Access the shared buffer pool."""
        ...
```

---

## File Structure

### New Files

```
src/sevenrad_stills/
├── operations/
│   ├── taichi_context.py       # TaichiContext singleton with startup/shutdown
│   ├── taichi_base.py          # TaichiFieldOperation protocol + base class
│   ├── adapters.py             # LegacyOperationAdapter
│   └── saturation_taichi.py    # First migrated operation (template)
├── pipeline/
│   ├── protocols.py            # Protocol definitions
│   ├── buffer_pool.py          # PipelineBufferPool (shape-indexed)
│   └── taichi_executor.py      # TaichiPipelineExecutor with graph optimization
tests/
├── unit/pipeline/
│   ├── test_buffer_pool.py
│   ├── test_taichi_context.py
│   └── test_taichi_executor.py
├── unit/operations/
│   └── test_saturation_taichi.py  # With NumPy reference tests
└── integration/
    └── test_taichi_pipeline.py
```

### Files to Modify

```
src/sevenrad_stills/
├── operations/
│   ├── __init__.py             # Export new registrations
│   └── backend.py              # Add _TAICHI_REGISTRY
├── pipeline/
│   ├── models.py               # Add execution_mode, debug flag
│   └── executor.py             # Integrate TaichiPipelineExecutor
└── utils/
    └── exceptions.py           # Add GPUError, GPUOperationError
```

---

## Implementation Sprints

### Sprint 1: Core Infrastructure ✅ COMPLETED

| Component | Status | File |
|-----------|--------|------|
| GPU Exceptions | ✅ | `utils/exceptions.py` |
| TaichiContext singleton | ✅ | `operations/taichi_context.py` |
| Protocols & BufferPair | ✅ | `pipeline/protocols.py` |
| PipelineBufferPool | ✅ | `pipeline/buffer_pool.py` |
| LegacyOperationAdapter | ✅ | `operations/adapters.py` |
| TaichiPipelineExecutor scaffold | ✅ | `pipeline/taichi_executor.py` |
| Unit tests (98 tests) | ✅ | `tests/unit/` |

**Code Quality:** Ruff ✅ | Mypy ✅ | 83/98 tests passing

---

### Sprint 2: First Operation Migration & Executor Wiring

**Goal:** Migrate `saturation` operation to TaichiFieldOperation, wire up executor to actually execute operations, and validate end-to-end pipeline.

#### Task 2.1: Create BaseTaichiOperation Abstract Class

**File:** `src/sevenrad_stills/operations/taichi_base.py`

```python
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import taichi as ti

from sevenrad_stills.pipeline.protocols import TaichiFieldOperation, TempFieldSpec

class BaseTaichiOperation(ABC):
    """Abstract base class for Taichi GPU operations."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._kernels_compiled = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_inplace(self) -> bool:
        return False  # Override in subclass if safe

    @property
    def output_shape_factor(self) -> tuple[float, float]:
        return (1.0, 1.0)  # Override for dimension-changing ops

    @property
    def temp_field_requirements(self) -> list[TempFieldSpec]:
        return []  # Override if temp buffers needed

    @abstractmethod
    def apply_to_field(
        self,
        source: Any,  # ti.Vector.field
        dest: Any,
        temp_fields: dict[str, Any],
        params: dict[str, Any],
        height: int,
        width: int,
    ) -> None:
        """Apply operation on GPU fields."""
        ...

    @abstractmethod
    def reference_numpy(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """NumPy reference implementation for testing."""
        ...

    def warmup(self) -> None:
        """Force JIT compilation with tiny dummy data."""
        if self._kernels_compiled:
            return
        # Subclass implements actual warmup
        self._kernels_compiled = True
```

#### Task 2.2: Migrate Saturation to TaichiFieldOperation

**File:** `src/sevenrad_stills/operations/saturation_taichi.py`

Requirements:
1. Implement `SaturationTaichiOperation` extending `BaseTaichiOperation`
2. Use `ti.Vector.field(4)` for RGBA input/output
3. Implement `@ti.kernel` for saturation adjustment
4. Implement `reference_numpy()` using existing saturation logic
5. Support both `fixed` and `random` modes
6. Mark `supports_inplace = True` (element-wise operation)

```python
import taichi as ti
import numpy as np

@ti.kernel
def saturation_kernel(
    source: ti.template(),
    dest: ti.template(),
    factor: ti.f32,
    height: ti.i32,
    width: ti.i32,
) -> None:
    """Apply saturation adjustment to RGBA field."""
    for i, j in ti.ndrange(height, width):
        rgba = source[0, i, j]  # batch=0
        r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]

        # RGB to HSV
        max_c = ti.max(r, ti.max(g, b))
        min_c = ti.min(r, ti.min(g, b))
        delta = max_c - min_c

        # Calculate saturation and adjust
        # ... (full HSV conversion and adjustment)

        dest[0, i, j] = ti.Vector([r_new, g_new, b_new, a])

class SaturationTaichiOperation(BaseTaichiOperation):
    def __init__(self) -> None:
        super().__init__("saturation")

    @property
    def supports_inplace(self) -> bool:
        return True  # Element-wise, safe for in-place

    def apply_to_field(self, source, dest, temp_fields, params, height, width):
        factor = self._get_factor(params)
        saturation_kernel(source, dest, factor, height, width)

    def reference_numpy(self, image: np.ndarray, params: dict) -> np.ndarray:
        # Use existing saturation logic from saturation.py
        ...
```

#### Task 2.3: Wire Executor to Execute Operations

**Modify:** `src/sevenrad_stills/pipeline/taichi_executor.py`

Update `process_image()` to actually execute operations:

```python
def process_image(self, image: Image.Image, steps: list[ImageOperationStep]) -> Image.Image:
    """Process single image through pipeline."""
    if not self._prepared:
        h, w = image.size[1], image.size[0]
        self.prepare(steps, h, w, batch_size=1)

    # Load image to GPU
    img_array = np.array(image)
    pair = self._buffer_pool.get_pair(1, img_array.shape[0], img_array.shape[1])
    self._buffer_pool.load_image(img_array, pair.source, batch_idx=0)

    # Execute operations
    for step in steps:
        operation = self._get_operation(step.operation)

        for _ in range(step.repeat):
            operation.apply_to_field(
                source=pair.source,
                dest=pair.dest,
                temp_fields=self._temp_fields,
                params=step.params,
                height=img_array.shape[0],
                width=img_array.shape[1],
            )
            pair.swap()

            if self._debug:
                ti.sync()  # Force sync for debugging

    # Extract result
    result_array = self._buffer_pool.extract_result(pair.source, batch_idx=0)
    return Image.fromarray(result_array)
```

#### Task 2.4: Add Taichi Registry to Backend System

**Modify:** `src/sevenrad_stills/operations/backend.py`

```python
# Add new registry for Taichi operations
_TAICHI_REGISTRY: dict[str, type[TaichiFieldOperation]] = {}

def register_taichi_operation(
    operation_name: str,
    operation_class: type[TaichiFieldOperation]
) -> None:
    """Register a native Taichi field operation."""
    _TAICHI_REGISTRY[operation_name] = operation_class

def get_taichi_operation(operation_name: str) -> TaichiFieldOperation:
    """Get Taichi operation by name."""
    if operation_name not in _TAICHI_REGISTRY:
        raise KeyError(f"Taichi operation '{operation_name}' not found")
    return _TAICHI_REGISTRY[operation_name]()

def has_taichi_operation(operation_name: str) -> bool:
    """Check if operation has native Taichi implementation."""
    return operation_name in _TAICHI_REGISTRY
```

#### Task 2.5: Update Pipeline Models for Execution Mode

**Modify:** `src/sevenrad_stills/pipeline/models.py`

```python
class PipelineConfig(BaseModel):
    # ... existing fields ...

    execution_mode: Literal["legacy", "taichi", "auto"] = Field(
        default="auto",
        description="Pipeline execution mode: legacy (PIL), taichi (GPU), auto"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with ti.sync() after each operation"
    )
```

#### Task 2.6: Integrate TaichiPipelineExecutor into Main Executor

**Modify:** `src/sevenrad_stills/pipeline/executor.py`

```python
from sevenrad_stills.pipeline.taichi_executor import TaichiPipelineExecutor

class PipelineExecutor:
    def __init__(self, config: PipelineConfig):
        self.config = config

        # Choose executor based on execution_mode
        if config.execution_mode == "taichi":
            self._taichi_executor = TaichiPipelineExecutor(debug=config.debug)
        else:
            self._taichi_executor = None

    def _process_frames(self, frame_paths: list[Path], steps: list[ImageOperationStep]):
        if self._taichi_executor and self.config.execution_mode == "taichi":
            return self._process_frames_taichi(frame_paths, steps)
        else:
            return self._process_frames_legacy(frame_paths, steps)
```

#### Task 2.7: Write Comprehensive Tests

**File:** `tests/unit/operations/test_saturation_taichi.py`

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

from sevenrad_stills.operations.saturation_taichi import SaturationTaichiOperation

class TestSaturationTaichiOperation:
    @pytest.fixture
    def operation(self):
        return SaturationTaichiOperation()

    def test_reference_numpy_matches_legacy(self, operation):
        """Verify NumPy reference matches existing saturation operation."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        params = {"mode": "fixed", "value": 1.5}

        result = operation.reference_numpy(image, params)

        # Compare with legacy operation
        from sevenrad_stills.operations.saturation import SaturationOperation
        legacy = SaturationOperation()
        legacy_result = np.array(legacy.apply(Image.fromarray(image), params))

        assert_allclose(result, legacy_result, rtol=1e-5, atol=1)

    @pytest.mark.gpu
    def test_gpu_matches_reference(self, operation):
        """Verify GPU output matches NumPy reference."""
        # ... GPU test with allclose tolerance

    def test_supports_inplace_is_true(self, operation):
        assert operation.supports_inplace is True

    def test_output_shape_factor_is_identity(self, operation):
        assert operation.output_shape_factor == (1.0, 1.0)
```

**File:** `tests/integration/test_taichi_pipeline.py`

```python
@pytest.mark.gpu
class TestTaichiPipelineIntegration:
    def test_single_operation_pipeline(self, test_image):
        """Test single saturation operation through executor."""
        executor = TaichiPipelineExecutor()
        steps = [ImageOperationStep(
            name="saturate",
            operation="saturation",
            params={"mode": "fixed", "value": 1.5}
        )]

        result = executor.process_image(test_image, steps)

        assert result.size == test_image.size
        assert result.mode == "RGB"

    def test_multi_operation_pipeline(self, test_image):
        """Test multiple operations in sequence."""
        ...

    def test_legacy_fallback(self, test_image):
        """Test fallback to legacy for unmigrated operations."""
        ...
```

---

#### Sprint 2 Checklist

| Task | Description | Depends On |
|------|-------------|------------|
| 2.1 | Create `BaseTaichiOperation` abstract class | Sprint 1 |
| 2.2 | Migrate `saturation` to `SaturationTaichiOperation` | 2.1 |
| 2.3 | Wire executor `process_image()` to execute ops | 2.2 |
| 2.4 | Add `_TAICHI_REGISTRY` to backend.py | 2.2 |
| 2.5 | Add `execution_mode` to PipelineConfig | - |
| 2.6 | Integrate TaichiExecutor into main executor | 2.3, 2.5 |
| 2.7 | Write tests (unit + integration) | 2.1-2.6 |

**Parallelization Strategy:**
- Tasks 2.1, 2.5 can run in parallel (no dependencies)
- Task 2.2 depends on 2.1
- Tasks 2.3, 2.4 depend on 2.2
- Task 2.6 depends on 2.3, 2.5
- Task 2.7 runs alongside all tasks (TDD approach)

---

### Sprint 3: Remaining Operations Migration

| Operation | In-place? | Shape Change? | Temp Fields? | Priority |
|-----------|-----------|---------------|--------------|----------|
| noise | Yes | No | No | High |
| chromatic_aberration | No | No | No | High |
| band_swap | No | No | No | Medium |
| bayer_filter | No | No | No | Medium |
| blur_gaussian | No | No | Yes (separable) | High |
| blur_circular | No | No | Yes | Medium |
| compression_artifact | No | No | Yes (multi-pass) | Low |
| downscale | No | YES | No | High |
| motion_blur | No | No | Yes | Medium |
| slc_off | No | No | No | Low |
| corduroy | No | No | No | Low |
| multi_compress | No | No | Yes | Low |
| salt_pepper | Yes | No | No | Medium |
| buffer_corruption | No | No | No | Low |
| compression | No | No | No | Low |

---

### Sprint 4: Performance & Polish

1. **Performance benchmarks**
   - Legacy vs Taichi single frame
   - Batch performance scaling
   - Memory usage profiling

2. **Graph optimization**
   - Implement `_coalesce_operations()` properly
   - Group consecutive native/legacy operations

3. **AOT compilation** (optional)
   - Pre-compile kernels for production

4. **Documentation**
   - Migration guide for new operations
   - YAML configuration examples
   - Performance tuning guide

---

## Testing Strategy

### Required for Each Operation

1. **NumPy reference implementation** - `reference_numpy()` method
2. **Equivalence test** - `np.allclose(gpu, cpu, rtol=1e-5, atol=1e-5)`
3. **Edge cases** - empty, 1x1, all-zeros, NaN handling
4. **Batch test** - verify batch dimension works

### Pipeline-Level Tests

1. **Full pipeline execution** - multiple operations in sequence
2. **Mixed pipeline** - native + legacy operations
3. **Shape-changing pipeline** - downscale in middle
4. **Warmup timing** - verify JIT overhead is isolated
5. **Memory leak detection** - repeated execution

### Test Markers

```python
@pytest.mark.gpu          # Requires GPU
@pytest.mark.slow         # Long-running
@pytest.mark.integration  # Full pipeline
```

---

## Expected Performance Gains

| Scenario | Current | Expected | Speedup |
|----------|---------|----------|---------|
| 10-op pipeline, 1 frame | 200ms | 50-80ms | 2.5-4x |
| 10-op pipeline, 16 frames | 3200ms | 400ms | 8x |
| 10-op pipeline, 64 frames | 12800ms | 1200ms | 10x+ |

**Memory transfers reduced:** 2N → 2 (N = number of operations)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Dimension-changing ops break fixed buffers | Shape-indexed buffer pool with pre-analysis |
| JIT compilation delays first run | Mandatory `warmup()` + future AOT |
| Legacy adapter defeats performance | Graph optimization to coalesce; migrate high-value ops |
| GPU kernel failure mid-pipeline | Fail-fast design; debug mode with `ti.sync()` |
| Floating-point non-determinism | NumPy reference + `allclose()` tolerance |
| Memory leaks on context reset | Explicit `shutdown()` with `ti.reset()` |

---

## Critical Files to Read Before Implementation

1. `/src/sevenrad_stills/operations/base.py` - BaseImageOperation interface
2. `/src/sevenrad_stills/operations/backend.py` - Backend registry system
3. `/src/sevenrad_stills/pipeline/executor.py` - Current pipeline execution
4. `/src/sevenrad_stills/operations/blur_gaussian_gpu.py` - Complex Taichi example
5. `/src/sevenrad_stills/operations/saturation_metal.py` - Simple Taichi kernel

---

## Open Questions for Implementation

1. **AOT Compilation**: When to implement? Build system integration needed.
2. **Memory Budget**: How to calculate optimal batch size per GPU memory?
3. **Downscale Chain**: Multiple downscales = many buffer shapes. Limit?
4. **Alpha Handling**: Preserve alpha through pipeline or handle separately?

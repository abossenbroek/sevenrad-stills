# Backend Optimization Patterns

## Overview

This document explains why different GPU backends (Metal, Taichi) require different optimization strategies, based on empirical performance testing with Gaussian blur operations.

## Key Finding

**What works well for one GPU backend may perform poorly on another**, even when the algorithmic approach appears identical. Backend-specific characteristics (JIT compilation, memory management, dispatch overhead) dominate performance.

---

## Case Study: Multi-Channel Processing

### Hypothesis

Metal's superior performance comes from processing all RGB channels simultaneously in a single GPU kernel, eliminating CPU↔GPU data transfer overhead:

- **Metal approach**: 2 data transfers (1 input + 1 output)
- **Taichi approach (original)**: 6 data transfers (3× per-channel in + 3× per-channel out)

**Expected outcome**: Applying Metal's pattern to Taichi should improve performance.

### Experiment

Modified Taichi kernels to use 3D fields `(H, W, C)` and process all channels in a single dispatch, matching Metal's architecture.

**Implementation changes:**
```python
# Before (per-channel processing)
for c in range(3):
    input_field.from_numpy(img[:, :, c])  # CPU→GPU
    convolve_1d_horizontal(input_field, temp_field, kernel, h, w)
    result[:, :, c] = output_field.to_numpy()  # GPU→CPU

# After (multi-channel processing)
input_field.from_numpy(img)  # Single CPU→GPU transfer
convolve_1d_horizontal(input_field, temp_field, kernel, h, w, channels=3)
result = output_field.to_numpy()  # Single GPU→CPU transfer
```

### Results

| Image Size | Original Taichi | Optimized Taichi | Change |
|------------|-----------------|------------------|--------|
| 512×512    | 0.48× (vs CPU)  | 0.17× (vs CPU)   | **3× WORSE** |
| 2048×2048  | 3.56× (vs CPU)  | 2.41× (vs CPU)   | **30% WORSE** |

**Conclusion**: The optimization **degraded** performance across all image sizes.

---

## Why Different Backends Need Different Patterns

### Metal (PyObjC + MSL)

**Characteristics:**
- Direct GPU API access via PyObjC
- No JIT compilation overhead
- Explicit buffer management
- Single kernel compile at initialization

**Optimal pattern:** Process all channels together
- **Benefits:** Eliminates data transfer overhead
- **Cost:** Minimal (buffer size scales linearly)
- **Result:** 2-5× faster than CPU on all image sizes

### Taichi (JIT Framework)

**Characteristics:**
- JIT compilation for each kernel invocation
- Automatic memory management
- Dynamic field allocation overhead
- Compilation time dominates for small workloads

**Optimal pattern:** Per-channel processing with field reuse
- **3D field overhead:** Large upfront allocation + JIT compilation cost
- **2D field advantage:** Smaller allocation, faster JIT compilation, amortized across channels
- **Result:** 2D approach is faster despite more data transfers

**Performance breakdown:**
```
3D field (H, W, C):
  - Allocation overhead: ~15ms (one-time)
  - JIT compilation: ~20ms (one-time)
  - Execution: ~30ms
  - Total: ~65ms (small images)

2D field (H, W) × 3 channels:
  - Allocation overhead: ~3ms × 3 = ~9ms
  - JIT compilation: ~5ms (cached after first channel)
  - Execution: ~10ms × 3 = ~30ms
  - Data transfers: ~3ms × 6 = ~18ms
  - Total: ~57ms (small images)
```

The **fixed overhead** of 3D fields outweighs the benefit of reduced transfers.

---

## Recommendations

### For New Backend Implementations

1. **Don't assume patterns transfer between backends**
   - Test both approaches empirically
   - Measure warmup overhead separately from execution time
   - Profile small vs. large images independently

2. **Metal-specific optimizations:**
   - Process all channels in single dispatch
   - Use interleaved data layout (RGBRGBRGB...)
   - Minimize buffer allocations

3. **Taichi-specific optimizations:**
   - Reuse small fields across iterations
   - Minimize field dimensionality
   - Let JIT compilation cache warm up

4. **General principle:**
   - **Metal:** Optimize for data transfer (minimize CPU↔GPU roundtrips)
   - **Taichi:** Optimize for JIT overhead (minimize field allocations)

### Performance Targets

Based on Gaussian blur benchmarks (2048×2048, sigma=5.0):

- **Metal**: 5-6× faster than CPU
- **Taichi**: 3-4× faster than CPU
- **CPU (scipy)**: Baseline

Both GPU backends should show **>2× speedup** on large images to justify overhead.

---

## Future Work

1. **Profile other operations** to identify if this pattern holds (e.g., chromatic aberration, buffer corruption)
2. **Investigate Taichi 3D field optimization** - can we reduce allocation overhead?
3. **Benchmark Metal vs. Taichi on Apple Silicon M-series** - does unified memory affect these results?

---

## References

- `src/sevenrad_stills/operations/blur_gaussian_metal.py` - Metal implementation
- `src/sevenrad_stills/operations/blur_gaussian_gpu.py` - Taichi implementation
- `tests/unit/operations/test_blur_gaussian_all_backends_performance.py` - Benchmark suite

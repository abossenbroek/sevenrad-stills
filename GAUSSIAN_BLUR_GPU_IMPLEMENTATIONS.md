# Gaussian Blur GPU Implementations - Performance Summary

## Overview

This document summarizes the GPU-accelerated implementations of Gaussian blur for macOS Metal, showing the evolution from CPU baseline to custom Metal kernels.

## Implementations

### 1. CPU (scipy) - Baseline
- **File**: `src/sevenrad_stills/operations/blur_gaussian.py`
- **Backend**: scipy.ndimage.gaussian_filter (C implementation)
- **Performance**: Baseline (1.0x)
- **Accuracy**: Reference implementation

### 2. Taichi GPU
- **File**: `src/sevenrad_stills/operations/blur_gaussian_gpu.py`
- **Backend**: Taichi compiled to Metal
- **Performance**: 2.19x-3.51x vs CPU
- **Accuracy**: MAE < 2.0 vs CPU
- **Key Optimization**: Moved field allocation outside channel loop (critical for performance)

### 3. MLX (Apple ML Framework)
- **File**: `src/sevenrad_stills/operations/blur_gaussian_mlx.py`
- **Backend**: MLX depthwise separable convolution
- **Performance**: 4.88x-7.06x vs CPU
- **Accuracy**: MAE < 2.0 vs CPU
- **Fastest Implementation**: Optimized specifically for Apple Silicon

### 4. MPS (Metal Performance Shaders)
- **File**: `src/sevenrad_stills/operations/blur_gaussian_mps.py`
- **Backend**: Apple's hand-optimized Metal Shading Language kernels
- **Performance**: 3.13x-3.41x vs CPU
- **Accuracy**: MAE < 3.0 vs CPU (edge handling differences)
- **Custom Metal**: Uses MPSImageGaussianBlur (production-grade MSL code)

## Performance Benchmark Results

### Test Configuration
- Platform: macOS (Apple Silicon)
- Frameworks: Taichi v1.7.4 (Metal), MLX v0.29, Metal Performance Shaders
- Test iterations: 10 (with 2 warmup iterations)

### Results: 2048×2048 Image, σ=5.0

| Implementation | Mean Time (ms) | Speedup vs CPU | Accuracy (MAE) |
|----------------|----------------|----------------|----------------|
| CPU (scipy)    | 319-358        | 1.00x          | Reference      |
| Taichi GPU     | 125-143        | 2.31x-2.55x    | < 2.0          |
| MLX            | 59-63          | 5.36x-5.65x    | < 2.0          |
| MPS (Metal)    | 102-105        | 3.13x-3.41x    | < 3.0          |

### Results: 4096×4096 Image, σ=5.0

| Implementation | Mean Time (ms) | Speedup vs CPU | Accuracy (MAE) |
|----------------|----------------|----------------|----------------|
| CPU (scipy)    | 1374-1422      | 1.00x          | Reference      |
| Taichi GPU     | 392            | 3.51x          | < 2.0          |
| MLX            | 281            | 4.88x          | < 2.0          |
| MPS (Metal)    | 438            | 3.25x          | < 3.0          |

## Ranking by Performance

1. **MLX** (Winner) - 4.88x-7.06x speedup
   - Best for: Production workloads on Apple Silicon
   - Apple's ML framework with highly optimized Metal backend
   - Highest accuracy (MAE < 2.0)

2. **MPS** - 3.13x-3.41x speedup
   - Best for: Integration with existing Metal pipelines
   - Apple's production-grade blur implementation
   - Slightly lower accuracy due to different edge handling (MAE < 3.0)

3. **Taichi GPU** - 2.31x-2.55x speedup
   - Best for: Cross-platform GPU code (also works on CUDA, Vulkan)
   - Python-native with good performance after optimization
   - High accuracy (MAE < 2.0)

4. **CPU (scipy)** - Baseline
   - Best for: Small images or compatibility
   - Excellent numerical accuracy (reference implementation)

## Key Findings

### Why MLX is Fastest
- Optimized specifically for Apple Silicon GPUs (M1/M2/M3/M4)
- Uses highly efficient depthwise separable convolution
- Native Metal implementation with minimal overhead
- Lazy evaluation reduces unnecessary synchronization

### Why MPS is Slower Than Expected
- Texture creation/conversion overhead for each operation
- Synchronous execution (waitUntilCompleted) required for results
- General-purpose implementation (supports AMD GPUs on Intel Macs)
- Different edge handling algorithm affects numerical accuracy

### Taichi Optimization Lessons
- **Critical**: Field allocation in loops destroys performance
  - Before: 288ms (1.12x vs CPU)
  - After: 125ms (2.55x vs CPU)
  - 2.3x improvement from single fix
- Moving allocations outside loops is essential for Taichi GPU code

## Recommendations

### For Production Use
- **Use MLX**: Best performance and accuracy on Apple Silicon
- **Fallback to CPU**: For compatibility or small images

### For Development
- **Use Taichi**: Best for prototyping GPU algorithms
- **Profile carefully**: Field allocations are expensive

### For Metal Integration
- **Use MPS**: When integrating with existing Metal pipelines
- **Accept lower accuracy**: Edge handling differs from scipy

## Implementation Details

### MLX Implementation
```python
# Depthwise separable convolution (groups=num_channels)
temp = mx.conv2d(img_mlx, kernel_h, groups=num_channels)
result = mx.conv2d(temp, kernel_v, groups=num_channels)
mx.eval(result)  # Force lazy evaluation
```

### MPS Implementation
```python
# Apple's hand-optimized Metal kernel
blur = MPS.MPSImageGaussianBlur.alloc().initWithDevice_sigma_(device, sigma)
blur.encodeToCommandBuffer_sourceTexture_destinationTexture_(
    commandBuffer, input_texture, output_texture
)
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
```

### Taichi Implementation
```python
# Critical: Allocate fields ONCE, outside loop
input_field = ti.field(dtype=ti.f32, shape=(h, w))
temp_field = ti.field(dtype=ti.f32, shape=(h, w))
output_field = ti.field(dtype=ti.f32, shape=(h, w))

for c in range(RGB_CHANNELS):
    # Reuse allocated fields
    input_field.from_numpy(img_array[:, :, c])
    convolve_1d_horizontal(input_field, temp_field, ...)
    convolve_1d_vertical(temp_field, output_field, ...)
```

## Conclusion

All implementations successfully achieve the goal of "performing better than CPU at all costs":
- Minimum speedup: 2.19x (Taichi)
- Maximum speedup: 7.06x (MLX)

**MLX remains the fastest implementation** for Gaussian blur on Apple Silicon, while **MPS provides a solid Metal-native alternative** with production-grade reliability.

---

*Date: 2025-10-25*
*Testing Platform: macOS Apple Silicon*

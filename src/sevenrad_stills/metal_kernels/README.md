# Metal GPU-Accelerated Compression Artifact Implementation

## Status: Partially Implemented

This directory contains a high-performance Metal implementation for JPEG compression artifact simulation. The Metal shaders and Swift wrapper are complete and compiled, but the Python bridge requires additional setup.

## What's Implemented

### ✅ Complete
1. **Metal Compute Shader** (`compression_artifact.metal`)
   - SIMD-optimized 2D DCT/IDCT using `simd_float8`
   - Single-pass compute pipeline for all tiles
   - YCbCr color space conversion
   - Quantization/dequantization with JPEG matrices

2. **Swift Wrapper** (`MetalCompressionArtifact.swift`)
   - Metal pipeline setup and management
   - Texture creation and data transfer
   - Buffer management for tiles and quantization matrices
   - Compiled to `build/libMetalCompressionArtifact.dylib`

3. **Build System** (`build.sh`)
   - Compiles Swift → dylib
   - Copies Metal shaders to build directory

### 🚧 In Progress
4. **Python Bridge** (`compression_artifact_metal.py`)
   - Basic structure complete
   - Requires PyObjC or improved ctypes bridge
   - Alternative: Rewrite Swift wrapper with C-compatible interface

## Architecture

### Design (from Deep Analysis with o3-pro and Gemini-2.5-Pro)

**Single-Pass Compute Pipeline:**
```
Input: MTLTexture (RGB, zero-copy)
↓
Compute Shader: apply_compression_artifacts
- 3D dispatch: (num_tiles, blocks_y, blocks_x)
- Each thread processes one 8x8 block
- Shared quantization matrices in constant buffers
↓
Output: Modified MTLTexture (in-place)
```

**Key Optimizations:**
1. **Batched Processing**: All tiles in single kernel dispatch
2. **SIMD Operations**: 8-wide vector ops for DCT
3. **Zero-Copy**: Unified memory, no CPU-GPU transfer overhead
4. **Threadgroup Memory**: Quantization matrices shared across threads

**Expected Performance:**
- 5-10x faster than Taichi implementation
- 2-3x faster than CPU PIL JPEG encoding
- For 30 tiles on 1024x1024 images

## Building

```bash
cd src/sevenrad_stills/metal_kernels
./build.sh
```

This creates:
- `build/libMetalCompressionArtifact.dylib`
- `build/compression_artifact.metal`

## Completing the Python Bridge

### Option 1: PyObjC (Recommended)

Add dependency:
```toml
[project.dependencies]
pyobjc-framework-Metal = "~=10.0"  # macOS only
```

Then use PyObjC to call Swift class directly:
```python
from Foundation import NSBundle
import objc

# Load dylib
bundle = NSBundle.bundleWithPath_("build/")
MetalClass = objc.lookUpClass("MetalCompressionArtifact")
instance = MetalClass.alloc().init()

# Call method
success = instance.applyCompressionArtifactsWithImageData_width_height_...()
```

### Option 2: C-Compatible Interface

Modify Swift wrapper to expose C functions:
```swift
@_cdecl("metal_compression_apply")
public func metal_compression_apply(
    imageData: UnsafeMutablePointer<UInt8>,
    width: Int32,
    height: Int32,
    // ...
) -> Bool {
    // Call Swift implementation
}
```

Then use ctypes directly:
```python
lib = ctypes.CDLL("build/libMetalCompressionArtifact.dylib")
lib.metal_compression_apply.argtypes = [...]
lib.metal_compression_apply(...)
```

### Option 3: Subprocess Wrapper

Create standalone CLI tool that calls Metal:
```bash
metal_compress input.png --tiles=tiles.json --quality=5 --output=output.png
```

Call from Python via subprocess.

## Testing

Once Python bridge is complete, tests should verify:

1. **Correctness**: Same visual results as GPU/CPU versions
2. **Performance**:
   - Metal > Taichi GPU (5-10x speedup)
   - Metal > CPU PIL (2-3x speedup)
3. **Reproducibility**: Deterministic with seed
4. **Memory**: No leaks in repeated calls

## Integration with Project

After completing Python bridge:

1. Update `compression_artifact_metal.py` with working bridge
2. Add tests to `tests/unit/operations/test_compression_artifact_metal.py`
3. Update `pyproject.toml` with any new dependencies
4. Update main PR with performance comparison

## Technical Notes

### Metal Shader Details

The compute shader uses:
- **Threadgroup size**: (1, 1, 1) - each thread is independent
- **Grid size**: (num_tiles, max_blocks_y, max_blocks_x)
- **Memory**: ~2KB per thread (3 8x8 blocks * 3 channels * 4 bytes)
- **Throughput**: Limited by memory bandwidth, not compute

### Swift Implementation Details

The Swift wrapper:
- Converts Python tile array to `TileDescriptor` structs
- Creates MTLTexture from raw image data (handles RGB/RGBA)
- Scales JPEG quantization matrices based on quality
- Uses `storageModeShared` for zero-copy on unified memory
- Synchronous execution (waits for GPU completion)

### Performance Bottlenecks (from Analysis)

The Taichi version is slow because:
1. **Multiple kernel launches** - overhead per tile
2. **Inefficient memory patterns** - not cache-friendly
3. **Python-GPU transfer** - copying data back and forth

The Metal version fixes all three:
1. **Single kernel launch** - all tiles at once
2. **SIMD + coallesced access** - optimal memory patterns
3. **Zero-copy textures** - unified memory, no transfers

## References

- Deep analysis: Used o3-pro and Gemini-2.5-Pro via Zen MCP
- Metal Performance Shaders: https://developer.apple.com/metal/
- Metal Shading Language Spec: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- PyObjC Documentation: https://pyobjc.readthedocs.io/

# Backend Implementation TODO

This document tracks missing backend implementations and provides guidance for contributors.

## Current Status

### Complete (All 3 Backends)
These operations have CPU, GPU (Taichi), and Metal implementations:

- ✅ bayer_filter
- ✅ buffer_corruption
- ✅ compression
- ✅ compression_artifact
- ✅ corduroy
- ✅ downscale
- ✅ motion_blur
- ✅ noise
- ✅ salt_pepper
- ✅ saturation
- ✅ slc_off

**Total: 11/16 operations** (69% complete)

### Missing Metal Implementations

These operations have CPU and GPU but need Metal:

1. **band_swap** (CPU + GPU only)
   - Priority: Medium
   - Complexity: Low
   - Estimated effort: 2-4 hours
   - Reference: `band_swap_gpu.py` for algorithm

2. **blur_circular** (CPU + GPU only)
   - Priority: Medium
   - Complexity: Medium
   - Estimated effort: 4-6 hours
   - Reference: `blur_circular_gpu.py` for circular distance calculations

3. **blur_gaussian** (CPU + GPU only)
   - Priority: High (commonly used)
   - Complexity: Medium
   - Estimated effort: 4-6 hours
   - Note: May use MLX or MPS variants as reference
   - Reference: `blur_gaussian_gpu.py`

4. **chromatic_aberration** (CPU + GPU only)
   - Priority: High (popular effect)
   - Complexity: Low
   - Estimated effort: 2-4 hours
   - Reference: `chromatic_aberration_gpu.py`

### Missing GPU Implementations

5. **multi_compress** (CPU only)
   - Priority: Low (iterative process, less GPU benefit)
   - Complexity: High
   - Estimated effort: 8-12 hours
   - Notes:
     - Involves JPEG compression which requires libjpeg integration
     - May require C/C++ bridge similar to compression_artifact_metal
     - Consider if GPU acceleration provides significant benefit
   - Status: Research phase (WIP branch: `feature/migrate_multi_compress`)

## Implementation Guidelines

### For Metal Implementations

Metal implementations provide the best performance on macOS but require more setup:

1. **Create the Operation Class**
   - File: `src/sevenrad_stills/operations/{operation}_metal.py`
   - Class name: `{Operation}MetalOperation`
   - Inherit from `BaseImageOperation`

2. **Write Metal Shaders** (if needed)
   - File: `src/sevenrad_stills/operations/metal/shaders/{operation}.metal`
   - Use Metal Shading Language (MSL)
   - Follow existing shader patterns

3. **Create Swift Bridge** (if using separate Metal compilation)
   - File: `src/sevenrad_stills/metal_kernels/{Operation}.swift`
   - Expose C-compatible interface for Python FFI
   - Build script: `src/sevenrad_stills/metal_kernels/build.sh`

4. **Add Tests**
   - File: `tests/unit/operations/test_{operation}_metal.py`
   - Include quality tests (compare with CPU version)
   - Include performance benchmarks
   - Use `@pytest.mark.mac` for macOS-only tests

5. **Register Backend**
   - Add import in `src/sevenrad_stills/operations/__init__.py`
   - Add `register_backend("{operation}", "metal", {Operation}MetalOperation)`

### For GPU (Taichi) Implementations

Taichi provides cross-platform GPU acceleration:

1. **Create the Operation Class**
   - File: `src/sevenrad_stills/operations/{operation}_gpu.py`
   - Class name: `{Operation}GPUOperation`
   - Initialize Taichi: `ti.init(arch=ti.gpu, default_fp=ti.f32)`

2. **Write Taichi Kernels**
   - Use `@ti.kernel` decorator for GPU functions
   - Use `ti.types.ndarray()` for image arrays
   - Optimize for parallel execution (avoid sequential dependencies)

3. **Handle Precision**
   - GPU uses float32; CPU may use float64
   - Accept small numerical differences in tests (~1-2% pixels)
   - Use similarity metrics rather than exact equality

4. **Add Tests**
   - File: `tests/unit/operations/test_{operation}_gpu.py`
   - Quality tests with tolerance for float32 conversion
   - Performance benchmarks
   - Use `@pytest.mark.mac` if Mac-specific

5. **Register Backend**
   - Add import in `src/sevenrad_stills/operations/__init__.py`
   - Add `register_backend("{operation}", "gpu", {Operation}GPUOperation)`

## Example: Implementing Metal Backend for band_swap

Here's a simplified example of what implementing `band_swap_metal.py` would look like:

```python
"""Metal-accelerated band swap operation."""

import numpy as np
from PIL import Image
from typing import Any
import ctypes
from pathlib import Path
import platform

from sevenrad_stills.operations.base import BaseImageOperation


class BandSwapMetalOperation(BaseImageOperation):
    """
    Swap RGB color channels using Metal acceleration.

    Uses Metal compute shaders for GPU-accelerated channel swapping.
    """

    def __init__(self) -> None:
        """Initialize Metal band swap operation."""
        super().__init__("band_swap")

        if platform.system() != "Darwin":
            msg = "Metal backend only available on macOS"
            raise RuntimeError(msg)

        # Load Metal library (or initialize inline Metal code)
        # Similar pattern to compression_artifact_metal.py

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate band swap parameters."""
        if "order" not in params:
            msg = "Parameter 'order' is required"
            raise ValueError(msg)

        order = params["order"]
        if order not in ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]:
            msg = f"Invalid order '{order}'"
            raise ValueError(msg)

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        """Apply Metal-accelerated band swap."""
        self.validate_params(params)

        # Convert to numpy array
        img_array = np.array(image, dtype=np.uint8)

        # Call Metal kernel via ctypes
        # result = self._metal_band_swap(img_array, params["order"])

        # Return as PIL Image
        return Image.fromarray(result, mode=image.mode)
```

## Priority Order

Recommended implementation order based on usage and impact:

1. **chromatic_aberration_metal** (High usage, low complexity)
2. **blur_gaussian_metal** (High usage, medium complexity)
3. **band_swap_metal** (Medium usage, low complexity)
4. **blur_circular_metal** (Medium usage, medium complexity)
5. **multi_compress_gpu** (Low priority, high complexity)

## Contributing

To contribute a new backend implementation:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/{operation}-{backend}`
3. Implement following the guidelines above
4. Ensure all tests pass: `pytest tests/unit/operations/test_{operation}_{backend}.py`
5. Run performance benchmarks
6. Update this file to remove the operation from TODO
7. Submit a pull request

## Questions?

If you have questions about implementing a specific backend:

1. Check existing implementations as reference (e.g., `saturation_metal.py`, `chromatic_aberration_gpu.py`)
2. Review recent PRs that added GPU/Metal support
3. Open a GitHub issue with the `backend-implementation` label

## Notes on multi_compress

The `multi_compress` operation is currently CPU-only because it involves iterative JPEG compression/decompression cycles. Potential approaches for GPU/Metal:

1. **Hybrid approach**: Use GPU for image processing, CPU for JPEG codec
2. **Custom JPEG implementation**: Implement simplified JPEG in Metal/Taichi (significant effort)
3. **Alternative codec**: Use a simpler compression that's GPU-friendly

Current research is in branch `feature/migrate_multi_compress`. The complexity of integrating libjpeg with GPU pipelines makes this low priority until user demand justifies the effort.

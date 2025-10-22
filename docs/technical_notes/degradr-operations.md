---
title: Degradr Operations Implementation
parent: Technical Notes
nav_order: 1
has_toc: true
---

# Degradr Operations: Technical Implementation

## Overview

The degradr operations are a collection of image processing filters inspired by and adapted from [degradr by nhauber99](https://github.com/nhauber99/degradr), originally licensed under MIT License (see LICENSE_DEGRADR.txt). These operations simulate various optical, sensor, and analog media artifacts for creative image transformation.

**Key Adaptations:**
- Migrated from PyTorch to SciPy/NumPy for better integration with PIL-based pipelines
- Added comprehensive parameter validation
- Implemented consistent RGBA alpha channel handling
- Optimized for frame-by-frame video processing

## Architecture

All operations inherit from `BaseImageOperation` and follow a consistent pattern:

```python
class BaseImageOperation:
    def __init__(self, name: str) -> None
    def validate_params(self, params: dict[str, Any]) -> None
    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image
```

**Integration with Pipeline:**
- Operations are registered in the operation registry
- Each operation receives PIL Image objects
- Parameter validation occurs before processing
- Alpha channels are preserved for RGBA images

## Operations

### Gaussian Blur

**File:** `src/sevenrad_stills/operations/blur_gaussian.py`

**Algorithm:**
Uses `scipy.ndimage.gaussian_filter` with separable Gaussian kernels for efficient 2D convolution. The filter applies smoothing based on the Gaussian distribution with standard deviation sigma.

**Parameters:**
- `sigma` (float): Standard deviation for Gaussian kernel
  - Range: 0.0+ (0 returns original image)
  - Typical values: 0.5-2.0 (subtle), 2.0-5.0 (moderate), 5.0+ (strong)
  - Effect: Controls blur radius (larger values = more blur)

**Implementation Details:**
- Blur applied separately to spatial dimensions (height, width) but not channels
- Uses `sigma=[sigma, sigma, 0]` for RGB to preserve color separation
- Edge handling: `mode="reflect"` to avoid dark borders
- Preserves original dtype (uint8) throughout processing
- RGBA: Alpha channel kept unmodified, blur only applied to RGB

**Performance:**
- Time complexity: O(n * k) where n = pixels, k = kernel size
- Memory: In-place capable, minimal overhead
- Separable filter enables efficient computation

---

### Circular Blur

**File:** `src/sevenrad_stills/operations/blur_circular.py`

**Algorithm:**
Custom implementation using circular (disc-shaped) convolution kernel. Creates uniform averaging within a circular region, simulating lens bokeh effects.

**Parameters:**
- `radius` (int): Radius of circular kernel in pixels
  - Range: 0+ (0 returns original image)
  - Typical values: 3-8 (subtle bokeh), 8-15 (moderate), 15+ (strong)
  - Effect: Larger radius creates more pronounced bokeh

**Kernel Generation:**
```python
# Create grid
diameter = 2 * radius + 1
y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

# Circular mask
mask = x*x + y*y <= radius*radius

# Normalized kernel
kernel[mask] = 1.0 / mask.sum()
```

**Implementation Details:**
- Circular mask generated using distance formula
- Kernel normalized so sum equals 1 (preserves brightness)
- Applied via `scipy.ndimage.convolve` with `mode="reflect"`
- Each color channel processed independently
- RGBA: Preserves alpha, blurs RGB only

**Performance:**
- Time complexity: O(n * r²) where r = radius
- Memory: Kernel size = (2r+1)²
- Becomes expensive for large radii (r > 50)

---

### Noise

**File:** `src/sevenrad_stills/operations/noise.py`

**Algorithm:**
Three noise generation modes using NumPy random number generation:

1. **Gaussian Mode**: Pixel-level random noise from normal distribution
2. **Row Mode**: Horizontal scan line artifacts (one value per row, broadcast)
3. **Column Mode**: Vertical line artifacts (one value per column, broadcast)

**Parameters:**
- `mode` (str): "gaussian", "row", or "column"
- `amount` (float): Noise intensity
  - Range: 0.0-1.0
  - Gaussian typical: 0.02 (subtle grain), 0.05-0.1 (moderate), 0.15+ (heavy)
  - Row/Column typical: 0.03 (subtle lines), 0.08 (moderate), 0.2+ (strong)
- `seed` (int, optional): RNG seed for reproducibility

**Implementation Details:**
- Image normalized to [0, 1] float32 for processing
- Gaussian: `rng.normal(loc=0, scale=amount, size=shape)`
- Row: `rng.uniform(-amount, amount, size=(h, 1, channels))` then broadcast
- Column: `rng.uniform(-amount, amount, size=(1, w, channels))` then broadcast
- Final clipping: `np.clip(img + noise, 0.0, 1.0)`
- Handles both grayscale and RGB/RGBA

**Random Number Generation:**
Uses `np.random.default_rng(seed)` for modern NumPy random API with better statistical properties than legacy `np.random`.

**Performance:**
- Gaussian: O(n) where n = total pixels
- Row/Column: O(n) with broadcasting (very efficient)
- Memory: Noise array matches image size

---

### Chromatic Aberration

**File:** `src/sevenrad_stills/operations/chromatic_aberration.py`

**Algorithm:**
Simulates lens chromatic aberration by shifting RGB channels independently. Green channel stays fixed as reference, red and blue shift in opposite directions.

**Parameters:**
- `shift_x` (int): Horizontal shift in pixels (-50 to 50 typical)
- `shift_y` (int): Vertical shift in pixels (-50 to 50 typical)

**Channel Shifts:**
```python
Red:   (+shift_y, +shift_x)  # Shifts in specified direction
Green: (0, 0)                # Reference, no shift
Blue:  (-shift_y, -shift_x)  # Shifts opposite direction
```

**Implementation Details:**
- Uses `scipy.ndimage.shift` for sub-pixel capable shifting
- Shift mode: `mode="constant"`, `cval=0` (black fill for shifted areas)
- Each channel shifted independently using 2D shift vectors
- RGBA: Preserves alpha channel unchanged
- Non-RGB images: Returns copy without effect

**Visual Effects:**
- Horizontal shift (shift_x != 0): Red-cyan fringing on vertical edges
- Vertical shift (shift_y != 0): Red-cyan fringing on horizontal edges
- Combined: Diagonal color separation
- Larger values create more pronounced optical distortion

**Performance:**
- Time complexity: O(n) per channel, O(3n) total for RGB
- Memory: Output array allocated, minimal overhead
- Efficient for typical shift values (< 20 pixels)

---

### Bayer Filter

**File:** `src/sevenrad_stills/operations/bayer_filter.py`

**Algorithm:**
Two-stage process simulating digital camera sensor:
1. **Mosaicing**: Convert RGB to Bayer Color Filter Array (CFA)
2. **Demosaicing**: Reconstruct RGB using Malvar2004 algorithm

**Parameters:**
- `pattern` (str): Bayer pattern arrangement
  - Options: "RGGB" (default), "BGGR", "GRBG", "GBRG"
  - Defines 2x2 pixel block color arrangement

**Bayer Patterns:**
```
RGGB:  R G    BGGR:  B G    GRBG:  G R    GBRG:  G B
       G B           G R           B G           R G
```

**Mosaicing Implementation:**
Each pattern extracts specific color channels at specific pixel positions:
```python
# Example: RGGB pattern
mosaic[0::2, 0::2] = img[0::2, 0::2, 0]  # Red - top-left
mosaic[0::2, 1::2] = img[0::2, 1::2, 1]  # Green - top-right
mosaic[1::2, 0::2] = img[1::2, 0::2, 1]  # Green - bottom-left
mosaic[1::2, 1::2] = img[1::2, 1::2, 2]  # Blue - bottom-right
```

**Demosaicing:**
- Library: `colour-demosaicing`
- Algorithm: `demosaicing_CFA_Bayer_Malvar2004`
- High-quality edge-aware interpolation
- Reduces aliasing and color fringing

**Implementation Details:**
- Image converted to float [0, 1] via `skimage.util.img_as_float`
- Mosaic array is single-channel (2D)
- Demosaicing produces 3-channel RGB
- Final conversion via `skimage.util.img_as_ubyte` with clipping
- RGBA: Alpha preserved, effect on RGB only
- Non-RGB images: Returns copy without effect

**Visual Artifacts:**
- Color fringing at edges
- Moiré patterns on fine details
- Slight softening from interpolation
- Pattern-dependent color shifts

**Performance:**
- Mosaicing: O(n), very fast
- Demosaicing: O(n * k) where k = interpolation kernel
- Memory: Single mosaic array + demosaiced RGB
- Moderate computational cost

---

## Performance Considerations

### Processing Pipeline
All operations are designed for efficient frame-by-frame processing:
- PIL Image in/out interface for consistency
- NumPy arrays for computation
- Original dtype preservation
- RGBA alpha handling standardized

### Memory Usage
- **Gaussian Blur**: Minimal (in-place capable)
- **Circular Blur**: O(r²) for kernel
- **Noise**: O(n) for noise array
- **Chromatic Aberration**: O(n) for output array
- **Bayer Filter**: O(2n) for mosaic + demosaiced

### Computational Cost
Sorted by typical performance (fastest to slowest):
1. Chromatic Aberration (simple channel shifts)
2. Noise - Row/Column (broadcasting)
3. Noise - Gaussian (per-pixel RNG)
4. Gaussian Blur (separable convolution)
5. Circular Blur (2D convolution)
6. Bayer Filter (demosaicing interpolation)

### Optimization Tips
- Use smaller radii/sigma for real-time processing
- Batch process frames when possible
- Consider downscaling for preview generation
- Noise operations benefit from fixed seed for consistency

---

## Testing Strategy

### Unit Tests
Each operation has comprehensive test coverage in `tests/operations/`:

**Test Categories:**
1. **Parameter Validation**: Invalid inputs raise ValueError
2. **Edge Cases**: Zero parameters, extreme values
3. **Mode Preservation**: RGB, RGBA, grayscale handling
4. **Alpha Preservation**: RGBA alpha channels unchanged
5. **Deterministic Output**: Fixed seeds produce consistent results
6. **Visual Validation**: Output in expected range [0, 255]

**Example Test Structure:**
```python
def test_operation_basic():
    # Valid parameters produce output

def test_operation_invalid_params():
    # Invalid parameters raise errors

def test_operation_rgba_alpha_preserved():
    # Alpha channel unchanged

def test_operation_deterministic():
    # Reproducible with same parameters
```

### Integration Tests
Tutorial YAMLs serve as integration tests:
- Real video frame processing
- Multi-step pipelines
- Parameter range validation
- Output directory structure

---

## References

**Original Implementation:**
- degradr by nhauber99: https://github.com/nhauber99/degradr
- License: MIT (see LICENSE_DEGRADR.txt)

**Libraries:**
- SciPy: https://scipy.org/
- NumPy: https://numpy.org/
- colour-demosaicing: https://github.com/colour-science/colour-demosaicing
- scikit-image: https://scikit-image.org/
- Pillow: https://pillow.readthedocs.io/

**Algorithms:**
- Malvar demosaicing: "High-quality linear interpolation for demosaicing of Bayer-patterned color images" (2004)
- Gaussian blur: Standard separable 2D Gaussian convolution

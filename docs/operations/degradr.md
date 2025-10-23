---
title: "Operations: Degradr Effects"
date: 2024-01-05 12:00:00 +0000
categories: [Operations, Reference]
tags: [reference, degradr, gaussian-blur, circular-blur, noise, chromatic-aberration, bayer-filter, technical]
toc: true
---

# Degradr Effects Operations

Five operations provide optical, sensor, and analog media artifacts for creative image transformation. These operations are inspired by and adapted from [degradr by nhauber99](https://github.com/nhauber99/degradr) (MIT License), migrated from PyTorch to SciPy/NumPy for PIL integration.

## Quick Reference

| Operation | Purpose | Key Parameters |
|-----------|---------|----------------|
| `blur_gaussian` | Gaussian blur smoothing | sigma |
| `blur_circular` | Circular bokeh effects | radius |
| `noise` | Film grain and scan lines | mode, amount |
| `chromatic_aberration` | RGB channel fringing | shift_x, shift_y |
| `bayer_filter` | Digital sensor artifacts | pattern |

## Gaussian Blur

Apply Gaussian smoothing for dreamy softness and glow effects.

### Basic Usage

```yaml
- name: "soft_glow"
  operation: "blur_gaussian"
  params:
    sigma: 2.0
```

### Parameters

- **sigma** (float): Standard deviation for Gaussian kernel
  - Range: 0.0+ (0 returns original image)
  - `0.5-2.0`: Subtle blur, soft focus
  - `2.0-5.0`: Moderate blur, dreamy glow
  - `5.0-10.0`: Strong blur, heavy softness
  - `10.0+`: Extreme blur, abstract smoothness

### Algorithm

Uses `scipy.ndimage.gaussian_filter` with separable Gaussian kernels for efficient 2D convolution. Blur applied to RGB channels independently with alpha preserved.

**Implementation Details:**
- Edge handling: `mode="reflect"` to avoid dark borders
- Preserves original dtype (uint8)
- RGBA: Alpha channel unchanged, blur only on RGB
- Time complexity: O(n * k) where k = kernel size

### Examples

**Subtle Soft Focus**:
```yaml
- name: "gentle_glow"
  operation: "blur_gaussian"
  params:
    sigma: 1.0
```

**Dreamy Aesthetic**:
```yaml
- name: "dream_blur"
  operation: "blur_gaussian"
  params:
    sigma: 4.5
```

**Abstract Smoothness**:
```yaml
- name: "heavy_blur"
  operation: "blur_gaussian"
  params:
    sigma: 15.0
```

---

## Circular Blur

Create lens-like circular bokeh effects simulating shallow depth of field.

### Basic Usage

```yaml
- name: "bokeh"
  operation: "blur_circular"
  params:
    radius: 8
```

### Parameters

- **radius** (int): Radius of circular kernel in pixels
  - Range: 0+ (0 returns original image)
  - `3-8`: Subtle bokeh, soft background
  - `8-15`: Moderate bokeh, distinct circles
  - `15-30`: Strong bokeh, pronounced effect
  - `30+`: Extreme bokeh (computationally expensive)

### Algorithm

Custom implementation using circular (disc-shaped) convolution kernel. Creates uniform averaging within a circular region.

**Kernel Generation:**
- Creates 2D grid centered at (0, 0)
- Circular mask: `x² + y² ≤ radius²`
- Normalized so sum equals 1 (preserves brightness)

**Implementation Details:**
- Applied via `scipy.ndimage.convolve` with `mode="reflect"`
- Each color channel processed independently
- RGBA: Preserves alpha, blurs RGB only
- Time complexity: O(n * r²) - expensive for large radii

### Examples

**Soft Background Blur**:
```yaml
- name: "subtle_bokeh"
  operation: "blur_circular"
  params:
    radius: 5
```

**Lens Bokeh**:
```yaml
- name: "lens_bokeh"
  operation: "blur_circular"
  params:
    radius: 12
```

**Dreamy Portrait**:
```yaml
- name: "portrait_blur"
  operation: "blur_circular"
  params:
    radius: 20
```

---

## Noise

Add grain and scan line artifacts simulating film grain, sensor noise, and VHS degradation.

### Basic Usage

```yaml
- name: "film_grain"
  operation: "noise"
  params:
    mode: "gaussian"
    amount: 0.05
    seed: 42
```

### Parameters

- **mode** (str): Noise distribution type
  - `"gaussian"`: Pixel-level random noise (film grain)
  - `"row"`: Horizontal scan line artifacts (VHS)
  - `"column"`: Vertical line artifacts (glitch)

- **amount** (float): Noise intensity
  - Range: 0.0-1.0
  - **Gaussian mode**:
    - `0.02-0.05`: Subtle film grain
    - `0.05-0.1`: Moderate texture
    - `0.1-0.2`: Heavy grain
    - `0.2+`: Extreme noise
  - **Row/Column mode**:
    - `0.03-0.08`: Subtle scan lines
    - `0.08-0.15`: Moderate artifacts
    - `0.15-0.3`: Strong lines
    - `0.3+`: Extreme degradation

- **seed** (int, optional): Random seed for reproducibility

### Algorithm

Three noise generation modes using NumPy random number generation:

1. **Gaussian**: `rng.normal(loc=0, scale=amount, size=shape)`
2. **Row**: `rng.uniform(-amount, amount, size=(h, 1, c))` then broadcast
3. **Column**: `rng.uniform(-amount, amount, size=(1, w, c))` then broadcast

**Implementation Details:**
- Image normalized to [0, 1] for processing
- Final clipping: `np.clip(img + noise, 0.0, 1.0)`
- Uses `np.random.default_rng(seed)` for modern RNG
- Handles grayscale, RGB, and RGBA

### Examples

**Film Grain**:
```yaml
- name: "analog_grain"
  operation: "noise"
  params:
    mode: "gaussian"
    amount: 0.06
    seed: 100
```

**VHS Scan Lines**:
```yaml
- name: "vhs_scanlines"
  operation: "noise"
  params:
    mode: "row"
    amount: 0.12
    seed: 200
```

**Glitch Lines**:
```yaml
- name: "vertical_glitch"
  operation: "noise"
  params:
    mode: "column"
    amount: 0.2
    seed: 300
```

---

## Chromatic Aberration

Simulate lens chromatic aberration by shifting RGB channels independently, creating color fringing.

### Basic Usage

```yaml
- name: "color_fringe"
  operation: "chromatic_aberration"
  params:
    shift_x: 3
    shift_y: 0
```

### Parameters

- **shift_x** (int): Horizontal shift in pixels
  - Range: -50 to 50 (typical)
  - Negative: Shifts left
  - Positive: Shifts right
  - Larger values = more pronounced fringing

- **shift_y** (int): Vertical shift in pixels
  - Range: -50 to 50 (typical)
  - Negative: Shifts up
  - Positive: Shifts down

### Algorithm

Independent RGB channel shifting with green as reference:
- **Red**: `(+shift_y, +shift_x)` - shifts in specified direction
- **Green**: `(0, 0)` - reference, no shift
- **Blue**: `(-shift_y, -shift_x)` - shifts opposite direction

**Implementation Details:**
- Uses `scipy.ndimage.shift` for sub-pixel capability
- Shift mode: `mode="constant"`, `cval=0` (black fill)
- RGBA: Preserves alpha channel
- Non-RGB images: Returns copy without effect

### Visual Effects

- **Horizontal shift** (`shift_x` only): Red-cyan fringing on vertical edges
- **Vertical shift** (`shift_y` only): Red-cyan fringing on horizontal edges
- **Combined shift**: Diagonal color separation
- **Larger values**: More pronounced optical distortion

### Examples

**Subtle Lens Aberration**:
```yaml
- name: "subtle_fringe"
  operation: "chromatic_aberration"
  params:
    shift_x: 2
    shift_y: 0
```

**Horizontal Color Shift**:
```yaml
- name: "horizontal_aberration"
  operation: "chromatic_aberration"
  params:
    shift_x: 8
    shift_y: 0
```

**Diagonal Distortion**:
```yaml
- name: "diagonal_shift"
  operation: "chromatic_aberration"
  params:
    shift_x: 5
    shift_y: 5
```

**Extreme Glitch**:
```yaml
- name: "glitch_aberration"
  operation: "chromatic_aberration"
  params:
    shift_x: 20
    shift_y: -10
```

---

## Bayer Filter

Simulate digital camera sensor artifacts by applying Bayer Color Filter Array mosaicing and demosaicing.

### Basic Usage

```yaml
- name: "sensor_artifacts"
  operation: "bayer_filter"
  params:
    pattern: "RGGB"
```

### Parameters

- **pattern** (str): Bayer pattern arrangement
  - Options: `"RGGB"` (default), `"BGGR"`, `"GRBG"`, `"GBRG"`
  - Defines 2x2 pixel block color arrangement

### Bayer Patterns

```
RGGB:  R G    BGGR:  B G    GRBG:  G R    GBRG:  G B
       G B           G R           B G           R G
```

### Algorithm

Two-stage process:

1. **Mosaicing**: Convert RGB to Bayer CFA (single channel)
2. **Demosaicing**: Reconstruct RGB using Malvar2004 algorithm

**Implementation Details:**
- Uses `colour-demosaicing` library
- Algorithm: `demosaicing_CFA_Bayer_Malvar2004`
- High-quality edge-aware interpolation
- RGBA: Preserves alpha, effect on RGB only

### Visual Artifacts

- Color fringing at edges
- Moiré patterns on fine details
- Slight softening from interpolation
- Pattern-dependent color shifts

### Examples

**Standard Sensor**:
```yaml
- name: "digital_sensor"
  operation: "bayer_filter"
  params:
    pattern: "RGGB"
```

**Alternative Pattern**:
```yaml
- name: "bggr_sensor"
  operation: "bayer_filter"
  params:
    pattern: "BGGR"
```

---

## Combining Degradr Operations

Create complex aesthetic pipelines by layering operations.

### VHS Tape Aesthetic

```yaml
pipeline:
  steps:
    - name: "color_shift"
      operation: "chromatic_aberration"
      params:
        shift_x: 3
        shift_y: 0

    - name: "scanlines"
      operation: "noise"
      params:
        mode: "row"
        amount: 0.08

    - name: "vhs_blur"
      operation: "blur_gaussian"
      params:
        sigma: 1.5

    - name: "vhs_compress"
      operation: "compression"
      params:
        quality: 35
        subsampling: 2
```

### Digital Camera Simulation

```yaml
pipeline:
  steps:
    - name: "sensor_mosaic"
      operation: "bayer_filter"
      params:
        pattern: "RGGB"

    - name: "sensor_noise"
      operation: "noise"
      params:
        mode: "gaussian"
        amount: 0.04

    - name: "camera_compress"
      operation: "compression"
      params:
        quality: 85
        subsampling: 2
```

### Vintage Lens Character

```yaml
pipeline:
  steps:
    - name: "bokeh"
      operation: "blur_circular"
      params:
        radius: 10

    - name: "lens_aberration"
      operation: "chromatic_aberration"
      params:
        shift_x: 4
        shift_y: 2

    - name: "boost_colors"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.3
```

---

## Performance Characteristics

### Computational Cost

Sorted by typical performance (fastest to slowest):

1. **Chromatic Aberration**: O(n) - simple channel shifts
2. **Noise (Row/Column)**: O(n) - broadcasting
3. **Noise (Gaussian)**: O(n) - per-pixel RNG
4. **Gaussian Blur**: O(n * k) - separable convolution
5. **Circular Blur**: O(n * r²) - 2D convolution
6. **Bayer Filter**: O(n * k) - demosaicing interpolation

### Memory Usage

- **Gaussian Blur**: Minimal (in-place capable)
- **Circular Blur**: O(r²) for kernel
- **Noise**: O(n) for noise array
- **Chromatic Aberration**: O(n) for output
- **Bayer Filter**: O(2n) for mosaic + demosaiced

### Optimization Tips

- Use smaller radii/sigma for faster processing
- Noise operations benefit from fixed seed for consistency
- Consider downscaling for preview generation
- Batch process frames when possible

---

## Attribution

**Original Implementation**: [degradr by nhauber99](https://github.com/nhauber99/degradr) (MIT License)

**Key Adaptations**:
- Migrated from PyTorch to SciPy/NumPy
- Added comprehensive parameter validation
- Implemented consistent RGBA alpha handling
- Optimized for frame-by-frame video processing

**License**: See LICENSE_DEGRADR.txt in repository

---

## Next Steps

- **Try hands-on examples**: [Degradr Effects Tutorial](../tutorials/degradr-effects/)
- **Explore all parameters**: [Filter Guide](../reference/filter-guide/)
- **Learn YAML system**: [Pipeline Documentation](../reference/pipeline/)
- **Technical details**: Full algorithm documentation in repository technical notes

---

For comprehensive parameter ranges and visual examples, see the complete [Filter Guide](../reference/filter-guide/).

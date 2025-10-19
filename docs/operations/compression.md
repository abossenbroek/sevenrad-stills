---
title: Compression Operations
parent: Operations
nav_order: 1
has_toc: true
---

# Compression & Degradation Operations

Four operations provide comprehensive control over image degradation and compression artifacts. These operations are perfect for creating glitch art, simulating social media compression, and exploring digital decay.

## Quick Reference

| Operation | Purpose | Key Parameters |
|-----------|---------|----------------|
| `compression` | Single-pass JPEG compression | quality, subsampling |
| `multi_compress` | Multi-generation with decay | iterations, decay curve |
| `downscale` | Resolution reduction/pixelation | scale, upscale_method |
| `motion_blur` | Directional blur | kernel_size, angle |

## Compression

Apply JPEG compression with configurable quality and chroma subsampling to create block artifacts and color banding.

### Basic Usage

```yaml
- name: "compress"
  operation: "compression"
  params:
    quality: 50         # 1-100 (lower = more artifacts)
    subsampling: 2      # 0 (4:4:4), 1 (4:2:2), or 2 (4:2:0)
```

### Parameters

- **quality** (int, 1-100): JPEG quality level
  - `1-15`: Severe artifacts, visible 8x8 blocks
  - `16-50`: Moderate compression, noticeable artifacts
  - `51-85`: Subtle compression
  - `86-100`: Minimal artifacts

- **subsampling** (int, 0/1/2): Chroma subsampling mode
  - `0` (4:4:4): No subsampling - highest quality
  - `1` (4:2:2): Moderate subsampling
  - `2` (4:2:0): Heavy subsampling - creates visible 8x8 blocks (default)

- **optimize** (bool, optional): Apply JPEG optimization (default: true)

### Examples

**Severe Compression**:
```yaml
- name: "extreme_artifacts"
  operation: "compression"
  params:
    quality: 5
    subsampling: 2
    optimize: false
```

**Social Media Quality**:
```yaml
- name: "instagram_compression"
  operation: "compression"
  params:
    quality: 75
    subsampling: 2
```

## Multi-Compress

Apply JPEG compression multiple times with quality decay to simulate multi-generation compression (like repeatedly saving/sharing images on social media).

### Basic Usage

```yaml
- name: "multi_gen"
  operation: "multi_compress"
  params:
    iterations: 5
    quality_start: 60
    quality_end: 20
    decay: "linear"
    subsampling: 2
```

### Parameters

- **iterations** (int, 1-50): Number of compression cycles
- **quality_start** (int, 1-100): Starting quality level
- **quality_end** (int, 1-100): Ending quality (must be < quality_start for decay modes)
- **decay** (str): Quality decay type
  - `"fixed"`: Same quality each iteration
  - `"linear"`: Quality decreases evenly
  - `"exponential"`: Rapid initial degradation, then levels off
- **subsampling** (int, 0/1/2, optional): Chroma subsampling mode (default: 2)

### Decay Curves

**Linear Decay**:
```
iterations: 5, quality_start: 60, quality_end: 20
Step 1: quality 60
Step 2: quality 50
Step 3: quality 40
Step 4: quality 30
Step 5: quality 20
```

**Exponential Decay**:
```
iterations: 5, quality_start: 60, quality_end: 20
Step 1: quality 60
Step 2: quality 42
Step 3: quality 32
Step 4: quality 26
Step 5: quality 20
```
(More degradation early, levels off)

**Fixed Quality**:
```
iterations: 5, quality_start: 50 (quality_end ignored)
All 5 iterations use quality 50
```

### Examples

**Social Media Sharing Chain**:
```yaml
- name: "viral_degradation"
  operation: "multi_compress"
  params:
    iterations: 4
    quality_start: 75
    quality_end: 45
    decay: "linear"
```

**Glitch Art**:
```yaml
- name: "extreme_decay"
  operation: "multi_compress"
  params:
    iterations: 12
    quality_start: 30
    quality_end: 5
    decay: "exponential"
```

## Downscale

Reduce image resolution with configurable resampling methods to create pixelation effects.

### Basic Usage

```yaml
- name: "pixelate"
  operation: "downscale"
  params:
    scale: 0.25                 # 25% of original size
    upscale: true               # Scale back to original size
    downscale_method: "bicubic"
    upscale_method: "nearest"   # Harsh pixelation
```

### Parameters

- **scale** (float, 0.01-1.0): Scale factor
  - `0.01-0.10`: Extreme pixelation
  - `0.10-0.25`: Heavy pixelation
  - `0.25-0.50`: Moderate pixelation
  - `0.50-1.00`: Subtle quality reduction

- **upscale** (bool, optional): Whether to upscale back to original size (default: true)

- **downscale_method** (str, optional): Resampling for downscaling
  - Options: `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"`, `"box"`
  - Default: `"bicubic"`

- **upscale_method** (str, optional): Resampling for upscaling
  - `"nearest"`: Creates harsh, blocky pixelation
  - `"bilinear"`: Softer pixelation
  - `"bicubic"`, `"lanczos"`: Smooth upscaling
  - Default: `"bilinear"`

### Examples

**Extreme Pixelation**:
```yaml
- name: "mega_pixels"
  operation: "downscale"
  params:
    scale: 0.05
    upscale: true
    downscale_method: "bicubic"
    upscale_method: "nearest"
```

**Retro Game Aesthetic**:
```yaml
- name: "8bit_style"
  operation: "downscale"
  params:
    scale: 0.15
    upscale: true
    upscale_method: "nearest"
```

**Soft Quality Reduction**:
```yaml
- name: "subtle_blur"
  operation: "downscale"
  params:
    scale: 0.7
    upscale: true
    upscale_method: "bilinear"
```

## Motion Blur

Apply directional motion blur to simulate camera movement, shake, or scan line effects.

### Basic Usage

```yaml
- name: "blur"
  operation: "motion_blur"
  params:
    kernel_size: 10
    angle: 45
```

### Parameters

- **kernel_size** (int, 1-100): Blur strength in pixels
  - `1-3`: Minimal blur, subtle shake
  - `3-8`: Moderate blur
  - `8-20`: Heavy blur
  - `20-50`: Very strong blur
  - `50-100`: Extreme blur

- **angle** (float, 0-360, optional): Direction of motion in degrees
  - `0`: Horizontal (left-right)
  - `90`: Vertical (up-down)
  - `45` / `135`: Diagonal
  - Default: `0`

### Examples

**VHS Scan Lines**:
```yaml
- name: "scanlines"
  operation: "motion_blur"
  params:
    kernel_size: 7
    angle: 0  # Horizontal
```

**Camera Shake**:
```yaml
- name: "shake"
  operation: "motion_blur"
  params:
    kernel_size: 3
    angle: 45
```

**Vertical Panning**:
```yaml
- name: "pan"
  operation: "motion_blur"
  params:
    kernel_size: 15
    angle: 90
```

## Using the Repeat Parameter

All operations support the `repeat` parameter (1-100) to apply the same operation multiple times sequentially:

```yaml
- name: "repeated_compression"
  operation: "compression"
  repeat: 5                 # Apply 5 times
  params:
    quality: 50
    subsampling: 2
```

### Repeat vs. Multi-Compress

- **`repeat` with `compression`**: Each iteration uses the same parameters (quality 50 five times)
- **`multi_compress`**: Supports quality decay curves (quality decreases across iterations)

**When to use each**:
- Use `repeat` for cumulative degradation with fixed settings
- Use `multi_compress` for progressive quality reduction

### Repeat Examples

**Repeated Downscaling**:
```yaml
- name: "cascade_downscale"
  operation: "downscale"
  repeat: 3
  params:
    scale: 0.8  # Each iteration: 80% of previous size
    upscale: false
```

**Multiple Blur Passes**:
```yaml
- name: "blur_stack"
  operation: "motion_blur"
  repeat: 3
  params:
    kernel_size: 2
    angle: 0
```

## Combining Operations

Create complex degradation patterns by chaining operations:

### Social Media Compression

```yaml
pipeline:
  steps:
    - name: "downscale"
      operation: "downscale"
      params:
        scale: 0.8

    - name: "multi_gen"
      operation: "multi_compress"
      params:
        iterations: 4
        quality_start: 75
        quality_end: 45
        decay: "linear"
```

### Glitch Art

```yaml
pipeline:
  steps:
    - name: "extreme_pixelation"
      operation: "downscale"
      params:
        scale: 0.08
        upscale: true
        upscale_method: "nearest"

    - name: "severe_compression"
      operation: "compression"
      params:
        quality: 5
        subsampling: 2

    - name: "multi_gen_glitch"
      operation: "multi_compress"
      params:
        iterations: 12
        quality_start: 30
        quality_end: 5
        decay: "exponential"
```

### VHS/Analog Degradation

```yaml
pipeline:
  steps:
    - name: "scanlines"
      operation: "motion_blur"
      params:
        kernel_size: 7
        angle: 0

    - name: "analog_compression"
      operation: "compression"
      params:
        quality: 35
        subsampling: 2

    - name: "resolution_loss"
      operation: "downscale"
      params:
        scale: 0.5
        upscale: true
        upscale_method: "bilinear"
```

## Best Practices

1. **Start subtle**: Begin with moderate parameters and increase intensity
2. **Preserve intermediates**: Check intermediate outputs to understand each step
3. **Experiment with order**: Operation order significantly affects results
4. **Use repeat wisely**: For progressive decay, prefer `multi_compress` over `repeat`

## Next Steps

- **Try hands-on examples**: [Compression Filters Tutorial](../tutorials/compression-filters/)
- **Explore all parameters**: [Filter Guide](../reference/filter-guide/)
- **Learn YAML system**: [Pipeline Documentation](../reference/pipeline/)

---

For detailed parameter ranges and visual examples, see the complete [Filter Guide](../reference/filter-guide/).

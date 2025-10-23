---
title: Filter Guide
parent: Reference
nav_order: 1
has_toc: true
---

# Filter Guide: Compression & Degradation Effects

This guide provides recommended parameter ranges for achieving specific visual degradation effects using the compression and degradation filters. All filters are deterministic and support elegant looping via the `repeat` parameter.

---

## Table of Contents

1. [JPEG Compression Artifacts](#jpeg-compression-artifacts)
2. [Multi-Generation Compression](#multi-generation-compression)
3. [Resolution Downscaling & Pixelation](#resolution-downscaling--pixelation)
4. [Motion Blur](#motion-blur)
5. [Combining Effects](#combining-effects)
6. [Using the Repeat Parameter](#using-the-repeat-parameter)

---

## JPEG Compression Artifacts

The `compression` operation applies JPEG compression with configurable quality and chroma subsampling to create block artifacts and color banding.

### Parameters

- `quality` (int, 1-100): JPEG quality level
- `subsampling` (int, 0/1/2): Chroma subsampling mode
  - `0` (4:4:4): No subsampling - highest quality
  - `1` (4:2:2): Moderate subsampling
  - `2` (4:2:0): Heavy subsampling - default, creates visible 8x8 blocks
- `optimize` (bool): Apply JPEG optimization (default: true)

### Recommended Ranges

#### Severe Compression Artifacts
*Heavy blocking/tiling, loss of high-frequency detail, color banding*

```yaml
- name: "severe_compression"
  operation: "compression"
  params:
    quality: 5          # Range: 1-15 for severe artifacts
    subsampling: 2      # 4:2:0 for maximum blocking
    optimize: false     # Disable optimization for heavier artifacts
```

**Best for**: Extreme degradation, visible 8x8 pixel blocks throughout

#### Moderate Compression
*Noticeable artifacts, especially in gradients and detailed areas*

```yaml
- name: "moderate_compression"
  operation: "compression"
  params:
    quality: 30         # Range: 16-50 for moderate compression
    subsampling: 2
```

**Best for**: Balanced degradation, some details preserved

#### Subtle Compression
*Mild artifacts, reduced file size with acceptable quality*

```yaml
- name: "subtle_compression"
  operation: "compression"
  params:
    quality: 70         # Range: 51-85 for subtle compression
    subsampling: 1      # 4:2:2 for less aggressive subsampling
```

**Best for**: Realistic compression without heavy artifacts

#### High Quality (Minimal Artifacts)
*Very subtle artifacts, primarily for file size reduction*

```yaml
- name: "high_quality"
  operation: "compression"
  params:
    quality: 95         # Range: 86-100
    subsampling: 0      # 4:4:4 - no chroma subsampling
```

**Best for**: Near-lossless compression

---

## Multi-Generation Compression

The `multi_compress` operation applies JPEG compression multiple times to simulate repeated save/load cycles, creating compound artifacts.

### Parameters

- `iterations` (int, 1-50): Number of compression cycles
- `quality_start` (int, 1-100): Starting quality level
- `quality_end` (int, 1-100): Ending quality level (for decay modes)
- `decay` (str): Quality decay type
  - `"fixed"`: Use same quality each iteration
  - `"linear"`: Quality decreases linearly
  - `"exponential"`: Quality decreases rapidly then levels off
- `subsampling` (int, 0/1/2): Chroma subsampling mode

### Recommended Ranges

#### Light Multi-Generation (2-3 cycles)
*Subtle compound artifacts*

```yaml
- name: "light_multi_gen"
  operation: "multi_compress"
  params:
    iterations: 3
    quality_start: 70
    quality_end: 50
    decay: "linear"
    subsampling: 2
```

**Visual effect**: Slight increase in blocking, some color banding

#### Moderate Multi-Generation (4-6 cycles)
*Noticeable compound degradation*

```yaml
- name: "moderate_multi_gen"
  operation: "multi_compress"
  params:
    iterations: 5
    quality_start: 50
    quality_end: 20
    decay: "linear"
    subsampling: 2
```

**Visual effect**: Clear blocking patterns, loss of fine detail

#### Heavy Multi-Generation (7-10 cycles)
*Severe compound artifacts*

```yaml
- name: "heavy_multi_gen"
  operation: "multi_compress"
  params:
    iterations: 8
    quality_start: 50
    quality_end: 10
    decay: "exponential"  # Rapid initial degradation
    subsampling: 2
```

**Visual effect**: Heavy blocking, significant color banding, loss of texture

#### Extreme Multi-Generation (10+ cycles)
*Maximum degradation*

```yaml
- name: "extreme_multi_gen"
  operation: "multi_compress"
  params:
    iterations: 15
    quality_start: 40
    quality_end: 5
    decay: "exponential"
    subsampling: 2
```

**Visual effect**: Extreme artifacts, abstract quality, barely recognizable details

---

## Resolution Downscaling & Pixelation

The `downscale` operation reduces image resolution to create pixelation effects.

### Parameters

- `scale` (float, 0.01-1.0): Scale factor for downscaling
- `upscale` (bool): Whether to upscale back to original size (default: true)
- `downscale_method` (str): Resampling method for downscaling
  - `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"`, `"box"`
- `upscale_method` (str): Resampling method for upscaling
  - `"nearest"` - Creates harsh, blocky pixelation
  - `"bilinear"` - Softer pixelation
  - `"bicubic"`, `"lanczos"` - Smooth upscaling

### Recommended Ranges

#### Extreme Pixelation
*Heavily degraded, architectural details lost*

```yaml
- name: "extreme_pixelation"
  operation: "downscale"
  params:
    scale: 0.05             # Range: 0.01-0.10
    upscale: true
    downscale_method: "bicubic"
    upscale_method: "nearest"  # Harsh blocks
```

**Visual effect**: Massive pixel blocks, abstract appearance

#### Heavy Pixelation
*Visible block structures, loss of detail*

```yaml
- name: "heavy_pixelation"
  operation: "downscale"
  params:
    scale: 0.15             # Range: 0.10-0.25
    upscale: true
    downscale_method: "bicubic"
    upscale_method: "nearest"
```

**Visual effect**: Large pixel blocks in windows, street furniture

#### Moderate Pixelation
*Noticeable quality reduction, some detail preserved*

```yaml
- name: "moderate_pixelation"
  operation: "downscale"
  params:
    scale: 0.35             # Range: 0.25-0.50
    upscale: true
    downscale_method: "bicubic"
    upscale_method: "nearest"
```

**Visual effect**: Pixelation visible in fine details

#### Subtle Quality Reduction
*Slight softening, reduced sharpness*

```yaml
- name: "subtle_reduction"
  operation: "downscale"
  params:
    scale: 0.70             # Range: 0.50-1.00
    upscale: true
    downscale_method: "bilinear"
    upscale_method: "bilinear"  # Soft upscaling
```

**Visual effect**: Soft focus effect, texture smoothing

---

## Motion Blur

The `motion_blur` operation applies directional blur to simulate camera movement or shake.

### Parameters

- `kernel_size` (int, 1-100): Blur strength in pixels
- `angle` (float, 0-360): Direction of motion in degrees
  - `0°`: Horizontal (left-right)
  - `90°`: Vertical (up-down)
  - `45°`/`135°`: Diagonal

### Recommended Ranges

#### Minimal Blur
*Very subtle shake effect*

```yaml
- name: "minimal_blur"
  operation: "motion_blur"
  params:
    kernel_size: 2          # Range: 1-3
    angle: 0                # Any angle
```

**Visual effect**: Barely perceptible blur, subtle edge softening

#### Subtle Blur
*Light camera shake*

```yaml
- name: "subtle_blur"
  operation: "motion_blur"
  params:
    kernel_size: 5          # Range: 3-8
    angle: 45               # Diagonal motion
```

**Visual effect**: Noticeable blur, simulates slight movement

#### Moderate Blur
*Noticeable motion effect*

```yaml
- name: "moderate_blur"
  operation: "motion_blur"
  params:
    kernel_size: 12         # Range: 8-20
    angle: 0                # Horizontal panning
```

**Visual effect**: Clear motion trail, edges significantly blurred

#### Heavy Blur
*Significant movement, dramatic effect*

```yaml
- name: "heavy_blur"
  operation: "motion_blur"
  params:
    kernel_size: 30         # Range: 20-50
    angle: 90               # Vertical motion
```

**Visual effect**: Strong motion blur, details heavily obscured

#### Extreme Blur
*Maximum motion effect*

```yaml
- name: "extreme_blur"
  operation: "motion_blur"
  params:
    kernel_size: 60         # Range: 50-100
    angle: 135              # Diagonal
```

**Visual effect**: Extreme streaking, abstract appearance

---

## Combining Effects

Combine multiple operations to achieve complex degradation patterns.

### Example 1: Compressed + Pixelated
*Simulates low-quality webcam or video chat*

```yaml
steps:
  - name: "compress"
    operation: "compression"
    params:
      quality: 20
      subsampling: 2

  - name: "pixelate"
    operation: "downscale"
    params:
      scale: 0.3
      upscale: true
      upscale_method: "nearest"
```

### Example 2: Multi-Gen Compression + Blur
*Simulates heavily degraded, repeatedly shared media*

```yaml
steps:
  - name: "multi_compress"
    operation: "multi_compress"
    params:
      iterations: 7
      quality_start: 60
      quality_end: 15
      decay: "exponential"

  - name: "slight_blur"
    operation: "motion_blur"
    params:
      kernel_size: 3
      angle: 0
```

### Example 3: Complete Degradation Pipeline
*All effects combined for maximum degradation*

```yaml
steps:
  - name: "downscale"
    operation: "downscale"
    params:
      scale: 0.25
      upscale: true
      upscale_method: "nearest"

  - name: "compress"
    operation: "compression"
    params:
      quality: 15
      subsampling: 2

  - name: "blur"
    operation: "motion_blur"
    params:
      kernel_size: 4
      angle: 45

  - name: "final_compress"
    operation: "multi_compress"
    params:
      iterations: 5
      quality_start: 30
      quality_end: 10
      decay: "linear"
```

---

## Using the Repeat Parameter

The `repeat` parameter allows any operation to be applied multiple times sequentially, creating elegant loops in YAML.

### Basic Repeat Usage

```yaml
- name: "repeated_compression"
  operation: "compression"
  repeat: 5                 # Apply 5 times
  params:
    quality: 50
    subsampling: 2
```

**Effect**: Simulates 5 save/load cycles at the same quality level.

### Repeat vs. Multi-Compress

- **`repeat` with `compression`**: Each iteration uses the same parameters
- **`multi_compress`**: Supports quality decay curves across iterations

Use `repeat` when you want consistent degradation each cycle. Use `multi_compress` when you want progressive quality reduction.

### Advanced Repeat Examples

#### Repeated Downscaling
*Progressive resolution loss*

```yaml
- name: "cascade_downscale"
  operation: "downscale"
  repeat: 3
  params:
    scale: 0.8              # Each iteration: 80% of previous size
    upscale: false
```

#### Repeated Blur + Compression
*Combine repeat on multiple steps*

```yaml
steps:
  - name: "blur_cycle"
    operation: "motion_blur"
    repeat: 3
    params:
      kernel_size: 2
      angle: 0

  - name: "compress_cycle"
    operation: "compression"
    repeat: 4
    params:
      quality: 40
      subsampling: 2
```

---

## Quick Reference Table

| Effect                    | Operation          | Key Parameters                                 |
|---------------------------|--------------------|------------------------------------------------|
| Severe blocking           | `compression`      | quality: 1-15, subsampling: 2                  |
| Multi-gen artifacts       | `multi_compress`   | iterations: 5-10, decay: exponential           |
| Heavy pixelation          | `downscale`        | scale: 0.1-0.25, upscale_method: nearest       |
| Minimal blur              | `motion_blur`      | kernel_size: 1-3                               |
| Moderate blur             | `motion_blur`      | kernel_size: 8-20                              |
| Complete degradation      | All combined       | See "Complete Degradation Pipeline" above      |

---

## Tips for Achieving Specific Goals

### Goal: Simulate Social Media Compression
- Use `multi_compress` with 3-5 iterations
- Start quality: 70, end quality: 40
- Linear decay

### Goal: Create Abstract, Glitch-Art Aesthetic
- Extreme pixelation (scale: 0.05-0.1)
- Severe compression (quality: 1-10)
- Heavy multi-generation (10+ iterations)

### Goal: Simulate VHS/Analog Video Degradation
- Moderate blur (kernel_size: 5-10, angle: 0 for scan lines)
- Moderate compression (quality: 30-50)
- Slight pixelation (scale: 0.4-0.6)

### Goal: Minimal, Subtle Degradation
- Compression quality: 70-85
- Minimal blur: kernel_size: 2-3
- Subtle downscale: scale: 0.7-0.9

---

## Practical Usage Examples

These real-world examples demonstrate complete pipelines for specific aesthetic goals. Each example has been integration-tested to ensure reliable results.

### Example 1: Social Media Compression Simulation

**Goal**: Simulate an image that has been shared multiple times on social media platforms, each time being recompressed.

**Use case**: Creating realistic degradation that mimics WhatsApp, Instagram, or Facebook sharing chains.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 10.0
  end: 12.0
  interval: 1.0

steps:
  - name: "social_media_compression"
    operation: "multi_compress"
    params:
      iterations: 4                 # Simulates 4 share cycles
      quality_start: 75             # First share at 75% quality
      quality_end: 45               # Final share at 45% quality
      decay: "linear"               # Consistent quality drop each share
      subsampling: 2                # Heavy subsampling for mobile compression

output:
  base_dir: "output/social_media"
  final_dir: "output/social_media/final"
  intermediate_dir: "output/social_media/intermediate"
```

**Expected result**: Noticeable JPEG blocking, color banding in gradients, loss of fine detail - characteristic of images that have been through multiple social media platforms.

---

### Example 2: Glitch Art Aesthetic

**Goal**: Create extreme digital artifacts for an abstract, glitch art aesthetic.

**Use case**: Artistic projects exploring digital degradation, book illustrations with surreal/distorted imagery.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 5.0
  end: 8.0
  interval: 0.5

steps:
  # Step 1: Extreme pixelation
  - name: "extreme_pixelation"
    operation: "downscale"
    params:
      scale: 0.08                   # Reduce to 8% of original size
      upscale: true
      downscale_method: "bicubic"   # Smooth downscaling
      upscale_method: "nearest"     # Harsh, blocky upscaling

  # Step 2: Severe compression
  - name: "severe_compression"
    operation: "compression"
    params:
      quality: 5                    # Very low quality
      subsampling: 2                # Maximum blocking

  # Step 3: Heavy multi-generation compression
  - name: "multi_generation_glitch"
    operation: "multi_compress"
    params:
      iterations: 12                # Many compression cycles
      quality_start: 30
      quality_end: 5
      decay: "exponential"          # Rapid initial degradation

output:
  base_dir: "output/glitch_art"
  final_dir: "output/glitch_art/final"
  intermediate_dir: "output/glitch_art/intermediate"
```

**Expected result**: Extreme pixel blocks, severe JPEG artifacts, abstract appearance where original details are barely recognizable. Perfect for glitch art or heavily stylized book illustrations.

---

### Example 3: VHS/Analog Video Degradation

**Goal**: Simulate the characteristic degradation of VHS tapes and analog video recordings.

**Use case**: Nostalgic aesthetic, retro video game art, 90s-inspired visuals.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 15.0
  end: 20.0
  interval: 0.25

steps:
  # Step 1: Horizontal motion blur (scan lines effect)
  - name: "scanline_blur"
    operation: "motion_blur"
    params:
      kernel_size: 7                # Moderate blur
      angle: 0                      # Horizontal motion

  # Step 2: Moderate compression
  - name: "analog_compression"
    operation: "compression"
    params:
      quality: 35                   # Medium-low quality
      subsampling: 2                # Heavy subsampling

  # Step 3: Slight pixelation
  - name: "resolution_reduction"
    operation: "downscale"
    params:
      scale: 0.5                    # Half resolution
      upscale: true
      downscale_method: "bilinear"  # Softer downscaling
      upscale_method: "bilinear"    # Softer upscaling

output:
  base_dir: "output/vhs_analog"
  final_dir: "output/vhs_analog/final"
  intermediate_dir: "output/vhs_analog/intermediate"
```

**Expected result**: Horizontal blur reminiscent of scan lines, moderate pixelation, compression artifacts - characteristic of VHS tape playback or analog video recording.

---

### Example 4: Subtle Documentary Degradation

**Goal**: Apply minimal degradation to give images a "found footage" or documentary quality without heavy distortion.

**Use case**: Photo essays, documentary-style books, subtle artistic treatment preserving most detail.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 30.0
  end: 35.0
  interval: 1.0

steps:
  # Step 1: Minimal blur for slight softness
  - name: "subtle_blur"
    operation: "motion_blur"
    params:
      kernel_size: 2                # Very minimal blur
      angle: 45                     # Slight diagonal shake

  # Step 2: Light compression
  - name: "light_compression"
    operation: "compression"
    params:
      quality: 75                   # Good quality with subtle artifacts
      subsampling: 1                # Moderate subsampling

  # Step 3: Slight quality reduction through multi-generation
  - name: "aging_effect"
    operation: "multi_compress"
    params:
      iterations: 2                 # Just 2 cycles
      quality_start: 70
      quality_end: 60
      decay: "linear"

output:
  base_dir: "output/documentary"
  final_dir: "output/documentary/final"
  intermediate_dir: "output/documentary/intermediate"
```

**Expected result**: Subtle softening, barely noticeable compression artifacts, maintains detail while giving images a slightly aged or "found" quality.

---

### Example 5: Progressive Cascade Degradation

**Goal**: Create a series of images showing progressive degradation stages, useful for comparison or artistic exploration.

**Use case**: Demonstrating degradation effects, creating visual sequences, artistic exploration of digital decay.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 0.0
  end: 2.0
  interval: 0.5

steps:
  # Pipeline 1: Light degradation
  - name: "stage_1_light"
    operation: "compression"
    params:
      quality: 70
      subsampling: 1

  # Pipeline 2: Moderate degradation (using repeat)
  - name: "stage_2_moderate"
    operation: "compression"
    repeat: 2                       # Apply compression twice
    params:
      quality: 50
      subsampling: 2

  # Pipeline 3: Heavy degradation
  - name: "stage_3_heavy"
    operation: "multi_compress"
    params:
      iterations: 5
      quality_start: 40
      quality_end: 20
      decay: "exponential"

output:
  base_dir: "output/cascade"
  final_dir: "output/cascade/final"
  intermediate_dir: "output/cascade/intermediate"
```

**Expected result**: Three distinct degradation stages visible in intermediate outputs, showing progression from light to heavy artifacts.

---

### Example 6: Architectural Detail Destruction

**Goal**: Heavily degrade architectural elements while maintaining recognizable structure.

**Use case**: Exploring abstraction of urban spaces, creating surreal architectural imagery.

```yaml
source:
  youtube_url: "https://youtube.com/watch?v=YOUR_VIDEO"

segment:
  start: 45.0
  end: 50.0
  interval: 1.0

steps:
  # Step 1: Heavy pixelation targeting architectural details
  - name: "pixelate_architecture"
    operation: "downscale"
    params:
      scale: 0.15                   # Heavy pixelation
      upscale: true
      downscale_method: "bicubic"
      upscale_method: "nearest"     # Blocky pixels

  # Step 2: Compression to create blocking artifacts
  - name: "block_artifacts"
    operation: "compression"
    params:
      quality: 20                   # Low quality
      subsampling: 2

  # Step 3: Minimal blur to soften harsh edges
  - name: "soften_edges"
    operation: "motion_blur"
    params:
      kernel_size: 3                # Minimal blur
      angle: 0

  # Step 4: Final multi-generation compression
  - name: "final_degradation"
    operation: "multi_compress"
    params:
      iterations: 5
      quality_start: 40
      quality_end: 15
      decay: "linear"

output:
  base_dir: "output/architecture"
  final_dir: "output/architecture/final"
  intermediate_dir: "output/architecture/intermediate"
```

**Expected result**: Large pixel blocks in windows and street furniture, heavy JPEG blocking, structural elements still recognizable but heavily abstracted.

---

## Using Repeat for Flexible Sequences

The `repeat` parameter provides elegant control over how many times an operation is applied, making it easy to create complex degradation sequences.

### Repeat Parameter Benefits

1. **Cleaner YAML**: No need to duplicate operation steps
2. **Consistent Parameters**: Same settings applied multiple times
3. **Range**: 1-100 repetitions supported per step
4. **Deterministic**: Always produces the same result with same parameters

### Example: Comparing Repeat vs. Multi-Compress

**Using repeat with standard compression**:
```yaml
- name: "repeated_compression"
  operation: "compression"
  repeat: 5
  params:
    quality: 50                     # Same quality each iteration
    subsampling: 2
```

**Using multi-compress with decay**:
```yaml
- name: "decaying_compression"
  operation: "multi_compress"
  params:
    iterations: 5
    quality_start: 50               # Starts at 50
    quality_end: 30                 # Ends at 30
    decay: "linear"                 # Progressive decay
```

**When to use each**:
- Use `repeat` for consistent, cumulative degradation
- Use `multi_compress` for progressive quality reduction with decay curves

---

## Notes on Non-Destructive Editing

All operations in this pipeline maintain incremental filenames to preserve the workflow history:

- Each step saves output with a unique name: `{step_name}_{frame_stem}_step{index:02d}.jpg`
- Intermediate results are saved to `intermediate_dir`
- Final results are saved to `final_dir`
- Original frames are never modified

This allows you to review each stage of degradation and adjust parameters as needed.

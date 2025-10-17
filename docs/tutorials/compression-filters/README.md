# Tutorial: Compression & Degradation Filters

This tutorial demonstrates compression and degradation filters through hands-on examples. Each tutorial processes a 3-second video segment at 15 frames per second, extracting 45 frames for processing.

## Prerequisites

- sevenrad-stills installed and configured
- YouTube video URL (or use the example: `https://www.youtube.com/watch?v=MzJaP-7N9I0`)
- Basic familiarity with YAML pipeline system (see [docs/PIPELINE.md](../../PIPELINE.md))

## Tutorial Overview

| Tutorial | Operations | Purpose | Complexity |
|----------|-----------|---------|------------|
| [01-social-media](#tutorial-1-social-media-compression) | multi_compress | Demonstrates multi-generation compression artifacts | Beginner |
| [02-glitch-art](#tutorial-2-extreme-degradation) | downscale, compression, multi_compress | Shows cumulative degradation from multiple operations | Intermediate |
| [03-vhs-analog](#tutorial-3-motion-blur-and-compression) | motion_blur, compression, downscale | Combines directional blur with compression | Intermediate |
| [04-progressive-cascade](#tutorial-4-progressive-degradation-stages) | compression (with repeat) | Demonstrates progressive quality reduction stages | Advanced |

## Common Segment Configuration

All tutorials use the same video segment for consistency:

```yaml
segment:
  start: 192.0    # 3 minutes 12 seconds
  end: 195.0      # 3 minutes 15 seconds
  interval: 0.0667  # 15 frames per second = 45 total frames
```

**Why these settings?**
- **3 seconds**: Enough variety without overwhelming output
- **15 fps**: Good balance between sample size and processing time
- **45 frames**: Substantial dataset to see filter effects across multiple frames

---

## Tutorial 1: Social Media Compression

**Goal**: Demonstrate multi-generation JPEG compression with progressive quality decay.

**Demonstrates**: How iterative compression creates cumulative artifacts and quality loss.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/compression-filters/01-social-media.yaml
```

### Expected Results

**Output**: `tutorials/01-social-media/final/` containing 45 images

**Visual Characteristics**:
- Noticeable JPEG blocking patterns (8x8 pixel blocks)
- Color banding in gradients and sky areas
- Loss of fine details in textures and edges
- Overall "compressed" look familiar from social media

**Example Output:**

![Social Media Compression Result](images/01-social-media-result.jpg)
*Result after 4 social media sharing cycles (quality 75 → 45)*

### Pipeline Breakdown

```yaml
steps:
  - name: "social_media_compression"
    operation: "multi_compress"
    params:
      iterations: 4                 # Simulates 4 share cycles
      quality_start: 75             # First share at 75% quality
      quality_end: 45               # Final share at 45% quality
      decay: "linear"               # Consistent quality drop
      subsampling: 2                # Heavy subsampling (4:2:0)
```

**Parameter Explanation**:
- `iterations: 4` - Each iteration represents one share/recompress cycle
- `quality_start: 75` - Mobile apps often compress to ~75% initially
- `quality_end: 45` - After 4 shares, quality degrades significantly
- `decay: "linear"` - Quality drops evenly: 75 → 65 → 55 → 45
- `subsampling: 2` - Heavy chroma subsampling creates visible 8x8 blocks

### What You'll Learn

- How multi-generation compression creates compound artifacts
- The impact of chroma subsampling on image quality
- Realistic degradation patterns from social media platforms

---

## Tutorial 2: Extreme Degradation

**Goal**: Apply extreme pixelation and compression to create heavy digital artifacts.

**Demonstrates**: Cumulative effects of combining downscaling, compression, and multi-generation processing.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/compression-filters/02-glitch-art.yaml
```

### Expected Results

**Output**: `tutorials/02-glitch-art/final/` containing 45 heavily distorted images

**Intermediate Steps** (preserved for inspection):
- `extreme_pixelation/` - 45 pixelated frames
- `severe_compression/` - 45 compressed frames

**Visual Characteristics**:
- Large pixel blocks from aggressive downscaling
- Severe JPEG compression artifacts (8x8 blocks)
- Color banding in smooth gradients
- Significant detail loss from multi-generation processing
- Heavy digital degradation

**Visual Progression:**

![Glitch Art Original](images/02-glitch-art-original.jpg)
*Original extracted frame*

![Glitch Art Step 1](images/02-glitch-art-step1-pixelation.jpg)
*After Step 1: Extreme Pixelation (scale: 0.08)*

![Glitch Art Step 2](images/02-glitch-art-step2-compression.jpg)
*After Step 2: Severe Compression (quality: 5)*

![Glitch Art Final](images/02-glitch-art-final.jpg)
*Final Result: After 12 multi-generation compression cycles*

### Pipeline Breakdown

```yaml
steps:
  # Step 1: Extreme pixelation
  - name: "extreme_pixelation"
    operation: "downscale"
    params:
      scale: 0.08                   # Reduce to 8% of original size
      upscale: true                 # Then scale back up
      downscale_method: "bicubic"   # Smooth reduction
      upscale_method: "nearest"     # Harsh, blocky enlargement

  # Step 2: Severe compression
  - name: "severe_compression"
    operation: "compression"
    params:
      quality: 5                    # Very low quality
      subsampling: 2                # Maximum blocking

  # Step 3: Multi-generation compression
  - name: "multi_generation_glitch"
    operation: "multi_compress"
    params:
      iterations: 12                # Many compression cycles
      quality_start: 30
      quality_end: 5
      decay: "exponential"          # Rapid initial degradation
```

**Parameter Explanation**:
- `scale: 0.08` - Reduces image to 8% (e.g., 1920x1080 → 154x86 → 1920x1080)
- `upscale_method: "nearest"` - Creates harsh pixel blocks instead of smooth interpolation
- `quality: 5` - Near-minimum JPEG quality for extreme artifacts
- `iterations: 12` - Aggressive multi-generation for compound distortion
- `decay: "exponential"` - Most degradation happens in early iterations

### What You'll Learn

- Combining multiple operations for cumulative effects
- How resampling methods (bicubic vs nearest) affect pixelation
- Using exponential decay for accelerated quality degradation
- Inspecting intermediate results to understand processing stages

---

## Tutorial 3: Motion Blur and Compression

**Goal**: Apply directional motion blur combined with compression and downscaling.

**Demonstrates**: Using motion blur at specific angles to create directional effects, combined with quality reduction.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/compression-filters/03-vhs-analog.yaml
```

### Expected Results

**Output**: `tutorials/03-vhs-analog/final/` containing 45 frames with VHS aesthetic

**Intermediate Steps**:
- `scanline_blur/` - Horizontal blur applied
- `analog_compression/` - JPEG compression added

**Visual Characteristics**:
- Horizontal motion blur (0° angle)
- Moderate JPEG compression artifacts
- Resolution reduction from 50% downscaling
- Soft edges from bilinear resampling
- Combined blur and compression effects

**Example Output:**

![Motion Blur Result](images/03-vhs-analog-result.jpg)
*Final result showing combined motion blur, compression, and downscaling*

### Pipeline Breakdown

```yaml
steps:
  # Step 1: Horizontal motion blur (scan lines)
  - name: "scanline_blur"
    operation: "motion_blur"
    params:
      kernel_size: 7                # Moderate blur
      angle: 0                      # Horizontal direction

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
      downscale_method: "bilinear"  # Softer downscale
      upscale_method: "bilinear"    # Softer upscale
```

**Parameter Explanation**:
- `kernel_size: 7` - Moderate blur strength for visible motion effect
- `angle: 0` - Pure horizontal motion (0° = left-right blur)
- `quality: 35` - Medium-low compression (creates moderate artifacts)
- `scale: 0.5` - Reduces to 50% resolution (e.g., 1920x1080 → 960x540 → 1920x1080)
- `bilinear` methods - Softer resampling creates smooth transitions instead of hard edges

### What You'll Learn

- Applying motion blur at specific angles (0° to 360°)
- Balancing compression and downscaling parameters
- Combining blur, compression, and resolution changes in sequence

---

## Tutorial 4: Progressive Degradation Stages

**Goal**: Create progressive quality reduction stages from light to heavy compression.

**Demonstrates**: Using the `repeat` parameter and comparing different compression intensities across stages.

### Running the Tutorial

```bash
sevenrad pipeline docs/tutorials/compression-filters/04-progressive-cascade.yaml
```

### Expected Results

**Output**: `tutorials/04-progressive-cascade/final/` containing 45 heavily degraded images

**Intermediate Steps** (showing progression):
- `stage_1_light/` - Light compression (45 frames)
- `stage_2_moderate/` - Moderate compression applied twice (45 frames)

**Visual Characteristics**:
- **Stage 1**: Subtle artifacts, most details preserved
- **Stage 2**: More noticeable blocking, some detail loss
- **Stage 3**: Heavy artifacts, significant quality reduction
- Clear visual progression across all three stages

**Progressive Degradation Stages:**

![Stage 1: Light](images/04-cascade-stage1-light.jpg)
*Stage 1: Light compression (quality: 70, subsampling: 1)*

![Stage 2: Moderate](images/04-cascade-stage2-moderate.jpg)
*Stage 2: Moderate compression applied twice (quality: 50, repeat: 2)*

![Stage 3: Heavy](images/04-cascade-stage3-heavy.jpg)
*Stage 3: Heavy multi-generation compression (5 iterations, exponential decay)*

### Pipeline Breakdown

```yaml
steps:
  # Stage 1: Light degradation
  - name: "stage_1_light"
    operation: "compression"
    params:
      quality: 70
      subsampling: 1

  # Stage 2: Moderate degradation (using repeat)
  - name: "stage_2_moderate"
    operation: "compression"
    repeat: 2                       # Apply compression twice
    params:
      quality: 50
      subsampling: 2

  # Stage 3: Heavy degradation
  - name: "stage_3_heavy"
    operation: "multi_compress"
    params:
      iterations: 5
      quality_start: 40
      quality_end: 20
      decay: "exponential"
```

**Parameter Explanation**:
- `quality: 70` - Light compression maintains good quality
- `subsampling: 1` - Moderate subsampling (4:2:2) less aggressive than 4:2:0
- `repeat: 2` - Applies the same compression twice for cumulative effect
- `quality: 50` - Lower quality for second stage
- `iterations: 5` - Final stage uses multi-compression with decay
- `decay: "exponential"` - Rapid quality drop for heavy final artifacts

### What You'll Learn

- Using the `repeat` parameter for cumulative degradation
- Difference between `repeat` and `multi_compress` with decay
- Creating visual progressions by chaining operations
- Inspecting intermediate results to understand each stage

---

## Advanced Topics

### The Repeat Parameter

The `repeat` parameter (1-100) applies any operation multiple times sequentially:

```yaml
- name: "repeated_compression"
  operation: "compression"
  repeat: 5                 # Apply 5 times
  params:
    quality: 50             # Same quality each iteration
```

**When to use `repeat` vs. `multi_compress`:**
- **Use `repeat`**: When you want consistent parameters each iteration
- **Use `multi_compress`**: When you want progressive quality decay

### Combining Effects

All filters can be combined in any order. Order matters:

```yaml
# Pixelate THEN compress (visible pixel blocks with compression)
steps:
  - operation: "downscale"
  - operation: "compression"

# Compress THEN pixelate (compression artifacts enlarged by pixelation)
steps:
  - operation: "compression"
  - operation: "downscale"
```

### Parameter Tuning Tips

**For subtle effects:**
- Compression quality: 70-85
- Downscale scale: 0.7-0.9
- Motion blur kernel_size: 2-3
- Multi_compress iterations: 2-3

**For extreme effects:**
- Compression quality: 1-15
- Downscale scale: 0.05-0.15
- Motion blur kernel_size: 20-50
- Multi_compress iterations: 10-20

See [docs/FILTER_GUIDE.md](../../FILTER_GUIDE.md) for comprehensive parameter ranges.

---

## Understanding Output Structure

All tutorials follow this non-destructive output structure:

```
tutorials/
├── 01-social-media/
│   ├── intermediate/          # No intermediate steps
│   └── final/
│       └── social_media_compression_*_step00.jpg (60 images)
│
├── 02-glitch-art/
│   ├── intermediate/
│   │   ├── extreme_pixelation/
│   │   │   └── extreme_pixelation_*_step00.jpg (60 images)
│   │   └── severe_compression/
│   │       └── severe_compression_*_step01.jpg (60 images)
│   └── final/
│       └── multi_generation_glitch_*_step02.jpg (60 images)
│
└── ... (similar structure for other tutorials)
```

**Key points:**
- Each step saves to its own directory
- Intermediate steps preserved for inspection
- Final output always in `final/` directory
- Filenames include step number and operation name
- Original frames never modified

---

## Troubleshooting

### Not Enough Frames Extracted

**Problem**: Expected 60 frames but got fewer

**Solutions**:
1. Check video duration - ensure video is longer than 3m18s
2. Verify YouTube URL is correct and accessible
3. Check segment times don't exceed video length

### Quality Not Degrading as Expected

**Problem**: Images don't show enough artifacts

**Solutions**:
1. Check `quality` parameter - lower values create more artifacts
2. Ensure `subsampling: 2` for maximum blocking
3. Increase `iterations` in multi_compress
4. Try exponential decay instead of linear

### Pipeline Runs Too Slowly

**Problem**: Processing 60 frames takes too long

**Solutions**:
1. Enable parallel processing (default): `sevenrad pipeline tutorial.yaml`
2. Reduce frame count by increasing `interval` (e.g., `interval: 0.2` for 30 frames)
3. Use fewer operations or lower iterations
4. Check system resources

### Operation Not Found Error

**Problem**: `Operation 'compression' not found`

**Solutions**:
1. Ensure you're using the latest version with compression filters
2. Check operation names match exactly: `compression`, `downscale`, `motion_blur`, `multi_compress`
3. Verify operations are registered in `src/sevenrad_stills/operations/__init__.py`

### YAML Syntax Errors

**Problem**: `Invalid YAML syntax` or indentation errors

**Solutions**:
1. Check YAML indentation - use spaces, not tabs
2. Ensure proper list syntax with `-` for each step
3. Validate YAML with online validator
4. Compare to working examples in this tutorial

---

## Next Steps

After completing these tutorials:

1. **Experiment with parameters** - Adjust quality levels, iterations, and scales
2. **Create custom pipelines** - Combine operations in different sequences
3. **Try different segments** - Process various video sections or still images
4. **Test parameter ranges** - Explore the full range of available settings
5. **Read comprehensive docs**:
   - [FILTER_GUIDE.md](../../FILTER_GUIDE.md) - Complete parameter reference
   - [PIPELINE.md](../../PIPELINE.md) - Pipeline system documentation

## Questions or Issues?

- Check [FILTER_GUIDE.md](../../FILTER_GUIDE.md) for parameter recommendations
- Review [PIPELINE.md](../../PIPELINE.md) for pipeline system details
- Examine working examples in `examples/` directory
- Report issues on the project repository

---

---

**Note**: All operations are deterministic and reproducible with the same parameters and source material.

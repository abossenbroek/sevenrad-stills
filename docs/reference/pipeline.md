---
title: YAML Pipeline System
parent: Reference
nav_order: 2
has_toc: true
---

# YAML Pipeline System

The YAML Pipeline System enables declarative configuration of video processing workflows, from YouTube download through frame extraction to image transformations.

## Overview

Pipelines are defined in YAML files that specify:
- **Source**: YouTube video URL
- **Segment**: Time range and extraction interval
- **Operations**: Sequence of image transformations
- **Output**: Directory structure for results

## Quick Start

### 1. Create a Pipeline YAML File

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

segment:
  start: 10.0    # Start at 10 seconds
  end: 30.0      # End at 30 seconds
  interval: 1.0  # Extract one frame per second

pipeline:
  steps:
    - name: "boost_saturation"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.5  # 150% saturation

output:
  base_dir: "./pipeline_output"
  intermediate_dir: "./pipeline_output/intermediate"
  final_dir: "./pipeline_output/final"
```

### 2. Run the Pipeline

```bash
sevenrad pipeline my_pipeline.yaml
```

## Configuration Reference

### Source Section

Specifies the video source.

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Fields:**
- `youtube_url` (string, required): YouTube video URL

### Segment Section

Defines which portion of the video to extract and process.

```yaml
segment:
  start: 10.0     # seconds
  end: 30.0       # seconds
  interval: 1.0   # seconds between frames
```

**Fields:**
- `start` (float, required): Start time in seconds (≥ 0)
- `end` (float, required): End time in seconds (> start)
- `interval` (float, required): Time between extracted frames (> 0)

**Example calculations:**
- `start: 10.0, end: 30.0, interval: 1.0` → 20 frames (1 per second for 20 seconds)
- `start: 0.0, end: 10.0, interval: 0.5` → 20 frames (2 per second for 10 seconds)

### Pipeline Section

Defines the sequence of image operations to apply.

```yaml
pipeline:
  steps:
    - name: "step_name"
      operation: "operation_type"
      params:
        # operation-specific parameters
```

**Fields:**
- `steps` (list, required): List of operation steps (at least one required)
- `name` (string, required): Human-readable step name (used in output filenames)
- `operation` (string, required): Operation type identifier
- `params` (dict, optional): Operation-specific parameters

### Output Section

Configures output directory structure.

```yaml
output:
  base_dir: "./pipeline_output"
  intermediate_dir: "./pipeline_output/intermediate"
  final_dir: "./pipeline_output/final"
```

**Fields:**
- `base_dir` (path, optional): Base directory for all outputs
- `intermediate_dir` (path, optional): Directory for intermediate step outputs
- `final_dir` (path, optional): Directory for final outputs

**Default values:**
- `base_dir`: `./pipeline_output`
- `intermediate_dir`: `./pipeline_output/intermediate`
- `final_dir`: `./pipeline_output/final`

## Available Operations

### Saturation

Adjusts image color saturation using either fixed or random values.

> **Note**: For comprehensive documentation on compression and degradation operations, see the [Compression Filters Tutorial](../tutorials/compression-filters/) and [Filter Guide](filter-guide/).

#### Fixed Mode

Apply a specific saturation multiplier to all frames.

```yaml
- name: "boost_colors"
  operation: "saturation"
  params:
    mode: "fixed"
    value: 1.5  # 1.5x saturation (150%)
```

**Parameters:**
- `mode`: `"fixed"`
- `value` (float): Saturation multiplier
  - `0.0`: Grayscale (no saturation)
  - `1.0`: Original saturation
  - `> 1.0`: Increased saturation
  - Must be ≥ 0

#### Random Mode

Apply random saturation variation within a range.

```yaml
- name: "vary_saturation"
  operation: "saturation"
  params:
    mode: "random"
    range: [-0.5, 0.5]  # Random variation ±50%
```

**Parameters:**
- `mode`: `"random"`
- `range` (list of 2 floats): `[min, max]` adjustment range
  - Values are added to 1.0 base saturation
  - Example: `[-0.5, 0.5]` produces values from 0.5 to 1.5
  - `min` must be ≥ -1.0
  - `max` must be > `min`

### Compression & Degradation Operations

Four operations provide comprehensive control over image degradation and compression artifacts. See [Compression Filters Tutorial](../tutorials/compression-filters/) for hands-on examples and [Filter Guide](filter-guide/) for complete parameter documentation.

#### Compression

Apply JPEG compression with configurable quality and chroma subsampling to create block artifacts.

```yaml
- name: "compress"
  operation: "compression"
  params:
    quality: 50         # 1-100 (lower = more artifacts)
    subsampling: 2      # 0 (4:4:4), 1 (4:2:2), or 2 (4:2:0)
    optimize: true      # Optional: JPEG optimization
```

**Parameters:**
- `quality` (int, 1-100): JPEG quality level (lower creates more artifacts)
- `subsampling` (int, 0/1/2): Chroma subsampling mode
  - `0` (4:4:4): No subsampling - highest quality
  - `1` (4:2:2): Moderate subsampling
  - `2` (4:2:0): Heavy subsampling - creates visible 8x8 blocks (default)
- `optimize` (bool, optional): Apply JPEG optimization (default: true)

**Use cases:** Single-pass compression, specific quality targets, testing subsampling modes

#### Multi-Compress

Apply JPEG compression multiple times with quality decay to simulate multi-generation compression.

```yaml
- name: "multi_gen"
  operation: "multi_compress"
  params:
    iterations: 5
    quality_start: 60
    quality_end: 20
    decay: "linear"     # "fixed", "linear", or "exponential"
    subsampling: 2
```

**Parameters:**
- `iterations` (int, 1-50): Number of compression cycles
- `quality_start` (int, 1-100): Starting quality level
- `quality_end` (int, 1-100): Ending quality (for decay modes, must be < quality_start)
- `decay` (str): Quality decay type
  - `"fixed"`: Same quality each iteration
  - `"linear"`: Quality decreases evenly
  - `"exponential"`: Rapid initial degradation, then levels off
- `subsampling` (int, 0/1/2, optional): Chroma subsampling mode (default: 2)

**Use cases:** Social media compression simulation, compound artifacts, progressive degradation

#### Downscale

Reduce image resolution with configurable resampling methods to create pixelation effects.

```yaml
- name: "pixelate"
  operation: "downscale"
  params:
    scale: 0.25
    upscale: true
    downscale_method: "bicubic"
    upscale_method: "nearest"
```

**Parameters:**
- `scale` (float, 0.01-1.0): Scale factor (e.g., 0.25 = 25% of original size)
- `upscale` (bool, optional): Whether to upscale back to original size (default: true)
- `downscale_method` (str, optional): Resampling for downscaling
  - Options: `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"`, `"box"`
  - Default: `"bicubic"`
- `upscale_method` (str, optional): Resampling for upscaling
  - `"nearest"`: Creates harsh, blocky pixelation
  - `"bilinear"`: Softer pixelation
  - `"bicubic"`, `"lanczos"`: Smooth upscaling
  - Default: `"bilinear"`

**Use cases:** Pixelation effects, resolution reduction, retro aesthetics

#### Motion Blur

Apply directional motion blur to simulate camera movement or shake.

```yaml
- name: "blur"
  operation: "motion_blur"
  params:
    kernel_size: 10
    angle: 45
```

**Parameters:**
- `kernel_size` (int, 1-100): Blur strength in pixels
  - 1-3: Minimal blur, subtle shake
  - 3-8: Moderate blur
  - 8-20: Heavy blur
  - 20+: Extreme blur
- `angle` (float, 0-360, optional): Direction of motion in degrees
  - `0`: Horizontal (left-right)
  - `90`: Vertical (up-down)
  - `45`/`135`: Diagonal
  - Default: `0`

**Use cases:** Camera shake simulation, scan line effects, motion artifacts

### Using the Repeat Parameter

All operations support the `repeat` parameter (1-100) to apply the same operation multiple times:

```yaml
- name: "repeated_compression"
  operation: "compression"
  repeat: 5                 # Apply 5 times
  params:
    quality: 50
```

**Repeat vs. Multi-Compress:**
- **`repeat`**: Applies same parameters each iteration (e.g., quality 50 five times)
- **`multi_compress`**: Applies changing parameters with decay curves

Use `repeat` for cumulative degradation with fixed settings. Use `multi_compress` for progressive quality decay.

## Multi-Step Pipelines

Operations are applied sequentially. Each step's output becomes the next step's input.

```yaml
pipeline:
  steps:
    # Step 1: Initial variation
    - name: "initial_variation"
      operation: "saturation"
      params:
        mode: "random"
        range: [-0.3, 0.3]

    # Step 2: Boost overall saturation
    - name: "boost"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.2

    # Step 3: Final random variation
    - name: "final_variation"
      operation: "saturation"
      params:
        mode: "random"
        range: [-0.2, 0.2]
```

**Output structure:**
```
pipeline_output/
├── intermediate/
│   ├── initial_variation/
│   │   └── initial_variation_*_step00.jpg
│   └── boost/
│       └── boost_*_step01.jpg
└── final/
    └── final_variation_*_step02.jpg
```

## Examples

### Example Pipelines

See the `examples/` directory for complete pipeline examples:

**Saturation Operations:**
- `saturation_pipeline.yaml`: Random saturation adjustment
- `saturation_fixed_pipeline.yaml`: Fixed saturation boost
- `multi_step_pipeline.yaml`: Multi-step processing chain

**Compression & Degradation Operations:**
- `severe_degradation_pipeline.yaml`: All effects combined for extreme artifacts
- `multi_generation_compression.yaml`: Multi-compress with decay curves
- `compression_block_sizes.yaml`: Subsampling modes demonstration
- `minimal_blur_sequence.yaml`: Minimal blur capabilities
- `flexible_sequence_repeat.yaml`: Repeat parameter showcase

### Interactive Tutorials

For hands-on learning with the compression filters, see the [Compression Filters Tutorial](../tutorials/compression-filters/):

1. **Social Media Compression** - Simulate sharing degradation
2. **Glitch Art Aesthetic** - Extreme digital artifacts
3. **VHS/Analog Degradation** - Nostalgic retro effects
4. **Progressive Cascade** - Degradation progression

All tutorials use the same 6-second segment (3m12s-3m18s at 10fps) for consistent learning.

## Non-Destructive Processing

Following the project's non-destructive editing principle:

1. Each step saves output to a separate directory
2. Intermediate results are preserved in `intermediate_dir`
3. Final output goes to `final_dir`
4. Incremental naming tracks processing steps

## Adding Custom Operations

To create a new operation:

1. Create a new class inheriting from `BaseImageOperation`
2. Implement `apply()` and `validate_params()` methods
3. Register the operation in `operations/__init__.py`

Example:

```python
from sevenrad_stills.operations.base import BaseImageOperation
from PIL import Image

class MyOperation(BaseImageOperation):
    def __init__(self) -> None:
        super().__init__("my_operation")

    def validate_params(self, params: dict[str, Any]) -> None:
        # Validate parameters
        pass

    def apply(self, image: Image.Image, params: dict[str, Any]) -> Image.Image:
        # Apply transformation
        return image

# Register it
from sevenrad_stills.operations import register_operation
register_operation(MyOperation)
```

## Troubleshooting

### Pipeline configuration errors

```bash
Pipeline configuration error: Pipeline must contain 'steps' key
```

Ensure your YAML has a `pipeline.steps` list with at least one step.

### Invalid YAML syntax

```bash
Pipeline configuration error: Invalid YAML syntax
```

Check YAML indentation and syntax. Use a YAML validator if needed.

### Operation not found

```bash
Operation 'foo' not found. Available: saturation
```

Check the operation name matches a registered operation. Use `saturation` for saturation adjustments.

### Validation errors

Parameter validation errors provide specific guidance:

```bash
Saturation operation requires 'mode' parameter
Fixed mode requires 'value' parameter
Range must be a list/tuple of two numbers
```

Follow the error message to correct your parameters.

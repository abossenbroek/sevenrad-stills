---
title: Getting Started
nav_order: 2
---

# Getting Started

This guide walks you through creating your first image transformation pipeline with **sevenrad-stills**.

## Prerequisites

- Complete the [Installation](installation) guide
- Virtual environment activated
- Basic understanding of YAML syntax

## Your First Pipeline

Let's create a simple pipeline that extracts frames from a YouTube video and applies saturation adjustment.

### Step 1: Create a Pipeline File

Create a file named `my-first-pipeline.yaml`:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=MzJaP-7N9I0"

segment:
  start: 10.0     # Start at 10 seconds
  end: 15.0       # End at 15 seconds
  interval: 1.0   # Extract 1 frame per second (5 total frames)

pipeline:
  steps:
    - name: "boost_saturation"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.5  # 150% saturation

output:
  base_dir: "./my_first_output"
  intermediate_dir: "./my_first_output/intermediate"
  final_dir: "./my_first_output/final"
```

### Step 2: Run the Pipeline

```bash
sevenrad pipeline my-first-pipeline.yaml
```

### Step 3: View Results

The pipeline will:

1. **Download** the YouTube video
2. **Extract** 5 frames (10s, 11s, 12s, 13s, 14s)
3. **Apply** 1.5x saturation boost
4. **Save** results to `my_first_output/final/`

Output structure:

```
my_first_output/
├── intermediate/
│   └── (no intermediate steps in this simple pipeline)
└── final/
    ├── boost_saturation_segment_000000_step00.jpg
    ├── boost_saturation_segment_000001_step00.jpg
    ├── boost_saturation_segment_000002_step00.jpg
    ├── boost_saturation_segment_000003_step00.jpg
    └── boost_saturation_segment_000004_step00.jpg
```

## Understanding the Pipeline

### Source Section

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=VIDEO_ID"
```

- Specifies the YouTube video to download
- Can be any publicly accessible video

### Segment Section

```yaml
segment:
  start: 10.0
  end: 15.0
  interval: 1.0
```

- `start`: Begin extraction at this timestamp (seconds)
- `end`: Stop extraction at this timestamp (seconds)
- `interval`: Time between frames (1.0 = 1 frame per second)

**Frame calculation**: `(end - start) / interval = (15 - 10) / 1.0 = 5 frames`

### Pipeline Section

```yaml
pipeline:
  steps:
    - name: "boost_saturation"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.5
```

- `name`: Human-readable step name (appears in output filenames)
- `operation`: Operation type (saturation, compression, downscale, etc.)
- `params`: Operation-specific parameters

### Output Section

```yaml
output:
  base_dir: "./my_first_output"
  intermediate_dir: "./my_first_output/intermediate"
  final_dir: "./my_first_output/final"
```

- `base_dir`: Root directory for all outputs
- `intermediate_dir`: Stores results from intermediate steps
- `final_dir`: Stores final processed images

## Multi-Step Pipeline

Let's create a more complex pipeline with multiple transformations:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=MzJaP-7N9I0"

segment:
  start: 10.0
  end: 13.0
  interval: 0.5  # 2 frames per second = 6 total frames

pipeline:
  steps:
    # Step 1: Downscale for pixelation
    - name: "pixelate"
      operation: "downscale"
      params:
        scale: 0.3
        upscale: true
        upscale_method: "nearest"

    # Step 2: Apply compression artifacts
    - name: "compress"
      operation: "compression"
      params:
        quality: 30
        subsampling: 2

    # Step 3: Boost saturation
    - name: "saturate"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.3

output:
  base_dir: "./multi_step_output"
  intermediate_dir: "./multi_step_output/intermediate"
  final_dir: "./multi_step_output/final"
```

Run it:

```bash
sevenrad pipeline multi-step-pipeline.yaml
```

### Output Structure

```
multi_step_output/
├── intermediate/
│   ├── pixelate/
│   │   └── pixelate_segment_*_step00.jpg (6 images)
│   └── compress/
│       └── compress_segment_*_step01.jpg (6 images)
└── final/
    └── saturate_segment_*_step02.jpg (6 images)
```

Each step builds on the previous:
1. **Original frame** → pixelate → `step00`
2. **step00** → compress → `step01`
3. **step01** → saturate → `step02` (final)

## Using the Repeat Parameter

Apply an operation multiple times with `repeat`:

```yaml
pipeline:
  steps:
    - name: "heavy_compression"
      operation: "compression"
      repeat: 5  # Apply 5 times
      params:
        quality: 50
        subsampling: 2
```

This simulates 5 save/load cycles at quality 50.

> **Note**: `repeat` applies the same parameters each time. For progressive quality decay, use `multi_compress` operation instead.

## CLI Options

### Parallel Processing

By default, pipelines process frames in parallel:

```bash
# Parallel processing (default)
sevenrad pipeline my-pipeline.yaml

# Custom worker count
sevenrad pipeline my-pipeline.yaml --workers 4

# Disable parallel processing
sevenrad pipeline my-pipeline.yaml --no-parallel
```

### Custom Configuration

Override default settings:

```bash
sevenrad pipeline my-pipeline.yaml --config /path/to/custom/config.toml
```

## Common Patterns

### Extract Many Frames

```yaml
segment:
  start: 0.0
  end: 60.0
  interval: 0.1  # 10 frames per second = 600 frames
```

### Extract Specific Moments

```yaml
# Extract frames at 5s, 10s, 15s, 20s
segment:
  start: 5.0
  end: 20.0
  interval: 5.0  # 4 frames
```

### Create Degradation Series

```yaml
pipeline:
  steps:
    - name: "stage_1_light"
      operation: "compression"
      params:
        quality: 70

    - name: "stage_2_moderate"
      operation: "compression"
      repeat: 2
      params:
        quality: 50

    - name: "stage_3_heavy"
      operation: "multi_compress"
      params:
        iterations: 5
        quality_start: 40
        quality_end: 20
        decay: "exponential"
```

This creates three distinct degradation stages visible in intermediate outputs.

## Next Steps

Now that you understand the basics:

1. **Explore Operations**:
   - [Compression & Degradation](operations/compression)
   - [Degradr Effects](operations/degradr)

2. **Follow Tutorials**:
   - [Compression Filters Tutorial](tutorials/compression-filters)
   - [Degradr Effects Tutorial](tutorials/degradr-effects)

3. **Read Reference Docs**:
   - [YAML Pipeline System](reference/pipeline)
   - [Filter Guide](reference/filter-guide)

## Troubleshooting

### "Operation not found"

Ensure operation names are correct:
- `saturation` (not `saturate`)
- `compression` (not `compress`)
- `downscale` (not `scale` or `resize`)

### "Segment validation failed"

- Ensure `end > start`
- Ensure `interval > 0`
- Check video is longer than your segment

### "No frames extracted"

- Verify YouTube URL is accessible
- Check segment times don't exceed video duration
- Ensure internet connection for download

### Slow Processing

- Reduce number of frames (increase `interval`)
- Use `--workers` to control parallelism
- Check system resources

---

**You're ready to explore!** Continue with the [Compression Filters Tutorial](tutorials/compression-filters) for hands-on examples.

---
title: GPU Acceleration
parent: Tutorials
nav_order: 4
has_toc: true
---

# Tutorial: GPU Acceleration with Taichi

This tutorial demonstrates how to use GPU acceleration for faster pipeline processing. The Taichi backend provides 10-15x speedups by executing operations on the GPU with minimal CPU-GPU data transfers.

## Prerequisites

- sevenrad-stills installed and configured ([Installation Guide](../installation/))
- GPU with Metal (macOS), CUDA (NVIDIA), or Vulkan support
- Taichi installed (`pip install taichi`)
- Basic familiarity with YAML pipeline system ([YAML Pipeline System](../reference/pipeline/))

## Tutorial Overview

| Tutorial | Operations | Speedup | Difficulty |
|----------|-----------|---------|------------|
| [01-single-operation](#tutorial-1-single-gpu-operation) | saturation | ~10x | Beginner |
| [02-multi-step](#tutorial-2-multi-step-gpu-pipeline) | noise, chromatic_aberration, blur_gaussian | ~12x | Intermediate |
| [03-all-operations](#tutorial-3-all-13-taichi-operations) | All 13 GPU operations | ~15x | Advanced |

## Key Benefits

**Zero-Copy Pipeline Architecture:**
- Image uploaded to GPU **once** at pipeline start
- All operations execute on GPU memory
- Final result downloaded **once** at pipeline end
- Eliminates per-operation CPU↔GPU transfers

---

## Tutorial 1: Single GPU Operation

**Goal**: Run a single operation on GPU to understand the basic configuration.

**Use Case**: Quick processing of high-resolution images or video frames.

### Running the Tutorial

Create a file `gpu-saturation.yaml`:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

segment:
  start: 192.0
  end: 195.0
  interval: 0.0667  # 15 fps = 45 frames

backend: "gpu"  # Enable GPU acceleration

pipeline:
  steps:
    - name: "desaturate"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 0.0  # Full grayscale

output:
  base_dir: "./output/gpu-saturation"
  intermediate_dir: "./output/gpu-saturation/intermediate"
  final_dir: "./output/gpu-saturation/final"
```

Run the pipeline:

```bash
sevenrad pipeline gpu-saturation.yaml
```

### Expected Results

**Output**: `./output/gpu-saturation/final/` containing 45 processed images

**Visual Transformation:**

| Original | After Saturation (GPU) |
|----------|----------------------|
| ![Original Frame]({{ site.baseurl }}/tutorials/gpu-acceleration/images/00-original.jpg) | ![Desaturated]({{ site.baseurl }}/tutorials/gpu-acceleration/images/01-saturation.jpg) |

*Left: Original extracted frame. Right: After GPU-accelerated saturation set to 0.0 (grayscale).*

### What You'll Learn

- The `backend: "gpu"` configuration key enables GPU processing
- Single operations work identically to CPU but execute faster
- Output quality matches CPU processing exactly

---

## Tutorial 2: Multi-Step GPU Pipeline

**Goal**: Chain multiple GPU operations to demonstrate zero-copy efficiency.

**Use Case**: Complex multi-effect processing where GPU acceleration provides maximum benefit.

### Running the Tutorial

Create a file `gpu-multi-step.yaml`:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

segment:
  start: 192.0
  end: 195.0
  interval: 0.0667

backend: "gpu"

pipeline:
  steps:
    # Step 1: Desaturate
    - name: "s1"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 0.0

    # Step 2: Add chromatic aberration
    - name: "s2"
      operation: "chromatic_aberration"
      params:
        shift_x: 7
        shift_y: 3

    # Step 3: Add noise
    - name: "s3"
      operation: "noise"
      params:
        mode: "gaussian"
        amount: 0.4
        seed: 42

    # Step 4: Apply Gaussian blur
    - name: "s4"
      operation: "blur_gaussian"
      params:
        sigma: 8.0

output:
  base_dir: "./output/gpu-multi-step"
  intermediate_dir: "./output/gpu-multi-step/intermediate"
  final_dir: "./output/gpu-multi-step/final"
```

Run the pipeline:

```bash
sevenrad pipeline gpu-multi-step.yaml
```

### Processing Flow

The GPU pipeline processes images efficiently:

```
Upload (once) → saturation → chromatic_aberration → noise → blur_gaussian → Download (once)
         GPU ─────────────────────────────────────────────────────────────────────
```

### Visual Progression

| Step | Operation | Result |
|------|-----------|--------|
| Original | - | ![Original]({{ site.baseurl }}/tutorials/gpu-acceleration/images/00-original.jpg) |
| Step 1 | saturation | ![Saturation]({{ site.baseurl }}/tutorials/gpu-acceleration/images/01-saturation.jpg) |
| Step 3 | noise | ![Noise]({{ site.baseurl }}/tutorials/gpu-acceleration/images/03-noise.jpg) |
| Step 4 | blur_gaussian | ![Blur]({{ site.baseurl }}/tutorials/gpu-acceleration/images/04-blur.jpg) |

*Progressive transformation through the GPU pipeline.*

### What You'll Learn

- Multiple operations chain efficiently on GPU
- Intermediate results stay in GPU memory
- No per-operation CPU-GPU transfers
- Warmup is performed once at pipeline start

---

## Tutorial 3: All 13 Taichi Operations

**Goal**: Demonstrate all available GPU-accelerated operations.

The project includes a comprehensive test pipeline with all 13 Taichi operations. This showcases the full range of GPU-accelerated effects.

### Available Taichi Operations

| Operation | Purpose | Key Parameters |
|-----------|---------|----------------|
| `saturation` | Color saturation adjustment | mode, value/range |
| `chromatic_aberration` | RGB channel shifting | shift_x, shift_y |
| `noise` | Gaussian/row/column noise | mode, amount, seed |
| `blur_gaussian` | Gaussian convolution blur | sigma |
| `blur_circular` | Circular bokeh blur | radius |
| `motion_blur` | Directional blur | kernel_size, angle |
| `downscale` | Resolution reduction | scale, methods |
| `salt_pepper` | Cosmic ray noise | amount, salt_vs_pepper |
| `band_swap` | Channel corruption | tile_count, permutation |
| `buffer_corruption` | Memory glitch effects | corruption_type, severity |
| `slc_off` | Satellite scan gaps | gap_width, scan_period |
| `corduroy` | Detector striping | strength, orientation |
| `bayer_filter` | Sensor mosaic | pattern |

### Running the Full Demo

```bash
sevenrad pipeline examples/all-effects-gpu.yaml
```

### Final Result

After all 16 operations (13 Taichi + 3 CPU-only):

![Final Result]({{ site.baseurl }}/tutorials/gpu-acceleration/images/final-result.jpg)

*Result after processing through all GPU-accelerated operations.*

---

## Backend Selection

### Configuration

Specify the backend in your YAML configuration:

```yaml
backend: "gpu"    # Use Taichi GPU acceleration
# backend: "cpu"  # Use CPU processing (default)
# backend: "metal" # Use Metal backend directly (macOS)
```

### Automatic Architecture Selection

The GPU backend automatically selects the best available architecture:

| Platform | Priority |
|----------|----------|
| macOS | Metal → Vulkan → CPU |
| Linux (NVIDIA) | CUDA → Vulkan → CPU |
| Linux (AMD) | Vulkan → CPU |
| Windows (NVIDIA) | CUDA → Vulkan → CPU |

### Performance Comparison

![Backend Comparison]({{ site.baseurl }}/tutorials/gpu-acceleration/images/backend-comparison.jpg)

*Visual comparison of CPU, GPU, and Metal backend outputs. All produce identical results.*

---

## Troubleshooting

### "Taichi is not available"

Install Taichi:

```bash
pip install taichi
```

### "Failed to initialize Metal backend"

Ensure your Mac has Metal support (macOS 10.14+). Try falling back to CPU:

```yaml
backend: "cpu"
```

### GPU Memory Errors

For large images or many operations, reduce batch size or image resolution:

```yaml
segment:
  interval: 0.2  # Extract fewer frames (5 fps instead of 15)
```

### Performance Tips

1. **Use warmup** for repeated processing - JIT compilation happens once
2. **Batch similar-sized images** - Buffer allocation is shape-based
3. **Process larger images** - GPU overhead is amortized on larger data
4. **Chain operations** - More operations = more GPU efficiency benefit

---

## Next Steps

- **[Taichi Pipeline Architecture](../reference/taichi-pipeline/)** - Technical details for developers
- **[YAML Pipeline System](../reference/pipeline/)** - Full pipeline configuration reference
- **[Satellite Operations](satellite-malfunctions/)** - More GPU-accelerated effects

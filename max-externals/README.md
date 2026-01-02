# SevenRad Max Externals

GPU image processing effects for Max/MSP/Jitter, ported from the SevenRad Taichi pipeline.

## Features

- **15 GPU shader effects** via `jit.gl.pix` for real-time video processing
- **2 C externals** for CPU-based tile and mask generation
- **Consistent RNG** - identical noise patterns across Python and Max implementations
- **Universal binaries** - native support for Apple Silicon and Intel Macs

## Requirements

- Max 8.5+ or Max 9
- macOS 10.13+ (x64 or arm64)

## Installation

### From Release (Recommended)

1. Download the latest release from the [releases page](https://github.com/abossenbroek/sevenrad-stills/releases)
2. Unzip to your Max Packages folder:
   - Max 9: `~/Documents/Max 9/Packages/`
   - Max 8: `~/Documents/Max 8/Packages/`
3. Restart Max

### From Source

```bash
# Clone with submodules
git clone --recursive https://github.com/abossenbroek/sevenrad-stills.git
cd sevenrad-stills/max-externals

# Build and install
make install
```

See [DEVELOPER.md](DEVELOPER.md) for detailed build instructions.

## Quick Start

### Using GPU Effects

All GPU effects are GenExpr shaders loaded via `jit.gl.pix`:

```
[jit.gl.pix @file sr.noise.genjit]
```

Connect to any Jitter video source:

```
[jit.grab] or [jit.movie]
     |
[jit.gl.pix @file sr.noise.genjit]
     |
[jit.gl.render]
```

### Example Patcher

Open the demo patcher to see effects in action:

`File > Open > ~/Documents/Max 9/Packages/sevenrad/examples/sr.demo.maxpat`

### Getting Help

Each effect has a help patcher. Select an object and press `?` or right-click > Open Help.

## Effects Reference

### Noise & Artifacts

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `sr.noise` | Gaussian, row, or column noise | `mode`, `amount`, `seed` |
| `sr.saltpepper` | Salt and pepper noise | `amount`, `salt_ratio`, `seed` |
| `sr.corduroy` | Scanline/stripe artifacts | `orientation`, `strength`, `density` |

### Blur Effects

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `sr.blur.h` | Horizontal Gaussian blur | `sigma` |
| `sr.blur.v` | Vertical Gaussian blur | `sigma` |
| `sr.blur.circular` | Bokeh-style circular blur | `radius` |
| `sr.motion` | Directional motion blur | `kernel_size`, `angle` |

### Color & Distortion

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `sr.saturation` | HSV saturation adjustment | `factor` (0=grayscale, 2=boost) |
| `sr.chromatic` | RGB channel separation | `shift_x`, `shift_y` |
| `sr.downscale` | Pixelation effect | `scale`, `pixelate` |

### Sensor Simulation

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `sr.bayer.mosaic` | Apply Bayer CFA pattern | `pattern` (RGGB, BGGR, etc.) |
| `sr.bayer.demosaic` | Reconstruct from Bayer | `pattern` |

### Glitch & Corruption

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| `sr.bandswap` | Channel permutation in tiles | `tile_count`, `permutation` |
| `sr.slcoff` | Landsat SLC-off wedge gaps | `gap_width`, `scan_period` |
| `sr.corruption` | XOR/invert/shuffle corruption | `mode`, `intensity`, `tile_count` |

### C Externals (CPU)

| External | Description | Purpose |
|----------|-------------|---------|
| `sr.tilegen` | Random tile bounds generator | Used with `sr.bandswap`, `sr.corruption` |
| `sr.maskgen` | SLC-off wedge mask generator | Used with `sr.slcoff` |

## Effect Chaining

For two-pass blur (better quality):

```
[video source]
     |
[jit.gl.pix @file sr.blur.h.genjit]
     |
[jit.gl.pix @file sr.blur.v.genjit]
     |
[output]
```

For glitch effects with random tiles:

```
[video source]
     |
[sr.tilegen @tile_count 5 @seed 42]
     |
[jit.gl.pix @file sr.bandswap.genjit]
     |
[output]
```

## Troubleshooting

### "No such file" error for .genjit

Ensure the package is installed in the correct location and Max has been restarted.

### Effects not rendering

- Check that `jit.gl.render` is present in your patch
- Verify OpenGL context is initialized (toggle rendering on/off)
- Try `@adapt 1` on `jit.gl.pix` for dimension matching

### Performance issues

- Reduce resolution with `@dim` attribute
- Use lower blur radius/sigma values
- Ensure GPU acceleration is enabled in Max preferences

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Credits

- Effects ported from [sevenrad-stills](https://github.com/abossenbroek/sevenrad-stills) Taichi implementation
- Built with [Cycling '74 Max SDK](https://github.com/Cycling74/max-sdk)

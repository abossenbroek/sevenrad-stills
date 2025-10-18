# Degradr Effects Tutorials

Learn creative image degradation through hands-on examples. These tutorials demonstrate operations inspired by the [degradr repository](https://github.com/nhauber99/degradr) (MIT licensed), adapted to use NumPy/SciPy/scikit-image for PIL integration.

## Quick Start

Choose any tutorial and run:

```bash
sevenrad pipeline docs/tutorials/degradr-effects/[tutorial].yaml
```

Output appears in `tutorials/degradr-effects/[tutorial-name]/` with intermediate steps preserved for inspection.

---

## Individual Operations

Learn each degradr operation through focused examples:

### Blur Effects

- **05-soft-glow.yaml** - Gaussian blur for dreamy softness
  - Learn: Control blur intensity with sigma parameter
  - Uses: Video segment 52-59s @ 15fps (105 frames)

- **06-dreamy-bokeh.yaml** - Circular bokeh lens effects
  - Learn: Create lens-like blur with circular kernels
  - Uses: Video segment 52-59s @ 15fps (105 frames)

### Texture & Grain

- **07-film-grain.yaml** - Add analog film texture
  - Learn: Apply Gaussian noise for film grain
  - Uses: Video segment 90-97s @ 15fps (105 frames)

- **08-vhs-lines.yaml** - VHS scan line artifacts
  - Learn: Create directional noise (row/column modes)
  - Uses: Video segment 90-97s @ 15fps (105 frames)

### Color Effects

- **09-color-shift.yaml** - Chromatic color fringing
  - Learn: Shift RGB channels for optical aberration
  - Uses: Video segment 52-59s @ 15fps (105 frames)

- **10-digital-mosaic.yaml** - Digital sensor patterns
  - Learn: Simulate Bayer filter sensor artifacts
  - Uses: Video segment 90-97s @ 15fps (105 frames)

---

## Creative Combinations

Master multi-operation pipelines for complete aesthetics:

- **01-vhs-scanlines.yaml** - Complete VHS tape aesthetic
  - Combines: chromatic_aberration + noise (row) + blur_gaussian + compression
  - Effect: Nostalgic 80s/90s VHS playback

- **02-sensor-noise.yaml** - Digital camera simulation
  - Combines: bayer_filter + noise (gaussian) + compression
  - Effect: High-ISO digital sensor output

- **03-lens-artifacts.yaml** - Vintage lens character
  - Combines: blur_circular + chromatic_aberration + saturation
  - Effect: Dreamy vintage photography

All combination tutorials use segment 192-195s @ 15fps (45 frames).

---

## Learning Path

1. **Start with Individual Operations** (05-10)
   - Understand each effect in isolation
   - Experiment with parameter ranges
   - See how each operation transforms images

2. **Explore Parameter Variations**
   - Try values outside tutorial examples
   - Observe subtle vs extreme effects
   - Find your preferred aesthetic range

3. **Study Creative Combinations** (01-03)
   - See how operations layer together
   - Learn effective operation ordering
   - Understand cumulative degradation

4. **Build Custom Pipelines**
   - Mix operations to create unique aesthetics
   - Develop your own artistic signatures
   - Document successful combinations

---

## What's Next?

### Documentation

- **Parameter Reference**: [FILTER_GUIDE.md](../../FILTER_GUIDE.md)
  - Complete parameter ranges for all operations
  - Detailed usage examples
  - Quick reference lookup

- **Technical Details**: [technical_notes/degradr-operations.md](../../technical_notes/degradr-operations.md)
  - Algorithm implementations
  - Performance characteristics
  - Testing strategies

- **System Guide**: [PIPELINE.md](../../PIPELINE.md)
  - Complete pipeline YAML syntax
  - Advanced features (repeat, conditionals)
  - Output structure

### Related Tutorials

- **[compression-filters](../compression-filters/)** - Multi-generation compression effects
  - JPEG compression cycles
  - Progressive quality decay
  - Digital artifact accumulation

---

## Attribution

**Original Implementation**: degradr by nhauber99 (MIT License)
**Adaptations**: Converted from PyTorch to NumPy/SciPy
**License**: See LICENSE_DEGRADR.txt

---

## Artistic Context

These effects explore digital media materiality through algorithmic transformation:

**VHS Scanlines**: Magnetic tape degradation as palimpsest of loss
**Sensor Noise**: Computational seeing through Bayer demosaicing
**Lens Artifacts**: Optical imperfection as aesthetic character

In the spirit of Rimbaud's systematic derangement and de Groen's clinical critique of late-stage capitalism's nostalgia aesthetics.

---

## Troubleshooting

**Effects too subtle?** Increase parameter values (shift, amount, radius, sigma)
**Effects too strong?** Reduce parameters or remove operations
**Slow processing?** Reduce frame count by increasing segment interval
**Operation not found?** Check operation names match exactly (use underscores)

---

**Happy experimenting with degradr-inspired effects!**

*Each transformation reveals the computational layers that construct seeing itself.*

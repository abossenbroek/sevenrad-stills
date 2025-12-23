# SevenRad Max 8 Externals

Max/MSP package providing GPU image effects ported from the SevenRad Taichi pipeline.

## Quick Start

### Prerequisites

- Max 8.5+
- CMake 3.19+ (`mise install cmake`)
- Xcode Command Line Tools (macOS): `xcode-select --install`

### Clone with Submodules

When cloning the parent repository, include submodules:

```bash
git clone --recursive https://github.com/your-repo/sevenrad-stills.git
```

Or if already cloned, initialize submodules:

```bash
git submodule update --init --recursive
```

### Build

```bash
cd max-externals
mkdir build && cd build
cmake ..
cmake --build .
```

The built externals will be copied to `max-externals/externals/`.

### Install

Copy the entire `max-externals/` folder to your Max Packages directory:

```bash
cp -r max-externals ~/Documents/Max\ 8/Packages/sevenrad
```

Restart Max to load the package.

## Working with Git Submodules

### After Cloning

If you cloned without `--recursive`, initialize the Max SDK submodule:

```bash
git submodule update --init --recursive
```

### Updating the Max SDK

To update to the latest Max SDK version:

```bash
cd max-externals/max-sdk
git fetch origin
git checkout <desired-tag-or-commit>
cd ../..
git add max-externals/max-sdk
git commit -m "chore: Update Max SDK to <version>"
```

### After Pulling Changes

If someone updated the submodule reference, sync your local copy:

```bash
git pull
git submodule update --init --recursive
```

### Submodule Status

Check submodule status:

```bash
git submodule status
```

## Package Contents

```
max-externals/
├── code/           # GenExpr GPU shaders (.genjit)
├── externals/      # Built C externals (.mxo)
├── help/           # Help patchers (.maxhelp)
├── max-sdk/        # Cycling74 Max SDK (submodule)
├── source/         # C source code
│   ├── common/     # Shared headers (RNG, utilities)
│   ├── sr.tilegen/ # Random tile generator
│   └── sr.maskgen/ # SLC-off mask generator
└── tests/          # Test framework
```

## Effects

| Effect | Type | Description |
|--------|------|-------------|
| sr.noise | GPU | Gaussian/row/column noise |
| sr.saturation | GPU | HSV saturation adjustment |
| sr.chromatic | GPU | Chromatic aberration |
| sr.blur.h/v | GPU | Two-pass Gaussian blur |
| sr.blur.circular | GPU | Circular (bokeh) blur |
| sr.motion | GPU | Directional motion blur |
| sr.saltpepper | GPU | Salt and pepper noise |
| sr.corduroy | GPU | Scanline artifacts |
| sr.bayer.mosaic/demosaic | GPU | Bayer filter simulation |
| sr.downscale | GPU | Pixelation effect |
| sr.bandswap | Hybrid | Channel permutation in tiles |
| sr.slcoff | Hybrid | Landsat SLC-off wedge gaps |
| sr.corruption | Hybrid | Buffer corruption effects |

## Testing

Generate Python reference images and compare Max outputs:

```bash
# From project root
source .venv/bin/activate
python max-externals/tests/generate_references.py
python max-externals/tests/compare_outputs.py
```

Run syntax validation:

```bash
pytest max-externals/tests/ -v
```

## Documentation

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed specifications.

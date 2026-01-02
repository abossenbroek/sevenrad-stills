# Developer Guide

This guide covers building, testing, and contributing to the SevenRad Max Externals package.

## Prerequisites

### Required Tools

- **CMake 3.19+** - Build system for C externals
- **Xcode Command Line Tools** - C compiler (macOS)
- **Python 3.10+** - For reference generation and testing
- **Max 8.5+ or Max 9** - For runtime testing

### Install with mise

```bash
# From project root
mise install  # Installs cmake, uv, etc.
```

### Manual Installation

```bash
# macOS
xcode-select --install
brew install cmake

# Or with mise
mise install cmake
```

## Project Structure

```
max-externals/
├── code/                   # GenExpr GPU shaders (.genjit)
│   ├── sr.noise.genjit
│   ├── sr.blur.h.genjit
│   └── ...
├── externals/              # Built C externals (.mxo) - gitignored
├── examples/               # Example patchers
│   └── sr.demo.maxpat
├── help/                   # Help patchers (.maxhelp)
├── max-sdk/                # Cycling74 Max SDK (git submodule)
├── source/                 # C source code
│   ├── common/             # Shared headers
│   │   ├── sr_rng.h        # PCG random number generator
│   │   └── sr_utils.h      # Utility macros
│   ├── sr.tilegen/         # Tile bounds generator
│   └── sr.maskgen/         # SLC-off mask generator
├── tests/                  # Test framework
│   ├── reference/          # Python-generated reference data
│   ├── actual/             # Max-generated outputs (gitignored)
│   ├── test_genexpr_syntax.py
│   ├── test_rng_consistency.py
│   └── ...
├── CMakeLists.txt          # CMake build configuration
├── Makefile                # Build automation
└── package-info.json       # Max package metadata
```

## Building

### Using Makefile (Recommended)

```bash
cd max-externals

# Build externals
make build

# Build and install to Max
make install

# Clean build artifacts
make clean

# Full rebuild
make rebuild
```

### Using CMake Directly

```bash
cd max-externals
mkdir build && cd build
cmake ..
cmake --build .
```

### Build Output

Built externals are placed in `externals/`:
- `sr.tilegen.mxo` - Universal binary (x64 + arm64)
- `sr.maskgen.mxo` - Universal binary (x64 + arm64)

## Installation

### Standard Install

```bash
make install
```

Copies the package to `~/Documents/Max 9/Packages/sevenrad/` (or Max 8).

### Development Install

```bash
make install-dev
```

Creates a symlink for live development - changes to source files are reflected immediately without reinstalling.

### Uninstall

```bash
make uninstall
```

## Testing

### Python Tests (No Max Required)

These tests validate shader syntax and RNG consistency without needing Max:

```bash
# Run all Python tests
make test

# Run specific test suites
make test-syntax   # GenExpr shader validation
make test-rng      # RNG consistency checks
```

### Max Integration Tests

Full integration testing requires Max:

```bash
# Show testing workflow
make test-max-info
```

**Workflow:**

1. `make install` - Build and install package
2. Restart Max
3. Open `tests/test_suite.maxpat` in Max
4. Click "Run All Tests" - generates outputs to `tests/actual/`
5. `make test-compare` - Compare against Python references

### Generating Reference Images

Reference images are generated from the Python/Taichi implementation:

```bash
make test-refs
```

This creates reference images and parameter JSON files in `tests/reference/`.

## GenExpr Shader Development

### File Format

GenExpr shaders use the `.genjit` extension and follow this structure:

```c
/**
 * Effect Name
 *
 * Description of what this effect does.
 *
 * Parameters:
 *   param_name - Description (range: min-max, default: value)
 */

// Parameter declarations
Param amount(0.5);
Param seed(0);

// Main processing
out = in1 * amount;
```

### PCG Random Number Generator

For deterministic random effects, use the PCG hash-based RNG:

```c
// Required constants (must match Python implementation)
#define PCG_MULT 747796405
#define PCG_INC 2891336453
#define PCG_FACTOR 277803737
#define COORD_PRIME_X 374761393
#define COORD_PRIME_Y 668265263

// Hash function
uint pcg_hash(uint state) {
    state = state * PCG_MULT + PCG_INC;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * PCG_FACTOR;
    return (word >> 22) ^ word;
}

// Generate float in [0, 1)
float rand_float(int x, int y, int seed) {
    uint h = uint(seed);
    h = pcg_hash(h + uint(x) * COORD_PRIME_X);
    h = pcg_hash(h + uint(y) * COORD_PRIME_Y);
    return float(h) / 4294967296.0;
}
```

### Testing Shaders

1. Add shader to `code/` directory
2. Run `make test-syntax` to validate syntax
3. Add test case to `tests/generate_references.py`
4. Run `make test-refs` to generate reference
5. Test in Max manually or via test suite

## C External Development

### Creating a New External

1. Create directory: `source/sr.neweffect/`
2. Create source file: `sr.neweffect.c`
3. Add to `CMakeLists.txt`:

```cmake
# For Max-only external
add_max_external(sr.neweffect
    source/sr.neweffect/sr.neweffect.c
)

# For Jitter external (needs matrix operations)
add_jitter_external(sr.neweffect
    source/sr.neweffect/sr.neweffect.c
)
```

### External Template

```c
#include "ext.h"
#include "ext_obex.h"

typedef struct _myobj {
    t_object ob;
    void *outlet;
    // Add attributes here
} t_myobj;

static t_class *myobj_class;

void *myobj_new(t_symbol *s, long argc, t_atom *argv);
void myobj_free(t_myobj *x);
void myobj_bang(t_myobj *x);

void ext_main(void *r) {
    t_class *c = class_new("sr.myobj",
        (method)myobj_new,
        (method)myobj_free,
        sizeof(t_myobj),
        0L, A_GIMME, 0);

    class_addmethod(c, (method)myobj_bang, "bang", 0);

    class_register(CLASS_BOX, c);
    myobj_class = c;
}

void *myobj_new(t_symbol *s, long argc, t_atom *argv) {
    t_myobj *x = (t_myobj *)object_alloc(myobj_class);
    x->outlet = outlet_new(x, NULL);
    return x;
}

void myobj_free(t_myobj *x) {
    // Cleanup
}

void myobj_bang(t_myobj *x) {
    outlet_bang(x->outlet);
}
```

### Shared Headers

Common functionality is in `source/common/`:

- `sr_rng.h` - PCG random number generator (matches Python implementation)
- `sr_utils.h` - Utility macros and helpers

## Git Submodule (Max SDK)

The Max SDK is included as a git submodule.

### Initial Setup

```bash
git submodule update --init --recursive
```

### After Pulling Changes

```bash
git pull
git submodule update --init --recursive
```

### Updating Max SDK

```bash
cd max-sdk
git fetch origin
git checkout v8.6.0  # or desired version
cd ..
git add max-sdk
git commit -m "chore: Update Max SDK to v8.6.0"
```

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make build` | Build C externals |
| `make clean` | Remove build artifacts |
| `make rebuild` | Clean and rebuild |
| `make install` | Build and install to Max Packages |
| `make install-dev` | Install as symlink for development |
| `make uninstall` | Remove from Max Packages |
| `make test` | Run all Python tests |
| `make test-syntax` | Validate GenExpr syntax |
| `make test-rng` | Test RNG consistency |
| `make test-refs` | Generate reference images |
| `make test-compare` | Compare Max output to references |
| `make find-max` | Show detected Max installation |
| `make help` | Show all targets |

## Troubleshooting

### CMake can't find Max SDK

```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Verify SDK exists
ls max-sdk/source/max-sdk-base/
```

### Linker errors for Jitter symbols

Use `add_jitter_external()` instead of `add_max_external()` in CMakeLists.txt for externals that use Jitter/matrix operations.

### Tests fail to import sevenrad_stills

```bash
# Ensure you're in the project root with venv activated
cd /path/to/sevenrad-stills
source .venv/bin/activate
pip install -e .
```

### Max doesn't see the package

1. Verify installation location: `ls ~/Documents/Max\ 9/Packages/sevenrad/`
2. Check package-info.json is present
3. Restart Max completely (Quit and reopen)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-effect`
3. Make changes and test: `make test && make install`
4. Test in Max manually
5. Commit with conventional commit message: `feat(max-externals): Add new effect`
6. Submit pull request

### Code Style

- C code: Follow Max SDK conventions
- GenExpr: Include documentation header with parameters
- Python: Follow PEP 8, use type annotations

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(max-externals): Add sr.neweffect shader
fix(max-externals): Fix sr.noise seed handling
docs(max-externals): Update README installation steps
```

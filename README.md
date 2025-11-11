# sevenrad-stills

A Python application for extracting movies from YouTube, taking stills and reworking these stills to create unique images as part of a book. The primary goal is to explore poetic interpretations of digital media through algorithmic transformation, where the poetic voice is Rimbaud and Dominique de Groen.

## Project Overview

This project uses modern Python practices and is structured for scalability and maintainability. The aesthetic is focused on abstraction and digital artifacts - think glitch art, data moshing, and poetic data visualization.

## Features

- Extract video content from YouTube
- Generate stills from video frames
- Apply algorithmic transformations to create unique imagery
- Non-destructive image editing with incremental versioning
- **Multi-backend support**: Choose between CPU, GPU (Taichi), or Metal (macOS) acceleration for image operations

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [mise](https://mise.jdx.dev/) for tool version management.

### Prerequisites

- Python 3.12+
- uv
- mise
- ffmpeg

### Setup

1. Clone the repository
2. Ensure mise is configured:
   ```bash
   mise install
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development

### Code Quality

This project enforces strict code quality standards:

- **Formatting**: `ruff format`
- **Linting**: `ruff check`
- **Type Checking**: `mypy src/`

All checks run automatically via pre-commit hooks on each commit.

### Running Tests

Run all unit tests (fast):
```bash
pytest
```

Run all tests including slow integration tests:
```bash
pytest -m "slow or integration"
```

Skip slow tests:
```bash
pytest -m "not slow"
```

Run only integration tests:
```bash
pytest tests/integration/ -v -s
```

**Note**: Integration tests download real YouTube videos and may take 30-60 seconds to complete.

### Project Structure

```
.
├── src/
│   └── sevenrad_stills/      # Main package
├── tests/                    # Test files
├── docs/                     # Documentation
│   ├── BACKEND_CONFIGURATION.md  # Backend selection guide
│   ├── BACKEND_TODO.md           # Missing implementations
│   └── reference/
│       └── filter-guide.md       # Complete operation reference
├── pyproject.toml            # Project configuration
├── .mise.toml                # Tool version management
└── .pre-commit-config.yaml   # Pre-commit hooks
```

## Backend Configuration

Sevenrad Stills supports three compute backends for image operations:

- **CPU** (default): Pure Python/NumPy implementations - universal compatibility
- **GPU**: Taichi-accelerated implementations - cross-platform GPU support
- **Metal**: Native Metal shaders - macOS only, maximum performance

### Selecting a Backend

Configure the backend in your YAML pipeline file:

```yaml
source:
  youtube_url: "https://www.youtube.com/watch?v=example"

# Choose your backend (cpu, gpu, or metal)
backend: "gpu"  # or "cpu" or "metal"

segment:
  start: 0.0
  end: 3.0
  interval: 0.5

pipeline:
  steps:
    - name: "saturation_boost"
      operation: "saturation"
      params:
        mode: "fixed"
        value: 1.5
```

### Backend Support Matrix

| Operation             | CPU | GPU | Metal | Notes                          |
|-----------------------|-----|-----|-------|--------------------------------|
| band_swap             | ✅  | ✅  | ❌    | Metal coming soon              |
| bayer_filter          | ✅  | ✅  | ✅    |                                |
| blur_circular         | ✅  | ✅  | ❌    | Metal coming soon              |
| blur_gaussian         | ✅  | ✅  | ❌    | Metal coming soon              |
| buffer_corruption     | ✅  | ✅  | ❌    | Metal needs wrapper class      |
| chromatic_aberration  | ✅  | ✅  | ❌    | Metal coming soon              |
| compression           | ✅  | ✅  | ✅    |                                |
| compression_artifact  | ✅  | ✅  | ✅    |                                |
| corduroy              | ✅  | ✅  | ✅    |                                |
| downscale             | ✅  | ✅  | ⚠️    | Metal has runtime issues       |
| motion_blur           | ✅  | ✅  | ⚠️    | Metal has runtime issues       |
| multi_compress        | ✅  | ❌  | ❌    | CPU only                       |
| noise                 | ✅  | ✅  | ✅    |                                |
| salt_pepper           | ✅  | ✅  | ✅    |                                |
| saturation            | ✅  | ✅  | ✅    |                                |
| slc_off               | ✅  | ✅  | ⚠️    | Metal has runtime issues       |

**Legend:**
- ✅ = Fully working
- ⚠️ = Implemented but has runtime errors
- ❌ = Not yet implemented

### Known Issues

**Metal Backend Issues** (pre-existing bugs, not related to backend configuration):
- `slc_off`: Runtime error - "converting to a C array"
- `motion_blur`: MLX library error - `module 'mlx.core' has no attribute 'flip'`
- `downscale`: Runtime error - "argument 0 must be None or objc.NULL"

**Workaround**: Use GPU backend for these operations until Metal implementations are fixed.

### When to Use Each Backend

**CPU**:
- Small images (< 1000x1000)
- Single-frame processing
- Operations without GPU/Metal support
- Maximum compatibility

**GPU (Taichi)**:
- Medium to large images (1000x1000+)
- Batch processing
- Cross-platform deployment
- Good balance of speed and compatibility

**Metal (macOS)**:
- Large images (2000x2000+)
- macOS-only deployment
- Maximum performance
- Apple Silicon or Intel Mac

For detailed information, see [docs/BACKEND_CONFIGURATION.md](docs/BACKEND_CONFIGURATION.md).

## Artistic Context

### Image Style
Images are edited in non-destructive mode, always tracking different steps with incremental filenames.

## Contributing

Please ensure all code follows:
- PEP 8 style guidelines
- PEP 484 type annotations
- Google-style docstrings
- All quality checks pass before committing

## License

[Add license information]

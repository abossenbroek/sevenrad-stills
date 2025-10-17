# sevenrad-stills

A Python application for extracting frames from YouTube videos and applying image processing operations through a flexible pipeline system.

## Project Overview

This project provides a command-line tool for downloading YouTube videos, extracting frames at specified intervals, and processing those frames through configurable image transformation pipelines. Built with modern Python practices for scalability and maintainability.

## Features

- **Video Extraction**: Download videos from YouTube and extract frames at custom intervals
- **Frame Processing**: Apply compression, degradation, blur, and scaling operations
- **Pipeline System**: Chain multiple operations in YAML-configured pipelines
- **Non-Destructive**: All operations preserve intermediate results with incremental versioning
- **Parallel Processing**: Process multiple frames concurrently for performance

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run a tutorial example
sevenrad pipeline docs/tutorials/compression-filters/01-social-media.yaml

# Create your own pipeline
cat > my_pipeline.yaml <<EOF
source:
  youtube_url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

segment:
  start: 10.0
  end: 13.0
  interval: 0.1  # Extract 1 frame every 0.1 seconds

output:
  base_dir: "output"

steps:
  - name: "compress"
    operation: "compression"
    params:
      quality: 50
      subsampling: 2
EOF

sevenrad pipeline my_pipeline.yaml
```

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
├── pyproject.toml            # Project configuration
├── .mise.toml                # Tool version management
└── .pre-commit-config.yaml   # Pre-commit hooks
```

## Available Operations

### Compression & Degradation
- **compression**: JPEG compression with configurable quality and chroma subsampling
- **multi_compress**: Multi-generation compression with progressive quality decay
- **downscale**: Resolution reduction with multiple resampling methods
- **motion_blur**: Directional blur with configurable angle and intensity

### Documentation
- [Filter Guide](docs/FILTER_GUIDE.md): Complete parameter reference for all operations
- [Pipeline System](docs/PIPELINE.md): YAML pipeline configuration guide
- [Tutorials](docs/tutorials/compression-filters/): Hands-on examples with real outputs

## Contributing

Please ensure all code follows:
- PEP 8 style guidelines
- PEP 484 type annotations
- Google-style docstrings
- All quality checks pass before committing

## License

[Add license information]

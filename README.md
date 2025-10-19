# sevenrad-stills

A Python application for extracting movies from YouTube, taking stills and reworking these stills to create unique images as part of a book. The primary goal is to explore poetic interpretations of digital media through algorithmic transformation, where the poetic voice is Rimbaud and Dominique de Groen.

## Project Overview

This project uses modern Python practices and is structured for scalability and maintainability. The aesthetic is focused on abstraction and digital artifacts - think glitch art, data moshing, and poetic data visualization.

## Features

- Extract video content from YouTube
- Generate stills from video frames
- Apply algorithmic transformations to create unique imagery
- Non-destructive image editing with incremental versioning

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

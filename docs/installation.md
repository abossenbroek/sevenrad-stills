---
title: Installation
nav_order: 1
---

# Installation

This guide walks you through setting up **sevenrad-stills** on your system using modern tooling.

## Prerequisites

Before installing sevenrad-stills, ensure you have the following:

### System Requirements

- **Python 3.12 or higher**
- **Git**
- **mise** - Tool version manager (manages Python, ffmpeg, and other tools)
- **uv** - Fast Python package installer (installed via mise)

### Operating Systems

- macOS (recommended)
- Linux
- Windows (via WSL)

## Step 1: Install mise

[mise](https://mise.jdx.dev/) is a polyglot tool version manager that handles Python, ffmpeg, and other dependencies.

### macOS (Homebrew)

```bash
brew install mise
```

### Linux

```bash
curl https://mise.run | sh
```

### Verify Installation

```bash
mise --version
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/abossenbroek/sevenrad-stills.git
cd sevenrad-stills
```

## Step 3: Install Dependencies with mise

The project includes a `.mise.toml` file that automatically installs:
- `uv` (Python package manager)
- `ffmpeg` (video processing)

```bash
mise install
```

This command reads `.mise.toml` and installs all required tools.

## Step 4: Create Virtual Environment

Use `uv` to create and activate a Python virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Step 5: Install Python Dependencies

Install sevenrad-stills and development dependencies:

```bash
uv pip install -e ".[dev]"
```

This installs:
- Core dependencies (Pillow, scikit-image, ffmpeg-python, yt-dlp, etc.)
- Development tools (pytest, ruff, mypy, pre-commit)

## Step 6: Install Pre-commit Hooks

Set up pre-commit hooks for code quality checks:

```bash
pre-commit install
```

Pre-commit will automatically run on every commit:
- `ruff format` - Code formatting
- `ruff check` - Linting
- `mypy` - Type checking

## Verification

Verify your installation:

### Check Python Version

```bash
python --version
# Should show Python 3.12.x or higher
```

### Check ffmpeg

```bash
ffmpeg -version
```

### Check sevenrad CLI

```bash
sevenrad --version
```

### Run Tests

```bash
pytest
```

If all tests pass, your installation is complete!

## Quick Test

Extract frames from a YouTube video:

```bash
sevenrad extract "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --fps 1 \
  --output-dir ./test_output
```

## Troubleshooting

### mise Not Found

Ensure mise is in your PATH. Add to your shell profile:

```bash
# For bash
echo 'eval "$(mise activate bash)"' >> ~/.bashrc

# For zsh
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
```

### Python Version Mismatch

If `python --version` shows the wrong version, ensure mise is properly activated:

```bash
mise doctor
```

### ffmpeg Not Found

Reinstall tools via mise:

```bash
mise install --force
```

### Import Errors

Ensure virtual environment is activated:

```bash
source .venv/bin/activate
```

### Pre-commit Failures

Update pre-commit hooks:

```bash
pre-commit autoupdate
```

## Next Steps

- [Getting Started](getting-started) - Run your first pipeline
- [Operations Reference](operations/compression) - Learn about available transformations
- [Tutorials](tutorials) - Hands-on examples

## Development Setup

For contributors, additional setup:

```bash
# Install all development dependencies
uv pip install -e ".[dev]"

# Run type checking
mypy src/

# Run tests with coverage
pytest --cov=sevenrad_stills --cov-report=html
```

## Updating

To update to the latest version:

```bash
git pull origin main
mise install
uv pip install -e ".[dev]"
```

---

**Installation complete!** You're ready to start transforming images. Continue to [Getting Started](getting-started) for your first pipeline.

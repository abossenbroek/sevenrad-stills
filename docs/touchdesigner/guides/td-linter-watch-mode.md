# td-linter Watch Mode Guide

Watch mode continuously monitors your `.toe.dir` project and re-lints automatically when files change.

## Prerequisites

Install the watch mode dependency:

```bash
pip install sevenrad-stills[td-linter-watch]
```

## Basic Usage

```bash
td-linter watch path/to/project.toe.dir
```

The watcher will:
1. Run an initial lint
2. Monitor for file changes
3. Re-lint after each change (with debouncing)
4. Display results in real-time

### Output

```
Watching project.toe.dir (Ctrl+C to stop)

[12:34:56] Run #1
  Violations: 0 errors, 2 warnings

[12:35:02] Run #2 (changed: op1.n)
  Violations: 0 errors, 1 warning
```

## Options

### Debounce Delay

Control how long to wait after file changes before re-linting:

```bash
# Default: 0.5 seconds
td-linter watch project.toe.dir --debounce 0.5

# Faster response (0.2 seconds)
td-linter watch project.toe.dir --debounce 0.2

# Slower response (1.0 second)
td-linter watch project.toe.dir --debounce 1.0
```

Lower values = faster feedback, but more CPU usage with rapid changes.

### Auto-Fix

Automatically apply fixes after each lint run:

```bash
td-linter watch project.toe.dir --fix
```

When `--fix` is enabled, fixable violations are automatically corrected.

### Screen Clearing

By default, the screen is cleared between runs:

```bash
# Clear screen (default)
td-linter watch project.toe.dir --clear

# Keep history visible
td-linter watch project.toe.dir --no-clear
```

### Rule Selection

Filter which rules to check:

```bash
# Only syntax and connection rules
td-linter watch project.toe.dir --select S,C

# Ignore performance rules
td-linter watch project.toe.dir --ignore F

# Use specific config
td-linter watch project.toe.dir --config strict.yaml
```

### Output Format

```bash
# JSON output for each run
td-linter watch project.toe.dir --format json

# SARIF output
td-linter watch project.toe.dir --format sarif
```

## Monitored File Types

Watch mode monitors these TouchDesigner file types:

| Extension | Description |
|-----------|-------------|
| `.n` | Node definition files |
| `.parm` | Parameter files |
| `.text` | Text content files |
| `.toc` | Table of contents/manifest |

Other files (like `.py` or `.glsl` embedded in text files) trigger re-linting when their parent `.text` file changes.

## Workflow Integration

### Development Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Terminal 1: Watch Mode                                     │
│  $ td-linter watch project.toe.dir --fix                    │
├─────────────────────────────────────────────────────────────┤
│  Terminal 2: Edit Files                                     │
│  $ vim project.toe.dir/operators/myop.n                     │
├─────────────────────────────────────────────────────────────┤
│  TouchDesigner: Hot Reload                                  │
│  Use toecollapse to reimport changes                        │
└─────────────────────────────────────────────────────────────┘
```

### With IDE

Run watch mode in a split terminal while editing in VS Code or another editor:

1. Open project directory in IDE
2. Run `td-linter watch project.toe.dir` in terminal
3. Edit files and see instant validation feedback

## Stopping Watch Mode

Press `Ctrl+C` to stop watching:

```
^C
Stopped watching. Ran 15 times.
```

## Troubleshooting

### "watchdog not installed"

Install the optional dependency:

```bash
pip install sevenrad-stills[td-linter-watch]
```

### High CPU Usage

Increase the debounce delay:

```bash
td-linter watch project.toe.dir --debounce 1.0
```

### Missing File Changes

Ensure the file extension is monitored (`.n`, `.parm`, `.text`, `.toc`).
Other files are ignored by the watcher.

### Permission Errors

On macOS, grant terminal access to the project directory if prompted.

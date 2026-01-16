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

## Event-Based API

For programmatic integration, use the `EventBasedWatcher` class which emits events for all actions.

### Basic Usage

```python
from pathlib import Path
from td_linter import lint
from td_linter.watch import EventBasedWatcher
from td_linter.events import EventType

def lint_project(toe_dir: Path):
    return lint(str(toe_dir))

watcher = EventBasedWatcher(
    Path("project.toe.dir"),
    lint_callback=lint_project,
    debounce_delay=0.5,
)

# Subscribe to specific events
watcher.events.subscribe(
    lambda e: print(f"Lint complete: {e.error_count} errors"),
    event_types={EventType.LINT_COMPLETE}
)

watcher.events.subscribe(
    lambda e: print(f"Error: {e.error_message}"),
    event_types={EventType.LINT_ERROR}
)

watcher.start()
```

### Available Events

| Event | When Fired | Useful Attributes |
|-------|------------|-------------------|
| `WatcherStartEvent` | Watcher starts | `toe_dir` |
| `WatcherStopEvent` | Watcher stops | `toe_dir`, `reason` |
| `FileChangeEvent` | File modified | `toe_dir`, `file_path` |
| `LintStartEvent` | Lint begins | `toe_dir` |
| `LintCompleteEvent` | Lint finishes | `violations`, `error_count`, `warning_count`, `duration_ms` |
| `LintErrorEvent` | Lint fails | `error`, `error_message` |

### Example: Custom Reporter

```python
import json
from datetime import datetime
from td_linter.watch import EventBasedWatcher
from td_linter.events import EventType, LintCompleteEvent

class JSONReporter:
    def __init__(self, output_file):
        self.output_file = output_file

    def on_lint_complete(self, event: LintCompleteEvent):
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": str(event.toe_dir),
            "errors": event.error_count,
            "warnings": event.warning_count,
            "duration_ms": event.duration_ms,
            "violations": [
                {"rule": v.rule, "message": v.message, "path": v.path}
                for v in event.violations
            ]
        }
        with open(self.output_file, "a") as f:
            f.write(json.dumps(report) + "\n")

reporter = JSONReporter("lint-results.jsonl")
watcher.events.subscribe(
    reporter.on_lint_complete,
    event_types={EventType.LINT_COMPLETE}
)
```

## Build Tool Integration

### Make

```makefile
# Makefile

.PHONY: lint lint-watch

lint:
	td-linter lint project.toe.dir --fail-on-warning

lint-watch:
	td-linter watch project.toe.dir --fix
```

### npm/package.json

```json
{
  "scripts": {
    "lint:td": "td-linter lint project.toe.dir",
    "lint:td:watch": "td-linter watch project.toe.dir",
    "lint:td:fix": "td-linter fix project.toe.dir"
  }
}
```

## Running as a Service

### systemd (Linux)

Create `/etc/systemd/system/td-linter-watch.service`:

```ini
[Unit]
Description=td-linter Watch Mode
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/td-linter watch project.toe.dir --no-clear
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable td-linter-watch
sudo systemctl start td-linter-watch
```

### launchd (macOS)

Create `~/Library/LaunchAgents/com.td-linter.watch.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.td-linter.watch</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/td-linter</string>
        <string>watch</string>
        <string>/path/to/project.toe.dir</string>
        <string>--no-clear</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.td-linter.watch.plist
```

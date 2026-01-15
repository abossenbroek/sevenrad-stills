# TDL-014: CLI Skeleton with Typer

---
id: TDL-014
status: pending
priority: high
phase: 1
depends_on: []
blocks: [TDL-050, TDL-054]
---

## Problem Statement

The linter needs a command-line interface. Users will invoke `td-linter lint <path>`, `td-linter rules`, etc. The CLI should follow modern conventions: subcommands, helpful error messages, colored output, and proper exit codes for CI integration.

## Acceptance Criteria

- [ ] `td-linter --help` shows usage
- [ ] `td-linter lint <path>` validates a .toe.dir
- [ ] `td-linter rules` lists available rules
- [ ] `td-linter init` creates default config
- [ ] `td-linter version` shows version info
- [ ] Exit code 0 for success, 1 for errors
- [ ] Colored output with `--no-color` fallback
- [ ] `--verbose` and `--quiet` flags work

## Files to Create

```
td_linter/
├── __main__.py            # Entry point
├── cli.py                 # CLI command definitions
└── __init__.py            # Version info
```

## Research Pointers

### Typer Framework

- https://typer.tiangolo.com/
- `pip install typer[all]` (includes rich)

Typer is built on Click but with type hints. Study:
- Command decoration
- Argument and Option handling
- Subcommands via `app.command()`
- Exit codes via `raise typer.Exit(1)`

### Rich for Output

- https://rich.readthedocs.io/
- Colored terminal output
- Tables, progress bars, tracebacks

### Command Structure

```
td-linter
├── lint <path>         # Main command
│   ├── --config        # Config file path
│   ├── --format        # text/json/sarif
│   ├── --pre-collapse  # Relaxed validation
│   └── --quiet         # Errors only
├── rules               # List rules
│   └── --category      # Filter by category
├── init                # Create config
│   └── --force         # Overwrite existing
└── version             # Show version
```

### CLI Design Best Practices

| Principle | Application |
|-----------|-------------|
| Fail fast | Validate path exists before processing |
| Clear errors | "Path not found: foo" not "Error" |
| Exit codes | 0=success, 1=lint errors, 2=bad args |
| Composable | JSON output for piping to other tools |
| Progressive | --quiet for CI, default for humans |

### Entry Point Configuration

In `pyproject.toml`:
```toml
[project.scripts]
td-linter = "td_linter.cli:app"
```

This makes `td-linter` available after `pip install`.

### Stub Implementation

Start with stub commands that:
1. Parse arguments correctly
2. Print "Not implemented yet"
3. Return correct exit codes

This lets you test CLI UX before implementing functionality.

## Testing CLI

- Use `typer.testing.CliRunner`
- Test help output
- Test error cases
- Test exit codes

## Definition of Done

All acceptance criteria checked. CLI structure ready for functionality to be added.

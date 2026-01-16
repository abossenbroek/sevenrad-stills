"""CLI for TouchDesigner Linter."""

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from td_linter import __version__
from td_linter.output import get_formatter

if TYPE_CHECKING:
    from td_linter.rules.base import LintRule, Violation
    from td_linter.rules.registry import RuleRegistry

app = typer.Typer(
    name="td-linter",
    help="Validate TouchDesigner .toe.dir expanded projects.",
    add_completion=False,
)
console = Console()


def _validate_toe_dir(path: Path) -> None:
    """Validate that path is a valid .toe.dir directory."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {path}")
        raise typer.Exit(1)

    if not path.name.endswith(".toe.dir"):
        console.print(
            f"[yellow]Warning:[/yellow] Path does not end with .toe.dir: {path}"
        )


def _apply_select_patterns(
    registry: "RuleRegistry",
    select: str,
    verbose: bool,
) -> list["LintRule"]:
    """Apply select patterns and return enabled rules."""
    select_patterns = [s.strip() for s in select.split(",")]
    enabled_rules = registry.select(select_patterns)
    if verbose:
        rule_ids = ", ".join(r.rule_id for r in enabled_rules)
        console.print(f"[dim]Selected: {rule_ids}[/dim]")
    return enabled_rules


def _apply_ignore_patterns(
    registry: "RuleRegistry",
    enabled_rules: list["LintRule"],
    ignore: str,
    verbose: bool,
) -> list["LintRule"]:
    """Apply ignore patterns and return filtered rules."""
    ignore_patterns = [i.strip() for i in ignore.split(",")]
    ignored_ids: set[str] = set()
    for pattern in ignore_patterns:
        if len(pattern) == 1:
            for rule in registry.by_category(pattern):
                ignored_ids.add(rule.rule_id)
        else:
            ignored_ids.add(pattern)
    filtered = [r for r in enabled_rules if r.rule_id not in ignored_ids]
    if verbose and ignored_ids:
        console.print(f"[dim]Ignored: {', '.join(ignored_ids)}[/dim]")
    return filtered


@app.command()
def lint(
    path: Path = typer.Argument(..., help="Path to .toe.dir directory"),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    select: str | None = typer.Option(
        None,
        "--select",
        "-s",
        help="Select categories/rules (comma-separated, e.g., S,C,G001)",
    ),
    ignore: str | None = typer.Option(
        None,
        "--ignore",
        "-i",
        help="Ignore categories/rules (comma-separated, e.g., F,P003)",
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json, sarif"
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable colored output"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only show errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    fail_on_warning: bool = typer.Option(
        False, "--fail-on-warning", help="Exit 1 on warnings (for strict CI)"
    ),
    validate_expressions: bool = typer.Option(
        False,
        "--validate-expressions",
        help="Validate Python expressions in .parm files",
    ),
    no_embedded: bool = typer.Option(
        False,
        "--no-embedded",
        help="Skip validation of embedded GLSL/Python code",
    ),
) -> None:
    """Validate a .toe.dir project."""
    _validate_toe_dir(path)

    if verbose:
        console.print(f"[dim]Linting: {path}[/dim]")

    # Import here to avoid circular imports and speed up --help
    from td_linter.linter import run_lint
    from td_linter.rules.registry import get_registry

    # Load configuration
    try:
        registry = get_registry(config)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from e

    # Apply CLI overrides for select/ignore
    enabled_rules = (
        _apply_select_patterns(registry, select, verbose)
        if select
        else registry.enabled()
    )

    if ignore:
        enabled_rules = _apply_ignore_patterns(registry, enabled_rules, ignore, verbose)

    if verbose:
        console.print(f"[dim]Running {len(enabled_rules)} rules[/dim]")

    violations = run_lint(
        path,
        validate_expressions=validate_expressions,
        validate_embedded=not no_embedded,
        rules=enabled_rules,
        config=registry.config,
    )

    # Format and output results
    formatter = get_formatter(output_format, no_color=no_color)
    project_path = str(path) if not quiet or violations else None
    output = formatter.format(violations, project_path=project_path)

    if output:
        console.print(output, highlight=False)

    # Determine exit code
    has_errors = any(v.severity == "error" for v in violations)
    has_warnings = any(v.severity == "warning" for v in violations)

    if has_errors or (fail_on_warning and has_warnings):
        raise typer.Exit(1)


@app.command()
def rules(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    category: str | None = typer.Option(
        None, "--category", help="Filter by category code (S, C, T, R, G, P, F)"
    ),
    show_disabled: bool = typer.Option(
        False, "--show-disabled", help="Include disabled rules in output"
    ),
) -> None:
    """List available lint rules."""
    from td_linter.rules.registry import get_registry

    try:
        registry = get_registry(config)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from e

    table = Table(title="Available Rules")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Name", style="white")
    table.add_column("Category", style="dim")
    table.add_column("Severity")
    table.add_column("Enabled", justify="center")

    all_info = registry.list_all_rules()

    for info in all_info:
        # Filter by category if specified
        if category and info["category_code"] != category.upper():
            continue

        # Skip disabled rules unless --show-disabled
        if not info["enabled"] and not show_disabled:
            continue

        severity = str(info["config_severity"])
        severity_style = {
            "error": "red",
            "warning": "yellow",
            "info": "blue",
        }.get(severity, "white")

        enabled_str = "[green]Yes[/green]" if info["enabled"] else "[dim]No[/dim]"

        table.add_row(
            str(info["rule_id"]),
            str(info["name"]),
            str(info["category"]),
            f"[{severity_style}]{severity}[/{severity_style}]",
            enabled_str,
        )

    console.print(table)

    # Show category legend
    legend = "S=syntax, C=connection, T=type, R=reference, G=glsl, P=python, F=perf"
    console.print(f"\n[dim]Categories: {legend}[/dim]")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
    preset: str = typer.Option(
        "recommended",
        "--preset",
        "-p",
        help="Preset to use: recommended, strict, minimal, pedantic",
    ),
) -> None:
    """Create default configuration file."""
    config_path = Path("td-linter.yaml")

    if config_path.exists() and not force:
        console.print(
            f"[yellow]Warning:[/yellow] Config exists at {config_path}. "
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    # Validate preset
    valid_presets = {"recommended", "strict", "minimal", "pedantic"}
    if preset not in valid_presets:
        options = ", ".join(valid_presets)
        console.print(
            f"[red]Error:[/red] Invalid preset '{preset}'. Options: {options}"
        )
        raise typer.Exit(1)

    default_config = f"""\
# td-linter configuration
# See: https://github.com/sevenrad/td-linter

# Schema version
version: "1.0.0"

# Extend a preset: recommended, strict, minimal, pedantic
extends: {preset}

# Select/ignore specific rules or categories (optional)
# select: [S, C, T, R, G, P]  # Enable only these
# ignore: [F001, F002]        # Disable these

# Rule-specific overrides (optional)
# rules:
#   C001:
#     severity: error
#   F001:
#     enabled: true
#     options:
#       max_depth: 15
"""
    config_path.write_text(default_config)
    console.print(f"[green]Created:[/green] {config_path} (preset: {preset})")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"td-linter {__version__}")


if __name__ == "__main__":
    app()

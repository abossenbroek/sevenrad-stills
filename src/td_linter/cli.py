"""CLI for TouchDesigner Linter."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from td_linter import __version__

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


@app.command()
def lint(
    path: Path = typer.Argument(..., help="Path to .toe.dir directory"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    output_format: str = typer.Option(
        "text", "--format", "-f", help="Output format: text, json"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only show errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Validate a .toe.dir project."""
    _validate_toe_dir(path)

    if verbose:
        console.print(f"[dim]Linting: {path}[/dim]")

    # Import here to avoid circular imports and speed up --help
    from td_linter.linter import run_lint

    violations = run_lint(path)

    if violations:
        for v in violations:
            color = "red" if v.severity == "error" else "yellow"
            console.print(f"[{color}]{v.rule}[/{color}]: {v.message}")
            if v.source_file and verbose:
                line_info = f":{v.line}" if v.line else ""
                console.print(f"  [dim]Location: {v.source_file}{line_info}[/dim]")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"[green]OK:[/green] {path.name} passed validation")


@app.command()
def rules() -> None:
    """List available lint rules."""
    from td_linter.linter import get_all_rules

    table = Table(title="Available Rules")
    table.add_column("ID", style="cyan")
    table.add_column("Description")
    table.add_column("Severity")

    for rule in get_all_rules():
        table.add_row(rule.id, rule.description, rule.severity)

    console.print(table)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
) -> None:
    """Create default configuration file."""
    config_path = Path(".td-linter.yaml")

    if config_path.exists() and not force:
        console.print(
            f"[yellow]Warning:[/yellow] Config exists at {config_path}. "
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    default_config = """\
# td-linter configuration
# See: https://github.com/sevenrad/td-linter

rules:
  # Structural validation
  no-invalid-cycles: error
  no-dangling-inputs: warning
  valid-operator-references: error
  type-compatibility: error
"""
    config_path.write_text(default_config)
    console.print(f"[green]Created:[/green] {config_path}")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"td-linter {__version__}")


if __name__ == "__main__":
    app()

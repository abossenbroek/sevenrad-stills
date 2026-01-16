"""Human-readable text output formatter."""

from collections import defaultdict
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console

from td_linter.output.base import OutputFormatter

if TYPE_CHECKING:
    from td_linter.rules.base import Violation


SEVERITY_COLORS = {
    "error": "red",
    "warning": "yellow",
    "info": "blue",
}


class TextFormatter(OutputFormatter):
    """Human-readable colored text output.

    Groups violations by source file and colors them by severity.
    Includes a summary at the end.
    """

    def __init__(self, no_color: bool = False) -> None:
        """Initialize the text formatter.

        Args:
            no_color: If True, disable colored output
        """
        self._no_color = no_color

    @property
    def name(self) -> str:
        """Return the formatter name."""
        return "text"

    @property
    def supports_color(self) -> bool:
        """Whether this formatter supports color output."""
        return True

    def format(
        self,
        violations: list["Violation"],
        project_path: str | None = None,
    ) -> str:
        """Format violations as human-readable text.

        Args:
            violations: List of violations to format
            project_path: Optional project path for success message

        Returns:
            Formatted text output with colors (unless no_color is set)
        """
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=not self._no_color,
            no_color=self._no_color,
            width=120,
        )

        if not violations:
            if project_path:
                console.print(f"[green]OK:[/green] {project_path} passed validation")
            return output.getvalue()

        # Group by source file
        by_file: dict[str, list["Violation"]] = defaultdict(list)
        for v in violations:
            key = str(v.source_file) if v.source_file else v.path
            by_file[key].append(v)

        # Output grouped violations
        for filepath, file_violations in sorted(by_file.items()):
            console.print(f"\n[bold]{filepath}[/bold]")
            for v in file_violations:
                color = SEVERITY_COLORS.get(v.severity, "white")
                line_str = f":{v.line}" if v.line else ""
                console.print(
                    f"  [{color}]{v.severity.upper()}[/{color}] "
                    f"{v.rule}{line_str}: {v.message}"
                )

        # Summary
        counts = {"error": 0, "warning": 0, "info": 0}
        for v in violations:
            sev = v.severity
            counts[sev] = counts.get(sev, 0) + 1

        console.print(
            f"\nSummary: {counts['error']} errors, "
            f"{counts['warning']} warnings, {counts['info']} info"
        )

        return output.getvalue()

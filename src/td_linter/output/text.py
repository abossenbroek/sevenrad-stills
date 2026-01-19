"""Human-readable text output formatter."""

from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from td_linter.output.base import OutputFormatter

if TYPE_CHECKING:
    from td_linter.rules.base import Violation


def _extract_node_name(violation: "Violation") -> tuple[str, str | None]:
    """Extract node name and container from violation path.

    Returns (node_name, container_path) where container_path is the parent path
    within the TD project (e.g., "project1" for "project1/chopexec1").
    """
    if violation.source_file:
        path = Path(violation.source_file)
        node_name = path.stem  # e.g., "chopexec1" from "chopexec1.text"

        # Find parent within .toe.dir
        parts = path.parts
        toe_dir_idx = None
        for i, part in enumerate(parts):
            if part.endswith(".toe.dir"):
                toe_dir_idx = i
                break

        if toe_dir_idx is not None and toe_dir_idx + 1 < len(parts) - 1:
            # Get container path (parts between .toe.dir and the file)
            container_parts = parts[toe_dir_idx + 1 : -1]
            container = "/".join(container_parts)
            return node_name, container

        return node_name, None

    # Fallback: extract from path string
    if "/" in violation.path:
        return violation.path.rsplit("/", 1)[-1], None
    return violation.path, None


SEVERITY_COLORS = {
    "error": "red",
    "warning": "yellow",
    "info": "blue",
}


def _get_container_rollups(
    violations: list["Violation"],
) -> list[tuple[str, int, int]]:
    """Find containers with multiple networks having errors inside.

    Returns a list of (container_path, network_count, error_count) tuples.
    Prefers showing sibling containers at a meaningful depth (like TD does).
    """
    # Count errors per unique path
    error_paths: set[str] = set()
    for v in violations:
        if v.severity == "error":
            error_paths.add(v.path)

    if not error_paths:
        return []

    # For each potential container, count distinct child networks with errors
    container_stats: dict[str, set[str]] = defaultdict(set)
    for path in error_paths:
        parts = path.split("/")
        for i in range(1, len(parts)):
            container = "/".join(parts[:i])
            container_stats[container].add(path)

    # Find containers with at least 3 networks with errors
    candidates = [
        (container, children)
        for container, children in container_stats.items()
        if len(children) >= 3
    ]

    if not candidates:
        return []

    # Group candidates by depth and find optimal depth level
    # Prefer depth where we have multiple sibling containers, each with errors
    by_depth: dict[int, list[tuple[str, set[str]]]] = defaultdict(list)
    for container, children in candidates:
        depth = container.count("/") + 1
        by_depth[depth].append((container, children))

    # Find the best depth: prefer depth 2 (like project1/EdgeBlend) if available
    # This matches how TD shows container-level errors
    best_depth = 2
    if 2 not in by_depth:
        # Fallback to lowest depth with containers
        for depth in sorted(by_depth.keys()):
            if depth >= 2:
                best_depth = depth
                break

    # Select containers at the best depth
    selected: list[tuple[str, int, int]] = []
    for container, children in by_depth.get(best_depth, []):
        error_count = sum(
            1 for v in violations if v.severity == "error" and v.path in children
        )
        selected.append((container, len(children), error_count))

    # Sort by error count descending
    selected.sort(key=lambda x: -x[2])

    # Limit to top 5 for readability
    return selected[:5]


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

        # Show container-level rollups first (like TD's "X networks with errors inside")
        rollups = _get_container_rollups(violations)
        if rollups:
            for container, network_count, error_count in rollups:
                console.print(
                    f"\n[bold red]/{container}[/bold red]: "
                    f"{network_count} networks with errors inside ({error_count} total errors)"
                )

        # Group by source file
        by_file: dict[str, list["Violation"]] = defaultdict(list)
        for v in violations:
            key = str(v.source_file) if v.source_file else v.path
            by_file[key].append(v)

        # Output grouped violations (individual errors)
        console.print("\n[dim]─── Individual errors ───[/dim]")
        for filepath, file_violations in sorted(by_file.items()):
            # Determine if we should show node-centric or file-centric header
            first_v = file_violations[0]
            node_name, container = _extract_node_name(first_v)
            language = first_v.context.get("language", "") if first_v.context else ""

            # Show node-centric header for embedded code (Python/GLSL)
            if language and node_name:
                lang_str = f" ({language})"
                container_str = f" in /{container}" if container else ""
                console.print(
                    f"\n[cyan]Node:[/cyan] [bold]{node_name}[/bold]{lang_str}{container_str}"
                )
                console.print(f"  [dim]{filepath}[/dim]")
            else:
                console.print(f"\n[bold]{filepath}[/bold]")

            for v in file_violations:
                color = SEVERITY_COLORS.get(v.severity, "white")
                line_str = f":{v.line}" if v.line else ""

                console.print(
                    f"  [{color}]{v.severity.upper()}[/{color}] "
                    f"{v.rule}{line_str}: {v.message}"
                )

                # Show source line with caret if available (LLVM-style)
                if v.context:
                    source_line = v.context.get("source_line")
                    if source_line is not None:
                        # Show the source line
                        console.print(f"    [dim]│[/dim] {source_line}")
                        # Show caret pointing to error position
                        raw_offset = v.context.get("offset") or v.context.get("column")
                        if isinstance(raw_offset, int) and raw_offset > 0:
                            caret_pos = " " * (raw_offset - 1) + "^"
                            console.print(f"    [dim]│[/dim] [{color}]{caret_pos}[/{color}]")

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

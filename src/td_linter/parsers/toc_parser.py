"""Parser for TouchDesigner .toc (table of contents) files."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedToc:
    """Parsed .toc file contents."""

    entries: list[str] = field(default_factory=list)
    special_entries: list[str] = field(default_factory=list)  # entries starting with .
    has_errors: bool = False
    source: Path | None = None


class TocParser:
    """
    Parser for TouchDesigner .toc files.

    TOC files are simple line-delimited lists of file paths relative to the
    .toe.dir root. Special entries start with '.' (e.g., .build, .start, .root).
    """

    def parse(self, file_path: Path) -> ParsedToc:
        """Parse a .toc file into structured data."""
        try:
            content = file_path.read_text()
            return self._parse_content(content, file_path)
        except Exception:
            return ParsedToc(has_errors=True, source=file_path)

    def parse_string(self, content: str) -> ParsedToc:
        """Parse a .toc file content string into structured data."""
        return self._parse_content(content, None)

    def _parse_content(self, content: str, source: Path | None) -> ParsedToc:
        """Parse TOC content into structured data."""
        entries: list[str] = []
        special_entries: list[str] = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("."):
                special_entries.append(line)
            else:
                entries.append(line)

        return ParsedToc(
            entries=entries,
            special_entries=special_entries,
            source=source,
        )

    def validate(self, toc: ParsedToc, toe_dir: Path) -> list[str]:
        """
        Validate that all TOC entries exist on disk.

        Returns a list of missing entries.
        """
        missing: list[str] = []
        for entry in toc.entries:
            if not (toe_dir / entry).exists():
                missing.append(entry)
        return missing

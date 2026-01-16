"""Auto-fix functionality for td-linter."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from td_linter.rules.base import Fix, Replacement, Violation


@dataclass
class FixResult:
    """Result of applying fixes."""

    applied: list[Fix] = field(default_factory=list)
    failed: list[tuple[Fix, Exception]] = field(default_factory=list)
    skipped: list[tuple[Fix, str]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Return number of successfully applied fixes."""
        return len(self.applied)

    @property
    def failure_count(self) -> int:
        """Return number of failed fixes."""
        return len(self.failed)

    @property
    def skipped_count(self) -> int:
        """Return number of skipped fixes."""
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        """Return total number of fixes attempted."""
        return self.success_count + self.failure_count + self.skipped_count


class FixApplier:
    """Applies fixes to files."""

    def __init__(self, dry_run: bool = False) -> None:
        """Initialize the fix applier.

        Args:
            dry_run: If True, don't modify files, just report what would change.
        """
        self.dry_run = dry_run
        self._file_cache: dict[Path, list[str]] = {}

    def apply(self, violations: Sequence[Violation]) -> FixResult:
        """Apply fixes from violations.

        Args:
            violations: Violations to apply fixes from.

        Returns:
            FixResult with applied, failed, and skipped fixes.
        """
        result = FixResult()

        # Filter to fixable violations
        fixable = [v for v in violations if v.fix is not None]

        if not fixable:
            return result

        # Group replacements by file
        by_file: dict[Path, list[tuple[Replacement, Fix]]] = {}
        for v in fixable:
            fix = v.fix
            if fix is None:
                continue
            for replacement in fix.replacements:
                path = replacement.file_path.resolve()
                if path not in by_file:
                    by_file[path] = []
                by_file[path].append((replacement, fix))

        # Apply fixes file by file
        for file_path, replacements_with_fixes in by_file.items():
            try:
                self._apply_to_file(file_path, replacements_with_fixes, result)
            except Exception as e:
                # Mark all fixes for this file as failed
                for _, fix in replacements_with_fixes:
                    if fix not in [f for f, _ in result.failed]:
                        result.failed.append((fix, e))

        return result

    def _apply_to_file(
        self,
        file_path: Path,
        replacements_with_fixes: list[tuple[Replacement, Fix]],
        result: FixResult,
    ) -> None:
        """Apply replacements to a single file.

        Replacements are sorted in reverse order (bottom-up) to preserve
        line numbers as we modify the file.
        """
        if not file_path.exists():
            for _, fix in replacements_with_fixes:
                result.skipped.append((fix, f"File not found: {file_path}"))
            return

        # Read file lines
        lines = self._read_file(file_path)

        # Sort replacements by line number (descending) to apply from bottom up
        replacements_with_fixes.sort(
            key=lambda x: (x[0].start_line, x[0].start_col or 0),
            reverse=True,
        )

        # Apply each replacement
        applied_fixes: set[int] = set()  # Track by id to avoid duplicates
        for replacement, fix in replacements_with_fixes:
            try:
                lines = self._apply_replacement(lines, replacement)
                if id(fix) not in applied_fixes:
                    applied_fixes.add(id(fix))
                    result.applied.append(fix)
            except Exception as e:
                result.failed.append((fix, e))

        # Write back if not dry run
        if not self.dry_run:
            self._write_file(file_path, lines)

    def _apply_replacement(
        self,
        lines: list[str],
        replacement: Replacement,
    ) -> list[str]:
        """Apply a single replacement to lines.

        Args:
            lines: Current file lines.
            replacement: Replacement to apply.

        Returns:
            Modified lines.

        Raises:
            ValueError: If line numbers are out of range.
        """
        start_line = replacement.start_line
        end_line = replacement.end_line

        # Validate line numbers (1-indexed)
        if start_line < 1 or end_line < 1:
            msg = f"Invalid line numbers: start={start_line}, end={end_line}"
            raise ValueError(msg)
        if start_line > len(lines) + 1 or end_line > len(lines) + 1:
            msg = f"Line numbers out of range: start={start_line}, end={end_line}, file has {len(lines)} lines"
            raise ValueError(msg)
        if start_line > end_line:
            msg = f"start_line ({start_line}) > end_line ({end_line})"
            raise ValueError(msg)

        # Convert to 0-indexed
        start_idx = start_line - 1
        end_idx = end_line  # end_line is inclusive, but slice is exclusive

        # Handle column-based replacement
        if replacement.start_col is not None and replacement.end_col is not None:
            # Single-line column replacement
            if start_line != end_line:
                msg = "Column replacement only supported for single-line changes"
                raise ValueError(msg)

            line = lines[start_idx]
            # Both columns are 1-indexed, convert to 0-indexed
            start_col = replacement.start_col - 1
            end_col = replacement.end_col - 1  # Exclusive end, convert to 0-indexed

            # Calculate effective line length (excluding trailing newline)
            line_content = line.rstrip("\n\r")
            trailing_newline = line[len(line_content) :]

            if start_col < 0 or end_col > len(line_content):
                msg = f"Column out of range: {start_col}:{end_col} for line of length {len(line_content)}"
                raise ValueError(msg)

            new_line = line_content[:start_col] + replacement.new_text + line_content[end_col:] + trailing_newline
            lines[start_idx] = new_line
        else:
            # Full line replacement
            new_lines = replacement.new_text.splitlines(keepends=True)
            # Ensure last line has newline if original did
            if new_lines and lines[start_idx:end_idx]:
                if not new_lines[-1].endswith("\n") and lines[end_idx - 1].endswith(
                    "\n"
                ):
                    new_lines[-1] += "\n"

            lines[start_idx:end_idx] = new_lines

        return lines

    def _read_file(self, file_path: Path) -> list[str]:
        """Read file lines, using cache if available."""
        resolved = file_path.resolve()
        if resolved not in self._file_cache:
            self._file_cache[resolved] = resolved.read_text().splitlines(keepends=True)
        return list(self._file_cache[resolved])  # Return copy

    def _write_file(self, file_path: Path, lines: list[str]) -> None:
        """Write lines back to file."""
        content = "".join(lines)
        file_path.write_text(content)
        # Invalidate cache
        resolved = file_path.resolve()
        if resolved in self._file_cache:
            del self._file_cache[resolved]

    def preview(self, violations: Sequence[Violation]) -> dict[Path, str]:
        """Preview fixes without applying them.

        Args:
            violations: Violations with fixes to preview.

        Returns:
            Dict mapping file paths to their new content after fixes.
        """
        result: dict[Path, str] = {}

        # Filter to fixable violations
        fixable = [v for v in violations if v.fix is not None]

        if not fixable:
            return result

        # Group replacements by file
        by_file: dict[Path, list[tuple[Replacement, Fix]]] = {}
        for v in fixable:
            fix = v.fix
            if fix is None:
                continue
            for replacement in fix.replacements:
                path = replacement.file_path.resolve()
                if path not in by_file:
                    by_file[path] = []
                by_file[path].append((replacement, fix))

        # Preview each file
        for file_path, replacements_with_fixes in by_file.items():
            if not file_path.exists():
                continue

            lines = self._read_file(file_path)

            # Sort replacements by line number (descending)
            replacements_with_fixes.sort(
                key=lambda x: (x[0].start_line, x[0].start_col or 0),
                reverse=True,
            )

            for replacement, _ in replacements_with_fixes:
                try:
                    lines = self._apply_replacement(lines, replacement)
                except Exception:
                    pass  # Skip failed replacements in preview

            result[file_path] = "".join(lines)

        return result

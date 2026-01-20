"""Auto-fix functionality for td-linter."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from td_linter.cache import BoundedLRUCache
from td_linter.rules.base import Fix, Replacement, Violation


class PathTraversalError(Exception):
    """Raised when a file path attempts to escape the project boundary."""

    def __init__(self, path: Path, boundary: Path) -> None:
        self.path = path
        self.boundary = boundary
        super().__init__(
            f"Path traversal detected: '{path}' is outside project boundary '{boundary}'"
        )


class ContentHashMismatchError(Exception):
    """Raised when file content has changed since fix was generated."""

    def __init__(self, file_path: Path, expected_hash: str, actual_hash: str) -> None:
        self.file_path = file_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Content hash mismatch for '{file_path}': "
            f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )


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
    """Applies fixes to files.

    Security:
        All file paths are validated against a project boundary to prevent
        path traversal attacks. Paths containing '..' or symlinks that resolve
        outside the boundary are rejected.

        Content hash verification ensures fixes are only applied to files that
        haven't changed since the fix was generated.
    """

    def __init__(
        self,
        dry_run: bool = False,
        project_root: Path | None = None,
        verify_hashes: bool = True,
    ) -> None:
        """Initialize the fix applier.

        Args:
            dry_run: If True, don't modify files, just report what would change.
            project_root: Optional project root for path validation. If provided,
                all file operations are restricted to this directory tree.
                This is a security measure to prevent path traversal attacks.
            verify_hashes: If True, verify content hashes before applying fixes.
                Disable with --force flag when you're sure you want to apply
                fixes to potentially modified files.
        """
        self.dry_run = dry_run
        self.verify_hashes = verify_hashes
        self._project_root = project_root.resolve() if project_root else None
        # Bounded cache for file contents - short TTL since files may change
        # Max 100 files to prevent memory issues during batch operations
        self._file_cache: BoundedLRUCache[Path, list[str]] = BoundedLRUCache(
            max_size=100,
            ttl_seconds=60.0,  # 1 minute TTL
        )

    def _validate_path(self, path: Path) -> Path:
        """Validate and canonicalize a file path.

        Args:
            path: Path to validate.

        Returns:
            Canonicalized absolute path.

        Raises:
            PathTraversalError: If the path escapes the project boundary.
        """
        # Resolve to absolute path, following symlinks
        # os.path.realpath follows ALL symlinks, unlike Path.resolve() which
        # may not follow symlinks on some platforms
        canonical = Path(os.path.realpath(path))

        # If no project root, only do basic validation
        if self._project_root is None:
            return canonical

        # Resolve project root the same way
        canonical_root = Path(os.path.realpath(self._project_root))

        # Check if path is within project root
        # Using is_relative_to for Python 3.9+ compatibility
        try:
            canonical.relative_to(canonical_root)
        except ValueError:
            # Path is not relative to root - this is a path traversal attempt
            raise PathTraversalError(path, self._project_root)

        return canonical

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _verify_content_hashes(
        self,
        replacements: list[Replacement],
    ) -> list[ContentHashMismatchError]:
        """Verify content hashes for all replacements.

        Args:
            replacements: List of replacements to verify.

        Returns:
            List of hash mismatch errors (empty if all valid).
        """
        errors: list[ContentHashMismatchError] = []
        verified_files: set[Path] = set()

        for replacement in replacements:
            # Skip if no hash to verify
            if replacement.content_hash is None:
                continue

            # Skip if already verified
            resolved = replacement.file_path.resolve()
            if resolved in verified_files:
                continue
            verified_files.add(resolved)

            # Verify hash
            try:
                actual_hash = self.compute_file_hash(resolved)
                if actual_hash != replacement.content_hash:
                    errors.append(
                        ContentHashMismatchError(
                            resolved, replacement.content_hash, actual_hash
                        )
                    )
            except OSError:
                # File doesn't exist or can't be read - will fail later
                pass

        return errors

    def apply(self, violations: Sequence[Violation]) -> FixResult:
        """Apply fixes from violations.

        Args:
            violations: Violations to apply fixes from.

        Returns:
            FixResult with applied, failed, and skipped fixes.

        Note:
            If a project_root was set, all file paths are validated to ensure
            they stay within the project boundary. Path traversal attempts
            (using '..' or symlinks) will result in failed fixes.

            If verify_hashes is True (default), content hashes are verified
            before applying any fixes. All fixes for a file will fail if the
            file content has changed since the fix was generated.
        """
        result = FixResult()

        # Filter to fixable violations
        fixable = [v for v in violations if v.fix is not None]

        if not fixable:
            return result

        # Collect all replacements for hash verification
        all_replacements: list[tuple[Replacement, Fix]] = []

        # Group replacements by file (with path validation)
        by_file: dict[Path, list[tuple[Replacement, Fix]]] = {}
        for v in fixable:
            fix = v.fix
            if fix is None:
                continue
            for replacement in fix.replacements:
                try:
                    # Validate and canonicalize the path
                    path = self._validate_path(replacement.file_path)
                except PathTraversalError as e:
                    result.failed.append((fix, e))
                    continue
                if path not in by_file:
                    by_file[path] = []
                by_file[path].append((replacement, fix))
                all_replacements.append((replacement, fix))

        # Verify content hashes if enabled (all-or-nothing)
        if self.verify_hashes:
            replacements_only = [r for r, _ in all_replacements]
            hash_errors = self._verify_content_hashes(replacements_only)
            if hash_errors:
                # Hash verification failed - mark affected fixes as failed
                failed_files = {e.file_path for e in hash_errors}
                for file_path in list(by_file.keys()):
                    if file_path in failed_files:
                        error = next(e for e in hash_errors if e.file_path == file_path)
                        for _, fix in by_file[file_path]:
                            if fix not in [f for f, _ in result.failed]:
                                result.failed.append((fix, error))
                        del by_file[file_path]

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
        cached = self._file_cache.get(resolved)
        if cached is None:
            lines = resolved.read_text().splitlines(keepends=True)
            self._file_cache.set(resolved, lines)
            return list(lines)  # Return copy
        return list(cached)  # Return copy

    def _write_file(self, file_path: Path, lines: list[str]) -> None:
        """Write lines back to file."""
        content = "".join(lines)
        file_path.write_text(content)
        # Invalidate cache
        resolved = file_path.resolve()
        self._file_cache.delete(resolved)

    def preview(self, violations: Sequence[Violation]) -> dict[Path, str]:
        """Preview fixes without applying them.

        Args:
            violations: Violations with fixes to preview.

        Returns:
            Dict mapping file paths to their new content after fixes.
            Paths that fail validation are silently skipped in preview.
        """
        result: dict[Path, str] = {}

        # Filter to fixable violations
        fixable = [v for v in violations if v.fix is not None]

        if not fixable:
            return result

        # Group replacements by file (with path validation)
        by_file: dict[Path, list[tuple[Replacement, Fix]]] = {}
        for v in fixable:
            fix = v.fix
            if fix is None:
                continue
            for replacement in fix.replacements:
                try:
                    path = self._validate_path(replacement.file_path)
                except PathTraversalError:
                    continue  # Skip paths that fail validation in preview
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

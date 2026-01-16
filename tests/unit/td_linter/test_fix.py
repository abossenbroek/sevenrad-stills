"""Unit tests for auto-fix functionality."""

from pathlib import Path

import pytest

from td_linter.fix import FixApplier, FixResult
from td_linter.rules.base import Fix, Replacement, Violation


class TestReplacement:
    """Tests for the Replacement dataclass."""

    def test_replacement_creation(self) -> None:
        """Replacement should store all fields correctly."""
        r = Replacement(
            file_path=Path("/test/file.txt"),
            start_line=1,
            end_line=1,
            start_col=5,
            end_col=10,
            new_text="replacement",
        )
        assert r.file_path == Path("/test/file.txt")
        assert r.start_line == 1
        assert r.end_line == 1
        assert r.start_col == 5
        assert r.end_col == 10
        assert r.new_text == "replacement"

    def test_replacement_defaults(self) -> None:
        """Replacement should have sensible defaults."""
        r = Replacement(
            file_path=Path("/test/file.txt"),
            start_line=1,
            end_line=1,
        )
        assert r.start_col is None
        assert r.end_col is None
        assert r.new_text == ""


class TestFix:
    """Tests for the Fix dataclass."""

    def test_fix_creation(self) -> None:
        """Fix should store description and replacements."""
        replacements = [
            Replacement(Path("/test.txt"), 1, 1, new_text="new"),
        ]
        fix = Fix(description="Test fix", replacements=replacements)
        assert fix.description == "Test fix"
        assert len(fix.replacements) == 1

    def test_fix_empty_replacements(self) -> None:
        """Fix should allow empty replacements list."""
        fix = Fix(description="Empty fix")
        assert fix.replacements == []


class TestViolationWithFix:
    """Tests for Violation with fix field."""

    def test_violation_without_fix(self) -> None:
        """Violation should have None fix by default."""
        v = Violation(rule="TEST", message="Test", path="/test")
        assert v.fix is None

    def test_violation_with_fix(self) -> None:
        """Violation should accept a Fix object."""
        fix = Fix(description="Fix it")
        v = Violation(rule="TEST", message="Test", path="/test", fix=fix)
        assert v.fix is not None
        assert v.fix.description == "Fix it"


class TestFixResult:
    """Tests for FixResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result should have zero counts."""
        result = FixResult()
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.skipped_count == 0
        assert result.total_count == 0

    def test_result_counts(self) -> None:
        """Result should correctly count all categories."""
        result = FixResult(
            applied=[Fix("a"), Fix("b")],
            failed=[(Fix("c"), Exception("error"))],
            skipped=[(Fix("d"), "reason")],
        )
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.skipped_count == 1
        assert result.total_count == 4


class TestFixApplierPositive:
    """Positive test cases for FixApplier."""

    def test_apply_single_line_deletion(self, tmp_path: Path) -> None:
        """Should delete a single line."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\nline3\n")

        fix = Fix(
            description="Delete line 2",
            replacements=[
                Replacement(file_path=file, start_line=2, end_line=2, new_text="")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "line1\nline3\n"

    def test_apply_single_line_replacement(self, tmp_path: Path) -> None:
        """Should replace a single line."""
        file = tmp_path / "test.txt"
        file.write_text("old line\n")

        fix = Fix(
            description="Replace line",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="new line\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "new line\n"

    def test_apply_multi_line_replacement(self, tmp_path: Path) -> None:
        """Should replace multiple lines with different content."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\nline3\nline4\n")

        fix = Fix(
            description="Replace lines 2-3",
            replacements=[
                Replacement(
                    file_path=file, start_line=2, end_line=3, new_text="new content\n"
                )
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "line1\nnew content\nline4\n"

    def test_apply_column_replacement(self, tmp_path: Path) -> None:
        """Should replace text at specific column positions."""
        file = tmp_path / "test.txt"
        file.write_text("hello world\n")

        fix = Fix(
            description="Replace 'world' with 'there'",
            replacements=[
                Replacement(
                    file_path=file,
                    start_line=1,
                    end_line=1,
                    start_col=7,
                    end_col=12,
                    new_text="there",
                )
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "hello there\n"

    def test_apply_insertion(self, tmp_path: Path) -> None:
        """Should insert new content."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline3\n")

        fix = Fix(
            description="Insert line 2",
            replacements=[
                Replacement(
                    file_path=file,
                    start_line=2,
                    end_line=2,
                    new_text="line2\nline3\n",
                )
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "line1\nline2\nline3\n"

    def test_apply_multiple_violations_same_file(self, tmp_path: Path) -> None:
        """Should apply multiple fixes to the same file (bottom-up order)."""
        file = tmp_path / "test.txt"
        file.write_text("a\nb\nc\nd\n")

        violations = [
            Violation(
                rule="T",
                message="m",
                path="/p",
                fix=Fix(
                    description="Replace line 2",
                    replacements=[
                        Replacement(file_path=file, start_line=2, end_line=2, new_text="B\n")
                    ],
                ),
            ),
            Violation(
                rule="T",
                message="m",
                path="/p",
                fix=Fix(
                    description="Replace line 4",
                    replacements=[
                        Replacement(file_path=file, start_line=4, end_line=4, new_text="D\n")
                    ],
                ),
            ),
        ]

        applier = FixApplier()
        result = applier.apply(violations)

        assert result.success_count == 2
        assert file.read_text() == "a\nB\nc\nD\n"

    def test_apply_multiple_files(self, tmp_path: Path) -> None:
        """Should apply fixes to multiple files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("old1\n")
        file2.write_text("old2\n")

        violations = [
            Violation(
                rule="T",
                message="m",
                path="/p",
                fix=Fix(
                    description="Fix file1",
                    replacements=[
                        Replacement(file_path=file1, start_line=1, end_line=1, new_text="new1\n")
                    ],
                ),
            ),
            Violation(
                rule="T",
                message="m",
                path="/p",
                fix=Fix(
                    description="Fix file2",
                    replacements=[
                        Replacement(file_path=file2, start_line=1, end_line=1, new_text="new2\n")
                    ],
                ),
            ),
        ]

        applier = FixApplier()
        result = applier.apply(violations)

        assert result.success_count == 2
        assert file1.read_text() == "new1\n"
        assert file2.read_text() == "new2\n"

    def test_dry_run_does_not_modify(self, tmp_path: Path) -> None:
        """Dry run should not modify files."""
        file = tmp_path / "test.txt"
        original_content = "original\n"
        file.write_text(original_content)

        fix = Fix(
            description="Would change",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="changed\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier(dry_run=True)
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == original_content  # Unchanged

    def test_preview_returns_modified_content(self, tmp_path: Path) -> None:
        """Preview should return what the file would look like."""
        file = tmp_path / "test.txt"
        file.write_text("original\n")

        fix = Fix(
            description="Change",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="changed\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        previews = applier.preview([violation])

        assert file.resolve() in previews
        assert previews[file.resolve()] == "changed\n"
        assert file.read_text() == "original\n"  # File unchanged


class TestFixApplierNegative:
    """Negative test cases for FixApplier."""

    def test_no_violations_returns_empty(self) -> None:
        """Empty violation list should return empty result."""
        applier = FixApplier()
        result = applier.apply([])
        assert result.total_count == 0

    def test_violations_without_fix_skipped(self) -> None:
        """Violations without fix field should be ignored."""
        violation = Violation(rule="T", message="m", path="/p", fix=None)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.total_count == 0

    def test_file_not_found_skipped(self, tmp_path: Path) -> None:
        """Missing files should be skipped with reason."""
        nonexistent = tmp_path / "does_not_exist.txt"
        fix = Fix(
            description="Fix missing file",
            replacements=[
                Replacement(file_path=nonexistent, start_line=1, end_line=1, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.skipped_count == 1
        assert "not found" in result.skipped[0][1].lower()

    def test_invalid_line_number_zero(self, tmp_path: Path) -> None:
        """Line number 0 should fail."""
        file = tmp_path / "test.txt"
        file.write_text("line1\n")

        fix = Fix(
            description="Invalid line",
            replacements=[
                Replacement(file_path=file, start_line=0, end_line=0, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_invalid_line_number_negative(self, tmp_path: Path) -> None:
        """Negative line numbers should fail."""
        file = tmp_path / "test.txt"
        file.write_text("line1\n")

        fix = Fix(
            description="Negative line",
            replacements=[
                Replacement(file_path=file, start_line=-1, end_line=-1, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_line_number_out_of_range(self, tmp_path: Path) -> None:
        """Line number beyond file length should fail."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n")

        fix = Fix(
            description="Out of range",
            replacements=[
                Replacement(file_path=file, start_line=100, end_line=100, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_start_line_after_end_line(self, tmp_path: Path) -> None:
        """start_line > end_line should fail."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n")

        fix = Fix(
            description="Invalid range",
            replacements=[
                Replacement(file_path=file, start_line=2, end_line=1, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_column_out_of_range(self, tmp_path: Path) -> None:
        """Column beyond line length should fail."""
        file = tmp_path / "test.txt"
        file.write_text("short\n")

        fix = Fix(
            description="Column out of range",
            replacements=[
                Replacement(
                    file_path=file,
                    start_line=1,
                    end_line=1,
                    start_col=1,
                    end_col=100,
                    new_text="x",
                )
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_multiline_column_replacement_fails(self, tmp_path: Path) -> None:
        """Column replacement spanning multiple lines should fail."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n")

        fix = Fix(
            description="Multiline column",
            replacements=[
                Replacement(
                    file_path=file,
                    start_line=1,
                    end_line=2,
                    start_col=1,
                    end_col=5,
                    new_text="x",
                )
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.failure_count == 1

    def test_empty_file_insertion_at_line_1(self, tmp_path: Path) -> None:
        """Inserting at line 1 in empty file should add content."""
        file = tmp_path / "empty.txt"
        file.write_text("")

        fix = Fix(
            description="Add content to empty file",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="new content\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        # Inserting at line 1 in empty file is valid - adds the content
        assert result.success_count == 1
        assert file.read_text() == "new content\n"

    def test_empty_file_line_2_fails(self, tmp_path: Path) -> None:
        """Replacement at line 2 in empty file should fail."""
        file = tmp_path / "empty.txt"
        file.write_text("")

        fix = Fix(
            description="Fix empty file at line 2",
            replacements=[
                Replacement(file_path=file, start_line=2, end_line=2, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        # Line 2 doesn't exist in empty file
        assert result.failure_count == 1

    def test_preview_nonexistent_file_skipped(self, tmp_path: Path) -> None:
        """Preview should skip nonexistent files."""
        nonexistent = tmp_path / "missing.txt"
        fix = Fix(
            description="Fix",
            replacements=[
                Replacement(file_path=nonexistent, start_line=1, end_line=1, new_text="x")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        previews = applier.preview([violation])

        assert len(previews) == 0


class TestFixApplierEdgeCases:
    """Edge case tests for FixApplier."""

    def test_file_without_trailing_newline(self, tmp_path: Path) -> None:
        """Should handle files without trailing newline."""
        file = tmp_path / "test.txt"
        file.write_text("no newline at end")

        fix = Fix(
            description="Add newline",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="with newline\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "with newline\n"

    def test_preserves_original_newline_style(self, tmp_path: Path) -> None:
        """Should preserve original line endings."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n")

        fix = Fix(
            description="Replace",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="new1\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        content = file.read_text()
        assert content == "new1\nline2\n"
        assert "\r" not in content

    def test_multiple_replacements_in_same_fix(self, tmp_path: Path) -> None:
        """Single fix with multiple replacements should work."""
        file = tmp_path / "test.txt"
        file.write_text("a\nb\nc\n")

        fix = Fix(
            description="Multiple replacements",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="A\n"),
                Replacement(file_path=file, start_line=3, end_line=3, new_text="C\n"),
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "A\nb\nC\n"

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Should handle unicode content correctly."""
        file = tmp_path / "test.txt"
        file.write_text("héllo wörld\n", encoding="utf-8")

        fix = Fix(
            description="Replace unicode",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=1, new_text="你好世界\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text(encoding="utf-8") == "你好世界\n"

    def test_file_with_only_newlines(self, tmp_path: Path) -> None:
        """Should handle file with only newlines."""
        file = tmp_path / "test.txt"
        file.write_text("\n\n\n")

        fix = Fix(
            description="Replace empty line",
            replacements=[
                Replacement(file_path=file, start_line=2, end_line=2, new_text="content\n")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == "\ncontent\n\n"

    def test_delete_all_content(self, tmp_path: Path) -> None:
        """Should be able to delete all content."""
        file = tmp_path / "test.txt"
        file.write_text("line1\nline2\n")

        fix = Fix(
            description="Delete all",
            replacements=[
                Replacement(file_path=file, start_line=1, end_line=2, new_text="")
            ],
        )
        violation = Violation(rule="T", message="m", path="/p", fix=fix)

        applier = FixApplier()
        result = applier.apply([violation])

        assert result.success_count == 1
        assert file.read_text() == ""

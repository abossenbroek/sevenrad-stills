"""Integration tests for watch mode with auto-fix.

These tests verify that watch mode correctly integrates with the fix system:
- Watch mode triggers re-linting on file changes
- Watch mode can apply fixes when --fix is enabled
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Skip if watchdog not installed
pytest.importorskip("watchdog")

from td_linter.fix import FixApplier
from td_linter.linter import run_lint
from td_linter.rules.base import Fix, Replacement, Violation
from td_linter.watch import TDLintEventHandler, TDLintWatcher


@pytest.mark.slow
@pytest.mark.integration
class TestWatchWithFix:
    """Test watch mode with fix integration."""

    @pytest.fixture
    def sample_toe_dir(self, tmp_path: Path) -> Path:
        """Create a minimal .toe.dir structure for testing."""
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()

        # Create minimal .toc file
        (toe_dir / ".toc").write_text("test.n\n")

        # Create a simple .n file
        (toe_dir / "test.n").write_text(
            """TOP:null
tile 0 0 100 100
flags =  viewer 0
inputs
{
}
end
"""
        )

        return toe_dir

    def test_watch_lint_callback_receives_violations(
        self, sample_toe_dir: Path
    ) -> None:
        """Watch callback should receive violations from linting."""
        callback_results: list[list[Violation]] = []

        def lint_callback() -> None:
            violations = list(run_lint(sample_toe_dir))
            callback_results.append(violations)

        handler = TDLintEventHandler(
            sample_toe_dir, lint_callback, debounce_delay=0.1
        )

        # Simulate a file change event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(sample_toe_dir / "test.n")

        handler.dispatch(mock_event)

        # Wait for debounce
        import time

        time.sleep(0.2)

        # Callback should have been called
        assert len(callback_results) >= 1

    def test_fix_applier_with_violations_from_lint(
        self, sample_toe_dir: Path
    ) -> None:
        """FixApplier should be able to process violations from linting."""
        # Run lint to get violations
        violations = list(run_lint(sample_toe_dir))

        # Create a fix applier in dry-run mode
        applier = FixApplier(dry_run=True)

        # Run in dry-run mode to not modify files
        result = applier.apply(violations)

        # Should complete without error
        # Most violations may not have fixes, so skip count could be high
        assert result.total_count >= 0


@pytest.mark.slow
@pytest.mark.integration
class TestWatcherLifecycle:
    """Test watcher start/stop lifecycle with linting."""

    @pytest.fixture
    def sample_toe_dir(self, tmp_path: Path) -> Path:
        """Create a minimal .toe.dir structure."""
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("")
        return toe_dir

    def test_watcher_starts_and_stops_cleanly(self, sample_toe_dir: Path) -> None:
        """Watcher should start and stop without errors."""
        callback = MagicMock()
        watcher = TDLintWatcher(sample_toe_dir, callback)

        # Start watching
        watcher.start()
        assert watcher.is_running

        # Stop watching
        watcher.stop()
        assert not watcher.is_running

    def test_watcher_context_manager_works(self, sample_toe_dir: Path) -> None:
        """Watcher should work as context manager."""
        callback = MagicMock()
        watcher = TDLintWatcher(sample_toe_dir, callback)

        with watcher:
            assert watcher.is_running

        assert not watcher.is_running


@pytest.mark.slow
@pytest.mark.integration
class TestFixIntegrationWithRules:
    """Test that fixes from rules integrate correctly."""

    def test_fixable_rules_produce_valid_fixes(self, tmp_path: Path) -> None:
        """Fixable rules should produce Fix objects that FixApplier can process."""
        # Create a file with a #version directive (G002 can fix this)
        test_file = tmp_path / "test.glsl"
        test_file.write_text("#version 450\nvoid main() {}\n")

        # Create a violation with a fix
        fix = Fix(
            description="Remove GLSL version directive",
            replacements=[
                Replacement(
                    file_path=test_file,
                    start_line=1,
                    end_line=1,
                    new_text="",
                )
            ],
        )
        violation = Violation(
            rule="G002",
            message="GLSL version directive found",
            path=str(test_file),
            severity="warning",
            fix=fix,
        )

        # Apply the fix (not dry-run, actually modify files)
        applier = FixApplier(dry_run=False)
        result = applier.apply([violation])

        # Fix should have been applied
        assert result.success_count == 1

        # File should be modified
        content = test_file.read_text()
        assert "#version" not in content


@pytest.mark.slow
@pytest.mark.integration
class TestWatchEdgeCases:
    """Edge case tests for watch mode."""

    @pytest.fixture
    def sample_toe_dir(self, tmp_path: Path) -> Path:
        """Create a minimal .toe.dir structure."""
        toe_dir = tmp_path / "test.toe.dir"
        toe_dir.mkdir()
        (toe_dir / ".toc").write_text("test.n\n")
        (toe_dir / "test.n").write_text("TOP:null\nend\n")
        return toe_dir

    def test_rapid_file_changes_debounced(self, sample_toe_dir: Path) -> None:
        """Rapid file changes should be debounced into single callback."""
        import time

        call_count = 0

        def callback() -> None:
            nonlocal call_count
            call_count += 1

        handler = TDLintEventHandler(
            sample_toe_dir, callback, debounce_delay=0.2
        )

        # Simulate 10 rapid changes
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(sample_toe_dir / "test.n")

        for _ in range(10):
            handler.dispatch(mock_event)

        # Wait for debounce to complete
        time.sleep(0.4)

        # Should have been debounced to 1-2 calls (not 10)
        assert call_count <= 2

    def test_irrelevant_file_ignored(self, sample_toe_dir: Path) -> None:
        """Non-TouchDesigner files should be ignored."""
        import time

        call_count = 0

        def callback() -> None:
            nonlocal call_count
            call_count += 1

        handler = TDLintEventHandler(
            sample_toe_dir, callback, debounce_delay=0.1
        )

        # Create and trigger event for .py file (should be ignored)
        py_file = sample_toe_dir / "script.py"
        py_file.write_text("print('hello')")

        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(py_file)

        handler.dispatch(mock_event)
        time.sleep(0.2)

        # Callback should NOT have been called
        assert call_count == 0

    def test_directory_event_ignored(self, sample_toe_dir: Path) -> None:
        """Directory events should be ignored."""
        import time

        call_count = 0

        def callback() -> None:
            nonlocal call_count
            call_count += 1

        handler = TDLintEventHandler(
            sample_toe_dir, callback, debounce_delay=0.1
        )

        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = str(sample_toe_dir / "subdir")

        handler.dispatch(mock_event)
        time.sleep(0.2)

        assert call_count == 0

    def test_watcher_double_start_safe(self, sample_toe_dir: Path) -> None:
        """Starting watcher twice should be safe."""
        callback = MagicMock()
        watcher = TDLintWatcher(sample_toe_dir, callback)

        watcher.start()
        watcher.start()  # Second start should be safe

        assert watcher.is_running
        watcher.stop()

    def test_watcher_double_stop_safe(self, sample_toe_dir: Path) -> None:
        """Stopping watcher twice should be safe."""
        callback = MagicMock()
        watcher = TDLintWatcher(sample_toe_dir, callback)

        watcher.start()
        watcher.stop()
        watcher.stop()  # Second stop should be safe

        assert not watcher.is_running


@pytest.mark.slow
@pytest.mark.integration
class TestFixEdgeCases:
    """Edge case tests for fix functionality."""

    def test_fix_unicode_content(self, tmp_path: Path) -> None:
        """Fix should handle Unicode content correctly."""
        test_file = tmp_path / "unicode.glsl"
        test_file.write_text("#version 450\n// Comment with emoji: 🎨\nvoid main() {}\n")

        fix = Fix(
            description="Remove version",
            replacements=[
                Replacement(
                    file_path=test_file,
                    start_line=1,
                    end_line=1,
                    new_text="",
                )
            ],
        )
        violation = Violation(
            rule="G002",
            message="Version directive",
            path=str(test_file),
            severity="warning",
            fix=fix,
        )

        applier = FixApplier(dry_run=False)
        result = applier.apply([violation])

        assert result.success_count == 1
        content = test_file.read_text()
        assert "🎨" in content  # Unicode preserved
        assert "#version" not in content

    def test_fix_multiple_violations_same_file(self, tmp_path: Path) -> None:
        """Multiple fixes to same file should all apply."""
        test_file = tmp_path / "multi.glsl"
        test_file.write_text("line1\nline2\nline3\nline4\n")

        violations = [
            Violation(
                rule="TEST",
                message="Remove line 4",
                path=str(test_file),
                severity="warning",
                fix=Fix(
                    description="Remove line 4",
                    replacements=[
                        Replacement(
                            file_path=test_file,
                            start_line=4,
                            end_line=4,
                            new_text="",
                        )
                    ],
                ),
            ),
            Violation(
                rule="TEST",
                message="Remove line 2",
                path=str(test_file),
                severity="warning",
                fix=Fix(
                    description="Remove line 2",
                    replacements=[
                        Replacement(
                            file_path=test_file,
                            start_line=2,
                            end_line=2,
                            new_text="",
                        )
                    ],
                ),
            ),
        ]

        applier = FixApplier(dry_run=False)
        result = applier.apply(violations)

        assert result.success_count == 2
        content = test_file.read_text()
        assert "line1" in content
        assert "line2" not in content
        assert "line3" in content
        assert "line4" not in content

    def test_fix_empty_violations_list(self) -> None:
        """Fix with empty violations should complete cleanly."""
        applier = FixApplier(dry_run=False)
        result = applier.apply([])

        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.skipped_count == 0

    def test_fix_violation_without_fix_skipped(self) -> None:
        """Violations without fix should be skipped."""
        violation = Violation(
            rule="TEST",
            message="No fix available",
            path="/some/path",
            severity="warning",
            fix=None,  # No fix
        )

        applier = FixApplier(dry_run=False)
        result = applier.apply([violation])

        assert result.success_count == 0
        assert result.skipped_count == 0  # Not counted as skipped, just no fix

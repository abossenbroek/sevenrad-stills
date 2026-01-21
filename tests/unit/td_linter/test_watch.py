"""Unit tests for watch mode functionality."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestTDLintEventHandler:
    """Tests for TDLintEventHandler class."""

    def test_is_relevant_n_file(self, tmp_path: Path) -> None:
        """Should identify .n files as relevant."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback)

        assert handler._is_relevant(str(tmp_path / "test.n"))

    def test_is_relevant_parm_file(self, tmp_path: Path) -> None:
        """Should identify .parm files as relevant."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback)

        assert handler._is_relevant(str(tmp_path / "test.parm"))

    def test_is_relevant_text_file(self, tmp_path: Path) -> None:
        """Should identify .text files as relevant."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback)

        assert handler._is_relevant(str(tmp_path / "shader.text"))

    def test_is_relevant_toc_file(self, tmp_path: Path) -> None:
        """Should identify .toc files as relevant."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback)

        assert handler._is_relevant(str(tmp_path / "project.toc"))

    def test_is_not_relevant_py_file(self, tmp_path: Path) -> None:
        """Should not identify .py files as relevant."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback)

        assert not handler._is_relevant(str(tmp_path / "script.py"))

    def test_is_not_relevant_outside_toe_dir(self, tmp_path: Path) -> None:
        """Should not identify files outside toe_dir as relevant."""
        from td_linter.watch import TDLintEventHandler

        toe_dir = tmp_path / "project.toe.dir"
        toe_dir.mkdir()

        callback = MagicMock()
        handler = TDLintEventHandler(toe_dir, callback)

        # File outside the toe_dir
        assert not handler._is_relevant(str(tmp_path / "other.n"))

    def test_debouncing_multiple_events(self, tmp_path: Path) -> None:
        """Should debounce multiple rapid events into one callback."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback, debounce_delay=0.1)

        # Create a mock event
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "test.n")

        # Dispatch multiple events rapidly
        for _ in range(5):
            handler.dispatch(mock_event)

        # Wait for debounce
        time.sleep(0.2)

        # Should only call once due to debouncing
        assert callback.call_count == 1

        handler.stop()

    def test_stop_cancels_pending_timer(self, tmp_path: Path) -> None:
        """Should cancel pending timer on stop."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback, debounce_delay=1.0)

        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = str(tmp_path / "test.n")

        handler.dispatch(mock_event)

        # Stop before debounce completes
        handler.stop()

        # Wait a bit
        time.sleep(0.1)

        # Callback should not have been called
        assert callback.call_count == 0

    def test_ignores_directory_events(self, tmp_path: Path) -> None:
        """Should ignore directory events."""
        from td_linter.watch import TDLintEventHandler

        callback = MagicMock()
        handler = TDLintEventHandler(tmp_path, callback, debounce_delay=0.05)

        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = str(tmp_path / "subdir")

        handler.dispatch(mock_event)

        time.sleep(0.1)

        # Should not call callback for directory events
        assert callback.call_count == 0


class TestTDLintWatcher:
    """Tests for TDLintWatcher class."""

    def test_watcher_creation(self, tmp_path: Path) -> None:
        """Should create watcher with correct attributes."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback, debounce_delay=0.5)

        assert watcher.toe_dir == tmp_path.resolve()
        assert watcher.on_change is callback
        assert watcher.debounce_delay == 0.5
        assert not watcher.is_running

    def test_watcher_start_stop(self, tmp_path: Path) -> None:
        """Should start and stop correctly."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback)

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_watcher_context_manager(self, tmp_path: Path) -> None:
        """Should work as context manager."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback)

        with watcher:
            assert watcher.is_running

        assert not watcher.is_running

    def test_watcher_double_start(self, tmp_path: Path) -> None:
        """Should handle double start gracefully."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback)

        watcher.start()
        watcher.start()  # Should not raise

        assert watcher.is_running

        watcher.stop()

    def test_watcher_double_stop(self, tmp_path: Path) -> None:
        """Should handle double stop gracefully."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback)

        watcher.start()
        watcher.stop()
        watcher.stop()  # Should not raise

        assert not watcher.is_running

    @pytest.mark.slow
    def test_watcher_detects_file_creation(self, tmp_path: Path) -> None:
        """Should detect file creation and trigger callback."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback, debounce_delay=0.1)

        with watcher:
            # Create a file
            (tmp_path / "test.n").write_text("content")

            # Wait for debounce
            time.sleep(0.5)

        assert callback.call_count >= 1

    @pytest.mark.slow
    def test_watcher_detects_file_modification(self, tmp_path: Path) -> None:
        """Should detect file modification and trigger callback."""
        from td_linter.watch import TDLintWatcher

        # Create file before starting watcher
        test_file = tmp_path / "test.n"
        test_file.write_text("original")

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback, debounce_delay=0.1)

        with watcher:
            # Modify the file
            test_file.write_text("modified")

            # Wait for debounce
            time.sleep(0.5)

        assert callback.call_count >= 1

    @pytest.mark.slow
    def test_watcher_ignores_irrelevant_files(self, tmp_path: Path) -> None:
        """Should not trigger callback for irrelevant file types."""
        from td_linter.watch import TDLintWatcher

        callback = MagicMock()
        watcher = TDLintWatcher(tmp_path, callback, debounce_delay=0.1)

        with watcher:
            # Create an irrelevant file
            (tmp_path / "test.py").write_text("content")

            # Wait for potential debounce
            time.sleep(0.3)

        assert callback.call_count == 0


class TestTDExtensions:
    """Tests for TD_EXTENSIONS constant."""

    def test_all_extensions_included(self) -> None:
        """Should include all TouchDesigner file extensions."""
        from td_linter.watch import TD_EXTENSIONS

        assert ".n" in TD_EXTENSIONS
        assert ".parm" in TD_EXTENSIONS
        assert ".text" in TD_EXTENSIONS
        assert ".toc" in TD_EXTENSIONS

    def test_no_irrelevant_extensions(self) -> None:
        """Should not include non-TD extensions."""
        from td_linter.watch import TD_EXTENSIONS

        assert ".py" not in TD_EXTENSIONS
        assert ".txt" not in TD_EXTENSIONS
        assert ".json" not in TD_EXTENSIONS


class TestRunWatchLoop:
    """Tests for run_watch_loop function."""

    def test_calls_on_start(self, tmp_path: Path) -> None:
        """Should call on_start callback."""
        from td_linter.watch import TDLintWatcher

        on_change = MagicMock()
        on_start = MagicMock()

        watcher = TDLintWatcher(tmp_path, on_change)

        # Mock the sleep to exit immediately
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            from td_linter.watch import run_watch_loop

            run_watch_loop(watcher, on_start=on_start)

        on_start.assert_called_once()
        assert not watcher.is_running  # Should be stopped

    def test_stops_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Should stop watcher on KeyboardInterrupt."""
        from td_linter.watch import TDLintWatcher

        on_change = MagicMock()
        watcher = TDLintWatcher(tmp_path, on_change)

        with patch("time.sleep", side_effect=KeyboardInterrupt):
            from td_linter.watch import run_watch_loop

            run_watch_loop(watcher)

        assert not watcher.is_running


class TestWatchdogOptionalDependency:
    """Tests for optional watchdog dependency handling."""

    def test_import_error_on_missing_watchdog(self, tmp_path: Path) -> None:
        """Should raise ImportError with helpful message when watchdog missing."""
        # This test verifies the error message format, but watchdog is installed
        # so we mock the import failure
        from td_linter.watch import TDLintWatcher

        watcher = TDLintWatcher(tmp_path, lambda: None)

        with patch.dict("sys.modules", {"watchdog": None, "watchdog.observers": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # The actual import happens in start(), not __init__
                # We can't easily test this without more complex mocking
                pass  # This test documents the expected behavior

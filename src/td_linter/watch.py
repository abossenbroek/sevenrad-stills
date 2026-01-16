"""Watch mode for continuous linting of .toe.dir projects.

This module provides file watching with event-based architecture.
Events are emitted for file changes, lint starts/completions, and errors.

Example:
    from td_linter.watch import EventBasedWatcher
    from td_linter.events import LintCompleteEvent, EventType

    watcher = EventBasedWatcher(toe_dir, lint_callback=run_lint)

    # Subscribe to lint events
    watcher.events.subscribe(
        lambda e: print(f"Lint found {e.error_count} errors"),
        event_types={EventType.LINT_COMPLETE}
    )

    watcher.start()
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING, Callable, Sequence

from td_linter.events import (
    EventStream,
    EventType,
    FileChangeEvent,
    LintCompleteEvent,
    LintErrorEvent,
    LintStartEvent,
    WatcherStartEvent,
    WatcherStopEvent,
    WatchEvent,
)

if TYPE_CHECKING:
    from watchdog.events import FileSystemEvent

    from td_linter.rules.base import Violation

# Relevant file extensions for TouchDesigner projects
TD_EXTENSIONS = frozenset({".n", ".parm", ".text", ".toc"})


class TDLintEventHandler:
    """Handle file system events and trigger linting with debouncing."""

    def __init__(
        self,
        toe_dir: Path,
        callback: Callable[[], None],
        debounce_delay: float = 0.5,
    ) -> None:
        """Initialize the event handler.

        Args:
            toe_dir: The .toe.dir directory being watched.
            callback: Function to call when relevant changes are detected.
            debounce_delay: Seconds to wait before triggering callback (debouncing).
        """
        self.toe_dir = toe_dir.resolve()
        self.callback = callback
        self.debounce_delay = debounce_delay
        self._debounce_timer: Timer | None = None
        self._pending_changes: set[Path] = set()

    def _is_relevant(self, path: str) -> bool:
        """Check if a file path is relevant for linting."""
        file_path = Path(path)

        # Must be within the toe_dir
        try:
            file_path.relative_to(self.toe_dir)
        except ValueError:
            return False

        # Check extension
        return file_path.suffix.lower() in TD_EXTENSIONS

    def _schedule_lint(self) -> None:
        """Schedule a lint operation, resetting the debounce timer."""
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()

        self._debounce_timer = Timer(self.debounce_delay, self._execute_callback)
        self._debounce_timer.start()

    def _execute_callback(self) -> None:
        """Execute the callback and clear pending changes."""
        self._pending_changes.clear()
        self.callback()

    def dispatch(self, event: FileSystemEvent) -> None:
        """Handle a file system event.

        Args:
            event: The file system event from watchdog.
        """
        # Ignore directory events
        if event.is_directory:
            return

        src_path = event.src_path
        if self._is_relevant(src_path):
            self._pending_changes.add(Path(src_path))
            self._schedule_lint()

        # Handle move/rename events (have dest_path)
        if hasattr(event, "dest_path") and event.dest_path:
            if self._is_relevant(event.dest_path):
                self._pending_changes.add(Path(event.dest_path))
                self._schedule_lint()

    def stop(self) -> None:
        """Stop any pending timer."""
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None


class TDLintWatcher:
    """Watch a .toe.dir directory for changes and trigger linting."""

    def __init__(
        self,
        toe_dir: Path,
        on_change: Callable[[], None],
        debounce_delay: float = 0.5,
    ) -> None:
        """Initialize the watcher.

        Args:
            toe_dir: The .toe.dir directory to watch.
            on_change: Callback to invoke when relevant changes are detected.
            debounce_delay: Seconds to wait before triggering callback.
        """
        self.toe_dir = toe_dir.resolve()
        self.on_change = on_change
        self.debounce_delay = debounce_delay
        self._observer: "Observer" = None  # type: ignore[assignment]
        self._handler: TDLintEventHandler | None = None
        self._running = False

    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        # Import here to handle optional dependency
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError as e:
            msg = (
                "watchdog is required for watch mode. "
                "Install it with: pip install td-linter[watch]"
            )
            raise ImportError(msg) from e

        # Create a wrapper class that inherits from FileSystemEventHandler
        handler = TDLintEventHandler(
            self.toe_dir,
            self.on_change,
            self.debounce_delay,
        )
        self._handler = handler

        class WatchdogHandler(FileSystemEventHandler):
            def on_any_event(inner_self, event: FileSystemEvent) -> None:  # noqa: N805
                handler.dispatch(event)

        self._observer = Observer()
        self._observer.schedule(
            WatchdogHandler(),
            str(self.toe_dir),
            recursive=True,
        )
        self._observer.start()
        self._running = True

    def stop(self) -> None:
        """Stop watching the directory."""
        if not self._running:
            return

        if self._handler:
            self._handler.stop()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)

        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether the watcher is currently running."""
        return self._running

    def __enter__(self) -> "TDLintWatcher":
        """Start watching on context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop watching on context manager exit."""
        self.stop()


def run_watch_loop(
    watcher: TDLintWatcher,
    on_start: Callable[[], None] | None = None,
    check_interval: float = 1.0,
) -> None:
    """Run the watch loop until interrupted.

    Args:
        watcher: The watcher instance to run.
        on_start: Optional callback to invoke after starting.
        check_interval: Seconds between checking if watcher is still alive.

    Raises:
        KeyboardInterrupt: When the user interrupts the loop.
    """
    watcher.start()

    if on_start:
        on_start()

    try:
        while watcher.is_running:
            time.sleep(check_interval)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()


# Type alias for lint callback
LintCallback = Callable[[Path], Sequence["Violation"]]


class EventBasedWatcher:
    """Watch a .toe.dir directory with event-based reporting.

    This watcher emits events for all significant actions, allowing
    consumers to handle errors, display progress, and log activity.

    Example:
        def lint_project(toe_dir: Path) -> list[Violation]:
            return run_lint(toe_dir)

        watcher = EventBasedWatcher(toe_dir, lint_callback=lint_project)

        # Handle lint completion
        watcher.events.subscribe(
            lambda e: print(f"Found {e.error_count} errors"),
            event_types={EventType.LINT_COMPLETE}
        )

        # Handle errors
        watcher.events.subscribe(
            lambda e: print(f"Error: {e.error_message}"),
            event_types={EventType.LINT_ERROR}
        )

        watcher.start()
    """

    def __init__(
        self,
        toe_dir: Path,
        lint_callback: LintCallback,
        debounce_delay: float = 0.5,
    ) -> None:
        """Initialize the event-based watcher.

        Args:
            toe_dir: The .toe.dir directory to watch.
            lint_callback: Function to call for linting. Takes toe_dir,
                returns list of violations.
            debounce_delay: Seconds to wait before triggering lint.
        """
        self.toe_dir = toe_dir.resolve()
        self._lint_callback = lint_callback
        self._debounce_delay = debounce_delay
        self._events = EventStream()
        self._watcher: TDLintWatcher | None = None

    @property
    def events(self) -> EventStream:
        """Get the event stream for subscribing to events."""
        return self._events

    def _on_change(self) -> None:
        """Handle file changes - run lint and emit events."""
        # Emit lint start
        self._events.emit(LintStartEvent(toe_dir=self.toe_dir))

        start_time = time.time()

        try:
            violations = self._lint_callback(self.toe_dir)
            duration_ms = (time.time() - start_time) * 1000

            # Emit lint complete
            self._events.emit(
                LintCompleteEvent(
                    toe_dir=self.toe_dir,
                    violations=list(violations),
                    duration_ms=duration_ms,
                )
            )
        except Exception as e:
            # Emit lint error
            self._events.emit(
                LintErrorEvent(
                    toe_dir=self.toe_dir,
                    error=e,
                    error_message=str(e),
                )
            )

    def start(self) -> None:
        """Start watching and emit start event."""
        self._watcher = TDLintWatcher(
            self.toe_dir,
            on_change=self._on_change,
            debounce_delay=self._debounce_delay,
        )
        self._watcher.start()
        self._events.emit(WatcherStartEvent(toe_dir=self.toe_dir))

    def stop(self, reason: str = "user_interrupt") -> None:
        """Stop watching and emit stop event."""
        if self._watcher:
            self._watcher.stop()
            self._watcher = None
        self._events.emit(WatcherStopEvent(toe_dir=self.toe_dir, reason=reason))

    @property
    def is_running(self) -> bool:
        """Return whether the watcher is currently running."""
        return self._watcher is not None and self._watcher.is_running

    def __enter__(self) -> "EventBasedWatcher":
        """Start watching on context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop watching on context manager exit."""
        self.stop()


def run_event_based_watch(
    toe_dir: Path,
    lint_callback: LintCallback,
    on_event: Callable[[WatchEvent], None] | None = None,
    debounce_delay: float = 0.5,
    check_interval: float = 1.0,
) -> None:
    """Run an event-based watch loop.

    This is a convenience function that sets up an EventBasedWatcher
    and runs until interrupted.

    Args:
        toe_dir: The .toe.dir directory to watch.
        lint_callback: Function to call for linting.
        on_event: Optional callback for all events.
        debounce_delay: Seconds to wait before triggering lint.
        check_interval: Seconds between checking if watcher is alive.
    """
    watcher = EventBasedWatcher(
        toe_dir,
        lint_callback=lint_callback,
        debounce_delay=debounce_delay,
    )

    if on_event:
        watcher.events.subscribe(on_event)

    try:
        watcher.start()
        while watcher.is_running:
            time.sleep(check_interval)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()

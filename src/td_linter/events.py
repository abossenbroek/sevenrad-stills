"""Event types for td-linter watch mode and other components.

This module provides an event-based architecture for handling changes,
errors, and lint results in td-linter. Events are emitted by producers
(like the file watcher) and consumed by listeners (like the CLI).

Example:
    # Create event stream
    stream = EventStream()

    # Subscribe to events
    stream.subscribe(lambda e: print(f"Got event: {e}"))

    # Emit events
    stream.emit(FileChangeEvent(path=Path("/project/file.n")))
    stream.emit(LintCompleteEvent(violations=[...]))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from td_linter.rules.base import Violation


class EventType(Enum):
    """Types of events in the watch system."""

    FILE_CHANGE = auto()
    LINT_START = auto()
    LINT_COMPLETE = auto()
    LINT_ERROR = auto()
    WATCHER_START = auto()
    WATCHER_STOP = auto()


@dataclass
class WatchEvent:
    """Base class for all watch-related events.

    Attributes:
        timestamp: When the event occurred (Unix timestamp).
        event_type: The type of event.
    """

    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> EventType:
        """Return the type of this event."""
        raise NotImplementedError


@dataclass
class FileChangeEvent(WatchEvent):
    """Event emitted when a file changes.

    Attributes:
        path: Path to the changed file.
        change_type: Type of change (created, modified, deleted, moved).
    """

    path: Path = field(default_factory=lambda: Path("."))
    change_type: str = "modified"

    @property
    def event_type(self) -> EventType:
        return EventType.FILE_CHANGE


@dataclass
class LintStartEvent(WatchEvent):
    """Event emitted when linting begins.

    Attributes:
        toe_dir: Path to the .toe.dir being linted.
        trigger: What triggered the lint (e.g., "file_change", "manual").
    """

    toe_dir: Path = field(default_factory=lambda: Path("."))
    trigger: str = "file_change"

    @property
    def event_type(self) -> EventType:
        return EventType.LINT_START


@dataclass
class LintCompleteEvent(WatchEvent):
    """Event emitted when linting completes successfully.

    Attributes:
        toe_dir: Path to the .toe.dir that was linted.
        violations: List of violations found.
        duration_ms: How long the lint took in milliseconds.
    """

    toe_dir: Path = field(default_factory=lambda: Path("."))
    violations: list["Violation"] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def event_type(self) -> EventType:
        return EventType.LINT_COMPLETE

    @property
    def error_count(self) -> int:
        """Return count of error-severity violations."""
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return count of warning-severity violations."""
        return sum(1 for v in self.violations if v.severity == "warning")


@dataclass
class LintErrorEvent(WatchEvent):
    """Event emitted when linting fails with an error.

    Attributes:
        toe_dir: Path to the .toe.dir that failed.
        error: The exception that occurred.
        error_message: Human-readable error message.
    """

    toe_dir: Path = field(default_factory=lambda: Path("."))
    error: Exception | None = None
    error_message: str = ""

    @property
    def event_type(self) -> EventType:
        return EventType.LINT_ERROR


@dataclass
class WatcherStartEvent(WatchEvent):
    """Event emitted when the file watcher starts.

    Attributes:
        toe_dir: Path to the .toe.dir being watched.
    """

    toe_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def event_type(self) -> EventType:
        return EventType.WATCHER_START


@dataclass
class WatcherStopEvent(WatchEvent):
    """Event emitted when the file watcher stops.

    Attributes:
        toe_dir: Path to the .toe.dir that was being watched.
        reason: Why the watcher stopped (e.g., "user_interrupt", "error").
    """

    toe_dir: Path = field(default_factory=lambda: Path("."))
    reason: str = "user_interrupt"

    @property
    def event_type(self) -> EventType:
        return EventType.WATCHER_STOP


# Type alias for event listeners
EventListener = Callable[[WatchEvent], None]


class EventStream:
    """Stream of events with subscription support.

    Provides a pub/sub mechanism for watch events. Subscribers receive
    events in the order they're emitted.

    Example:
        stream = EventStream()

        # Subscribe to all events
        stream.subscribe(lambda e: print(e))

        # Subscribe to specific event types
        stream.subscribe(
            lambda e: handle_error(e),
            event_types={EventType.LINT_ERROR}
        )

        # Emit an event
        stream.emit(FileChangeEvent(path=Path("file.n")))
    """

    def __init__(self) -> None:
        """Initialize the event stream."""
        self._listeners: list[tuple[EventListener, set[EventType] | None]] = []

    def subscribe(
        self,
        listener: EventListener,
        event_types: set[EventType] | None = None,
    ) -> Callable[[], None]:
        """Subscribe to events.

        Args:
            listener: Callback to invoke for matching events.
            event_types: Optional set of event types to filter.
                If None, receives all events.

        Returns:
            Unsubscribe function - call to remove the subscription.
        """
        entry = (listener, event_types)
        self._listeners.append(entry)

        def unsubscribe() -> None:
            if entry in self._listeners:
                self._listeners.remove(entry)

        return unsubscribe

    def emit(self, event: WatchEvent) -> None:
        """Emit an event to all matching subscribers.

        Args:
            event: The event to emit.
        """
        for listener, event_types in self._listeners:
            if event_types is None or event.event_type in event_types:
                try:
                    listener(event)
                except Exception:
                    # Don't let listener errors break the event stream
                    pass

    def clear(self) -> None:
        """Remove all subscribers."""
        self._listeners.clear()

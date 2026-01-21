"""Unit tests for the events module."""

import time
from pathlib import Path

import pytest

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


class TestEventTypes:
    """Tests for event type enumeration."""

    def test_all_event_types_defined(self) -> None:
        """Should have all expected event types."""
        expected = {
            "FILE_CHANGE",
            "LINT_START",
            "LINT_COMPLETE",
            "LINT_ERROR",
            "WATCHER_START",
            "WATCHER_STOP",
        }
        actual = {e.name for e in EventType}
        assert actual == expected


class TestFileChangeEvent:
    """Tests for FileChangeEvent."""

    def test_creates_with_defaults(self) -> None:
        """Should create with default values."""
        event = FileChangeEvent()
        assert event.change_type == "modified"
        assert event.timestamp > 0

    def test_creates_with_path(self) -> None:
        """Should store path correctly."""
        event = FileChangeEvent(path=Path("/test/file.n"))
        assert event.path == Path("/test/file.n")

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = FileChangeEvent()
        assert event.event_type == EventType.FILE_CHANGE


class TestLintStartEvent:
    """Tests for LintStartEvent."""

    def test_creates_with_toe_dir(self) -> None:
        """Should store toe_dir correctly."""
        event = LintStartEvent(toe_dir=Path("/project.toe.dir"))
        assert event.toe_dir == Path("/project.toe.dir")

    def test_trigger_default(self) -> None:
        """Should have file_change as default trigger."""
        event = LintStartEvent()
        assert event.trigger == "file_change"

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = LintStartEvent()
        assert event.event_type == EventType.LINT_START


class TestLintCompleteEvent:
    """Tests for LintCompleteEvent."""

    def test_creates_with_violations(self) -> None:
        """Should store violations correctly."""
        from td_linter.rules.base import Violation

        violations = [
            Violation(rule="R001", message="msg", path="/op", severity="error"),
            Violation(rule="R002", message="msg", path="/op", severity="warning"),
        ]
        event = LintCompleteEvent(violations=violations)
        assert len(event.violations) == 2

    def test_error_count(self) -> None:
        """Should count errors correctly."""
        from td_linter.rules.base import Violation

        violations = [
            Violation(rule="R001", message="msg", path="/op", severity="error"),
            Violation(rule="R002", message="msg", path="/op", severity="error"),
            Violation(rule="R003", message="msg", path="/op", severity="warning"),
        ]
        event = LintCompleteEvent(violations=violations)
        assert event.error_count == 2

    def test_warning_count(self) -> None:
        """Should count warnings correctly."""
        from td_linter.rules.base import Violation

        violations = [
            Violation(rule="R001", message="msg", path="/op", severity="error"),
            Violation(rule="R002", message="msg", path="/op", severity="warning"),
            Violation(rule="R003", message="msg", path="/op", severity="warning"),
        ]
        event = LintCompleteEvent(violations=violations)
        assert event.warning_count == 2

    def test_duration_ms(self) -> None:
        """Should store duration correctly."""
        event = LintCompleteEvent(duration_ms=123.45)
        assert event.duration_ms == 123.45

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = LintCompleteEvent()
        assert event.event_type == EventType.LINT_COMPLETE


class TestLintErrorEvent:
    """Tests for LintErrorEvent."""

    def test_creates_with_error(self) -> None:
        """Should store error correctly."""
        error = ValueError("test error")
        event = LintErrorEvent(error=error, error_message="test error")
        assert event.error is error
        assert event.error_message == "test error"

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = LintErrorEvent()
        assert event.event_type == EventType.LINT_ERROR


class TestWatcherStartEvent:
    """Tests for WatcherStartEvent."""

    def test_creates_with_toe_dir(self) -> None:
        """Should store toe_dir correctly."""
        event = WatcherStartEvent(toe_dir=Path("/project.toe.dir"))
        assert event.toe_dir == Path("/project.toe.dir")

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = WatcherStartEvent()
        assert event.event_type == EventType.WATCHER_START


class TestWatcherStopEvent:
    """Tests for WatcherStopEvent."""

    def test_creates_with_reason(self) -> None:
        """Should store reason correctly."""
        event = WatcherStopEvent(reason="error")
        assert event.reason == "error"

    def test_default_reason(self) -> None:
        """Should have user_interrupt as default reason."""
        event = WatcherStopEvent()
        assert event.reason == "user_interrupt"

    def test_event_type(self) -> None:
        """Should return correct event type."""
        event = WatcherStopEvent()
        assert event.event_type == EventType.WATCHER_STOP


class TestEventStream:
    """Tests for EventStream."""

    def test_subscribe_and_receive(self) -> None:
        """Subscriber should receive emitted events."""
        stream = EventStream()
        received: list[WatchEvent] = []

        stream.subscribe(lambda e: received.append(e))
        stream.emit(FileChangeEvent(path=Path("/test.n")))

        assert len(received) == 1
        assert isinstance(received[0], FileChangeEvent)

    def test_multiple_subscribers(self) -> None:
        """Multiple subscribers should all receive events."""
        stream = EventStream()
        received1: list[WatchEvent] = []
        received2: list[WatchEvent] = []

        stream.subscribe(lambda e: received1.append(e))
        stream.subscribe(lambda e: received2.append(e))
        stream.emit(LintStartEvent())

        assert len(received1) == 1
        assert len(received2) == 1

    def test_event_type_filtering(self) -> None:
        """Should filter by event type."""
        stream = EventStream()
        errors_only: list[WatchEvent] = []

        stream.subscribe(
            lambda e: errors_only.append(e),
            event_types={EventType.LINT_ERROR},
        )

        stream.emit(LintStartEvent())
        stream.emit(LintErrorEvent())
        stream.emit(LintCompleteEvent())

        assert len(errors_only) == 1
        assert isinstance(errors_only[0], LintErrorEvent)

    def test_unsubscribe(self) -> None:
        """Unsubscribe function should stop receiving events."""
        stream = EventStream()
        received: list[WatchEvent] = []

        unsubscribe = stream.subscribe(lambda e: received.append(e))

        stream.emit(LintStartEvent())
        unsubscribe()
        stream.emit(LintStartEvent())

        assert len(received) == 1  # Only first event received

    def test_clear(self) -> None:
        """Clear should remove all subscribers."""
        stream = EventStream()
        received: list[WatchEvent] = []

        stream.subscribe(lambda e: received.append(e))
        stream.clear()
        stream.emit(LintStartEvent())

        assert len(received) == 0

    def test_listener_error_does_not_break_stream(self) -> None:
        """Listener errors should not prevent other listeners."""
        stream = EventStream()
        received: list[WatchEvent] = []

        def failing_listener(e: WatchEvent) -> None:
            raise ValueError("Listener error")

        stream.subscribe(failing_listener)
        stream.subscribe(lambda e: received.append(e))

        stream.emit(LintStartEvent())

        # Second listener should still receive the event
        assert len(received) == 1


class TestEventTimestamp:
    """Tests for event timestamp functionality."""

    def test_timestamp_auto_generated(self) -> None:
        """Events should have auto-generated timestamps."""
        before = time.time()
        event = FileChangeEvent()
        after = time.time()

        assert before <= event.timestamp <= after

    def test_timestamp_can_be_overridden(self) -> None:
        """Timestamp can be explicitly set."""
        event = FileChangeEvent(timestamp=12345.0)
        assert event.timestamp == 12345.0

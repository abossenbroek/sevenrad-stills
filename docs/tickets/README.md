# Tickets Overview

This directory contains individual ticket documentation for the macOS Swift drum machine-inspired image processing application.

## Active Tickets

### EPIC-001: Video Source & Frame Extraction (13 points)

| Ticket | Name | Points | Dependencies | Status |
|--------|------|--------|--------------|--------|
| [TICKET-001](./TICKET-001.md) | YouTube URL Input & Validation | 2 | None | Ready |
| [TICKET-002](./TICKET-002.md) | Local Video File Picker | 1 | None | Ready |
| [TICKET-003](./TICKET-003.md) | Video Metadata Display | 2 | 001 OR 002 | Blocked |
| [TICKET-004](./TICKET-004.md) | Timeline Scrubber Component | 5 | 003 | Blocked |
| [TICKET-005](./TICKET-005.md) | Frame Extraction Service Integration | 3 | 004 | Blocked |

## Ticket Structure

Each ticket follows this format:
- Context (1-2 sentences)
- User Story (As/I want/So that)
- Acceptance Criteria (testable checkboxes)
- Technical Notes (implementation details)
- Story Points
- Dependencies
- Related (parent epic, related tickets)

## Development Workflow

1. TICKET-001 and TICKET-002 can be developed in parallel (both have no dependencies)
2. TICKET-003 requires either video source to be complete
3. TICKET-004 requires metadata to render timeline
4. TICKET-005 completes the epic and enables EPIC-003

## Navigation

- [Epics Directory](../epics/)
- [EPIC-001 Overview](../epics/EPIC-001.md)

# [TICKET-041] Frame Timeline Component

## Context
Display extracted frames as thumbnails in horizontal scrollable timeline to support frame-by-frame navigation and selection for step sequencer interface.

## User Story
As a technical artist,
I want to see all extracted frames as thumbnails in a timeline,
So that I can quickly navigate to specific frames and understand my sequence structure.

## Acceptance Criteria
- [ ] Display extracted frames as 40×30px thumbnails in horizontal scrollable container
- [ ] Show current playhead position with orange border highlight
- [ ] Click frame thumbnail to jump to that position
- [ ] Keyboard navigation: Left/Right arrows step through frames
- [ ] Display frame counter: "Frame 12/45" in header
- [ ] Timeline scrolls to keep current frame visible during playback
- [ ] Thumbnails load from frame extraction cache (no re-extraction)

## Technical Notes
- Use NSScrollView with horizontal-only scrolling
- Thumbnails scale from full-resolution frames (cache thumbnails for performance)
- Sync scroll position with playback head position
- Grid below timeline must align with frame thumbnails

## Related
- EPIC-008: Frame-Level Step Sequencer
- TICKET-005: Frame extraction (provides source frames)

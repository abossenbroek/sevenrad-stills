# [TICKET-042] Step Sequencer Grid Component

## Context
Grid layout for toggling effects on/off per frame, inspired by RYTM's step sequencer interface. 17 rows (effects) × N columns (frames).

## User Story
As a technical artist,
I want to toggle effects on/off for individual frames via a grid interface,
So that I can create rhythmic visual patterns and dynamic effect progressions.

## Acceptance Criteria
- [ ] Grid displays 17 rows (one per effect) × N columns (one per frame)
- [ ] Cell states: active (orange fill), inactive (grey), hovered (border highlight)
- [ ] Click cell to toggle effect on/off for that frame
- [ ] Shift+click to toggle entire row (all frames for one effect)
- [ ] Shift+click column header to toggle entire column (all effects for one frame)
- [ ] Grid scrolls horizontally synchronized with frame timeline
- [ ] Effect labels show effect name and multiplier (e.g., "Saturation ×4")
- [ ] Visual separator every 4th frame (RYTM-style pattern markers)

## Technical Notes
- Grid uses NSGridView or custom layout with absolute positioning
- Cell size: 44×44px (matching HTML mockup)
- Bulk operations: Shift-click range selection, invert, clear all, every 2nd
- Grid state persists in Pipeline model's frame_overrides

## Related
- EPIC-008: Frame-Level Step Sequencer
- TICKET-012: Effect model (provides effect list)
- TICKET-041: Frame timeline (provides frame count)

# [EPIC-008] Frame-Level Step Sequencer

## Context
RYTM-inspired pattern sequencer for frame-by-frame effect control. Users can toggle effects on/off for specific frames to create dynamic visual progressions and rhythmic patterns, similar to the Elektron RYTM's 64-step pattern sequencer applied to video frames.

## User Story
As a technical artist,
I want to enable/disable effects on specific frames in my sequence,
So that I can create dynamic visual progressions and rhythmic effect patterns.

## Scope
- Frame timeline component displaying extracted frames as thumbnails
- Step sequencer grid (17 effects × N frames) with toggle cells
- Playback controls with variable speed (1x/2x/4x/8x)
- Per-frame effect overrides (enable/disable effects for specific frames)
- YAML schema extension to support frame_overrides structure
- Real-time preview playback respecting effect toggles

Out of scope:
- Audio sync and beat detection
- MIDI controller integration
- Auto-pattern generation or AI assistance
- Timeline editing (cut/copy/paste frames)
- Effect parameter automation curves

## Success Criteria
- [ ] Users can view all extracted frames as thumbnails in horizontal timeline
- [ ] Users can toggle individual effects on/off for any frame via grid interface
- [ ] Playback engine applies frame overrides correctly (skips disabled effects)
- [ ] Pipeline YAML serialization includes frame_overrides data
- [ ] Playback runs smoothly at 1x-8x speed with visible effect changes
- [ ] Grid interface supports keyboard navigation and bulk operations

## Related Tickets
- TICKET-041: Frame Timeline Component
- TICKET-042: Step Sequencer Grid Component
- TICKET-043: Playback Controls
- TICKET-044: Frame Override Data Model
- TICKET-045: Step Sequencer Preview Integration

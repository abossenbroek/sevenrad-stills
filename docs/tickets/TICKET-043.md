# [TICKET-043] Playback Controls

## Context
Transport controls for playing back frame sequences with effect overrides, similar to DAW or drum machine transport.

## User Story
As a technical artist,
I want to play back my frame sequence with effects toggling on/off,
So that I can preview the final visual rhythm before rendering.

## Acceptance Criteria
- [ ] Play/Pause button with keyboard shortcut (Spacebar)
- [ ] Speed selector dropdown: 1x, 2x, 4x, 8x
- [ ] Loop toggle (repeat sequence continuously)
- [ ] Step forward/backward buttons (advance 1 frame)
- [ ] Playback respects frame overrides (skips disabled effects per frame)
- [ ] Current frame indicator updates during playback
- [ ] Stop button resets playhead to frame 0

## Technical Notes
- Playback timer calculates frame interval based on speed multiplier
- Each frame renders with effect overrides applied before display
- Debounce rendering if backend too slow (skip frames to maintain playback speed)
- Playback state: stopped, playing, paused

## Related
- EPIC-008: Frame-Level Step Sequencer
- TICKET-041: Frame timeline (playhead position)
- TICKET-042: Step sequencer grid (provides overrides)

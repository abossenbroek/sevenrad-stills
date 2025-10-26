# [TICKET-045] Step Sequencer Preview Integration

## Context
Integrate step sequencer with preview rendering engine to display real-time playback with per-frame effect overrides applied.

## User Story
As a technical artist,
I want to see my frame sequence play back with effects toggling on/off,
So that I can verify my visual rhythm before committing to full render.

## Acceptance Criteria
- [ ] Render preview for each frame during playback
- [ ] Apply frame overrides before sending pipeline to backend
- [ ] Debounce during fast playback (skip frames if backend render time exceeds frame interval)
- [ ] Show playback progress indicator
- [ ] Cache rendered frames for smooth replay (clear cache when overrides change)
- [ ] Display rendering error if backend fails (fallback to original frame)

## Technical Notes
- Modify Pipeline JSON sent to backend: filter effects based on frame_overrides
- Example: If frame 5 disables effect "uuid1", remove that effect from pipeline JSON for frame 5
- Cache key: frame_index + pipeline_hash + overrides_hash
- Debounce logic: if render time > (1/fps) * speed_multiplier, skip to next frame
- Preview resolution: 960×540 (performance)

## Related
- EPIC-008: Frame-Level Step Sequencer
- TICKET-027: Preview rendering (extends this)
- TICKET-043: Playback controls (triggers rendering)
- TICKET-044: Frame override data model (provides override data)

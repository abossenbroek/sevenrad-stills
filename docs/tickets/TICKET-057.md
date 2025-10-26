# [TICKET-057] Adaptive Preview Resolution

## Context
Preview render performance depends on output resolution. Smaller viewports don't need full-resolution previews. Calculate appropriate resolution based on viewport size to maintain 1-second SLA.

## User Story
As a user,
I want preview renders to complete quickly regardless of window size,
So that I can iterate on effects without delays.

## Acceptance Criteria
- [ ] Resolution calculator method created: calculatePreviewResolution(viewportSize:)
- [ ] Default resolution 960×540 for standard viewport (≤1200px width)
- [ ] Scales up to 1280×720 for large viewports (1201-1600px)
- [ ] Scales down to 640×360 for small viewports (≤800px)
- [ ] Maintains aspect ratio of source video
- [ ] Passes resolution to render_preview XPC call as width/height params
- [ ] Updates when window size changes (debounced)

## Technical Notes
- Reference backend-contract.md section 10.1 for scaling guidelines
- Resolution passed in render_preview params as {"width": 960, "height": 540}
- Backend should use these values to scale preview output

## Related
- Parent: EPIC-005
- Consumed by: TICKET-027

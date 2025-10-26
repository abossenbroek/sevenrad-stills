# [TICKET-052] Conditional Processing UI

## Context
Users need to apply effects conditionally (every Nth frame, probability-based) to create rhythmic or randomized visual patterns. This provides generative control without manual keyframing.

## User Story
As a user,
I want to control when effects apply using conditions and probability,
So that I can create rhythmic patterns and randomized variations across frames.

## Acceptance Criteria
- [ ] Trig condition dropdown: "All Frames", "Every 2nd", "Every 3rd", "Every 4th", "Fill (last frame only)"
- [ ] Probability slider: 0-100% chance effect applies to each eligible frame
- [ ] "Use Seed" checkbox for reproducible randomness
- [ ] Visual indicator in frame timeline when condition prevents effect (greyed-out frame)
- [ ] Preview badge shows estimated frame application count (e.g., "Will apply to ~34/45 frames")
- [ ] Collapsible "Advanced Modulation" section containing these controls

## Technical Notes
- Reference HTML mockup lines 632-676 for UI structure
- Trig conditions map to backend values: "all", "every_2nd", "every_3rd", "every_4th", "fill"
- Probability stored as float 0.0-1.0 internally, displayed as integer 0-100%
- Seed defaults to current timestamp if checkbox enabled
- Frame timeline visual update requires TICKET-041 integration
- Preview calculation: (total_frames / trig_interval) * probability

## Related
- Parent: EPIC-010
- Dependencies: TICKET-009 (parameter panel), TICKET-041 (frame timeline)

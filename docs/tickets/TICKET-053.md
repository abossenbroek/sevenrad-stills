# [TICKET-053] Parameter Ramping

## Context
Parameters need to evolve across frames to create dynamic, changing effects. Parameter ramping provides this without manual keyframe animation.

## User Story
As a user,
I want to ramp parameters from start to end values across frames,
So that I can create evolving effects like progressive blurring or color shifts.

## Acceptance Criteria
- [ ] "Ramp [parameter] across frames" checkbox per parameter
- [ ] Start value input field (defaults to current parameter value)
- [ ] End value input field (user-specified target)
- [ ] Curve selector dropdown: "Linear", "Ease-In", "Ease-Out", "Exponential"
- [ ] Mini graph canvas showing selected curve shape
- [ ] Ramp applies across all frames in sequence (frame 0 = start, last frame = end)
- [ ] Validation prevents start/end values outside parameter bounds

## Technical Notes
- Reference HTML mockup lines 680-716 for UI layout
- Canvas graph dimensions: 100% width × 60px height
- Draw grid (dotted #333), then curve line (#ff6b35)
- Curve types map to interpolation functions:
  - Linear: `lerp(start, end, t)`
  - Ease-In: `lerp(start, end, t^2)`
  - Ease-Out: `lerp(start, end, 1 - (1-t)^2)`
  - Exponential: `lerp(start, end, t^3)`
- Ramp checkbox appears below each parameter slider
- Collapsed by default unless ramp is enabled

## Related
- Parent: EPIC-010
- Dependencies: TICKET-009 (parameter panel), TICKET-017 (parameter controls)

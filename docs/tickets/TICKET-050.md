# [TICKET-050] Scene Comparison View

## Context
Scene comparison view displays two scenes side-by-side with diff visualization, enabling users to analyze differences between effect configurations before loading.

## User Story
As a technical artist,
I want to compare two scenes side-by-side,
So that I can understand the differences between effect configurations.

## Acceptance Criteria
- [ ] Split view panel displays two scene configurations side-by-side
- [ ] Dropdown selectors for Scene A and Scene B (populated scenes only)
- [ ] Diff visualization highlights added effects (green), removed effects (red), modified parameters (yellow)
- [ ] Show effect name, repeat count, and parameters for each scene
- [ ] "Load Scene A" and "Load Scene B" buttons apply selected scene
- [ ] Empty state shown if fewer than 2 populated scenes exist
- [ ] Comparison view accessible from toolbar or scene panel

## Technical Notes
- Diff algorithm compares effect chains and parameter values
- Modified effects show parameter differences inline (e.g., "Quality: 45 → 60")
- Comparison view updates automatically when scene selections change
- Load buttons behave identically to standard scene load operation

## Related
- EPIC-009: Scene Preset System
- TICKET-046: Scene Slot UI Component (dependency)
- TICKET-048: Scene Load Operation (dependency)

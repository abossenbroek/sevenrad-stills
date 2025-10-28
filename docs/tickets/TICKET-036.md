# [TICKET-036] Tooltips and Help Text

## Context
Add contextual tooltips throughout the app to help users understand features and discover keyboard shortcuts. Tooltips should appear on hover with minimal delay.

## User Story
As a user,
I want to see helpful tooltips when hovering over UI elements,
So that I can learn features and understand what controls do.

## Acceptance Criteria
- [ ] Effect pads show tooltip: "[Effect Name]: [One-line description]."
- [ ] Parameter controls show tooltip: "[Parameter Name]: [Description]. Range: [min-max]. [Shortcut]"
- [ ] Keyboard shortcuts displayed using registry: Reference `docs/design/keyboard-shortcuts.md`
- [ ] Tooltips appear after 500ms hover delay and dismiss after 5s or on mouse move
- [ ] All tooltips use sentence case and end with period
- [ ] Tooltip format validated against template before commit

## Technical Notes
- Use SwiftUI's `.help()` modifier for native tooltip implementation
- Tooltip template: `"[Control Name]: [Description]. [Shortcut if applicable]"`
- Examples:
  - Pad: `"Pixelate: Reduces image resolution to create blocky effect."`
  - Parameter: `"Intensity: Controls effect strength. Range: 0-100. Drag to adjust."`
  - Button: `"Move Layer Up: Reorder layer toward top of stack. Cmd+↑"`
- Consume keyboard shortcut registry for accurate shortcut display
- Store tooltip text in localization-ready format (even if not translating yet)
- Keep tooltip text under 80 characters for readability

## Related
- Parent: EPIC-007
- Dependencies: All UI implementation tickets
- Related: TICKET-035 (keyboard shortcuts documented in tooltips)
- Related: TICKET-038 (first-run tutorial)

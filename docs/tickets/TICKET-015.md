# [TICKET-015] Layer Selection and Highlighting

## Context
Users need clear visual feedback about which layer's parameters they are editing. Selection state drives the parameter panel display.

## User Story
As a user,
I want to see which layer is selected,
So that I know which effect I am currently editing.

## Acceptance Criteria
- [ ] Click on layer selects it
- [ ] Selected layer shows orange accent border/background
- [ ] Only one layer selected at a time
- [ ] Selection state tracked in app state
- [ ] Deselect when clicking empty area
- [ ] Selection persists during reordering

## Technical Notes
- Use `@State var selectedLayerID: UUID?`
- Apply conditional modifier: `.border(.orange, width: 2)` when selected
- Clear selection on pipeline clear or layer removal

## Related
Parent: EPIC-003
Depends on: TICKET-008 (layer stack UI)
Blocks: TICKET-017 (parameter binding needs selection)

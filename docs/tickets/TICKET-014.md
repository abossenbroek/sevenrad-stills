# [TICKET-014] Layer Reordering (Drag-and-Drop)

## Context
Critical for effect pipeline control - processing order determines final output. Users must be able to visually reorganize layers to experiment with different effect sequences.

## User Story
As a user,
I want to drag layers up or down in the stack,
So that I can change the order effects are applied.

## Acceptance Criteria
- [ ] Layers support drag gesture in layer stack
- [ ] Drop target highlights during drag
- [ ] Pipeline array reorders on drop
- [ ] UI animates layer movement
- [ ] Selection follows dragged layer
- [ ] Works for all layer positions (top, middle, bottom)

## Technical Notes
- SwiftUI: Use `onDrag()` and `onDrop()` modifiers
- NSTableView alternative: Implement drag-and-drop delegate
- Update pipeline.effects array indices on reorder
- Consider smooth animation with `.animation(.default)`

## Related
Parent: EPIC-003
Depends on: TICKET-008 (layer stack UI), TICKET-013 (pipeline state)

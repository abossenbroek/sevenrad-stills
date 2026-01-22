# [TICKET-008] Layer Stack Panel Component

## Context
Build the layer stack visualization that displays the effect pipeline as a vertical stack of cards, enabling users to see, reorder, and manage their effect chain.

## User Story
As a user,
I want to see my effect pipeline as a stack of layers,
So that I can understand the processing order and reorder effects by dragging.

## Acceptance Criteria
- [ ] Vertical scrollable list displaying effect layers top-to-bottom
- [ ] Each layer card: 260x56px, 10px vertical spacing
- [ ] Layer card shows: effect name, bypass toggle, delete button
- [ ] Drag-and-drop reordering with visual feedback (ghost card)
- [ ] Selected layer highlighted with #ff6b35 left border (4px)
- [ ] Empty state: "Click a pad to add effect" centered message
- [ ] Bypass toggle grays out layer (#5a5a5a) when inactive
- [ ] Delete button shows on hover, confirms before removing
- [ ] Maximum 16 layers enforced with visual indicator
- [ ] Stack updates immediately when pad clicked

## Technical Notes
- Implement as `LayerStackView` with `LayerCardView` subcomponent
- Use `onDrag` and `onDrop` modifiers for reordering
- State: `@Binding var layers: [EffectLayer]`
- Create `EffectLayer` model: `id`, `effectId`, `isBypassed`, `parameters`
- Animate list changes with `.animation(.spring())`

## Related
- Parent: EPIC-002
- Depends: TICKET-006, TICKET-011
- Blocks: EPIC-003

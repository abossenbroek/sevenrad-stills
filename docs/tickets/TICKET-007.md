# [TICKET-007] 16-Pad Effect Grid Component

## Context
Create the signature 16-pad effect palette styled after Elektron RYTM, serving as the primary interaction point for selecting and applying effects to the layer pipeline.

## User Story
As a user,
I want to select effects from a tactile 16-pad grid,
So that I can quickly compose my effect pipeline like programming a drum pattern.

## Acceptance Criteria
- [ ] 4x4 grid of 64x64px pads with 8px spacing
- [ ] Each pad displays effect name in Menlo font, 11pt
- [ ] Pad states: inactive (#2a2a2a), hover (#3a3a3a), active (#ff6b35), selected (#ff6b35 with pulse)
- [ ] Smooth 0.15s easeInOut transitions between states
- [ ] Pads clickable with visual feedback (scale 0.95 on press)
- [ ] Grid scrollable if more than 16 effects defined
- [ ] Effect icons/labels configurable via data model
- [ ] Accessibility: VoiceOver announces effect name and state
- [ ] Grid responds to window width changes gracefully

## Technical Notes
- Implement as `EffectPadGridView` with `EffectPadView` subcomponent
- Use `LazyVGrid` with flexible spacing
- State management via `@Binding` for selected effect
- Create `EffectPad` model: `id`, `name`, `icon`, `category`
- Reference TICKET-011 for exact color values and typography

## Related
- Parent: EPIC-002
- Depends: TICKET-006, TICKET-011
- Blocks: EPIC-003

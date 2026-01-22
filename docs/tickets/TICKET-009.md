# [TICKET-009] Parameter Panel Component

## Context
Develop the contextual parameter control panel that displays adjustable parameters for the currently selected layer, using RYTM-style knobs and sliders.

## User Story
As a user,
I want to adjust effect parameters using visual controls,
So that I can fine-tune each effect in my pipeline.

## Acceptance Criteria
- [ ] Panel displays parameters for selected layer
- [ ] Empty state: "Select a layer to edit parameters" when no selection
- [ ] Each parameter shows: label, value display, control (slider/knob)
- [ ] Sliders: horizontal, 200px wide, RYTM orange track (#ff6b35)
- [ ] Value labels update in real-time as controls move
- [ ] Parameters grouped by category with subtle dividers
- [ ] Numeric input allowed by clicking value display
- [ ] Reset button per parameter (returns to default)
- [ ] Panel scrollable when parameters exceed viewport height
- [ ] Parameter changes debounced (50ms) to prevent excessive updates

## Technical Notes
- Implement as `ParameterPanelView` with reusable `ParameterControlView`
- Support control types: slider, knob, toggle, dropdown
- Use `@Binding` for parameter values to enable two-way sync
- Create `EffectParameter` model: `id`, `name`, `type`, `value`, `min`, `max`, `default`
- Knob rendering via custom `Shape` or prebuilt library

## Related
- Parent: EPIC-002
- Depends: TICKET-006, TICKET-011
- Blocks: EPIC-003

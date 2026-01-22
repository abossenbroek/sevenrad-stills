# [TICKET-017] Parameter Binding and Validation

## Context
Core editing functionality - users adjust sliders to modify effect behavior. Parameters must validate against backend constraints and update pipeline state reactively.

## User Story
As a user,
I want to adjust parameter sliders,
So that I can fine-tune the selected effect's behavior.

## Acceptance Criteria
- [ ] Parameter panel shows controls for selected layer
- [ ] Sliders bound to effect parameter values
- [ ] Min/max ranges enforced from EffectRegistry
- [ ] Value updates immediately in pipeline state (NOT preview rendering)
- [ ] Display current value next to slider
- [ ] Support float, int, bool, enum, color parameter types
- [ ] Validate values before updating state
- [ ] ColorPicker control for color parameters
- [ ] Serialize color values to hex string format (#FF0000)

## Technical Notes
- Use `Slider(value: $effect.params["factor"], in: 0...3)`
- Access schema from `EffectRegistry.getParameters(effectName)`
- For enums, use `Picker` instead of Slider
- For bools, use `Toggle`
- For colors, use `ColorPicker` and serialize to hex: `String(format: "#%02X%02X%02X", r, g, b)`
- Round float values to schema `step` precision
- This ticket updates pipeline state only - preview rendering handled by EPIC-005

## Related
Parent: EPIC-003
Depends on: TICKET-009 (parameter panel UI), TICKET-012 (effect models), TICKET-015 (selection)

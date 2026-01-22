# [TICKET-051] Large REPEAT Rotary Control

## Context
The REPEAT control needs prominent visual treatment to emphasize its importance in the effect workflow. RYTM-style rotary controls provide tactile, intuitive adjustment for critical parameters.

## User Story
As a user,
I want a large, prominent REPEAT rotary control,
So that I can quickly adjust effect repetition with precision and visual feedback.

## Acceptance Criteria
- [ ] Large rotary knob UI positioned prominently in parameter panel
- [ ] Value range: 1-100 with visual arc showing current position
- [ ] Quick preset buttons: ×1, ×2, ×4, ×8, ×16
- [ ] Click value display to enable numeric keyboard entry
- [ ] Scroll wheel adjusts value (Shift key for fine control, ±1 increments)
- [ ] Orange arc visualization (RYTM-style) indicates current value
- [ ] Visual tick marks at 1, 10, 25, 50, 75, 100
- [ ] Active preset button highlights when value matches preset

## Technical Notes
- Reference HTML mockup line 595-628 for visual design
- Rotary control should occupy special section with border/background
- Use `@State` binding to sync with effect model `repeat` property
- Scroll wheel: normal = ±5, Shift = ±1
- Keyboard entry validates range (1-100), rejects invalid input
- Orange color: `#ff6b35` (matches design system)

## Related
- Parent: EPIC-010
- Dependencies: TICKET-009 (parameter panel), TICKET-011 (design system)

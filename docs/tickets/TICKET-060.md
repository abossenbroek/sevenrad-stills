# [TICKET-060] Pipeline Validation and Error Display

## Context
Invalid parameters cause backend errors at render time. Client-side validation prevents submission of malformed pipelines and provides immediate feedback to users.

## User Story
As a user,
I want to see validation errors before rendering,
So that I can fix issues without waiting for backend failures.

## Acceptance Criteria
- [ ] Client-side validation using EffectRegistry schemas
- [ ] Call backend `validate_pipeline` endpoint before render
- [ ] Display validation errors in UI with red border on invalid parameter
- [ ] Show error message below invalid parameter control
- [ ] Prevent render button activation if validation fails
- [ ] Validate on parameter change and before render
- [ ] Clear error state when parameter becomes valid

## Technical Notes
- Validate parameter ranges: `value >= min && value <= max`
- Validate enums: `allowedValues.contains(value)`
- Call `POST /validate` with pipeline JSON before `POST /render`
- Use `.border(Color.red, width: 2)` modifier on invalid controls
- Display backend validation errors using `Text` below parameter
- Disable render button with `.disabled(!isValid)` modifier

## Related
Parent: EPIC-003
Depends on: TICKET-012 (effect models), TICKET-017 (parameter binding)

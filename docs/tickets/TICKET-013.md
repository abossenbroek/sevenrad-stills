# [TICKET-013] Add Effect to Pipeline Action

## Context
Implements the primary user interaction: tapping an effect pad to add it to the processing pipeline. Connects the pad grid UI to pipeline state management.

## User Story
As a user,
I want to tap an effect pad,
So that the effect is added to my pipeline with default parameters.

## Acceptance Criteria
- [ ] Tap on effect pad triggers add action
- [ ] New effect instance created with UUID and defaults
- [ ] Effect appended to end of pipeline array
- [ ] Layer stack UI updates to show new layer
- [ ] Default parameters loaded from EffectRegistry
- [ ] Repeat count initialized to 1
- [ ] Effect enabled by default

## Technical Notes
- Use SwiftUI `@State` or ObservableObject for pipeline state
- Pad tap handler calls `pipeline.addEffect(effectName)`
- Effect defaults from `EffectRegistry.getDefaults(effectName)`

## Related
Parent: EPIC-003
Depends on: TICKET-007 (pad grid UI), TICKET-012 (effect models)

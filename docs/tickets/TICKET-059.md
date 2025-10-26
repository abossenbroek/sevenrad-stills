# [TICKET-059] Toggle Effect Enable/Disable

## Context
Users need to temporarily disable effects without removing them from the pipeline. This allows experimentation and comparison while preserving layer configuration and order.

## User Story
As a user,
I want to toggle effects on/off without removing them,
So that I can test variations while preserving my pipeline structure.

## Acceptance Criteria
- [ ] Toggle switch displayed on each layer in stack
- [ ] Toggle updates `effect.enabled` boolean in pipeline state
- [ ] Disabled layers show greyed-out appearance (opacity 0.5)
- [ ] Disabled effects remain in pipeline array
- [ ] Backend receives disabled effects in JSON but skips processing
- [ ] Toggle state persists during drag-and-drop reordering
- [ ] Layer stack shows visual distinction between enabled/disabled

## Technical Notes
- Add `enabled: Bool` property to Effect model (default: true)
- Use SwiftUI `Toggle` with `.toggleStyle(.switch)`
- Apply `.opacity(effect.enabled ? 1.0 : 0.5)` to layer card
- Backend validates but ignores disabled effects in processing
- Position toggle in layer card header alongside effect name

## Related
Parent: EPIC-003
Depends on: TICKET-012 (effect models), TICKET-014 (reordering)

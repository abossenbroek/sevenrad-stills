# [TICKET-018] Repeat Count Configuration

## Context
Many effects (blur, saturation, pixelsort) compound when applied multiple times. Simple repeat count allows users to intensify effects without adding duplicate layers. This is a simple stepper control (1-100) that applies the effect N times uniformly across all frames.

## User Story
As a user,
I want to set a repeat count for an effect,
So that it applies multiple times in sequence for stronger results.

## Acceptance Criteria
- [ ] Stepper control in parameter panel for repeat count
- [ ] Range constrained to 1-100
- [ ] Layer stack displays "×N" indicator when repeat > 1
- [ ] Repeat count updates pipeline state
- [ ] Default repeat is 1
- [ ] Stepper shows current value
- [ ] Backend validation allows max 100 repeats

## Technical Notes
- Use SwiftUI `Stepper` with range: `1...100`
- Display in layer: `Text("×\(effect.repeat)").font(.caption).foregroundColor(.secondary)`
- Position indicator in top-right corner of layer card
- Backend enforces this limit (see backend-contract.md section 11.2)
- NOTE: This is simple uniform repeat - frame-level step sequencer is EPIC-008 (separate feature)

## Related
Parent: EPIC-003
Depends on: TICKET-009 (parameter panel UI), TICKET-017 (parameter binding)

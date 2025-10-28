# [TICKET-044] Frame Override Data Model

## Context
Extend Pipeline YAML schema to support per-frame effect overrides, enabling serialization and persistence of step sequencer state.

## User Story
As a technical artist,
I want my frame-level effect toggles to be saved with the pipeline,
So that I can close the app and resume work without losing my sequencer patterns.

## Acceptance Criteria
- [ ] Pipeline model includes frame_overrides array with frame_index and effect_overrides
- [ ] Each override specifies effect_id and enabled boolean
- [ ] YAML serialization writes frame_overrides section
- [ ] YAML deserialization reads frame_overrides and validates frame indices
- [ ] Frame indices must be within extracted frame count (validation error otherwise)
- [ ] Effect IDs must match existing pipeline effects (validation error otherwise)
- [ ] Default behavior: all effects enabled if no override specified

## Technical Notes
```yaml
frame_overrides:
  - frame_index: 0
    effect_overrides:
      - effect_id: "550e8400-e29b-41d4-a716-446655440000"
        enabled: false
  - frame_index: 5
    effect_overrides:
      - effect_id: "550e8400-e29b-41d4-a716-446655440001"
        enabled: true
```
- Sparse representation: only store overrides, not full grid state
- Validation: frame_index < frame_count, effect_id exists in pipeline.effects

## Related
- EPIC-008: Frame-Level Step Sequencer
- TICKET-012: Pipeline model (extends this)

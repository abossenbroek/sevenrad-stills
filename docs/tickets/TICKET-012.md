# [TICKET-012] Effect Model and Registry

## Context
Foundation for all pipeline operations. Defines Swift data structures for effects, parameters, and pipelines. Creates a registry matching the 17 effects defined in the Python backend API contract.

## User Story
As a developer,
I want strongly-typed Swift models for effects and pipelines,
So that the app can validate and manipulate effect configurations safely.

## Acceptance Criteria
- [ ] `Effect` struct with id (UUID), name, enabled, repeat, params
- [ ] `EffectParameter` struct with type, min/max, default, step
- [ ] `Pipeline` struct with effects array and source metadata
- [ ] `EffectRegistry` singleton containing all 17 effect definitions
- [ ] Parameter schemas match backend-contract.md specifications
- [ ] Models conform to `Codable` for JSON serialization
- [ ] All 17 effects from Appendix A in backend-contract.md registered

## Technical Notes
- Reference `/docs/api/backend-contract.md` Appendix A for complete effect list
- Use `AnyCodable` wrapper for heterogeneous parameter values
- Effect names must match backend exactly (case-sensitive)
- UUID generation for effect instances on creation
- Parameter types: float, int, bool, enum, color

State Management Architecture:
- `Pipeline` struct: Codable data model (effects array, metadata)
- `PipelineManager` class: ObservableObject managing pipeline state
- Published property: `@Published var pipeline: Pipeline`
- Methods: `addEffect()`, `removeEffect()`, `reorderEffects()`, `updateParameter()`
- Owned by root app view, injected via `@EnvironmentObject`

## Related
Parent: EPIC-003

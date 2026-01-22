# [EPIC-003] Effect Pipeline Management

## Context
The core of the macOS Swift drum machine UI - enabling users to build, reorder, and configure effect chains. This epic establishes the foundation for all subsequent work: without a working pipeline, users cannot create, edit, or render processed images. This implements the creative heart of the application where users compose visual transformations.

## User Story
As a user,
I want to add, reorder, and configure effects in a pipeline,
So that I can create complex image transformations through layered processing.

## Scope
- Add effects from pad grid to pipeline
- Visual layer stack showing effect order
- Drag-and-drop layer reordering
- Parameter editing for selected layers
- Enable/disable effects without removal
- Repeat count configuration (1-100)
- Layer selection and highlighting
- Remove layers from pipeline

Out of scope:
- Undo/redo functionality (EPIC-004)
- Real-time preview rendering (EPIC-005)
- Pipeline persistence/saving (EPIC-006)
- Backend XPC preview rendering (EPIC-005) - this epic handles pipeline STATE updates only
- Effect preset management
- Frame-level step sequencer (EPIC-008)

## Success Criteria
- [ ] User can tap any of 17 effect pads to add effect to pipeline
- [ ] Layer stack displays all active effects in processing order
- [ ] User can drag layers to reorder pipeline
- [ ] Selected layer shows orange accent with parameter panel
- [ ] Parameter sliders update effect configuration in real-time
- [ ] Repeat count stepper allows 1-100 iterations
- [ ] Trash icon removes layer from pipeline
- [ ] Disabled effects remain in pipeline but skip processing

## Related Tickets
- TICKET-012: Effect Model and Registry
- TICKET-013: Add Effect to Pipeline Action
- TICKET-014: Layer Reordering (Drag-and-Drop)
- TICKET-015: Layer Selection and Highlighting
- TICKET-016: Remove Layer Action
- TICKET-017: Parameter Binding and Validation
- TICKET-018: Repeat Count Configuration
- TICKET-059: Toggle Effect Enable/Disable
- TICKET-060: Pipeline Validation and Error Display

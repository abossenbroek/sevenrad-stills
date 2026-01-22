# [TICKET-019] Command Pattern Infrastructure

## Context
All pipeline modifications must be captured as commands to enable undo/redo and time-travel functionality. This ticket establishes the foundation for the history system by implementing the Command pattern for all effect operations.

## Architecture Decision
This implementation uses Command Pattern with replay-based undo.
Commands store minimal deltas, NOT full state snapshots.
Undo works by: reset pipeline + replay commands[0:index-1]
Commands do NOT need undo() methods - restoration is handled by replay.

## User Story
As a developer,
I want all pipeline modifications to follow the Command pattern,
So that each action can be executed, described, and replayed for history management.

## Acceptance Criteria
- [ ] Command protocol defined with execute() and description() methods
- [ ] AddEffectCommand implementation complete
- [ ] RemoveEffectCommand implementation complete
- [ ] ChangeParameterCommand implementation complete
- [ ] ReorderEffectCommand implementation complete
- [ ] Each command returns JSON-serializable state delta
- [ ] Unit tests verify command execution and description generation

## Technical Notes
- Commands store minimal state deltas, not full pipeline snapshots
- Description format shows new value only (e.g., "Reverb: Mix 40%", "Add Delay")
- Commands operate on pipeline model established in TICKET-012
- Consider using Codable for state serialization

## Related
Parent: EPIC-004
Dependencies: TICKET-012 (pipeline model)
Blocks: TICKET-020 (history state manager)

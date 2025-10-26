# [TICKET-023] Time-Travel State Restoration

## Context
When users click a history item to restore a previous state, the system must reset the pipeline and replay all commands up to that point. If changes are made after time-travel, future states are discarded to maintain linear history.

## User Story
As a user,
I want to click any history state to restore that exact pipeline configuration,
So that I can explore different creative directions without manually recreating settings.

## Acceptance Criteria
- [ ] Clicking history item resets pipeline to initial state
- [ ] Commands replay sequentially up to selected index
- [ ] Preview re-render is debounced - triggers only after restoration completes
- [ ] Show progress indicator if restoration takes >200ms
- [ ] Future states marked visually as "unreachable"
- [ ] Future states are permanently discarded after branching
- [ ] Restoration completes within 100ms for typical command stacks

## Technical Notes
- Reset pipeline by clearing all effects
- Replay commands using execute() methods from TICKET-019
- Use HistoryManager.restore_to_index() from TICKET-020
- Consider debouncing preview re-render for rapid state changes

## Related
Parent: EPIC-004
Dependencies: TICKET-020 (history state manager), TICKET-021 (history panel UI)

# [TICKET-021] History Panel UI Component

## Context
Users need a visual interface to see their action history and navigate to previous states. This panel provides a Photoshop-style history list with clear indicators of the current state.

## User Story
As a user,
I want to see all my actions listed in a collapsible history panel,
So that I can understand what I've done and quickly restore any previous state.

## Acceptance Criteria
- [ ] Collapsible sidebar panel (240pt wide when open)
- [ ] List displays command descriptions in chronological order
- [ ] Current state indicated with orange dot or highlight
- [ ] Click any history item to restore that state
- [ ] Panel toggle button with keyboard shortcut (Cmd+H)
- [ ] Future states shown with reduced opacity after time-travel
- [ ] Smooth animations for state changes
- [ ] State changes trigger automatic UI updates via observation

## Technical Notes
- Use SwiftUI List with custom row styling
- Integrate with HistoryManager @Published properties
- Panel state (open/closed) persists in UserDefaults
- Consider grouping commands by time (e.g., "5 minutes ago")

## Related
Parent: EPIC-004
Dependencies: TICKET-020 (history state manager)

# [TICKET-022] Undo/Redo Keyboard Shortcuts

## Context
Standard keyboard shortcuts for undo and redo are essential for efficient workflow. Users expect Cmd+Z and Cmd+Shift+Z to work immediately without needing to use the history panel.

## User Story
As a user,
I want to use Cmd+Z to undo and Cmd+Shift+Z to redo,
So that I can quickly reverse or reapply actions using familiar keyboard shortcuts.

## Acceptance Criteria
- [ ] Cmd+Z triggers HistoryManager.undo()
- [ ] Cmd+Shift+Z triggers HistoryManager.redo()
- [ ] UI updates immediately to reflect state changes
- [ ] Shortcuts disabled when at history boundaries (no undo/redo available)
- [ ] Menu items show enabled/disabled state correctly
- [ ] State changes trigger automatic UI updates via observation

## Technical Notes
- Register keyboard shortcuts in SwiftUI using .keyboardShortcut()
- Update menu items in App menu to show shortcuts
- Consider visual feedback (brief animation) on undo/redo

## Related
Parent: EPIC-004
Dependencies: TICKET-020 (history state manager)

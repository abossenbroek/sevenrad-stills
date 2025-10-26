# [TICKET-011b] Application Shell & Menu Bar

## Context
Create the macOS application entry point with standard menu bar structure, providing the foundation for window management and global application commands.

## User Story
As a user,
I want standard macOS menus and application lifecycle management,
So that the app behaves like a native macOS application.

## Acceptance Criteria
- [ ] @main App struct with WindowGroup scene containing MainWindowView
- [ ] Standard macOS menu bar with File, Edit, View, Window, Help menus
- [ ] File menu: New Session, Open Pipeline..., Save Pipeline..., Quit (Cmd+Q)
- [ ] Edit menu: Undo (Cmd+Z), Redo (Cmd+Shift+Z), Cut, Copy, Paste
- [ ] View menu: Zoom In/Out/Reset, Toggle Effect Grid, Toggle Layer Stack, Toggle Parameters
- [ ] Window menu: Minimize, Zoom, Bring All to Front (system-provided)
- [ ] Help menu: Application Help, link to documentation
- [ ] Window delegate handles willClose, didBecomeMain, didResignMain events
- [ ] Commands disabled appropriately when no session active

## Technical Notes
- Implement in `Sources/App/VideoEffectApp.swift`
- Use `.commands()` modifier for menu bar customization
- Create `AppCommands.swift` for CommandGroup definitions
- Window delegate pattern: `NSWindowDelegate` conformance
- Menu items send actions via AppState/MainViewModel

## Related
- Parent: EPIC-002
- Depends: TICKET-011c (requires AppState)
- Blocks: TICKET-006 (provides WindowGroup container)

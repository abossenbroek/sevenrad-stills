# Keyboard Shortcut Registry

## Purpose
Centralized documentation and conflict detection for all keyboard shortcuts across the application.

## Shortcut Definitions

| Shortcut | Scope | Action | Ticket | Notes |
|----------|-------|--------|--------|-------|
| `Cmd+Z` | Global | Undo | TICKET-022 | Standard macOS undo |
| `Cmd+Shift+Z` | Global | Redo | TICKET-022 | Standard macOS redo |
| `Cmd+S` | Global | Save Project | TICKET-008 | Standard save |
| `Cmd+O` | Global | Open Project | TICKET-008 | Standard open |
| `Cmd+N` | Global | New Project | TICKET-008 | Standard new |
| `Cmd+W` | Global | Close Window | System | macOS standard |
| `Cmd+Q` | Global | Quit Application | System | macOS standard |
| `Tab` | Panel Focus | Cycle Forward | TICKET-035 | Moves through: Grid → Layers → Parameters → Transport |
| `Shift+Tab` | Panel Focus | Cycle Backward | TICKET-035 | Reverse panel cycle |
| `Arrow Keys` | Pattern Grid | Navigate Pads | TICKET-035 | 2D navigation, wraps at edges |
| `Arrow Up/Down` | Layer List | Navigate Layers | TICKET-035 | Vertical list navigation |
| `Enter` | Pattern Grid | Activate Pad | TICKET-035 | Trigger selected pad |
| `Space` | Layer List | Toggle Mute/Solo | TICKET-035 | On selected layer |
| `Cmd+Up` | Layer List | Move Layer Up | TICKET-039 | Accessibility alternative to drag |
| `Cmd+Down` | Layer List | Move Layer Down | TICKET-039 | Accessibility alternative to drag |
| `Delete` | Layer List | Remove Layer | TICKET-035 | Remove selected layer |
| `Escape` | Modal/Dialog | Cancel/Close | TICKET-035 | Dismiss modal, cancel operation |
| `Cmd+,` | Global | Open Preferences | Future | Standard preferences shortcut |

## Conflict Detection

### Rules
1. No two shortcuts in the same scope may use identical key combinations
2. Global shortcuts override panel-specific shortcuts
3. System shortcuts (Cmd+W, Cmd+Q) cannot be overridden
4. Modal dialogs trap focus and prevent global shortcuts during display

### Validation Checklist
- [ ] All shortcuts documented in this registry
- [ ] No duplicate shortcuts within same scope
- [ ] All shortcuts tested with VoiceOver enabled
- [ ] Shortcuts displayed in tooltips (see TICKET-036)
- [ ] Shortcuts displayed in menu bar (where applicable)

## Implementation Guidelines

### SwiftUI Integration
```swift
// Example: Register global shortcut
.keyboardShortcut("z", modifiers: .command)

// Example: Panel-specific shortcut with focus check
.keyboardShortcut(.upArrow, modifiers: .command)
.disabled(!isFocused)
```

### Tooltip Integration
All shortcuts must be documented in tooltips:
```
Format: "[Control Name]: [Description]. [Shortcut]"
Example: "Move Layer Up: Reorder layer toward top of stack. Cmd+↑"
```

## Future Considerations
- User-customizable shortcuts (not in current scope)
- Vim-style navigation modes (hjkl as alternative to arrows)
- Quick-access number keys for pad selection (1-9 for grid positions)

## Related Tickets
- TICKET-022: Undo/Redo implementation
- TICKET-035: Keyboard navigation system
- TICKET-036: Tooltip documentation
- TICKET-039: Accessibility keyboard alternatives

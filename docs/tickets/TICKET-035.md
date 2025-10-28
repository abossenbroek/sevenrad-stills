# [TICKET-035] Keyboard Navigation

## Context
Implement comprehensive keyboard navigation for power users and accessibility. Users should be able to operate the entire app without touching the mouse.

## User Story
As a user,
I want to navigate and control the app using only my keyboard,
So that I can work efficiently and the app is accessible.

## Acceptance Criteria
- [ ] Tab key cycles focus through panels: Grid → Layers → Parameters → Transport (wraps to first)
- [ ] Shift+Tab cycles backward through panels
- [ ] Pattern grid: Arrow keys navigate in 2D, wrap at edges, Enter activates selected pad
- [ ] Layer list: Arrow Up/Down navigate layers, Space toggles mute/solo, Delete removes layer
- [ ] Modal dialogs: Trap focus within dialog, Escape closes, restore previous focus on dismiss
- [ ] Focused elements show 2pt orange border with 4pt padding (meets WCAG 2.1 AA contrast)
- [ ] All shortcuts registered in centralized keyboard shortcut registry

## Technical Notes
- Use SwiftUI's `.focusable()` and `@FocusState` for focus management
- Implement custom keyboard shortcuts using `.keyboardShortcut()` modifier
- Reference `docs/design/keyboard-shortcuts.md` for shortcut registry
- Modal focus trapping: Use `.focusScope()` and prevent Tab from escaping
- Test with VoiceOver enabled to ensure compatibility
- Grid navigation: Track `@State selectedGridPosition: (x: Int, y: Int)`, clamp/wrap on arrow keys

## Related
- Parent: EPIC-007
- Dependencies: All UI implementation tickets
- Related: TICKET-036 (keyboard shortcuts need tooltip documentation)

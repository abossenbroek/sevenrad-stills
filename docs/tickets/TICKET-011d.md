# [TICKET-011d] Core Accessibility Foundation

## Context
Implement accessibility features across all UI components to ensure the application is usable by everyone, including users with disabilities, meeting macOS accessibility standards.

## User Story
As a user with accessibility needs,
I want full keyboard navigation and screen reader support,
So that I can use the application effectively.

## Acceptance Criteria
- [ ] All interactive elements have meaningful accessibility labels
- [ ] Focus order follows logical reading order: Effect Grid → Layer Stack → Parameters → Preview
- [ ] Keyboard navigation: Tab moves between panels, Arrow keys navigate within panels
- [ ] VoiceOver announces state changes: layer added, effect selected, parameter changed
- [ ] Color contrast meets WCAG 2.1 AA standards (4.5:1 for normal text, 3:1 for large text)
- [ ] Drag-and-drop keyboard alternative: Cmd+Up/Down moves selected layer in stack
- [ ] Focus indicators visible on all focusable elements (2px accent color ring)
- [ ] All functionality accessible without mouse/trackpad

## Technical Notes
- Use `.accessibilityLabel()`, `.accessibilityHint()`, `.accessibilityValue()` modifiers
- Implement custom `.accessibilityAction()` for drag-and-drop alternative
- Use `.focusable()` and `.focusSection()` for keyboard navigation
- Add `.accessibilityElement(children: .contain)` for container views
- Test with VoiceOver enabled (Cmd+F5)
- Verify contrast ratios using Accessibility Inspector

## Related
- Parent: EPIC-002
- Depends: TICKET-011, TICKET-011c (requires DesignSystem, AppState)
- Blocks: All component tickets (TICKET-007, 008, 009, 010)

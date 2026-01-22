# [TICKET-039] Accessibility Compliance (WCAG 2.1 AA)

## Context
Ensure the app meets WCAG 2.1 AA accessibility standards for color contrast, keyboard access, and reduced motion support.

## User Story
As a user with accessibility needs,
I want the app to meet accessibility standards,
So that I can use it regardless of vision or motor limitations.

## Acceptance Criteria
- [ ] Text contrast audit: All text meets 4.5:1 ratio minimum (3:1 for text larger than 18pt)
- [ ] UI component contrast: Interactive elements meet 3:1 ratio against adjacent colors
- [ ] Reduced motion: Respect `NSWorkspace.shared.accessibilityDisplayShouldReduceMotion`
- [ ] Touch targets: All interactive elements minimum 44x44pt clickable area
- [ ] Accessibility labels: All buttons and controls have descriptive `accessibilityLabel`
- [ ] Accessibility hints: Non-obvious controls have `accessibilityHint` explaining usage
- [ ] Keyboard alternative for drag-and-drop: Cmd+Up/Down to reorder layers

## Technical Notes
- Use WebAIM Contrast Checker or similar tool for color audit
- Test with System Preferences > Accessibility > Display > Reduce Motion enabled
- SwiftUI: Check `.accessibilityReduceMotion` environment value
- Minimum touch target: Use `.frame(minWidth: 44, minHeight: 44)` with `.contentShape(Rectangle())`
- Test with Accessibility Inspector (Xcode > Open Developer Tool)
- VoiceOver compatibility: Tab navigation must work with VoiceOver enabled

## Related
- Parent: EPIC-007
- Related: TICKET-035 (keyboard navigation is accessibility foundation)

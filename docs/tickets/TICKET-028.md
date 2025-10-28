# [TICKET-028] Real-Time Preview Updates

## Context
Users adjust parameters frequently. Preview updates must be debounced to avoid overwhelming the backend while maintaining responsive feedback.

## User Story
As a user,
I want the preview to update smoothly when I adjust parameters,
So that I can explore effects without lag or jank.

## Acceptance Criteria
- [ ] Parameter changes debounced with 300ms delay
- [ ] Loading spinner shown when render starts
- [ ] Preview image replaced when render completes
- [ ] Preview updates complete within 1 second (from last change to display)
- [ ] Multiple rapid changes only trigger one render (last value wins)

## Technical Notes
- Use Combine debounce operator or custom timer
- Track isRendering state to show/hide spinner
- Cancel in-flight renders when new request starts
- Consider using @Published properties for reactive updates
- Test with rapid slider adjustments

## Related
- Parent: EPIC-005
- Depends on: TICKET-027, TICKET-017

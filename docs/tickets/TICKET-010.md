# [TICKET-010] Preview Panel Content

## Context
Create the preview content view that displays the current video frame with applied effects, providing immediate visual feedback for the effect pipeline. This view receives size constraints from its parent container.

## User Story
As a user,
I want to see a preview of my effect pipeline on the current frame,
So that I can evaluate my artistic choices in real-time.

## Acceptance Criteria
- [ ] PreviewContentView receives size from parent, maintains aspect ratio
- [ ] Image centered in available space with #0a0a0a background
- [ ] Placeholder state: "No frame loaded" with RYTM styling
- [ ] Image updates when frame position changes
- [ ] Image updates when effect parameters change (inherits debounce from AppState)
- [ ] Zoom controls: Fit, 50%, 100%, 200% (toolbar buttons)
- [ ] Pan support when zoomed (click-drag or trackpad gesture)
- [ ] Loading indicator during effect processing (RYTM spinner style)
- [ ] Display format info: resolution, color space (small footer text)

## Technical Notes
- Implement as `PreviewContentView` with `ZoomableImageView`
- Use `NSImage` or `CGImage` for rendering
- State: `@EnvironmentObject var appState: AppState`
- Implement zoom with `scaleEffect()` modifier
- Pan with `DragGesture()` and `offset()` modifier
- Does not manage debouncing (handled by AppState in TICKET-011c)

## Related
- Parent: EPIC-002
- Depends: TICKET-006 (receives size from container), TICKET-011c (binds to AppState)
- Blocks: EPIC-003, EPIC-005

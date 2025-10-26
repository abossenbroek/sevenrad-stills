# [TICKET-040] Operation Progress UI

## Context
Provide visual feedback for long-running operations so users know the app is working and how long to wait.

## User Story
As a user,
I want to see progress for long operations,
So that I know the app is working and can estimate completion time.

## Acceptance Criteria
- [ ] Progress bars for operations exceeding 2 seconds (frame extraction, video download, export)
- [ ] Time estimates displayed: "Extracting frames... 45% complete, ~30s remaining"
- [ ] Cancel buttons available for all long operations
- [ ] Background operation status indicators in menu bar or status area
- [ ] Progress UI updates at least once per second

## Technical Notes
- Use SwiftUI `ProgressView` with determinate progress (0.0-1.0)
- Time estimation: Track recent progress rate, extrapolate remaining time
- Cancel mechanism: Use `Task` cancellation for async operations
- Status area: Show mini progress indicator with tooltip on hover
- Consider macOS notification for completion of background operations
- Test with slow network/disk to verify progress accuracy

## Related
- Parent: EPIC-007
- Dependencies: TICKET-001 (video download), TICKET-002 (frame extraction), TICKET-008 (export)

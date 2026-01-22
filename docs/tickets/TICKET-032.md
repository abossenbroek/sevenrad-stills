# [TICKET-032] Progress Tracking UI

## Context
Batch rendering can take significant time for large frame sequences. This ticket implements a modal progress dialog that shows real-time rendering status with cancellation confirmation and Finder integration on completion.

## User Story
As a user,
I want to see progress during batch rendering,
So that I know how long the process will take and can cancel if needed.

## Acceptance Criteria
- [ ] Display modal progress sheet during render operation
- [ ] Show "Processing frame N/M..." text with current and total counts
- [ ] Display NSProgressIndicator with percentage (0-100%)
- [ ] Provide Cancel button with confirmation dialog
- [ ] Show cancellation confirmation: "Cancel rendering? N frames completed, M remaining."
- [ ] Keep partial results in output directory if cancelled (non-destructive)
- [ ] Subscribe to TICKET-031's pre-throttled progress publisher
- [ ] Update UI on main thread (assume input is already throttled to 10/sec)
- [ ] Show "Reveal in Finder" button on completion
- [ ] Show completion state (success/cancelled/error) before dismissing

## Technical Notes
- Depends on TICKET-031 for progress stream
- Does NOT implement throttling (assumes pre-throttled input from TICKET-031)
- Use Combine to subscribe to progress publisher from TICKET-031
- Use NSProgressIndicator in indeterminate mode initially
- Switch to determinate mode when total frame count is known
- Cancellation preserves all completed frames (non-destructive behavior)
- "Reveal in Finder" uses TICKET-061 integration

## Related
Parent Epic: EPIC-006
Dependencies: TICKET-031, TICKET-061

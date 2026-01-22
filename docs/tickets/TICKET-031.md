# [TICKET-031] Batch Rendering Queue

## Context
Users need to apply their pipeline to all frames in a session with non-destructive output. This ticket implements batch rendering with incremental filename generation, progress throttling, and validation checks matching the project's image editing standards.

## User Story
As a user,
I want to render my entire frame sequence with the current pipeline,
So that I can generate final processed output for all frames without overwriting originals.

## Acceptance Criteria
- [ ] Generate incremental filenames: `{pipeline_name}_{frame_stem}_v{version:03d}.{ext}`
- [ ] Auto-increment version number if output files exist (non-destructive mode)
- [ ] Preserve original frame files (never overwrite source frames)
- [ ] Save pipeline YAML alongside rendered frames in output directory
- [ ] Warn if frame count >100 frames (show estimated render time)
- [ ] Validate disk space before rendering (100 frames × 5MB = 500MB minimum)
- [ ] Call backend render_sequence() via XPC with frame array and pipeline JSON
- [ ] Receive raw progress callbacks from backend and throttle to max 10 updates/sec
- [ ] Publish throttled progress via Combine for UI subscription
- [ ] Support cancellation requests that terminate backend processing
- [ ] Handle completion with success/failure status

## Technical Notes
- Depends on TICKET-027 for XPC communication
- Depends on TICKET-005 for session frame list
- Threading model: Receives raw callbacks, throttles to 10/sec, publishes via Combine
- TICKET-032 subscribes to this publisher (does NOT implement throttling)
- Backend returns progress as stream of (current, total) tuples
- Filename versioning: scan output dir for existing files, increment v001 → v002 → v003
- Disk space check: use FileManager.attributesOfFileSystem(forPath:)
- Estimated render time based on previous frame processing metrics

## Related
Parent Epic: EPIC-006
Dependencies: TICKET-027, TICKET-005
Blocks: TICKET-032

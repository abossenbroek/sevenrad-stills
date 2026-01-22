# [TICKET-005] Frame Extraction Service Integration

## Context
After segment selection, users initiate frame extraction by delegating to the Python backend. This ticket integrates with the backend extraction service, handles progress updates, and manages the transition to the next workflow stage. This is the completion point of EPIC-001.

## User Story
As a user,
I want to extract frames from my selected segment with visible progress,
So that I can proceed to the image processing workflow with confidence that extraction is working.

## Acceptance Criteria
- [ ] Extract button triggers XPC call to backend with segment parameters
- [ ] Passes video path, start time, end time, and hardcoded interval (1.0 seconds) to backend
- [ ] Checks available disk space before starting extraction
- [ ] Receives progress updates via callback/stream
- [ ] Updates progress indicator showing frame count or percentage
- [ ] Shows estimated time remaining during extraction
- [ ] Allows user to cancel extraction mid-process
- [ ] On cancellation: cleanup partial frames and reset session state
- [ ] Handles completion successfully and stores session_id
- [ ] Transitions to Performance view upon successful completion
- [ ] Shows clear error message if extraction fails
- [ ] Disables extract button during active extraction

## Technical Notes
- XPC call to backend: `extract_frames(video_path: String, start_time: Double, end_time: Double, interval: Double, progress_callback: ProgressHandler) -> Result<SessionInfo, Error>`
- SessionInfo should contain:
  - session_id: String
  - frame_count: Int
  - output_folder: String
- **Hardcoded frame interval: 1.0 seconds** (can be made configurable in future epic)
- Progress callback should receive:
  - current_frame: Int
  - total_frames: Int
  - elapsed_time: Double
- Use async/await for XPC call with progress streaming
- Cancel operation via XPC: `cancel_extraction(session_id: String) -> Result<Void, Error>`
- Cancellation cleanup behavior: delete partial frames and reset session directory to clean state
- Disk space check: verify at least 1GB free space before starting extraction
- Store session_id in app state for EPIC-003 (Performance View)
- Consider showing thumbnail of first extracted frame as confirmation

## Dependencies
Requires: TICKET-004 (needs segment selection)

## Related
Parent: EPIC-001
Enables: EPIC-003 (Performance & Processing - next workflow stage)

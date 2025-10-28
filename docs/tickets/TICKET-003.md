# [TICKET-003] Video Metadata Display

## Context
Before selecting a segment and extracting frames, users need to understand the video's properties (duration, resolution, frame rate). This information informs their segment selection and sets expectations for the extraction process. For YouTube videos, metadata is cached from the download operation; for local files, metadata must be fetched separately.

## User Story
As a user,
I want to see video metadata after loading a video,
So that I can make informed decisions about segment selection and frame extraction.

## Acceptance Criteria
- [ ] For YouTube path (TICKET-001): Use cached metadata from DownloadResult
- [ ] For local file path (TICKET-002): Fetch metadata via XPC call to backend
- [ ] Validates metadata: duration > 0, resolution within reasonable limits (e.g., max 4K)
- [ ] Displays duration in human-readable format (MM:SS or HH:MM:SS for longer videos)
- [ ] Displays resolution (e.g., "1920×1080" or "4K")
- [ ] Displays frame rate (e.g., "30 fps" or "29.97 fps")
- [ ] Shows loading state while fetching metadata (local files only)
- [ ] Shows clear error message if metadata fetch fails or validation fails
- [ ] Metadata remains visible throughout segment selection workflow

## Technical Notes
- XPC call to backend (local files only): `get_video_metadata(file_path: String) -> Result<VideoMetadata, Error>`
- VideoMetadata struct should contain:
  - duration_seconds: Double
  - width: Int
  - height: Int
  - fps: Double
- Format duration using DateComponentsFormatter for consistency
- Validation checks:
  - duration_seconds > 0
  - width and height between 1 and 3840 (4K max)
  - fps > 0 and < 120
- Consider displaying additional metadata (codec, file size) as stretch goal
- For YouTube path: metadata already available from TICKET-001's DownloadResult
- For local file path: fetch metadata immediately after TICKET-002 completes

## Dependencies
Requires: TICKET-055 (session must be initialized first)
Requires: TICKET-001 OR TICKET-002 (video must be loaded first)

## Related
Parent: EPIC-001
Enables: TICKET-004 (timeline needs duration)

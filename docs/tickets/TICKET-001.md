# [TICKET-001] YouTube URL Input & Validation

## Context
Users need a simple, reliable way to load videos from YouTube as their primary content source. This is the first entry point in the video loading workflow and must validate URLs before delegating download operations to the Python backend.

## User Story
As a technical artist,
I want to paste a YouTube URL into the application,
So that the video can be downloaded and prepared for frame extraction.

## Acceptance Criteria
- [ ] Text field accepts and displays user input for URLs
- [ ] Validates YouTube URL format using regex before submission (supports youtube.com and youtu.be formats)
- [ ] Shows clear error message for invalid URL formats
- [ ] Passes validated URL to backend via XPC for download
- [ ] Shows loading indicator during download process
- [ ] Handles backend errors (network failures, invalid video ID, age-restricted content)
- [ ] Stores complete DownloadResult (file path and metadata) in app state

## Technical Notes
- Use SwiftUI TextField with .onSubmit modifier
- URL validation regex should match:
  - `https://www.youtube.com/watch?v=VIDEO_ID`
  - `https://youtu.be/VIDEO_ID`
  - Optional query parameters after VIDEO_ID
- XPC call to backend: `download_youtube_video(url: String) -> Result<DownloadResult, Error>`
- DownloadResult struct should contain:
  - file_path: String (local path to downloaded video)
  - metadata: VideoMetadata (duration, resolution, fps from YouTube)
- Store DownloadResult in app state for subsequent operations
- Consider debouncing or button-based submission to prevent excessive validation

## Dependencies
Requires: TICKET-055 (session must be initialized first)

## Related
Parent: EPIC-001

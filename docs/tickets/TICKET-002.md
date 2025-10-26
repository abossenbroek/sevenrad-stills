# [TICKET-002] Local Video File Picker

## Context
Users working with local video files need a native macOS file picker to select content without relying on YouTube. This provides an alternative video source and supports offline workflows.

## User Story
As a user,
I want to select a video file from my filesystem,
So that I can use local content for frame extraction without downloading from YouTube.

## Acceptance Criteria
- [ ] Button triggers native macOS file picker (NSOpenPanel)
- [ ] File picker filters for common video formats (.mp4, .mov, .avi, .mkv, .webm)
- [ ] Selected file path is validated (file exists and is readable)
- [ ] File path is stored in app state for subsequent operations
- [ ] Shows error if selected file is not a valid video or is inaccessible
- [ ] Transitions to metadata display upon successful selection

## Technical Notes
- Use NSOpenPanel with allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .avi]
- SwiftUI integration via fileImporter modifier on button
- Validate file accessibility using FileManager.default.isReadableFile(atPath:)
- Store absolute file path as URL in app state
- No need to copy file - work directly with user's file location

## Story Points
1

## Dependencies
None

## Related
Parent: EPIC-001
Alternative to: TICKET-001

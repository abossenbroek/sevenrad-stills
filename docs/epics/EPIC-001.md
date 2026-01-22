# [EPIC-001] Video Source & Frame Extraction

## Context
This epic establishes the foundational workflow for loading videos and extracting frames in the drum machine-inspired image processing app. Users need the ability to source video content either from YouTube URLs or local filesystem, inspect video properties, select specific segments, and extract frames for processing. This is the first interaction point in the application and sets up the session-based workflow for subsequent image processing operations.

## User Story
As a technical artist,
I want to load a video from YouTube or my filesystem and extract frames from a selected segment,
So that I can begin my algorithmic image transformation workflow with source material.

## Scope
- YouTube URL input with validation and download delegation
- Local video file selection via native macOS file picker
- Video metadata display (duration, resolution, fps)
- Timeline scrubber with draggable start/end markers for segment selection
- Frame extraction with progress tracking and cancellation
- Session folder creation and management
- Transition to next workflow stage upon completion

Out of scope:
- Python/MLX backend implementation (assume exists)
- Video playback and preview functionality
- Frame interval configuration UI (use sensible defaults)
- Advanced video filters or preprocessing
- Multi-video batch processing
- Video format transcoding
- Frame thumbnail grid display (handled by EPIC-003)

## Success Criteria
- [ ] User can paste a YouTube URL and successfully load video metadata
- [ ] User can select a local video file and successfully load video metadata
- [ ] Video metadata (duration, resolution, fps) displays correctly
- [ ] Timeline scrubber allows selection of start/end segment with visual feedback
- [ ] Frame extraction completes successfully with progress indicator
- [ ] Extracted frames are stored in a session folder with valid session_id
- [ ] Application transitions to Performance view after successful extraction
- [ ] All error states are handled gracefully with clear user feedback

## Priority
P0 (MVP Critical)

## Dependencies
None - this is a foundational epic that can start immediately

## Can Be Parallelized With
EPIC-002 (UI Shell & Navigation)

## Related Tickets
- TICKET-055: Session Initialization & Management
- TICKET-001: YouTube URL Input & Validation
- TICKET-002: Local Video File Picker
- TICKET-003: Video Metadata Display
- TICKET-004: Timeline Scrubber Component
- TICKET-005: Frame Extraction Service Integration

## Technical Notes
- All backend communication happens via XPC service boundary
- Backend API contract defined in `docs/api/backend-contract.md`
- Session management follows existing Python backend patterns
- Frame extraction interval defaults to 1 frame per second (configurable later)
- Timeline scrubber should use SwiftUI Geometry Reader for responsive layout

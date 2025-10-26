# [TICKET-055] Session Initialization & Management

## Context
All frame extraction and processing operations require a session context with a unique identifier and directory structure. Session creation must happen before any video operations, and cleanup must occur on app exit to prevent resource leaks.

## User Story
As a user,
I want the application to automatically manage my work session,
So that all extracted frames and processing artifacts are organized and cleaned up properly.

## Acceptance Criteria
- [ ] XPC call to backend `create_session()` executes on app launch
- [ ] Store session_id, base_dir, and directory structure in app state
- [ ] Session directory structure is validated (frames/, processed/, exports/ subdirectories exist)
- [ ] Session cleanup XPC call executes on app exit (onDisappear or AppDelegate termination)
- [ ] Handles session creation errors gracefully (disk space, permissions)
- [ ] Shows clear error message if session cannot be created
- [ ] App cannot proceed to video loading without valid session

## Technical Notes
- XPC call to backend: `create_session() -> Result<SessionInfo, Error>`
- SessionInfo struct should contain:
  - session_id: String (UUID format)
  - base_dir: String (absolute path)
  - subdirectories: [String] (frames, processed, exports)
- XPC cleanup call: `cleanup_session(session_id: String) -> Result<Void, Error>`
- Store session in SwiftUI EnvironmentObject or ObservableObject for global access
- Session should be created in App.init() or first view's onAppear
- Consider showing session_id in debug builds for troubleshooting

## Dependencies
None - must execute before all other tickets

## Related
Parent: EPIC-001
Enables: TICKET-001, TICKET-002, TICKET-003, TICKET-004, TICKET-005

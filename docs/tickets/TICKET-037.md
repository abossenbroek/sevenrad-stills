# [TICKET-037] Error Message Refinement

## Context
Present errors from TICKET-029 in user-friendly format. This ticket owns error messaging and UI presentation while TICKET-029 owns error detection and retry logic.

## User Story
As a user,
I want error messages that clearly explain what went wrong and how to fix it,
So that I can resolve issues quickly without confusion.

## Acceptance Criteria
- [ ] All error messages follow format: "What happened" + "Why" + "What to do"
- [ ] Error alerts include actionable buttons (e.g., "Choose Different File", "Retry", "Cancel")
- [ ] Consistent error UI styling (alert dialog with red accent, error icon)
- [ ] Technical errors logged to console with stack trace while showing user-friendly message
- [ ] Network/file errors include specific details (e.g., "Could not load 'video.mp4': File not found")

## Technical Notes
- Boundary with TICKET-029:
  - TICKET-029 owns: Error detection, retry logic, throwing typed errors (XPCError, ValidationError, TimeoutError)
  - TICKET-037 owns: Error presentation, user messaging, alert UI
- Create `ErrorPresenter` utility to standardize error formatting
- Use `.alert()` modifier with custom `LocalizedError` types
- Example error messages:
  - XPC: "Rendering service disconnected. The app will retry automatically."
  - Validation: "Invalid parameter: Intensity must be between 0 and 100. Current value: 150."
  - File not found: "Could not load 'video.mp4': File not found. Choose a different file or check file permissions."
- Use `LocalizedError` protocol for i18n-ready messages (implement `errorDescription`, `failureReason`, `recoverySuggestion`)
- Log errors using `os_log` with appropriate subsystem/category

## Related
- Parent: EPIC-007
- Dependencies: TICKET-029 (error handling implementation)

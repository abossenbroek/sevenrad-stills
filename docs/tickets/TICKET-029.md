# [TICKET-029] Error Handling and Retry Logic

## Context
Backend communication can fail for many reasons. The app must handle errors gracefully and maintain usability even when rendering fails.

## User Story
As a user,
I want clear feedback when rendering fails,
So that I understand what went wrong and can continue working.

## Acceptance Criteria
- [ ] XPC connection errors caught and displayed to user
- [ ] Pipeline validation errors shown with specific parameter issues
- [ ] Rendering timeout errors handled gracefully
- [ ] Backend crash/disconnect triggers automatic retry with exponential backoff
- [ ] User sees last successful preview when new render fails
- [ ] Error messages are user-friendly (no stack traces)

## Technical Notes
- Catch specific error types: XPCError, ValidationError, TimeoutError
- Retry strategy: 1s, 2s, 4s delays before giving up
- Log detailed errors for debugging while showing simple messages to users
- Consider adding "Retry" button in error state
- Preserve last valid preview image as fallback

## Related
- Parent: EPIC-005
- Depends on: TICKET-027

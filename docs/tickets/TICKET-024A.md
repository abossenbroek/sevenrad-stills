# [TICKET-024A] XPC Async Callback Handler

## Context
The XPC client needs to handle asynchronous responses from the Python backend. Callback handlers must match responses to requests using request IDs and provide clean async/await interfaces.

## User Story
As a developer,
I want to register callbacks for XPC responses,
So that I can handle asynchronous render notifications and progress updates.

## Acceptance Criteria
- [ ] XPCCallbackHandler class created with registerCallback(requestID:handler:) method
- [ ] Generates unique request IDs (UUID) for each XPC call
- [ ] Tracks in-flight requests with request ID → completion handler mapping
- [ ] Provides async/await wrapper around callback-based XPC responses
- [ ] Implements cancel(requestID:) method to cancel in-flight requests
- [ ] Implements timeout mechanism (5 seconds default) for stalled requests
- [ ] Cleans up completion handlers after response received or timeout
- [ ] Thread-safe access to callback registry

## Technical Notes
- Use Dictionary<String, CompletionHandler> for callback storage
- Wrap callbacks in Task/Continuation for async/await compatibility
- Use DispatchQueue or actor for thread safety
- Timeout should throw RequestTimeoutError

## Related
- Parent: EPIC-005
- Blocks: TICKET-027
- Consumed by: TICKET-024

# [TICKET-024] XPC Connection & Communication

## Context
The Swift app needs to communicate with the Python backend via XPC for rendering operations. This ticket creates the foundation for all backend communication with request tracking.

## User Story
As a developer,
I want a reliable XPC connection to the Python backend,
So that the app can request preview and export renders.

## Acceptance Criteria
- [ ] XPCClient class created with connect() and disconnect() methods
- [ ] Connection established to Python XPC service on app launch
- [ ] Connection state tracked (connected, disconnected, error)
- [ ] Request ID generation (UUID) for each XPC call
- [ ] Tracks in-flight requests for cancellation and timeout
- [ ] Callback registration delegates to XPCCallbackHandler (TICKET-024A)
- [ ] Connection errors logged with descriptive messages
- [ ] Automatic reconnection attempted after connection loss
- [ ] XPCClient accessible as singleton or environment object

## Technical Notes
- Reference docs/api/backend-contract.md for service name and protocol
- Use NSXPCConnection for macOS XPC communication
- Consider using Combine for connection state updates
- Integrate XPCCallbackHandler from TICKET-024A for async responses
- Log connection events for debugging

## Related
- Parent: EPIC-005
- Depends on: TICKET-024A
- Blocks: TICKET-025, TICKET-027

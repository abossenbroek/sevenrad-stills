# [TICKET-027] Preview Request/Response Handling

## Context
Orchestrates the full preview rendering workflow: validate pipeline, serialize to JSON, send XPC request, wait for completion, read image from shared memory.

## User Story
As a developer,
I want to request preview renders from the backend,
So that users can see their effect changes in real-time.

## Acceptance Criteria
- [ ] renderPreview(pipeline: Pipeline) async method created
- [ ] Validates pipeline using validate_pipeline XPC call before rendering
- [ ] Serializes pipeline to JSON using Pipeline.toJSON()
- [ ] Calculates preview resolution using TICKET-030 logic
- [ ] Creates shared memory buffer before XPC call
- [ ] Calls backend render_preview() via XPCClient with resolution params
- [ ] Passes pipeline JSON, resolution, and shared memory buffer ID
- [ ] Uses async/await from XPCCallbackHandler (TICKET-024A) for response
- [ ] Reads width/height from buffer header (offsets 0 and 4)
- [ ] Reads CGImage from shared memory using SharedMemory class
- [ ] Cleans up shared memory buffer after reading (shm_unlink)
- [ ] Returns CGImage or throws descriptive error
- [ ] Times out after 5 seconds if no response

## Technical Notes
- Use async/await for clean asynchronous code
- Generate unique buffer IDs per request: "preview_buffer_{sessionID}_{UUID}"
- Buffer size = 8 + (width × height × 4) bytes
- Validation errors should prevent render attempt
- Reference docs/api/backend-contract.md sections 3.1 and 5.1

## Related
- Parent: EPIC-005
- Depends on: TICKET-024, TICKET-024A, TICKET-025, TICKET-026, TICKET-057
- Blocks: TICKET-028

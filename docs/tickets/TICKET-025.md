# [TICKET-025] Shared Memory Integration

## Context
Preview images must transfer from Python backend to Swift app without expensive memory copies. Shared memory provides zero-copy image data access with explicit buffer lifecycle management.

## User Story
As a developer,
I want to read rendered images from shared memory,
So that preview updates are fast and efficient.

## Acceptance Criteria
- [ ] SharedMemory class created with static create(name:size:) method
- [ ] Frontend creates buffer with shm_open(O_CREAT | O_RDWR)
- [ ] Reads width (UInt32) from buffer offset 0
- [ ] Reads height (UInt32) from buffer offset 4
- [ ] Reads RGBA pixel data starting at offset 8
- [ ] Converts raw bytes to CGImage with dimensions from buffer header
- [ ] Cleans up buffer with shm_unlink() after image read
- [ ] Memory mapping cleaned up properly on dealloc
- [ ] Returns nil if buffer doesn't exist or is invalid

## Technical Notes
- Frontend owns buffer lifecycle: create before request, cleanup after read
- Use shm_open() and mmap() for shared memory access
- Reference docs/api/backend-contract.md section 8 for complete protocol
- Buffer format: [width:UInt32][height:UInt32][RGBA pixels...]
- Buffer size = 8 + (width × height × 4) bytes

## Related
- Parent: EPIC-005
- Depends on: TICKET-024
- Blocks: TICKET-027

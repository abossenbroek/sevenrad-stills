# [EPIC-005] Preview Rendering & Backend Integration

## Context
The drum machine app requires real-time visual feedback when users adjust effect parameters. This epic implements XPC communication with the Python backend to render preview frames with ~1 second latency, using shared memory for efficient image transfer.

## User Story
As a user,
I want to see my parameter changes reflected in the preview window within 1 second,
So that I can immediately understand how effects transform my source video.

## Scope
- XPC service connection management with Python backend
- Python XPC service with Taichi/MLX GPU-accelerated effects
- Shared memory buffer implementation for zero-copy image transfer
- Pipeline model to JSON serialization matching backend schema
- Request/response handling for preview rendering
- Adaptive preview resolution based on viewport size
- Real-time preview updates with debouncing
- Error handling and retry logic

Out of scope:
- Full video rendering (covered in EPIC-006)
- Preview window UI (covered in TICKET-010)
- Pipeline model definition (covered in TICKET-012 through TICKET-018)

## Success Criteria
- [ ] Preview updates within 1 second of parameter change
- [ ] Loading indicator displays during render operations
- [ ] Images transfer via shared memory without copy overhead
- [ ] Connection errors handled gracefully with automatic retry
- [ ] Clear error messages displayed when rendering fails
- [ ] App reconnects to backend if XPC connection drops

## Related Tickets
- TICKET-024: XPC Connection & Communication
- TICKET-024A: XPC Async Callback Handler
- TICKET-025: Shared Memory Integration
- TICKET-026: Pipeline JSON Serialization
- TICKET-027: Preview Request/Response Handling
- TICKET-028: Real-Time Preview Updates
- TICKET-029: Error Handling and Retry Logic
- TICKET-057: Adaptive Preview Resolution
- TICKET-058: Backend XPC Service (Python + Taichi)

## Dependencies
- TICKET-010: Preview window UI must exist
- TICKET-012 through TICKET-018: Pipeline model must be defined
- docs/api/backend-contract.md: API specification

## Blocks
- EPIC-006: Export functionality requires working backend integration

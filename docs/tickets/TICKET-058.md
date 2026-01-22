# [TICKET-058] Backend XPC Service (Python + Taichi)

## Context
Python backend service receives XPC requests from Swift frontend, executes Taichi/MLX GPU-accelerated effects, and writes results to shared memory buffers.

## User Story
As a backend service,
I want to execute GPU-accelerated effects using Taichi,
So that preview renders complete within 1 second.

## Acceptance Criteria
- [ ] Python XPC service listens for connections from Swift app
- [ ] Implements all 7 core methods: create_session, list_available_effects, get_effect_parameters, validate_pipeline, render_preview, render_sequence, export_yaml
- [ ] All 17 effects ported to Taichi kernels (see Appendix A in backend-contract.md)
- [ ] render_preview writes RGBA pixels to shared memory buffer
- [ ] Shared memory buffer format: [width:UInt32][height:UInt32][RGBA pixels...]
- [ ] render_sequence supports version_increment parameter for incremental filenames
- [ ] Incremental filename format: {pipeline_name}_{frame_stem}_v{version:03d}.{ext}
- [ ] Progress callbacks sent for long-running operations (>500ms)

## Technical Notes
- Use Taichi for GPU acceleration: https://github.com/taichi-dev/taichi
- Install: `pip install taichi`
- Taichi kernels compile to Metal on macOS for Apple Silicon GPU
- Reference backend-contract.md for complete XPC protocol spec
- Use mmap for shared memory writing (shm_open from Swift side)
- Each effect should be a separate Taichi kernel

## Related
- Parent: EPIC-005
- Blocks: EPIC-006
- See: docs/api/backend-contract.md

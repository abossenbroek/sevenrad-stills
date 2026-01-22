# [EPIC-006] Export & Rendering

## Context
Users need the ability to persist their creative pipeline work and generate final output. This epic implements YAML export for pipeline configurations and batch rendering of full frame sequences, enabling users to save their work and produce finished results from their drum machine sessions.

## User Story
As a user,
I want to export my pipeline as YAML and render complete frame sequences,
So that I can save my creative configurations and generate final output for my projects.

## Scope
- YAML export of current pipeline configuration with validation
- Batch rendering of full frame sequences with applied pipeline
- Incremental filename generation with version tracking (non-destructive)
- Progress tracking with cancellation support
- Export dialog UI with multiple output options
- File system integration (save panels, Finder integration)

Out of scope:
- Video encoding (frame sequences only)
- Preset management or library
- Cloud export or sharing features
- Undo/redo for export operations
- Custom output formats beyond YAML

## Success Criteria
- [ ] Pipeline can be exported as valid YAML matching backend format
- [ ] Frame sequences can be batch-rendered with pipeline applied
- [ ] Progress is visible with frame count and cancellation option
- [ ] Users can locate output files via Finder integration
- [ ] Export dialog provides clear options for YAML vs rendering
- [ ] All operations handle errors gracefully with user notifications

## Related Tickets
- TICKET-030: YAML Pipeline Export
- TICKET-031: Batch Rendering Queue
- TICKET-032: Progress Tracking UI
- TICKET-033: Export Dialog UI
- TICKET-061: Finder Integration for Exports

## Dependencies
- TICKET-026: Pipeline JSON serialization for export
- TICKET-027: Backend communication layer for XPC calls
- TICKET-005: Session frame management for render input

## Priority
P2 (Polish Features)

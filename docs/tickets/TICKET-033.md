# [TICKET-033] Export Dialog UI

## Context
Users need a clear interface to choose between exporting YAML configurations or rendering full frame sequences with appropriate format and quality settings. This ticket implements a modal export dialog that acts as a pure UI router to export workflows.

## User Story
As a user,
I want a clear export dialog with multiple output options,
So that I can easily choose between YAML export and frame rendering with appropriate settings.

## Acceptance Criteria
- [ ] Modal sheet presented from main window toolbar/menu
- [ ] Radio buttons for "Export YAML" vs "Render Sequence"
- [ ] Add output format selection (PNG/JPG) for render operations
- [ ] Add quality slider (0-100%) when JPG format selected
- [ ] Add output directory selection via NSSavePanel for renders
- [ ] Show filename pattern preview before render: `{pipeline_name}_{frame_stem}_v{version:03d}.{ext}`
- [ ] Display disk space validation (show available space vs estimated usage)
- [ ] Disable options when prerequisites are missing (e.g., no frames loaded)
- [ ] Export and Cancel buttons with appropriate keyboard shortcuts
- [ ] Trigger TICKET-030 flow when YAML selected (contains zero rendering logic)
- [ ] Trigger TICKET-031/032 flow when Render selected (contains zero rendering logic)

## Technical Notes
- Pure UI router: triggers TICKET-030/031, contains ZERO rendering logic
- Use NSStackView for clean layout
- Disk space check: call FileManager.attributesOfFileSystem(forPath:)
- Estimated usage: frame_count × 5MB (conservative estimate)
- Quality slider only visible when JPG format selected
- Sheet should be presented modally to prevent state changes during export

## Related
Parent Epic: EPIC-006
Dependencies: None
Triggers: TICKET-030, TICKET-031

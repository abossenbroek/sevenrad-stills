# [EPIC-002] Core UI Shell

## Context
The Core UI Shell establishes the foundational user interface for the macOS drum machine app, implementing a 4-panel layout inspired by Elektron RYTM aesthetics. This epic delivers the main window structure, effect pad grid, layer management, parameter controls, and preview system that all future functionality will build upon.

## User Story
As a video artist,
I want to see a professional drum-machine-style interface with clear visual zones,
So that I can intuitively navigate between effect selection, layer composition, parameter adjustment, and preview.

## Scope
- Main window with 4-panel responsive layout
- 16-pad effect grid with RYTM-inspired styling
- Layer stack panel for pipeline visualization
- Parameter panel with contextual controls
- Preview window for current frame display
- Elektron RYTM design system implementation

Out of scope:
- Actual effect processing logic (covered in EPIC-003)
- Video playback controls (covered in EPIC-005)
- File import/export (covered in EPIC-001)
- Preset management
- Keyboard shortcuts

## Success Criteria
- [ ] Main window renders with all 4 panels properly constrained
- [ ] 16-pad grid displays with correct RYTM colors and states
- [ ] Layer stack shows placeholder layers with reorder capability
- [ ] Parameter panel updates based on selected effect/layer
- [ ] Preview window displays test images without distortion
- [ ] All components follow RYTM design system specifications
- [ ] UI remains responsive during state changes
- [ ] Interface works on macOS 13.0+

## Related Tickets
- TICKET-006: Main Window Layout Structure
- TICKET-007: 16-Pad Effect Grid Component
- TICKET-008: Layer Stack Panel Component
- TICKET-009: Parameter Panel Component
- TICKET-010: Preview Panel Content
- TICKET-011: Elektron RYTM Design System Implementation
- TICKET-011b: Application Shell & Menu Bar
- TICKET-011c: Application State Management & View Model
- TICKET-011d: Core Accessibility Foundation

## Metadata
**Priority**: P0 (MVP Critical)
**Dependencies**: None
**Blocks**: EPIC-003, EPIC-005

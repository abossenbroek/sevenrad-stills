# [TICKET-046] Scene Slot UI Component

## Context
RYTM-style scene slot UI displays 6 visible scene slots with bank switching to access all 12 scenes (A-L). Each slot shows scene letter, name, and effect count.

## User Story
As a technical artist,
I want to see available scene slots at a glance,
So that I can quickly identify and access saved effect configurations.

## Acceptance Criteria
- [ ] Scene slot row displays 6 scene slots (A-F or G-L depending on bank)
- [ ] Bank toggle button switches between A-F and G-L scene banks
- [ ] Each slot shows scene letter, editable name, and populated indicator
- [ ] Empty slots display "Empty" placeholder with dashed border
- [ ] Filled slots display effect count badge (e.g., "4 FX")
- [ ] Active scene highlighted with orange border and glow
- [ ] Left-click slot to load scene
- [ ] Right-click slot opens context menu: Save, Rename, Clear, Duplicate
- [ ] Hover state shows slot can be interacted with

## Technical Notes
- Scene slot component matches mockup 04-scene-presets.html styling
- Context menu positioned near clicked slot
- Active scene state tracked in application state
- Inline name editing triggers on double-click or "Rename" action

## Related
- EPIC-009: Scene Preset System
- TICKET-011: State Management System (dependency)

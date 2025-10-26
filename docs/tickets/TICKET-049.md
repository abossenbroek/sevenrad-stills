# [TICKET-049] Scene Management Operations

## Context
Scene management provides operations for renaming, clearing, duplicating, importing, and exporting scenes. These operations enable users to organize and share scene presets.

## User Story
As a technical artist,
I want to manage my saved scenes,
So that I can organize, share, and reuse effect configurations.

## Acceptance Criteria
- [ ] Rename scene: Inline text editing in scene slot (double-click or context menu)
- [ ] Clear scene: Delete scene YAML file and reset slot to empty state
- [ ] Duplicate scene: Copy scene YAML to another slot (choose target via dialog)
- [ ] Import scene: Open file picker to load external .yaml scene file
- [ ] Export scene: Save scene YAML to user-selected location
- [ ] All operations show confirmation notifications
- [ ] Clear operation requires confirmation dialog to prevent accidental deletion

## Technical Notes
- Rename updates scene name in YAML metadata without changing effect configuration
- Duplicate operation allows selecting target slot from available empty slots
- Import validates YAML structure before loading
- Export defaults filename to `{scene_name}.yaml` with sanitized name
- Context menu shows relevant operations (e.g., no "Clear" for empty slots)

## Related
- EPIC-009: Scene Preset System
- TICKET-046: Scene Slot UI Component (dependency)
- TICKET-047: Scene Save Operation (dependency)
- TICKET-048: Scene Load Operation (dependency)

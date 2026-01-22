# [TICKET-048] Scene Load Operation

## Context
Scene load reads YAML file from session directory, deserializes pipeline state, and replaces current pipeline with saved configuration. Load operation integrates with undo system for reversibility.

## User Story
As a technical artist,
I want to load a saved scene,
So that I can restore a previously saved effect configuration.

## Acceptance Criteria
- [ ] Load operation reads scene YAML from `{session_dir}/scenes/scene_{letter}.yaml`
- [ ] Deserialize YAML and replace current pipeline state with saved configuration
- [ ] Show confirmation notification: "Loaded scene A: {scene_name}"
- [ ] If scene file missing or corrupted, show error dialog with details
- [ ] Scene load operation creates undo history entry for reverting
- [ ] Active scene indicator updates to loaded scene slot
- [ ] Load accessible via slot click and context menu "Load" option

## Technical Notes
- YAML parsing validates effect names exist in registered effect registry
- If effect not found, skip that effect and log warning
- Load operation clears current pipeline before applying scene state
- Undo entry captures previous pipeline state before load

## Related
- EPIC-009: Scene Preset System
- TICKET-026: YAML Serialization (dependency)
- TICKET-019: Undo/Redo System (dependency)
- TICKET-046: Scene Slot UI Component (dependency)

# [TICKET-047] Scene Save Operation

## Context
Scene save captures the current pipeline state (all effects, parameters, repeat counts) and serializes to YAML file in session directory. Users save scenes to preserve effect configurations for later recall.

## User Story
As a technical artist,
I want to save my current effect pipeline to a scene slot,
So that I can preserve and recall this configuration later.

## Acceptance Criteria
- [ ] Save operation captures complete pipeline state (effects, parameters, repeat counts)
- [ ] Pipeline state serialized to YAML file at `{session_dir}/scenes/scene_{letter}.yaml`
- [ ] Show confirmation notification: "Scene A saved: {scene_name}"
- [ ] If slot already populated, show overwrite confirmation dialog
- [ ] Auto-name new scenes as "Scene A", "Scene B", etc. (user can rename after)
- [ ] Scene metadata includes creation timestamp and last modified timestamp
- [ ] Save operation accessible via context menu and "Save Current as New Scene" button

## Technical Notes
- YAML structure includes effect chain, parameters, repeat counts, metadata
- Session directory automatically creates `scenes/` subdirectory if missing
- Overwrite confirmation skippable via "Don't ask again" checkbox (session-scoped)

## Related
- EPIC-009: Scene Preset System
- TICKET-026: YAML Serialization (dependency)
- TICKET-046: Scene Slot UI Component (dependency)

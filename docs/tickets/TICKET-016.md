# [TICKET-016] Remove Layer Action

## Context
Users need ability to remove unwanted effects from pipeline. Simple deletion mechanism with visual feedback.

## User Story
As a user,
I want to click a trash icon on a layer,
So that the effect is removed from my pipeline.

## Acceptance Criteria
- [ ] Trash icon button visible on each layer
- [ ] Click trash removes layer from pipeline
- [ ] Layer stack updates immediately
- [ ] If selected layer removed, clear selection
- [ ] Removal animates smoothly
- [ ] Cannot remove if only one layer remains (optional safety)

## Technical Notes
- Button with SF Symbol `trash` icon
- Call `pipeline.removeEffect(id: UUID)`
- Check if `selectedLayerID == removedID`, set to nil if true
- Consider confirmation dialog for destructive action (optional)

## Related
Parent: EPIC-003
Depends on: TICKET-008 (layer stack UI)

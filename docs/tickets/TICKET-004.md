# [TICKET-004] Timeline Scrubber Component

## Context
The timeline scrubber is the core UI element for segment selection. Users need an intuitive, visual way to select start and end points within the video timeline. This component should feel responsive and provide clear visual feedback about the selected segment.

## User Story
As a user,
I want to drag start and end markers on a timeline,
So that I can precisely select which portion of the video to extract frames from.

## Acceptance Criteria
- [ ] Displays horizontal timeline scaled to video duration
- [ ] Shows draggable start marker (left boundary)
- [ ] Shows draggable end marker (right boundary)
- [ ] Displays selected segment duration in real-time
- [ ] Enforces constraint: end time must be greater than start time
- [ ] Markers snap to timeline during drag operations
- [ ] Shows time labels at marker positions (MM:SS format)
- [ ] Visual distinction between selected and unselected portions of timeline
- [ ] Responsive layout that adapts to window width

## Technical Notes
- Implement as custom SwiftUI View using GeometryReader for responsive sizing
- Use DragGesture for marker interactions
- State management:
  - startTime: Double (in seconds)
  - endTime: Double (in seconds)
  - duration: Double (from TICKET-003 metadata)
- Minimum segment duration: 1 second (prevent accidental zero-length selections)
- Default state: start=0, end=full duration
- Stretch goal: Show thumbnail preview at marker positions using backend thumbnail generation
- Consider color coding: selected segment in accent color, unselected in gray
- Ensure touch target size for markers is at least 44x44 points (Apple HIG)

## Story Points
5

## Dependencies
Requires: TICKET-003 (needs video duration)

## Related
Parent: EPIC-001
Enables: TICKET-005 (segment selection feeds into extraction)

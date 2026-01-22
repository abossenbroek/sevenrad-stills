# [TICKET-038] Empty States & Onboarding

## Context
Provide helpful guidance when panels are empty and introduce new users to the app's workflow through a first-run tutorial.

## User Story
As a user,
I want clear guidance when starting with an empty project,
So that I understand how to begin and what each panel does.

## Acceptance Criteria
- [ ] Pattern grid empty state: "No frame loaded. Load a video to begin."
- [ ] Layer list empty state: "No effects added. Click a pad above to add effect."
- [ ] Parameter panel empty state: "Select a layer to edit parameters."
- [ ] History panel empty state: "No actions yet. Changes will appear here."
- [ ] First-run tutorial modal highlights: pad grid, layer list, parameter controls, history panel
- [ ] Tutorial includes "Don't show again" checkbox stored in UserDefaults
- [ ] Empty states use consistent styling: SF Symbols icon, muted secondary text, centered layout

## Technical Notes
- Use SwiftUI `ContentUnavailableView` (macOS 14+) for empty states
- Store tutorial preference: `UserDefaults.standard.bool(forKey: "hasSeenTutorial")`
- Tutorial modal uses `.sheet()` with overlay highlights on each panel
- Empty state icons: `video.slash`, `square.stack.3d.up.slash`, `slider.horizontal.3`, `clock.arrow.circlepath`
- Fallback for macOS 13: Custom `VStack` with `Image(systemName:)` and `Text`

## Related
- Parent: EPIC-007
- Related: TICKET-036 (tutorial and tooltips complement each other)

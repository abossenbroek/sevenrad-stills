# [TICKET-011c] Application State Management & View Model

## Context
Establish centralized state management and business logic layer, providing the data model and operations that all UI components bind to.

## User Story
As a developer,
I want centralized state with clear business logic separation,
So that UI components remain simple and testable.

## Acceptance Criteria
- [ ] AppState class (ObservableObject) with @Published properties: session, currentFrame, effectPalette, effectLayers, selectedLayerIndex
- [ ] MainViewModel class with business logic: addLayer(), removeLayer(), reorderLayer(), selectLayer()
- [ ] Debounced parameter change handler (single 100ms debounce timer)
- [ ] Document binding pattern: @EnvironmentObject for AppState in all views
- [ ] State persistence hooks (prepare for save/load, no implementation yet)
- [ ] Clear separation: AppState holds data, MainViewModel holds operations
- [ ] Unit tests for MainViewModel operations (add/remove/reorder layers)

## Technical Notes
- Create `Sources/State/AppState.swift`
- Create `Sources/ViewModels/MainViewModel.swift`
- Use Combine for debouncing: `publisher.debounce(for: .milliseconds(100), scheduler: RunLoop.main)`
- MainViewModel holds weak reference to AppState
- Inject AppState via `.environmentObject()` in App struct

## Related
- Parent: EPIC-002
- Blocks: TICKET-011b, TICKET-006, TICKET-007, TICKET-008, TICKET-009, TICKET-010

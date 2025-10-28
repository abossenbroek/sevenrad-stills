# [TICKET-034] Smooth Animations and Transitions

## Context
Add polished animations to all UI interactions. Animations should feel responsive and natural, enhancing the user experience without causing delays or distractions.

## User Story
As a user,
I want to see smooth animations when interacting with the UI,
So that the app feels polished and responsive.

## Acceptance Criteria
- [ ] Pad press triggers 150ms spring animation (scale 0.95 -> 1.0, subtle glow)
- [ ] Layer reorder shows 300ms ease-in-out animation with smooth position transitions
- [ ] Panel expand/collapse animates with 200ms ease-out timing
- [ ] Preview updates instantly without fade transitions (performance critical)
- [ ] Loading spinners have smooth fade-in (100ms) and fade-out (200ms) transitions
- [ ] All animations maintain 60fps on 2018 MacBook Pro (Intel i5, 8GB RAM)
- [ ] Frame drops detected trigger fallback to simpler animations

## Technical Notes
- Use SwiftUI's `withAnimation()` for declarative animations
- Specify spring damping ratio of 0.7 for pad press animations
- Consider using `.matchedGeometryEffect()` for layer reordering
- Disable animations using `@Environment(\.disableAnimations)` during bulk operations
- Profile with Instruments: Core Animation FPS template
- Fallback strategy: If FPS < 50 for 3+ frames, disable spring animations and use linear
- Performance baseline: Test with 10-layer stack and rapid pad triggering

## Related
- Parent: EPIC-007
- Dependencies: All UI implementation tickets

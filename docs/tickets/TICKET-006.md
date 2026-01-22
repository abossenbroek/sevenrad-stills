# [TICKET-006] Main Window Layout Structure

## Context
Establish the foundational window layout structure with a 4-panel container system. This ticket is layout-only: it creates empty container shells that accept content views as children, without implementing the actual content.

## User Story
As a user,
I want to see a well-organized 4-panel interface,
So that each functional area has dedicated screen space.

## Acceptance Criteria
- [ ] Main window opens with minimum size 1200x800, default 1440x900
- [ ] 4-panel layout implemented using SwiftUI GeometryReader and HStack/VStack
- [ ] Left panel container: 320px fixed width, accepts EffectGridContentView
- [ ] Center-left panel container: 280px fixed width, accepts LayerStackContentView
- [ ] Center-right panel container: flexible width, min 300px, accepts ParameterContentView
- [ ] Right panel container: 400px fixed width, accepts PreviewContentView
- [ ] Panels separated by 1px dividers (#2a2a2a)
- [ ] Window background uses RYTM dark gray (#1a1a1a)
- [ ] Layout responds gracefully to window resize
- [ ] Container views pass size constraints to children

## Technical Notes
- Use `NSWindow` with `titlebarAppearsTransparent` for modern macOS look
- Implement custom window chrome matching RYTM aesthetic
- Create `MainWindowView` with container views: `PanelContainer`
- Containers accept content views as `@ViewBuilder` children
- Content views created by TICKET-007, 008, 009, 010

## Related
- Parent: EPIC-002
- Depends: TICKET-011b (requires WindowGroup), TICKET-011c (requires AppState)
- Blocks: TICKET-007, TICKET-008, TICKET-009, TICKET-010

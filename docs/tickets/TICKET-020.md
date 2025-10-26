# [TICKET-020] History State Manager

## Context
The history system requires a centralized manager to maintain the command stack, handle undo/redo operations, and manage state restoration. This is the core business logic for the history feature.

## User Story
As a developer,
I want a HistoryManager class to track and manage all executed commands,
So that the application can support undo, redo, and time-travel functionality.

## Acceptance Criteria
- [ ] HistoryManager class with command stack implementation
- [ ] execute_command() method adds commands and executes them
- [ ] execute_command() shows "History branched" notification when called after time-travel
- [ ] undo() method reverts last command and updates current index
- [ ] redo() method re-applies next command if available
- [ ] restore_to_index() method replays commands to specific state
- [ ] Failed command replay logs warning, skips command, continues restoration
- [ ] Branching logic: executing after time-travel discards future commands
- [ ] @Published properties for observable state changes (commands, currentIndex, canUndo, canRedo)
- [ ] @Published pipeline state triggers automatic preview re-render
- [ ] Maximum history depth enforced with FIFO deletion when limit reached
- [ ] Unit tests cover undo, redo, branching, error handling scenarios

## Technical Notes
- Use Swift @Published properties for observable state:
  - @Published var commands: [Command]
  - @Published var currentIndex: Int
  - @Published var canUndo: Bool { currentIndex > 0 }
  - @Published var canRedo: Bool { currentIndex < commands.count }
- Store commands as lightweight JSON-serializable configurations
- Current state index tracks position in command stack
- Maximum history depth can be configurable (default: unlimited)

## Related
Parent: EPIC-004
Dependencies: TICKET-019 (command pattern)
Blocks: TICKET-021, TICKET-022, TICKET-023

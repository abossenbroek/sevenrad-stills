# [TICKET-061] Finder Integration for Exports

## Context
After export operations complete, users need quick access to their output files. This ticket implements Finder integration using NSWorkspace to reveal exported files or open containing directories.

## User Story
As a user,
I want to quickly locate my exported files,
So that I can immediately work with the rendered frames or YAML configurations.

## Acceptance Criteria
- [ ] Show "Reveal in Finder" button on render completion
- [ ] Use NSWorkspace.shared.selectFile(_:inFileViewerRootedAtPath:) to highlight file
- [ ] Offer "Reveal in Finder" option on YAML export completion
- [ ] Alternative "Open Output Folder" button for opening containing directory
- [ ] Handle errors gracefully if file no longer exists at expected path

## Technical Notes
- Use NSWorkspace.shared.selectFile(_:inFileViewerRootedAtPath:) for single file selection
- Use NSWorkspace.shared.open(URL) for directory opening
- Verify file exists before attempting reveal operation
- Consider adding to success notifications for both YAML and render completions

## Related
Parent Epic: EPIC-006
Related: TICKET-030, TICKET-032

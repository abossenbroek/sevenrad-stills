# [TICKET-030] YAML Pipeline Export

## Context
Users need to persist their pipeline configurations for future sessions. This ticket implements YAML export with validation support, allowing export of both valid and invalid pipelines with appropriate warnings.

## User Story
As a user,
I want to export my current pipeline as a YAML file,
So that I can save my configuration and reload it in future sessions.

## Acceptance Criteria
- [ ] Call backend validate_pipeline() before export to check pipeline validity
- [ ] Present NSSavePanel for user to choose file location
- [ ] Show filename pattern preview: `{pipeline_name}_{timestamp}.yaml`
- [ ] Write YAML file matching backend's expected pipeline format
- [ ] Add validation errors as comments in exported YAML if pipeline is invalid
- [ ] Allow export of invalid pipelines with warning dialog (for debugging)
- [ ] Show success notification with file path on completion
- [ ] Handle errors (write failures, validation failures) with user alerts

## Technical Notes
- Depends on TICKET-026 for pipeline JSON serialization
- Use existing XPC service client from TICKET-027
- Call validate_pipeline() first, capture any validation errors
- If invalid: show warning dialog "Pipeline has errors. Export anyway for debugging?"
- If exported despite errors, add "# VALIDATION ERRORS:" section at top of YAML
- YAML format must be compatible with backend's pipeline loader

## Related
Parent Epic: EPIC-006
Dependencies: TICKET-026, TICKET-027
Related: TICKET-061

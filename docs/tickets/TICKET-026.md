# [TICKET-026] Pipeline JSON Serialization

## Context
The Python backend expects pipeline configuration as JSON matching a specific schema. The Swift Pipeline model must serialize correctly.

## User Story
As a developer,
I want to convert Pipeline models to JSON,
So that the backend can execute the effect pipeline.

## Acceptance Criteria
- [ ] Pipeline.toJSON() method returns valid JSON string
- [ ] JSON includes source video path
- [ ] Effects array contains id, name, params, and repeat for each effect
- [ ] Parameters serialize correctly (float, int, bool, enum, color)
- [ ] Enum values serialize as strings matching backend expectations
- [ ] Color values serialize as hex strings (#RRGGBB)
- [ ] Output matches schema in docs/api/backend-contract.md

## Technical Notes
- Use Codable for type-safe serialization
- Create custom CodingKeys for any naming differences
- Validate against backend schema examples
- Consider adding unit tests for serialization edge cases

## Related
- Parent: EPIC-005
- Depends on: TICKET-012
- Blocks: TICKET-027

# sevenrad-stills

Photo/video processing toolkit with TouchDesigner integration.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `src/td_linter/` | TouchDesigner project validator - see `src/td_linter/CLAUDE.md` |
| `docs/touchdesigner/` | TD documentation, fixtures, and specifications |

## Key Commands

```bash
# Run td-linter on a .toe file
uv run td-linter lint project.toe

# Run tests
uv run pytest tests/unit/td_linter/ -v

# Run regression tests for TD-verified fixtures
uv run pytest tests/integration/td_linter/test_fixture_regression.py -v -m "slow or integration"
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/td-check-lint <file>` | Validate linter against real TD project |

## Conventions

### Commits
- Use conventional commits: `feat:`, `fix:`, `docs:`, etc.
- Include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`

### Changelog
- Maintain `CHANGELOG.md` following Keep a Changelog format
- Document test fixture attributions under "Test Fixtures" section

### Binary Files
- `.toe` files tracked with git-lfs (see `.gitattributes`)

## Component Documentation

For detailed instructions, see:
- **td-linter**: `src/td_linter/CLAUDE.md`

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

# Run tests (or: make test)
uv run pytest tests/unit/td_linter/ -v

# Run regression tests (or: make test-regression)
uv run pytest tests/integration/td_linter/test_fixture_regression.py -v -m "slow or integration"

# Expand fixtures for CI (requires TouchDesigner)
make expand-fixtures
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests (excludes integration) |
| `make test-regression` | Run TD linter regression tests |
| `make expand-fixtures` | Expand .toe → .toe.dir (requires TD) |
| `make clean-fixtures` | Remove expanded .toe.dir directories |
| `make docs` | Serve documentation locally |

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
- `fixtures-expanded.zip` tracked with git-lfs (pre-expanded fixtures for CI)

## Component Documentation

For detailed instructions, see:
- **td-linter**: `src/td_linter/CLAUDE.md`

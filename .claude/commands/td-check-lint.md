---
description: Validate td-linter against real TouchDesigner projects. Downloads/copies .toe file, asks user if TD opens it, then verifies linter matches. Fixes false positives automatically.
argument-hint: <url-or-filepath>
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, AskUserQuestion, Task
---

# TD Linter Validation Command

Validate that td-linter produces correct results for a real TouchDesigner project.

**Ground truth**: If TD opens a file successfully, any linter errors are false positives in our linter.

## Input

File or URL: `$ARGUMENTS`

## Workflow

### Step 1: Acquire the .toe file

If `$ARGUMENTS` is a URL:
- Download with `curl -L -o /tmp/test_project.toe "$ARGUMENTS"`

If `$ARGUMENTS` is a local path:
- Copy to fixtures: `cp "$ARGUMENTS" docs/touchdesigner/fixtures/projects/`

**Important**: Handle filenames with special characters (spaces, #, etc.) by quoting paths.

### Step 2: Ask user to verify in TouchDesigner

Use the `AskUserQuestion` tool to ask:

**Question**: "Does this .toe file open successfully in TouchDesigner?"
**Options**:
- "Yes, opens without errors"
- "No, TD reports errors"
- "TD crashes or won't open"

### Step 3: Run td-linter directly on the .toe file

```bash
# Lint the .toe file directly (auto-discovers TD, expands, lints, cleans up)
uv run td-linter lint "<path-to-toe-file>"

# If debugging is needed, keep the expanded files:
uv run td-linter lint "<path-to-toe-file>" --keep-files-after-expand
```

The linter automatically:
1. Finds TouchDesigner installation (env var, common locations, PATH)
2. Expands .toe to .toe.dir
3. Runs validation
4. Cleans up temporary files

Capture the output and exit code.

### Step 4: Compare results

| TD Opens? | Linter Passes? | Action |
|-----------|----------------|--------|
| Yes | Yes | Consistent - keep as regression fixture |
| Yes | No | **FALSE POSITIVE** - fix the linter |
| No | No | Consistent - linter correctly caught errors |
| No | Yes | Linter missed real errors - investigate |

### Step 5: If false positive detected

1. **Re-run with debug flag**:
   ```bash
   uv run td-linter lint "<path>" --keep-files-after-expand
   ```

2. **Diagnose**: Read the linter error messages

3. **Categorize** the error type:
   - `G0xx` (GLSL): Check `embedded/glsl_validator.py` preamble, consult https://docs.derivative.ca/Write_a_GLSL_TOP
   - `S0xx` (Syntax): Check `grammars/n_file.lark` or `grammars/parm_file.lark` for missing directives
   - `P0xx` (Python): Check `embedded/python_validator.py` and `embedded/constants.py` for missing builtins
   - Language detection: Check `embedded/language_detector.py` for binary header handling

4. **Read** the actual file that triggered the error

5. **Find** the unrecognized pattern

6. **Fix** the grammar/preamble/validator

7. **Re-run** linter to verify fix

8. **Commit** with message: `fix(td-linter): Handle <pattern> in <component>`

### Step 6: Keep fixture for regression

If the file is from a URL or user wants to keep it:

1. Move to `docs/touchdesigner/fixtures/projects/`
2. Track with git-lfs (already configured for .toe files in .gitattributes)
3. Add attribution to commit message:
   ```
   Test fixture:
   - Add <filename>.toe
     Source: <url>
     Author: <author name>
   ```
4. Update CHANGELOG.md with attribution under "Test Fixtures" section
5. Add to regression test list in `tests/integration/td_linter/test_fixture_regression.py`:
   ```python
   TD_VERIFIED_FIXTURES = [
       # ... existing fixtures ...
       ("<filename>.toe", "<author> - <source description>"),
   ]
   ```

## Success Criteria

- Linter result matches TD behavior
- No false positives on valid TD projects
- Regression fixture preserved with proper attribution

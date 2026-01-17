---
description: Validate td-linter against real TouchDesigner projects. Downloads/copies .toe file, runs linter first, only asks user to verify in TD if linter fails. Fixes false positives automatically.
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
- Copy to fixtures: `cp /tmp/test_project.toe docs/touchdesigner/fixtures/projects/`

If `$ARGUMENTS` is a local path:
- Copy to fixtures: `cp "$ARGUMENTS" docs/touchdesigner/fixtures/projects/`

**Important**: Handle filenames with special characters (spaces, #, etc.) by quoting paths.

### Step 2: Run td-linter FIRST

```bash
# Lint the .toe file directly (auto-discovers TD, expands, lints, cleans up)
uv run td-linter lint "<path-to-toe-file>"
```

The linter automatically:
1. Finds TouchDesigner installation (env var, common locations, PATH)
2. Expands .toe to .toe.dir
3. Runs validation
4. Cleans up temporary files

Capture the output and exit code.

### Step 3: Branch based on linter result

**If linter PASSES (exit code 0)**:
- Skip TD verification (assume it would open fine)
- Go directly to Step 6 (add as regression fixture)
- Commit with message: `test(td-linter): Add <filename>.toe regression fixture`

**If linter FAILS (exit code non-zero)**:
- Continue to Step 4 to verify with TouchDesigner

### Step 4: Ask user to verify in TouchDesigner (only if linter failed)

Open the file in TD:
```bash
open "<path-to-toe-file>"
```

Use the `AskUserQuestion` tool to ask:

**Question**: "The linter reported errors. Does this .toe file open successfully in TouchDesigner?"
**Options**:
- "Yes, opens without errors" → FALSE POSITIVE in linter, go to Step 5
- "No, TD reports errors" → Linter is correct, delete fixture and stop
- "TD crashes or won't open" → Linter is correct, delete fixture and stop

### Step 5: Fix false positive (only if TD opens but linter fails)

1. **Re-run with debug flag**:
   ```bash
   uv run td-linter lint "<path>" --keep-files-after-expand
   ```

2. **Diagnose**: Read the linter error messages

3. **Categorize** the error type:
   - `R0xx` (Reference): Check `graph/builder.py` for operator resolution
   - `C0xx` (Connection): Check `rules/builtin/connection.py`
   - `G0xx` (GLSL): Check `embedded/glsl_validator.py` preamble, consult https://docs.derivative.ca/Write_a_GLSL_TOP
   - `S0xx` (Syntax): Check `grammars/n_file.lark` or `grammars/parm_file.lark` for missing directives
   - `P0xx` (Python): Check `embedded/python_validator.py` and `embedded/constants.py` for missing builtins
   - Language detection: Check `embedded/language_detector.py` for binary header handling

4. **Read** the actual file that triggered the error (in the .toe.dir)

5. **Find** the unrecognized pattern

6. **Fix** the grammar/preamble/validator

7. **Re-run** linter to verify fix (should now pass)

8. **Clean up** expanded files: `rm -rf "<path>.dir"`

9. **Commit** with message: `fix(td-linter): Handle <pattern> in <component>`

### Step 6: Keep fixture for regression

1. File should already be in `docs/touchdesigner/fixtures/projects/`
2. Track with git-lfs (already configured for .toe files in .gitattributes)
3. Add to regression test list in `tests/integration/td_linter/test_fixture_regression.py`:
   ```python
   TD_VERIFIED_FIXTURES = [
       # ... existing fixtures ...
       ("<filename>.toe", "<author> - <source description>"),
   ]
   ```
4. Update CHANGELOG.md with attribution under "Test Fixtures" section
5. Commit with attribution:
   ```
   Test fixture:
   - Add <filename>.toe
     Source: <url>
     Author: <author name>
   ```

## Success Criteria

- Linter result matches TD behavior
- No false positives on valid TD projects
- Regression fixture preserved with proper attribution

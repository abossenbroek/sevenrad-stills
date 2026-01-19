# .toe Files Pending TD Verification

From: https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner

## Failed Linting - Need TD Verification

## All files verified - no pending files

## Already Verified - Real Errors in TD

| File | Errors | Notes |
|------|--------|-------|
| Full_blend.toe | 56 | TD shows "9 networks with errors inside" |
| Textport_1.toe | 1 | Python syntax error in chopexec1: `print(this will error)` |
| 12-6_example_2.toe | 4 | GLSL compile errors, missing panel1 refs |
| 12-6_example_3.toe | 3 | Missing panel1 references |

## Fixed False Positives

| File | Issue | Fix |
|------|-------|-----|
| Basic_2D_buffers.toe | MRT fragColor[5] conflict | Added filter for `redeclaring` and `overlapping location` |

## How to Verify

```bash
/td-check-lint "/tmp/td-intro-book/TouchDesigner Example Files/12.5/Basic_2D_buffers.toe"
```

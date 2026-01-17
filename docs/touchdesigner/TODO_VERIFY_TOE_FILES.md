# .toe Files Pending TD Verification

From: https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner

## Failed Linting - Need TD Verification

| File | Errors | Path |
|------|--------|------|
| Basic_2D_buffers.toe | 2 | TouchDesigner Example Files/12.5/ |
| 12-6_example_2.toe | 4 | TouchDesigner Example Files/12.6/ |
| 12-6_example_3.toe | 3 | TouchDesigner Example Files/12.6/ |
| Textport_1.toe | 1 | TouchDesigner Example Files/9.2/ |

## Already Verified - Real Errors in TD

| File | Errors | Notes |
|------|--------|-------|
| Full_blend.toe | 56 | TD shows "9 networks with errors inside" |

## How to Verify

```bash
/td-check-lint "/tmp/td-intro-book/TouchDesigner Example Files/12.5/Basic_2D_buffers.toe"
```

# TDL-001: Sample Collection Campaign

---
id: TDL-001
status: pending
priority: critical
phase: 0
depends_on: []
blocks: [TDL-010, TDL-011, TDL-020]
---

## Problem Statement

The linter specification contains assumptions about .toe.dir format that are based on limited samples. A grammar built on insufficient examples will fail on real-world projects. We need a diverse corpus of .toe.dir samples spanning different TD versions, project types, and complexity levels.

## Acceptance Criteria

- [ ] 50+ .toe.dir samples collected
- [ ] Samples span TD 2022.x, 2023.x, and 2024.x versions
- [ ] Samples include: simple projects, complex networks, GLSL-heavy, Python-heavy
- [ ] `samples/catalog.yaml` documents each sample (source, version, characteristics)
- [ ] At least 5 samples with known issues (for negative test cases)
- [ ] No proprietary/licensed content without permission

## Files to Create

```
samples/
├── catalog.yaml           # Metadata for all samples
├── td2022/
│   └── [samples].toe.dir/
├── td2023/
│   └── [samples].toe.dir/
├── td2024/
│   └── [samples].toe.dir/
└── edge-cases/
    └── [problematic samples].toe.dir/
```

## Research Pointers

### Where to Find Samples

1. **GitHub Search Queries**:
   - `"toe.dir" extension:n` (finds .n files in expanded projects)
   - `touchdesigner expanded project`
   - `"toeexpand" OR "toecollapse"`

2. **Community Resources**:
   - TouchDesigner Forum (derivative.ca/community)
   - TD Discord servers
   - Patreon/tutorial creators who share project files

3. **Official Sources**:
   - Derivative example projects
   - TouchDesigner built-in templates

### How to Expand .toe Files

If you find .toe files but not .toe.dir:
```bash
# In TouchDesigner Python console
project.expand('/path/to/output.toe.dir')
```

Or use the `toeexpand` command-line tool.

## Catalog Schema

Design a YAML schema that captures:
- Source URL or attribution
- TouchDesigner version used to create/expand
- Project characteristics (family usage: TOP-heavy, CHOP-heavy, etc.)
- File count and complexity metrics
- Known issues or edge cases

## Quality Criteria for Samples

| Criterion | Why It Matters |
|-----------|----------------|
| Version diversity | Format may differ between TD versions |
| Complexity range | Simple projects miss edge cases |
| Feature coverage | Need GLSL, Python, all operator families |
| Real-world origin | Synthetic samples miss organic complexity |

## Anti-Patterns to Avoid

- Don't just collect your own projects (bias)
- Don't skip samples that fail to parse (those are the valuable ones)
- Don't strip metadata (we need context)

## Definition of Done

All acceptance criteria checked. Corpus ready for grammar development.

# TD-026a: band_swap C++ TOP Fallback

---
id: TD-026a
status: dormant
priority: medium
phase: 5
depends_on: [TD-026]
blocks: []
---

## Description

C++ TOP implementation for band_swap effect if GLSL compute shader fails.

**Status: DORMANT** - Activate only if TD-026 (GLSL compute) tests fail.

## Activation Trigger

Activate this ticket if:
- TD-026 GLSL compute shader fails on Metal
- Performance is unacceptable in GLSL
- MoltenVK translation produces incorrect results

## Acceptance Criteria (if activated)

- [ ] C++ TOP project created using TD SDK
- [ ] Implements band_swap algorithm in C++
- [ ] Performance matches or exceeds GLSL target
- [ ] Packaged as .tox operator

## C++ TOP Setup

```cpp
// Requires TouchDesigner SDK
#include "TOP_CPlusPlusBase.h"

class BandSwapTOP : public TOP_CPlusPlusBase {
public:
    void execute(TOP_Output* output) override {
        // Band swap implementation
    }
};
```

## Why C++ May Be Needed

- Random tile access pattern may not map well to fragment shaders
- Compute shader dispatch limits on Metal
- Need for shared memory between workgroups

## References

- [03-REMEDIATION-PLAN.md](../../03-REMEDIATION-PLAN.md) - Phase 5.2
- RF-007 red team finding
- [TouchDesigner C++ TOP docs](https://docs.derivative.ca/Write_a_CPlusPlus_TOP)

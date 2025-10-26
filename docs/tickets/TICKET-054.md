# [TICKET-054] Conditional Effect Execution (Backend)

## Context
Backend must process trig conditions and parameter ramping during render. This includes evaluating frame eligibility and calculating interpolated parameter values.

## User Story
As a backend system,
I want to execute effects conditionally based on trig rules and apply ramped parameters,
So that frontend-configured advanced modulation works correctly during rendering.

## Acceptance Criteria
- [ ] Backend evaluates trig conditions (every_nth, probability, fill)
- [ ] Probability uses seeded random for reproducibility if seed provided
- [ ] Parameter ramping calculates per-frame values using specified curve
- [ ] YAML schema extended with `trig_condition`, `probability`, `seed` fields
- [ ] Parameter ramp schema: `enabled`, `start`, `end`, `curve`
- [ ] Validation rejects ramp start/end values outside parameter bounds
- [ ] Render skips effect application when trig condition evaluates false

## Technical Notes
- Update `backend-contract.md` Section 3 (Pipeline JSON Schema)
- Add to effect schema:
  ```yaml
  effects:
    - name: blur
      repeat: 4
      trig_condition: every_2nd  # "all", "every_2nd", "every_3rd", "every_4th", "fill"
      probability: 0.75  # 0.0-1.0
      seed: 12345  # optional, for reproducible randomness
      parameters:
        radius:
          value: 10
          ramp:
            enabled: true
            start: 5
            end: 20
            curve: exponential_in  # "linear", "ease_in", "ease_out", "exponential"
  ```
- Trig condition logic:
  - `all`: apply every frame
  - `every_2nd`: apply if frame_index % 2 == 0
  - `every_3rd`: apply if frame_index % 3 == 0
  - `every_4th`: apply if frame_index % 4 == 0
  - `fill`: apply only if frame_index == total_frames - 1
- Probability: `random.seed(seed); random.random() < probability`
- Ramp interpolation: `start + (end - start) * curve_function(t)` where `t = frame_index / (total_frames - 1)`

## Related
- Parent: EPIC-010
- Dependencies: Backend API contract update

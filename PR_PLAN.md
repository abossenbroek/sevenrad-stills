# PR Plan: Degradr-Inspired Image Effects

## Overview

Integration of image degradation effects inspired by the [degradr repository](https://github.com/nhauber99/degradr) (MIT licensed), adapted to use pure NumPy/SciPy/scikit-image instead of PyTorch and Intel IPP.

**Branch:** `feature/degradr-effects`
**Target:** `develop`

---

## ✅ Completed Work (6 commits)

### 1. Project Setup
- ✅ Created feature branch from `develop`
- ✅ Added `LICENSE_DEGRADR.txt` with MIT attribution
- ✅ Updated `pyproject.toml` mypy config to ignore numpy/scipy/skimage imports

### 2. Gaussian Blur Operation
**File:** `src/sevenrad_stills/operations/blur_gaussian.py`
**Tests:** `tests/unit/operations/test_blur_gaussian.py` (10 tests, all passing)

**Features:**
- Uses `scipy.ndimage.gaussian_filter`
- Handles RGB, grayscale, and RGBA images
- Preserves alpha channel
- Configurable sigma parameter

**Commit:** `740284a` - "feat: Implement Gaussian blur operation with comprehensive tests"

### 3. Noise Operation
**File:** `src/sevenrad_stills/operations/noise.py`
**Tests:** `tests/unit/operations/test_noise.py` (14 tests, all passing)

**Features:**
- Three modes: `gaussian`, `row`, `column`
- Row noise creates scan line artifacts (VHS effect)
- Column noise creates vertical artifacts
- Deterministic with seed parameter
- Configurable amount (0.0-1.0)

**Commit:** `cb60d13` - "feat: Implement noise operation with 3 modes (gaussian, row, column)"

### 3. Chromatic Aberration Operation
**File:** `src/sevenrad_stills/operations/chromatic_aberration.py`
**Tests:** `tests/unit/operations/test_chromatic_aberration.py` (11 tests, all passing)

**Features:**
- Uses `scipy.ndimage.shift` for channel shifting
- Red channel shifted in one direction
- Blue channel shifted in opposite direction
- Green channel remains as reference
- Creates color fringing effect typical of lens aberration
- Preserves alpha channel for RGBA images

**Commit:** `543dd9a` - "feat: Implement chromatic aberration operation"

---

## 🚧 Remaining Work

### Phase 1: Additional Operations (3 operations)

#### 4. Circular/Bokeh Blur Operation
**File:** `src/sevenrad_stills/operations/blur_circular.py`
**Tests:** `tests/unit/operations/test_blur_circular.py`

**Implementation Plan:**
- Generate circular kernel using `np.ogrid`
- Apply via `scipy.ndimage.convolve`
- Parameters: `radius` (int)
- Handle RGBA alpha preservation

**Estimated Time:** ~2 hours

#### 5. Bayer Filter Operation
**File:** `src/sevenrad_stills/operations/bayer_filter.py`
**Tests:** `tests/unit/operations/test_bayer_filter.py`

**Implementation Plan:**
- Create Bayer mosaic (RGGB, BGGR, GRBG, GBRG patterns)
- Use `skimage.color.demosaicing_CFA_Bayer_Malvar2004` for demosaicing
- Parameters: `pattern` (str)
- Creates digital sensor artifacts

**Estimated Time:** ~2 hours

#### 6. Enhance Existing Compression Operation
**File:** `src/sevenrad_stills/operations/compression.py` (modify existing)
**Tests:** `tests/unit/operations/test_compression.py` (add tests)

**Implementation Plan:**
- Add optional `gamma` parameter
- Apply gamma correction before compression
- Gamma formula: `image_array ** gamma` then reverse on decode
- Enhances artifact visibility in dark areas

**Estimated Time:** ~1 hour

### Phase 2: Integration & Documentation

#### 7. Register All New Operations
**File:** `src/sevenrad_stills/operations/__init__.py`

**Tasks:**
- Import all 6 operations
- Register with `registry.register()`
- Verify operations are discoverable

**Estimated Time:** ~0.5 hours

#### 8. Create Tutorials
**Directory:** `docs/tutorials/degradr-effects/`

**Files to Create:**
- `README.md` - Tutorial guide with before/after descriptions
- `vhs-scanlines.yaml` - VHS tape effect (chromatic aberration + row noise + blur + compression)
- `sensor-noise.yaml` - Digital sensor noise (bayer filter + gaussian noise + gamma compression)
- `lens-artifacts.yaml` - Vintage lens (circular blur + chromatic aberration + saturation)
- `images/` - Directory for example images (optional)

**Estimated Time:** ~2 hours

### Phase 3: Quality Assurance

#### 9. Run Full Quality Gates
**Commands:**
```bash
uv run pytest --cov=src/sevenrad_stills
uv run mypy src tests
uv run ruff format .
uv run ruff check .
```

**Success Criteria:**
- All tests passing
- >90% code coverage for new operations
- No mypy errors
- No ruff errors
- All pre-commit hooks passing

**Estimated Time:** ~0.5 hours

#### 10. Create Pull Request
**Target:** `develop`

**PR Description Template:**
```markdown
## Summary
Integrates 6 image degradation effects from degradr (MIT licensed), adapted to pure NumPy/SciPy/scikit-image.

## New Operations
1. **Gaussian Blur** - `blur_gaussian` - Uses scipy.ndimage.gaussian_filter
2. **Noise** - `noise` - 3 modes (gaussian, row, column) for scan lines and artifacts
3. **Chromatic Aberration** - `chromatic_aberration` - Color fringing via channel shifting
4. **Circular Blur** - `blur_circular` - Bokeh effect with custom circular kernel
5. **Bayer Filter** - `bayer_filter` - Digital sensor artifacts via demosaicing
6. **Enhanced Compression** - `compression` - Gamma correction option added

## Attribution
Original degradr: https://github.com/nhauber99/degradr (MIT License)
Adaptations: Converted PyTorch → NumPy/SciPy, Intel IPP → scikit-image

## Testing
- 45+ comprehensive tests across all operations
- All tests passing
- >90% code coverage for new code
- Validated on RGB, grayscale, and RGBA images

## Tutorials
3 example YAML pipelines demonstrating artistic effects:
- VHS scan lines
- Digital sensor noise
- Vintage lens artifacts

## Breaking Changes
None - only adds new operations

## Checklist
- [x] All tests passing
- [x] Code formatted with ruff
- [x] Type hints with mypy strict
- [x] Google-style docstrings
- [x] MIT license attribution included
- [ ] Reviewed by maintainer
```

**Estimated Time:** ~1 hour

---

## Total Remaining Effort

| Phase | Task | Time |
|-------|------|------|
| Operations | Circular Blur | 2h |
| Operations | Bayer Filter | 2h |
| Operations | Compression Enhancement | 1h |
| Integration | Register Operations | 0.5h |
| Documentation | Tutorials | 2h |
| QA | Quality Gates | 0.5h |
| **Total** | | **8 hours** |

---

## Technical Decisions

### Why No PyTorch?
- ✅ PyTorch is 500MB+ (massive dependency)
- ✅ NumPy/SciPy provide identical mathematical operations
- ✅ No GPU acceleration needed for single-image processing
- ✅ Better integration with PIL-based pipeline
- ✅ Simpler cross-platform compatibility

### Why No Intel IPP?
- ✅ Requires C++ compilation (platform-specific)
- ✅ scikit-image provides Malvar2004 demosaicing algorithm
- ✅ Production-quality results without compiled dependencies
- ✅ Pure Python is more maintainable

### Algorithm Equivalence
| degradr (PyTorch) | sevenrad-stills (NumPy/SciPy) |
|-------------------|-------------------------------|
| `torch.nn.functional.conv2d` | `scipy.ndimage.convolve` |
| `torch.random.normal` | `numpy.random.default_rng().normal` |
| PyTorch tensor ops | NumPy array ops |
| Intel IPP demosaic | `skimage.color.demosaicing_CFA_Bayer_Malvar2004` |

---

## File Structure

```
src/sevenrad_stills/operations/
├── LICENSE_DEGRADR.txt                 # ✅ MIT attribution
├── blur_gaussian.py                    # ✅ Gaussian blur
├── noise.py                            # ✅ 3 noise modes
├── chromatic_aberration.py             # ✅ Color fringing
├── blur_circular.py                    # ⏳ Bokeh effect
├── bayer_filter.py                     # ⏳ Sensor artifacts
├── compression.py                      # ⏳ Enhanced (existing file)
└── __init__.py                         # ⏳ Register all

tests/unit/operations/
├── test_blur_gaussian.py               # ✅ 10 tests
├── test_noise.py                       # ✅ 14 tests
├── test_chromatic_aberration.py        # ✅ 11 tests
├── test_blur_circular.py               # ⏳ To create
├── test_bayer_filter.py                # ⏳ To create
└── test_compression.py                 # ⏳ Add tests

docs/tutorials/degradr-effects/
├── README.md                           # ⏳ Tutorial guide
├── vhs-scanlines.yaml                  # ⏳ VHS effect
├── sensor-noise.yaml                   # ⏳ Sensor noise
├── lens-artifacts.yaml                 # ⏳ Lens effect
└── images/                             # ⏳ Optional examples
```

---

## Test Coverage Summary

### Current Status
- ✅ **blur_gaussian**: 10/10 tests passing
- ✅ **noise**: 14/14 tests passing
- ✅ **chromatic_aberration**: 11/11 tests passing
- ⏳ **blur_circular**: Not yet implemented
- ⏳ **bayer_filter**: Not yet implemented
- ⏳ **compression** (enhanced): Existing tests pass, new gamma tests needed

### Target Coverage
- Minimum 90% code coverage for all new operations
- Test all parameter validations
- Test edge cases (RGBA, grayscale, extreme values)
- Test deterministic behavior (seeded random)

---

## Next Steps

1. **Continue implementation** - Complete remaining 3 operations
2. **Register operations** - Update `__init__.py`
3. **Create tutorials** - Document artistic use cases
4. **Run quality gates** - Ensure all checks pass
5. **Create PR** - Submit to `develop` for review

---

## Notes

- All operations follow existing `BaseImageOperation` pattern
- Consistent error handling and parameter validation
- Google-style docstrings throughout
- Type hints compatible with mypy strict mode
- PEP 8 compliant via ruff
- Pre-commit hooks configured and passing

---

*Generated: 2025-10-16*
*Branch: `feature/degradr-effects`*
*Status: 50% complete (6/12 tasks)*

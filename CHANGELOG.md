# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### td-linter

- **Direct .toe file linting**: Run `td-linter lint project.toe` to lint binary .toe files directly. The linter automatically expands the file, validates it, and cleans up temporary files.
- **TouchDesigner auto-discovery**: Automatically finds TouchDesigner installations on macOS, Windows, and Linux via common install locations, `TOUCHDESIGNER_PATH` env var, or PATH lookup.
- **`--td-path` CLI option**: Explicitly specify TouchDesigner installation path.
- **`--keep-files-after-expand` CLI option**: Keep expanded .toe.dir files after linting for debugging.
- **`td_tools` module**: New Python API for TouchDesigner tool discovery (`find_touchdesigner()`, `expand_toe()`, `collapse_toe()`).
- **`exports` block support**: Grammar now parses `exports` blocks in .n files.

### Fixed

#### td-linter

- **`extrainputs` block parsing**: Grammar now parses `extrainputs` blocks in .n files, including multi-level parent references like `../..` and `../../..`. Fixes false positives where operators with `extrainputs` failed to parse, causing their references to be flagged as missing.
- **GLSL false positives**: Updated shader preambles to GLSL 460 with extensions (`GL_GOOGLE_include_directive`, `GL_ARB_gpu_shader5`). Filtered false positives for scalar swizzles, include directives, and non-constant initializers.
- **Python false positives**: Added missing TouchDesigner builtins (`OP`, `math`, `textDAT`, `tableDAT`, operator type classes).
- **Binary header parsing**: Fixed `.text` file parsing to correctly handle 27-byte binary headers in TouchDesigner files.
- **Documentation detection**: Added heuristic to avoid misclassifying prose/documentation as GLSL code.

### Test Fixtures

- Added `MakingSimpleParticleSystemsWithTOPS Marco Kornke.toe` as regression fixture.
  - Source: [Interactive Immersive - TouchDesigner TOPs Particle System](https://interactiveimmersive.ac-page.com/touchdesigner-tops-particle-system)
  - Author: Marco Kornke
- Added `3D Waveform.toe` as regression fixture.
  - Source: [Introduction to TouchDesigner](https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner)
  - Author: Interactive Immersive HQ
- Added `Audio Responsive Geometry.toe` as regression fixture.
  - Source: [Introduction to TouchDesigner](https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner)
  - Author: Interactive Immersive HQ
- Added `Rendering_1.toe` as regression fixture.
  - Source: [Introduction to TouchDesigner](https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner)
  - Author: Interactive Immersive HQ
- Added `Instancing.toe` as regression fixture.
  - Source: [Introduction to TouchDesigner](https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner)
  - Author: Interactive Immersive HQ
- Added `Scripting_1.toe`, `Perform_mode.toe`, `Cooking_1.toe`, `Basic_3D.toe`, `common_chops.toe`, `Phong.toe`, `UI.toe`, `Color Picker.toe`, `01_Moving_particles_with_textures.toe`, `Video Switcher.toe` as regression fixtures.
  - Source: [Introduction to TouchDesigner](https://github.com/interactiveimmersivehq/Introduction-to-touchdesigner)
  - Author: Interactive Immersive HQ

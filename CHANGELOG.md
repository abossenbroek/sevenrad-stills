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

- **GLSL false positives**: Updated shader preambles to GLSL 460 with extensions (`GL_GOOGLE_include_directive`, `GL_ARB_gpu_shader5`). Filtered false positives for scalar swizzles, include directives, and non-constant initializers.
- **Python false positives**: Added missing TouchDesigner builtins (`OP`, `math`, `textDAT`, `tableDAT`, operator type classes).
- **Binary header parsing**: Fixed `.text` file parsing to correctly handle 27-byte binary headers in TouchDesigner files.
- **Documentation detection**: Added heuristic to avoid misclassifying prose/documentation as GLSL code.

### Test Fixtures

- Added `MakingSimpleParticleSystemsWithTOPS Marco Kornke.toe` as regression fixture.
  - Source: [Interactive Immersive - TouchDesigner TOPs Particle System](https://interactiveimmersive.ac-page.com/touchdesigner-tops-particle-system)
  - Author: Marco Kornke

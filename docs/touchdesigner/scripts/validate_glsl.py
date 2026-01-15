#!/usr/bin/env python3
"""
GLSL shader validation for TouchDesigner compatibility.

This script validates GLSL shaders by prepending the TouchDesigner preamble
and running them through glslangValidator. This catches syntax and semantic
errors before testing in actual TouchDesigner.

Usage:
    python validate_glsl.py <shader_file> [shader_file...]

Exit codes:
    0 - All shaders passed validation
    1 - One or more shaders failed validation

Requirements:
    - glslangValidator (install via: brew install glslang)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# TouchDesigner fragment shader preamble - simulates TD's built-in declarations
# TODO: Update with real preamble extracted from TouchDesigner 2022.20000+
# See TD-001 ticket for extraction procedure
TD_PREAMBLE = """
#version 330 core

// TouchDesigner built-in uniforms (synthetic - update with real values)
uniform sampler2D sTD2DInputs[8];
uniform vec4 uTDOutputInfo;
uniform int uTDPass;

// Input from vertex shader
in vec2 vUV;

// Standard output
layout(location = 0) out vec4 fragColor;

// TouchDesigner helper stubs
vec4 TDOutputSwizzle(vec4 c) { return c; }

// --- USER SHADER BEGINS BELOW ---
"""

# TouchDesigner compute shader preamble (GLSL 430+)
# TODO: Update with real preamble extracted from TouchDesigner 2022.20000+
TD_COMPUTE_PREAMBLE = """
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// TouchDesigner compute shader stubs (synthetic - update with real values)
uniform sampler2D sTD2DInputs[8];
layout(rgba32f) uniform image2D sTD2DOutputs[8];

// --- USER SHADER BEGINS BELOW ---
"""


def validate_shader(shader_path: Path) -> tuple[bool, str]:
    """
    Validate a GLSL shader file.

    Args:
        shader_path: Path to the shader file to validate

    Returns:
        Tuple of (passed, message) where passed is True if validation succeeded

    """
    content = shader_path.read_text()

    # Check for forbidden #version directive
    if "#version" in content:
        return (
            False,
            f"ERROR: {shader_path}: Contains #version directive (TouchDesigner auto-injects this)",
        )

    # Determine shader type
    suffix = shader_path.suffix.lower()
    if suffix == ".comp":
        preamble = TD_COMPUTE_PREAMBLE
        stage = "comp"
    else:
        preamble = TD_PREAMBLE
        stage = "frag"

    # Create temp file with preamble
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(preamble)
        f.write(content)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["glslangValidator", "-S", stage, temp_path], capture_output=True, text=True
        )

        if result.returncode == 0:
            return True, f"OK: {shader_path}"
        else:
            errors = result.stdout + result.stderr
            return False, f"ERRORS in {shader_path}:\n{errors}"
    finally:
        Path(temp_path).unlink()


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_glsl.py <shader_file> [shader_file...]")
        sys.exit(1)

    all_passed = True
    for path in sys.argv[1:]:
        passed, message = validate_shader(Path(path))
        print(message)
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""
GLSL shader validation for TouchDesigner .text files.

Validates GLSL shaders using glslangValidator with TouchDesigner-specific
preambles. Filters warnings for TD-injected uniforms and functions.

Security Note:
    The glslangValidator subprocess is run with timeout enforcement and
    optional sandboxing. On macOS, sandbox-exec is used when available.
    On Linux, bubblewrap is used when available. This provides defense
    in depth against malicious shader content or compromised toolchain.
"""

import re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from td_linter.rules.base import Violation
from td_linter.sandbox import Sandbox, SandboxError, SandboxTimeoutError


class ShaderType(Enum):
    """Type of GLSL shader."""

    FRAGMENT = "frag"
    COMPUTE = "comp"


@dataclass
class GLSLError:
    """GLSL validation error from glslangValidator."""

    line: int  # Adjusted line number (user code, not with preamble)
    column: Optional[int]
    message: str
    severity: str  # "error" or "warning"
    raw_line: int  # Original line from glslangValidator


# TouchDesigner fragment shader preamble
# Provides minimal declarations for syntax validation
TD_FRAGMENT_PREAMBLE = """\
#version 330 core

// TouchDesigner built-in uniforms (synthetic - for syntax validation)
uniform sampler2D sTD2DInputs[8];
uniform sampler3D sTD3DInputs[8];
uniform vec4 uTDOutputInfo;
uniform int uTDPass;

// Input from vertex shader
in vec2 vUV;

// Standard output
layout(location = 0) out vec4 fragColor;

// TouchDesigner helper stubs
vec4 TDOutputSwizzle(vec4 c) { return c; }
float TDAlphaOfOutput(vec4 c) { return c.a; }

// --- USER SHADER BEGINS BELOW ---
"""

# TouchDesigner compute shader preamble (GLSL 430+)
TD_COMPUTE_PREAMBLE = """\
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

// TouchDesigner compute shader stubs (synthetic - for syntax validation)
uniform sampler2D sTD2DInputs[8];
uniform sampler3D sTD3DInputs[8];
layout(rgba32f) uniform image2D sTD2DOutputs[8];

// --- USER SHADER BEGINS BELOW ---
"""

# Patterns to filter (TD injects these at runtime or preamble declares)
# These warnings are expected and should not be reported
FILTERED_PATTERNS = [
    re.compile(r"undefined uniform", re.IGNORECASE),
    re.compile(r"undeclared identifier", re.IGNORECASE),
    re.compile(r"use of undeclared", re.IGNORECASE),
    re.compile(r"unknown variable", re.IGNORECASE),
    re.compile(r"redefinition", re.IGNORECASE),  # Preamble may declare same vars
    re.compile(r"local_size", re.IGNORECASE),  # Layout qualifier conflict
    re.compile(r"cannot change previously set", re.IGNORECASE),  # Preamble conflict
    re.compile(r"compilation terminated", re.IGNORECASE),  # Follow-on error
]

# Patterns that indicate compute shader
COMPUTE_SHADER_PATTERNS = [
    re.compile(r"\bgl_GlobalInvocationID\b"),
    re.compile(r"\bgl_LocalInvocationID\b"),
    re.compile(r"\bgl_WorkGroupID\b"),
    re.compile(r"\bimageStore\b"),
    re.compile(r"\bimageLoad\b"),
    re.compile(r"\blayout\s*\([^)]*local_size", re.IGNORECASE),
]

# Pattern to parse glslangValidator output
# Format: ERROR: 0:15: 'foo' : message
ERROR_PATTERN = re.compile(r"(ERROR|WARNING):\s*\d+:(\d+):\s*(.+)")


class GLSLValidator:
    """
    Validate GLSL shaders using glslangValidator.

    Uses minimal TouchDesigner preambles for syntax validation and
    filters warnings for TD-injected symbols.
    """

    def __init__(self, glslang_path: Optional[Path] = None) -> None:
        """
        Initialize validator.

        Args:
            glslang_path: Optional path to glslangValidator executable.
                         If not provided, searches PATH.

        """
        self._glslang_path = glslang_path

    def is_available(self) -> bool:
        """
        Check if glslangValidator is available.

        Returns:
            True if glslangValidator is found and executable.

        """
        path = self._get_glslang_path()
        if path is None:
            return False

        try:
            # Use sandbox with short timeout for version check
            sandbox = Sandbox(timeout=5)
            result = sandbox.run(
                [str(path), "--version"],
                validate_binary=False,  # Don't require hash match for availability check
            )
            return result.return_code == 0
        except (SandboxTimeoutError, SandboxError, OSError):
            return False

    def validate(
        self,
        content: str,
        source_file: Optional[Path] = None,
        shader_type: Optional[ShaderType] = None,
    ) -> Iterator[Violation]:
        """
        Validate GLSL content and yield violations.

        Args:
            content: GLSL shader content (without .text header).
            source_file: Optional source file path for error reporting.
            shader_type: Type of shader (auto-detected if not specified).

        Yields:
            Violation objects for each error found.

        """
        if not self.is_available():
            yield Violation(
                rule="glsl-validator-unavailable",
                message="glslangValidator not found. Install via: brew install glslang",
                path=str(source_file) if source_file else "<unknown>",
                severity="warning",
                source_file=source_file,
            )
            return

        # Check for forbidden patterns first
        yield from self._check_forbidden_patterns(content, source_file)

        # Auto-detect shader type if not specified
        if shader_type is None:
            shader_type = self._detect_shader_type(content)

        # Get preamble and count its lines
        preamble = self._get_preamble(shader_type)
        preamble_lines = preamble.count("\n")

        # Run glslangValidator
        full_shader = preamble + content
        errors = self._run_glslang(full_shader, shader_type)

        # Parse and filter errors
        for error in self._parse_errors(errors, preamble_lines):
            # Skip filtered warnings
            if self._should_filter(error):
                continue

            yield Violation(
                rule="glsl-syntax-error",
                message=error.message,
                path=str(source_file) if source_file else "<unknown>",
                severity="error" if error.severity == "error" else "warning",
                source_file=source_file,
                line=error.line,
            )

    def _get_glslang_path(self) -> Optional[Path]:
        """Get path to glslangValidator executable."""
        if self._glslang_path:
            return self._glslang_path

        path = shutil.which("glslangValidator")
        return Path(path) if path else None

    def _get_preamble(self, shader_type: ShaderType) -> str:
        """Get the appropriate preamble for the shader type."""
        if shader_type == ShaderType.COMPUTE:
            return TD_COMPUTE_PREAMBLE
        return TD_FRAGMENT_PREAMBLE

    def _detect_shader_type(self, content: str) -> ShaderType:
        """
        Auto-detect shader type from content.

        Args:
            content: GLSL shader content.

        Returns:
            Detected shader type (COMPUTE or FRAGMENT).

        """
        for pattern in COMPUTE_SHADER_PATTERNS:
            if pattern.search(content):
                return ShaderType.COMPUTE
        return ShaderType.FRAGMENT

    def _check_forbidden_patterns(
        self,
        content: str,
        source_file: Optional[Path],
    ) -> Iterator[Violation]:
        """
        Check for forbidden patterns in shader content.

        TouchDesigner injects #version, so user shaders should not include it.
        """
        for line_num, line in enumerate(content.split("\n"), start=1):
            stripped = line.strip()
            if stripped.startswith("#version"):
                msg = (
                    "Shader contains #version directive "
                    "(TouchDesigner auto-injects this)"
                )
                yield Violation(
                    rule="glsl-forbidden-version",
                    message=msg,
                    path=str(source_file) if source_file else "<unknown>",
                    severity="error",
                    source_file=source_file,
                    line=line_num,
                )

    def _run_glslang(self, full_shader: str, shader_type: ShaderType) -> str:
        """
        Run glslangValidator and return output.

        Args:
            full_shader: Complete shader with preamble.
            shader_type: Type of shader.

        Returns:
            Combined stdout and stderr from glslangValidator.

        Security:
            The subprocess is run in a sandbox when platform support is
            available (macOS sandbox-exec, Linux bubblewrap). Timeout is
            enforced to prevent DoS via malicious shaders.
        """
        glslang_path = self._get_glslang_path()
        if glslang_path is None:
            return ""

        # Write shader to temp file
        suffix = f".{shader_type.value}"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
        ) as f:
            f.write(full_shader)
            temp_path = Path(f.name)

        try:
            # Use sandbox for security
            sandbox = Sandbox(timeout=30)
            result = sandbox.run(
                [str(glslang_path), "-S", shader_type.value, str(temp_path)],
                validate_binary=False,  # Don't require hash match
            )
            return result.stdout + result.stderr
        except SandboxTimeoutError:
            return "ERROR: glslangValidator timed out (possible infinite loop)"
        except SandboxError as e:
            return f"ERROR: glslangValidator failed: {e}"
        except OSError as e:
            return f"ERROR: glslangValidator failed: {e}"
        finally:
            temp_path.unlink(missing_ok=True)

    def _parse_errors(
        self,
        output: str,
        preamble_lines: int,
    ) -> Iterator[GLSLError]:
        """
        Parse glslangValidator output into GLSLError objects.

        Args:
            output: Raw output from glslangValidator.
            preamble_lines: Number of lines in the preamble.

        Yields:
            GLSLError objects with adjusted line numbers.

        """
        for line in output.split("\n"):
            match = ERROR_PATTERN.match(line.strip())
            if match:
                severity = match.group(1).lower()
                raw_line = int(match.group(2))
                message = match.group(3)

                # Adjust line number by subtracting preamble
                adjusted_line = raw_line - preamble_lines
                if adjusted_line < 1:
                    # Error is in preamble, skip
                    continue

                yield GLSLError(
                    line=adjusted_line,
                    column=None,
                    message=message,
                    severity=severity,
                    raw_line=raw_line,
                )

    def _should_filter(self, error: GLSLError) -> bool:
        """
        Check if error should be filtered (TD-specific warning).

        Args:
            error: The GLSL error to check.

        Returns:
            True if the error should be filtered out.

        """
        return any(pattern.search(error.message) for pattern in FILTERED_PATTERNS)

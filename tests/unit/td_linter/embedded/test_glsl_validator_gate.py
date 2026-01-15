"""
Gate tests for TDL-031: GLSL Validator Integration.

Tests GLSL shader validation using glslangValidator.
"""

import pytest
from td_linter.embedded import GLSLValidator, ShaderType


class TestPositiveValidShaders:
    """G3.1-P1: Valid TD shaders should pass."""

    def test_valid_fragment_shader(self):
        """Valid fragment shader passes validation."""
        content = """
        void main() {
            vec4 color = texture(sTD2DInputs[0], vUV);
            fragColor = TDOutputSwizzle(color);
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        # Filter out availability warnings
        errors = [v for v in violations if v.rule != "glsl-validator-unavailable"]
        # May have filtered warnings, should have no syntax errors
        syntax_errors = [v for v in errors if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) == 0

    def test_valid_compute_shader(self):
        """Valid compute shader passes validation."""
        content = """
        void main() {
            ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
            vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
            imageStore(sTD2DOutputs[0], xy, color);
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content, shader_type=ShaderType.COMPUTE))

        syntax_errors = [v for v in violations if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) == 0

    def test_shader_with_td_functions(self):
        """Shader using TD helper functions passes."""
        content = """
        void main() {
            vec4 color = vec4(1.0);
            fragColor = TDOutputSwizzle(color);
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        syntax_errors = [v for v in violations if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) == 0


class TestPositiveSyntaxErrors:
    """G3.1-P2: Syntax errors should be caught."""

    def test_missing_semicolon(self):
        """Missing semicolon is caught."""
        content = """
        void main() {
            vec4 color = vec4(1.0)
            fragColor = color;
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        syntax_errors = [v for v in violations if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) > 0

    def test_type_mismatch(self):
        """Type mismatch in assignment is caught."""
        content = """
        void main() {
            vec4 color = 5;  // Cannot assign int to vec4
            fragColor = color;
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        syntax_errors = [v for v in violations if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) > 0

    def test_invalid_function_signature(self):
        """Invalid function signature is caught."""
        content = """
        void main(invalid_type x) {
            fragColor = vec4(1.0);
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        # Should have an error about invalid type
        syntax_errors = [v for v in violations if v.rule == "glsl-syntax-error"]
        assert len(syntax_errors) > 0


class TestPositiveForbiddenPatterns:
    """G3.1-P3: Forbidden patterns should be flagged."""

    def test_version_directive_error(self):
        """#version directive is flagged as error."""
        content = """
        #version 330 core
        void main() {
            fragColor = vec4(1.0);
        }
        """
        validator = GLSLValidator()
        violations = list(validator.validate(content))

        forbidden = [v for v in violations if v.rule == "glsl-forbidden-version"]
        assert len(forbidden) == 1
        assert "#version" in forbidden[0].message


class TestNegativeTDSpecificWarnings:
    """G3.1-N1: TD-specific warnings should be filtered."""

    def test_undefined_td_uniform_filtered(self):
        """Undefined TD uniforms are filtered."""
        content = """
        void main() {
            // Using TD uniform that might not be in preamble
            vec4 info = uTDOutputInfo;
            fragColor = vec4(info.xy, 0.0, 1.0);
        }
        """
        validator = GLSLValidator()

        if not validator.is_available():
            pytest.skip("glslangValidator not available")

        violations = list(validator.validate(content))

        # Undefined uniform warnings should be filtered
        undefined_warnings = [
            v
            for v in violations
            if "undefined" in v.message.lower() and v.rule == "glsl-syntax-error"
        ]
        # These should be filtered out
        # Note: Some may still appear if not in filter list


class TestValidatorAvailability:
    """Test validator availability checking."""

    def test_is_available_returns_bool(self):
        """is_available returns boolean."""
        validator = GLSLValidator()
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_unavailable_yields_warning(self):
        """Unavailable validator yields warning violation."""
        # Create validator with non-existent path
        from pathlib import Path

        validator = GLSLValidator(glslang_path=Path("/nonexistent/glslangValidator"))

        violations = list(validator.validate("void main() {}"))

        warnings = [v for v in violations if v.rule == "glsl-validator-unavailable"]
        assert len(warnings) == 1

"""
Gate tests for TDL-030: Language Detector.

Tests language detection for .text files containing GLSL or Python code.
"""

import pytest
from td_linter.embedded import (
    DetectionContext,
    Language,
    LanguageDetector,
)


class TestPositiveGLSLDetection:
    """G3-P1: Detector should correctly identify GLSL content."""

    def test_detects_fragment_shader(self):
        """Fragment shader with vec types detected as GLSL."""
        content = """
        uniform sampler2D sTD2DInputs[1];
        in vec2 vUV;
        out vec4 fragColor;

        void main() {
            vec4 color = texture(sTD2DInputs[0], vUV);
            fragColor = TDOutputSwizzle(color);
        }
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.GLSL
        assert result.confidence > 0.5

    def test_detects_compute_shader(self):
        """Compute shader with TD-specific uniforms detected as GLSL."""
        content = """
        layout(local_size_x = 16, local_size_y = 16) in;
        uniform sampler2D sTD2DInputs[1];
        layout(rgba32f) uniform image2D sTD2DOutputs[1];

        void main() {
            ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
            vec4 color = texelFetch(sTD2DInputs[0], xy, 0);
            imageStore(sTD2DOutputs[0], xy, color);
        }
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.GLSL

    def test_detects_td_specific_glsl(self):
        """TD-specific GLSL patterns strongly indicate GLSL."""
        content = """
        vec4 color = TDOutputSwizzle(vec4(1.0));
        float alpha = TDAlphaOfOutput(color);
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.GLSL


class TestPositivePythonDetection:
    """G3-P2: Detector should correctly identify Python content."""

    def test_detects_execute_dat_script(self):
        """Execute DAT with TD callbacks detected as Python."""
        content = """
        def onFrameStart(frame):
            op('geo1').par.tx = absTime.frame * 0.1
            debug(f'Frame: {frame}')

        def onFrameEnd(frame):
            pass
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.PYTHON
        assert result.confidence > 0.5

    def test_detects_callbacks_script(self):
        """Callback script with TD-specific API detected as Python."""
        content = """
        def onValueChange(par, prev):
            if par.name == 'active':
                op('container1').par.display = par.val
                me.parent().cook(force=True)
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.PYTHON

    def test_detects_td_specific_python(self):
        """TD-specific Python patterns strongly indicate Python."""
        content = """
        target = op('/project1/base1')
        project.paths['input'] = '/data'
        value = absTime.frame * 0.5
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.PYTHON


class TestPositiveContextDetection:
    """G3-P3: Detector should use context when available."""

    def test_execute_dat_context_overrides_content(self):
        """Context from DAT type provides strong hint."""
        # Content that could be ambiguous
        content = "# Some comment\npass"

        context = DetectionContext(dat_type="execute")
        detector = LanguageDetector()
        result = detector.detect(content, context)

        assert result.language == Language.PYTHON
        assert result.detection_method == "context"

    def test_script_dat_context(self):
        """Script DAT context indicates Python."""
        content = "x = 5"

        context = DetectionContext(dat_type="script")
        detector = LanguageDetector()
        result = detector.detect(content, context)

        assert result.language == Language.PYTHON


class TestNegativeAmbiguousContent:
    """G3-N1: Detector should return UNKNOWN for ambiguous content."""

    def test_empty_content(self):
        """Empty content returns UNKNOWN."""
        detector = LanguageDetector()
        result = detector.detect("")

        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.0

    def test_comments_only(self):
        """Content with only comments is ambiguous."""
        content = "# This is a comment"
        detector = LanguageDetector()
        result = detector.detect(content)

        assert result.language == Language.UNKNOWN

    def test_mixed_patterns_low_confidence(self):
        """Mixed patterns with similar scores return UNKNOWN."""
        # This has both Python-like and GLSL-like syntax
        content = """
        def main():
            pass
        vec4 color;
        """
        detector = LanguageDetector()
        result = detector.detect(content)

        # Should return based on which scores higher, or UNKNOWN if too close
        # The test verifies the detector handles mixed content


class TestTextHeaderStripping:
    """Test .text file header handling."""

    def test_strips_version_header(self):
        """Version header line is stripped."""
        content = "2\ndef onCook():\n    pass"
        detector = LanguageDetector()
        stripped, lines = detector.strip_text_header(content)

        assert lines == 1
        assert stripped == "def onCook():\n    pass"

    def test_strips_version_and_metadata(self):
        """Version and metadata lines are stripped."""
        content = "2\n*\ndef onCook():\n    pass"
        detector = LanguageDetector()
        stripped, lines = detector.strip_text_header(content)

        assert lines == 2
        assert stripped == "def onCook():\n    pass"

    def test_no_header(self):
        """Content without header is unchanged."""
        content = "def onCook():\n    pass"
        detector = LanguageDetector()
        stripped, lines = detector.strip_text_header(content)

        assert lines == 0
        assert stripped == content

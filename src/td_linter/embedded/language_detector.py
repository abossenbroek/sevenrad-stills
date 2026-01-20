"""
Language detection for TouchDesigner .text files.

Determines whether a .text file contains GLSL shaders, Python scripts,
or unknown content.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from td_linter.embedded.constants import (
    DETECTION_THRESHOLD,
    GLSL_PATTERNS,
    PYTHON_DAT_TYPES,
    PYTHON_PATTERNS,
)


class Language(Enum):
    """Detected language of a .text file."""

    GLSL = "glsl"
    PYTHON = "python"
    UNKNOWN = "unknown"


@dataclass
class DetectionContext:
    """
    Context from parent .n file for improved detection.

    When available, context provides strong hints about the expected
    language based on the DAT operator type.
    """

    dat_type: Optional[str] = None  # e.g., "execute", "text", "script"
    parent_path: Optional[Path] = None  # Path to parent .n file
    docked_to: Optional[str] = None  # If docked to a GLSL TOP


@dataclass
class DetectionResult:
    """Result of language detection."""

    language: Language
    confidence: float  # 0.0 to 1.0
    glsl_score: int
    python_score: int
    detection_method: str  # "context" or "content"


class LanguageDetector:
    """
    Detect language of .text file content.

    Uses a combination of context (DAT type) and content analysis
    (pattern matching) to determine the language.
    """

    def detect(
        self,
        content: str,
        context: Optional[DetectionContext] = None,
    ) -> DetectionResult:
        """
        Detect language from content and optional context.

        Args:
            content: The raw content of the .text file (including header).
            context: Optional context from the parent .n file.

        Returns:
            DetectionResult with language, confidence, and detection method.

        """
        # Strip .text header before analysis
        stripped_content, _ = self.strip_text_header(content)

        # Context-based detection takes priority
        if context and context.dat_type:
            dat_type_lower = context.dat_type.lower()
            if dat_type_lower in PYTHON_DAT_TYPES:
                # Check if content has strong GLSL indicators
                # (e.g., text DAT containing shader code)
                glsl_score = self._score_glsl(stripped_content)
                python_score = self._score_python(stripped_content)

                # If strongly GLSL despite DAT type, use content
                if glsl_score > python_score + DETECTION_THRESHOLD * 2:
                    return DetectionResult(
                        language=Language.GLSL,
                        confidence=min(1.0, glsl_score / 20.0),
                        glsl_score=glsl_score,
                        python_score=python_score,
                        detection_method="content",
                    )

                return DetectionResult(
                    language=Language.PYTHON,
                    confidence=0.9,
                    glsl_score=glsl_score,
                    python_score=python_score,
                    detection_method="context",
                )

        # Content-based detection
        glsl_score = self._score_glsl(stripped_content)
        python_score = self._score_python(stripped_content)

        # Check for documentation/plain text indicators
        # If content looks like prose (long lines without code structure), skip
        if self._looks_like_documentation(stripped_content):
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=0.8,
                glsl_score=glsl_score,
                python_score=python_score,
                detection_method="content",
            )

        # Determine language based on score difference
        score_diff = abs(glsl_score - python_score)

        if score_diff < DETECTION_THRESHOLD:
            # Too close to call
            confidence = 0.0 if (glsl_score == 0 and python_score == 0) else 0.3
            return DetectionResult(
                language=Language.UNKNOWN,
                confidence=confidence,
                glsl_score=glsl_score,
                python_score=python_score,
                detection_method="content",
            )

        if glsl_score > python_score:
            # Calculate confidence based on how decisive the score is
            max_score = max(glsl_score, 1)
            confidence = min(1.0, score_diff / max_score * 0.5 + 0.5)
            return DetectionResult(
                language=Language.GLSL,
                confidence=confidence,
                glsl_score=glsl_score,
                python_score=python_score,
                detection_method="content",
            )
        else:
            max_score = max(python_score, 1)
            confidence = min(1.0, score_diff / max_score * 0.5 + 0.5)
            return DetectionResult(
                language=Language.PYTHON,
                confidence=confidence,
                glsl_score=glsl_score,
                python_score=python_score,
                detection_method="content",
            )

    def strip_text_header(self, content: str) -> tuple[str, int]:
        """
        Remove .text file header (version line and binary metadata).

        TouchDesigner .text files have a binary header format:
            Bytes 0-1:   "2\\n" (version)
            Byte 2:      "*" (metadata marker)
            Bytes 3-24:  Fixed binary header (22 bytes of flags/config)
            Bytes 25-26: Content length (big-endian 16-bit integer)
            Bytes 27+:   Actual content

        The total header size is 27 bytes when the binary format is used.

        Args:
            content: Raw file content.

        Returns:
            Tuple of (stripped_content, lines_stripped).

        """
        if not content:
            return content, 0

        # Handle binary content after the '*' marker
        # The format is: "2\n*" + 24 bytes binary + actual content
        if len(content) >= 27 and content.startswith("2\n*"):
            # Check if this is the binary format by looking for null bytes
            # in the expected header region (bytes 3-24)
            header_region = content[3:25]
            if any(ord(c) < 32 and ord(c) != ord('\t') for c in header_region):
                # Binary format detected - skip exactly 27 bytes
                stripped = content[27:]
                # Count lines in header for line number adjustment
                lines_stripped = content[:27].count("\n")
                return stripped, lines_stripped

        # Fallback: line-based stripping for simpler/text-only headers
        lines = content.split("\n")

        if len(lines) < 1:
            return content, 0

        # First line should be a version number (typically "2")
        first_line = lines[0].strip()
        if not first_line.isdigit():
            # Not a standard .text file header
            return content, 0

        # Check if second line is metadata (starts with '*')
        if len(lines) >= 2 and lines[1].startswith("*"):
            return "\n".join(lines[2:]), 2

        return "\n".join(lines[1:]), 1

    def _score_glsl(self, content: str) -> int:
        """
        Count weighted GLSL pattern matches.

        Args:
            content: Content to analyze (without header).

        Returns:
            Total score based on pattern matches.

        """
        score = 0
        for pattern, weight in GLSL_PATTERNS.items():
            matches = pattern.findall(content)
            score += len(matches) * weight
        return score

    def _score_python(self, content: str) -> int:
        """
        Count weighted Python pattern matches.

        Args:
            content: Content to analyze (without header).

        Returns:
            Total score based on pattern matches.

        """
        score = 0
        for pattern, weight in PYTHON_PATTERNS.items():
            matches = pattern.findall(content)
            score += len(matches) * weight
        return score

    def _looks_like_documentation(self, content: str) -> bool:
        """
        Check if content appears to be documentation/prose rather than code.

        Documentation files often mention code keywords but lack actual
        code structure (braces, semicolons, function definitions).

        Args:
            content: Content to analyze.

        Returns:
            True if content appears to be documentation.

        """
        if not content.strip():
            return False

        lines = content.strip().split("\n")
        if not lines:
            return False

        # Documentation indicators
        doc_indicators = 0

        # Check for prose-like content: sentences, natural language patterns
        first_line = lines[0].strip()
        if first_line and first_line[0].isupper():
            # Starts with capital letter (like a sentence)
            words = first_line.split()
            if len(words) > 5:  # Long opening line
                doc_indicators += 1

        # Check for common documentation patterns
        content_lower = content.lower()
        if "how to" in content_lower:
            doc_indicators += 2
        if "example:" in content_lower:
            doc_indicators += 1
        if "instructions" in content_lower:
            doc_indicators += 1
        if "readme" in content_lower:
            doc_indicators += 2

        # Code indicators (absence suggests documentation)
        code_indicators = 0

        # Check for function definitions/declarations
        if "void " in content and "(" in content and ")" in content:
            code_indicators += 2
        if "def " in content and ":" in content:
            code_indicators += 2
        if "class " in content and ":" in content:
            code_indicators += 2

        # Check for code structure
        if "{" in content and "}" in content:
            code_indicators += 1
        if ";" in content:
            # Count semicolons as percentage of lines
            semicolon_lines = sum(1 for line in lines if ";" in line)
            if semicolon_lines > len(lines) * 0.3:
                code_indicators += 2

        # If high documentation indicators and low code indicators, it's docs
        return doc_indicators >= 2 and code_indicators < 2

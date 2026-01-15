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
        Remove .text file header (version line).

        TouchDesigner .text files have a header format:
            2                  (version number)
            *                  [optional metadata line]
            [actual content]

        Args:
            content: Raw file content.

        Returns:
            Tuple of (stripped_content, lines_stripped).

        """
        if not content:
            return content, 0

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

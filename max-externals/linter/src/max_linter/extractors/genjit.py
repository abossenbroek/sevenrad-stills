"""Extractor for shader code from .genjit files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedShader:
    """Shader code extracted from a .genjit file."""

    code: str
    language: str  # "genexpr" or "glsl"
    source_file: Path
    box_id: str | None = None


class GenjitExtractor:
    """Extracts shader code from .genjit files.

    .genjit files are JSON patcher files that contain GenExpr or GLSL code
    in codebox objects.
    """

    # GLSL/XML markers that indicate GLSL format instead of GenExpr
    GLSL_MARKERS = [
        "<jit.gl.pix>",
        "</jit.gl.pix>",
        '<param name="',
        '<language name="glsl"',
        "#version",
        "uniform ",
        "void main()",
        "gl_FragColor",
        "texture2DRect",
        "varying ",
        "<![CDATA[",
    ]

    def __init__(self) -> None:
        """Initialize extractor."""
        pass

    def extract(self, filepath: Path) -> list[ExtractedShader]:
        """Extract shader code from a .genjit file.

        Args:
            filepath: Path to the .genjit file

        Returns:
            List of extracted shaders (usually just one)
        """
        try:
            content = filepath.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            return []
        except OSError as e:
            logger.error(f"Cannot read {filepath}: {e}")
            return []

        shaders: list[ExtractedShader] = []

        # Find codebox objects
        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            maxclass = box.get("maxclass", "")

            if maxclass == "codebox":
                code = box.get("code", "")
                if not code.strip():
                    continue

                # Detect language
                language = self._detect_language(code)

                shaders.append(
                    ExtractedShader(
                        code=code,
                        language=language,
                        source_file=filepath,
                        box_id=box.get("id"),
                    )
                )

        return shaders

    def _detect_language(self, code: str) -> str:
        """Detect if code is GLSL or GenExpr.

        Args:
            code: Shader code

        Returns:
            "glsl" or "genexpr"
        """
        # Strip comments for detection
        code_no_comments = self._strip_comments(code)

        for marker in self.GLSL_MARKERS:
            if marker in code_no_comments:
                return "glsl"

        return "genexpr"

    def _strip_comments(self, code: str) -> str:
        """Strip C-style comments from code.

        Args:
            code: Source code

        Returns:
            Code with comments replaced by spaces
        """
        result = []
        i = 0
        in_block_comment = False

        while i < len(code):
            if in_block_comment:
                if code[i : i + 2] == "*/":
                    in_block_comment = False
                    result.append("  ")
                    i += 2
                else:
                    result.append("\n" if code[i] == "\n" else " ")
                    i += 1
            elif code[i : i + 2] == "/*":
                in_block_comment = True
                result.append("  ")
                i += 2
            elif code[i : i + 2] == "//":
                while i < len(code) and code[i] != "\n":
                    result.append(" ")
                    i += 1
            else:
                result.append(code[i])
                i += 1

        return "".join(result)

    def extract_all(self, directory: Path) -> list[ExtractedShader]:
        """Extract shaders from all .genjit files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of all extracted shaders
        """
        shaders: list[ExtractedShader] = []

        for filepath in directory.glob("*.genjit"):
            shaders.extend(self.extract(filepath))

        return shaders

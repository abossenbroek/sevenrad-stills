"""Extractor for shader code from .genjit files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractedParam:
    """Parameter extracted from a .genjit file.

    Attributes:
        name: Parameter name (e.g., "shift_x")
        default: Default value
        min_val: Minimum value (optional)
        max_val: Maximum value (optional)
    """

    name: str
    default: float
    min_val: float | None = None
    max_val: float | None = None


@dataclass
class ExtractedShader:
    """Shader code extracted from a .genjit file."""

    code: str
    language: str  # "genexpr" or "glsl"
    source_file: Path
    box_id: str | None = None
    params: list[ExtractedParam] = field(default_factory=list)
    had_crlf: bool = False  # True if original code had Windows line endings


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

        # Extract params from the patcher (shared across all codeboxes)
        params = self._extract_params(data)

        shaders: list[ExtractedShader] = []

        # Find codebox objects
        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            maxclass = box.get("maxclass", "")

            if maxclass == "codebox":
                code = box.get("code", "")
                if not code.strip():
                    continue

                # Detect and normalize line endings (CRLF -> LF)
                had_crlf = "\r\n" in code
                if had_crlf:
                    code = code.replace("\r\n", "\n")
                    logger.warning(
                        f"{filepath}: Codebox '{box.get('id', 'unknown')}' "
                        "has Windows line endings (\\r\\n) - normalizing to LF"
                    )

                # Detect language
                language = self._detect_language(code)

                shaders.append(
                    ExtractedShader(
                        code=code,
                        language=language,
                        source_file=filepath,
                        box_id=box.get("id"),
                        params=params,
                        had_crlf=had_crlf,
                    )
                )

        return shaders

    def _extract_params(self, data: dict[str, Any]) -> list[ExtractedParam]:
        """Extract param declarations from .genjit patcher boxes.

        Searches for newobj boxes with text starting with "param " and parses
        the parameter name, default value, and optional min/max bounds.

        Format: param name default [min max]
        Examples:
            - param shift_x 5.0
            - param intensity 0.5 0.0 1.0

        Args:
            data: Parsed .genjit JSON data

        Returns:
            List of ExtractedParam objects
        """
        params: list[ExtractedParam] = []

        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            if box.get("maxclass") != "newobj":
                continue

            text = box.get("text", "")
            if not text.startswith("param "):
                continue

            parts = text.split()
            if len(parts) < 3:
                # Need at least: param name default
                continue

            try:
                param = ExtractedParam(
                    name=parts[1],
                    default=float(parts[2]),
                    min_val=float(parts[3]) if len(parts) > 3 else None,
                    max_val=float(parts[4]) if len(parts) > 4 else None,
                )
                params.append(param)
            except (ValueError, IndexError):
                # Skip malformed param declarations
                logger.warning(f"Malformed param declaration: {text}")

        return params

    def validate_param_ranges(
        self, params: list[ExtractedParam]
    ) -> list[tuple[str, str]]:
        """Validate that param defaults are within their specified bounds.

        Args:
            params: List of extracted parameters

        Returns:
            List of (param_name, error_message) tuples for invalid params
        """
        errors: list[tuple[str, str]] = []

        for p in params:
            if p.min_val is not None and p.default < p.min_val:
                errors.append(
                    (p.name, f"default {p.default} is less than min {p.min_val}")
                )
            if p.max_val is not None and p.default > p.max_val:
                errors.append(
                    (p.name, f"default {p.default} is greater than max {p.max_val}")
                )

        return errors

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

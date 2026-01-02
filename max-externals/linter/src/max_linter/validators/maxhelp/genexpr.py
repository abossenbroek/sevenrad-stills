"""GenExpr validation mixin for Max help patchers.

This module provides the GenExprValidatorMixin class that validates
GenExpr shader code in Max/MSP help patchers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max_linter.genexpr import GenExprValidator


class GenExprValidatorMixin:
    """Mixin providing GenExpr validation methods.

    Validates GenExpr shader format and syntax.

    Rules:
        genjit-format: Invalid genjit file format
        genexpr-syntax: GenExpr syntax errors
        security: Path traversal attempts
        shader-name: Unusual shader name characters
    """

    # These attributes must be provided by the composing class
    filepath: Path | None
    genexpr_validator: GenExprValidator | None

    def error(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an error."""
        raise NotImplementedError

    def warning(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record a warning."""
        raise NotImplementedError

    def info(self, rule: str, message: str, object_id: str | None = None) -> None:
        """Record an informational message."""
        raise NotImplementedError

    def _extract_gen_shader(self, text: str) -> str | None:
        """Extract @gen shader name from jit.gl.pix text."""
        match = re.search(r"@gen\s+(\S+)", text)
        if match:
            return match.group(1)
        return None

    def _strip_comments(self, code: str) -> str:
        """Strip C-style comments from code to avoid false positives in GLSL detection.

        Removes:
        - Single-line comments: // ...
        - Block comments: /* ... */

        Args:
            code: Source code string

        Returns:
            Code with comments replaced by whitespace (preserves line numbers)
        """
        result = []
        i = 0
        in_block_comment = False

        while i < len(code):
            if in_block_comment:
                # Look for end of block comment
                if code[i : i + 2] == "*/":
                    in_block_comment = False
                    result.append("  ")  # Replace */ with spaces
                    i += 2
                else:
                    # Preserve newlines for line number tracking
                    result.append("\n" if code[i] == "\n" else " ")
                    i += 1
            elif code[i : i + 2] == "/*":
                # Start of block comment
                in_block_comment = True
                result.append("  ")  # Replace /* with spaces
                i += 2
            elif code[i : i + 2] == "//":
                # Single-line comment - skip to end of line
                while i < len(code) and code[i] != "\n":
                    result.append(" ")
                    i += 1
            else:
                result.append(code[i])
                i += 1

        return "".join(result)

    def _find_genjit_file(self, shader_name: str) -> Path | None:  # noqa: PLR0911
        """Find the .genjit file for a shader name.

        Security: Validates shader_name to prevent path traversal attacks.
        """
        if self.filepath is None:
            return None

        # Validate shader_name
        if not shader_name or not shader_name.strip():
            return None

        shader_name = shader_name.strip()

        # Check for path traversal attempts
        if "/" in shader_name or "\\" in shader_name or ".." in shader_name:
            self.error(
                "security",
                f"Shader name '{shader_name}' contains path separators - "
                "possible path traversal attack",
            )
            return None

        # Check for unreasonably long names
        if len(shader_name) > 255:
            self.error("security", "Shader name exceeds 255 characters")
            return None

        # Only allow alphanumeric, dots, underscores, hyphens
        if not re.match(r"^[a-zA-Z0-9._-]+$", shader_name):
            self.warning(
                "shader-name",
                f"Shader name '{shader_name}' contains unusual characters",
            )

        # Look in code/ directory relative to help/
        code_dir = self.filepath.parent.parent / "code"
        genjit_file = code_dir / f"{shader_name}.genjit"

        # Verify the resolved path is still within code_dir
        try:
            genjit_file_resolved = genjit_file.resolve(strict=False)
            code_dir_resolved = code_dir.resolve(strict=True)

            if not genjit_file_resolved.is_relative_to(code_dir_resolved):
                self.error("security", "Shader path escapes code directory")
                return None
        except (ValueError, OSError):
            return None

        if genjit_file_resolved.exists():
            return genjit_file_resolved
        return None

    def _parse_genjit_params(self, genjit_path: Path) -> list[dict[str, Any]]:
        """Parse parameters from a .genjit file.

        Returns list of dicts with 'name', 'default', 'min', 'max' keys.
        """
        params: list[dict[str, Any]] = []

        try:
            content = genjit_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            return params

        # Find boxes with text starting with "param "
        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            text = box.get("text", "")
            if text.startswith("param "):
                parts = text.split()
                if len(parts) >= 5:
                    # Format: "param name default min max"
                    params.append(
                        {
                            "name": parts[1],
                            "default": parts[2],
                            "min": float(parts[3]),
                            "max": float(parts[4]),
                        }
                    )
                elif len(parts) >= 3:
                    # Legacy format without bounds - still accept but warn
                    params.append(
                        {
                            "name": parts[1],
                            "default": parts[2],
                            "min": None,
                            "max": None,
                        }
                    )

        return params

    def _validate_genexpr_code(
        self, code: str, shader_name: str, declared_params: set[str] | None = None
    ) -> bool:
        """Validate GenExpr shader code using the GenExprValidator.

        Args:
            code: GenExpr shader code to validate
            shader_name: Name of the shader for error messages
            declared_params: Set of parameter names declared in the .genjit file

        Returns:
            True if valid, False if errors found
        """
        if not self.genexpr_validator:
            self.info(
                "genexpr-validation",
                f"GenExprValidator not available - skipping syntax validation "
                f"for {shader_name}",
            )
            return True

        valid = True

        # Import here to avoid circular imports at module level
        from max_linter.results import DiagnosticSeverity as LSPSev

        diagnostics = self.genexpr_validator.validate(code, declared_params)

        for diag in diagnostics:
            # Convert LSP diagnostic severity to our severity
            if diag.severity == LSPSev.ERROR:
                self.error(
                    "genexpr-syntax",
                    f"GenExpr error in {shader_name}.genjit "
                    f"line {diag.range.start.line + 1}: {diag.message}",
                )
                valid = False
            elif diag.severity == LSPSev.WARNING:
                self.warning(
                    "genexpr-syntax",
                    f"GenExpr warning in {shader_name}.genjit "
                    f"line {diag.range.start.line + 1}: {diag.message}",
                )
            else:
                # INFORMATION or HINT
                self.info(
                    "genexpr-syntax",
                    f"GenExpr info in {shader_name}.genjit "
                    f"line {diag.range.start.line + 1}: {diag.message}",
                )

        return valid

    def _validate_genjit_format(
        self, genjit_path: Path, shader_name: str
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Validate that a .genjit file uses proper GenExpr format, not GLSL/XML.

        Detects GLSL/XML markers that indicate wrong shader format:
        - XML tags: <jit.gl.pix>, <param name=, <language name="glsl"
        - GLSL keywords: #version, uniform, void main(), gl_FragColor, texture2DRect

        Validates GenExpr requirements:
        - Has proper 'param name default' objects (not XML <param>)
        - Codebox uses GenExpr syntax (in1, out1, sample, norm, dim)
        - Uses GenExprValidator for syntax and semantic validation

        Args:
            genjit_path: Path to the .genjit file
            shader_name: Name of the shader for error messages

        Returns:
            Tuple of (is_valid, params_list)
        """
        valid = True
        params: list[dict[str, Any]] = []

        try:
            content = genjit_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self.error(
                "genjit-format",
                f"Invalid JSON in {shader_name}.genjit: {e.msg}",
            )
            return False, params
        except OSError as e:
            self.error(
                "genjit-format",
                f"Cannot read {shader_name}.genjit: {e}",
            )
            return False, params

        # Find codebox and param objects
        codebox_content: str | None = None
        has_param_objects = False

        for box_wrapper in data.get("patcher", {}).get("boxes", []):
            box = box_wrapper.get("box", {})
            maxclass = box.get("maxclass", "")
            text = box.get("text", "")

            # Check for proper param objects
            if maxclass == "newobj" and text.startswith("param "):
                has_param_objects = True
                parts = text.split()
                if len(parts) >= 3:
                    params.append({"name": parts[1], "default": parts[2]})

            # Get codebox content
            if maxclass == "codebox":
                codebox_content = box.get("code", "")

        if codebox_content is None:
            self.error(
                "genjit-format",
                f"No codebox found in {shader_name}.genjit",
            )
            return False, params

        # Strip comments before checking for GLSL markers
        code_without_comments = self._strip_comments(codebox_content)

        # GLSL/XML markers that indicate wrong format
        glsl_xml_markers = [
            (
                "<jit.gl.pix>",
                "XML wrapper <jit.gl.pix> (use GenExpr codebox instead)",
            ),
            ("</jit.gl.pix>", "XML closing tag </jit.gl.pix>"),
            ('<param name="', "XML parameter declaration <param name="),
            ('<language name="glsl"', "GLSL language declaration"),
            ("#version", "GLSL #version directive"),
            ("uniform ", "GLSL uniform declaration"),
            ("void main()", "GLSL main function"),
            ("gl_FragColor", "GLSL gl_FragColor output"),
            ("texture2DRect", "GLSL texture2DRect function"),
            ("varying ", "GLSL varying declaration"),
            ("<![CDATA[", "XML CDATA section"),
        ]

        for marker, description in glsl_xml_markers:
            if marker in code_without_comments:
                self.error(
                    "genjit-format",
                    f"GLSL/XML format detected in {shader_name}.genjit: "
                    f"{description}. Convert to GenExpr format with "
                    "'param name default' objects.",
                )
                valid = False

        # Check for GenExpr requirements (only if not already detected as GLSL)
        if valid:
            genexpr_markers = ["out1", "in1", "sample(", "norm", "dim"]
            has_genexpr = any(marker in codebox_content for marker in genexpr_markers)

            if not has_genexpr:
                self.warning(
                    "genjit-format",
                    f"No GenExpr markers found in {shader_name}.genjit "
                    "(expected: out1, in1, sample, norm, dim)",
                )

            # Check for param objects if parameters are used
            if not has_param_objects and params:
                self.warning(
                    "genjit-format",
                    f"No 'param' objects found in {shader_name}.genjit. "
                    "Parameters should be declared as 'param name default' objects.",
                )

            # Validate GenExpr code using GenExprValidator
            param_names = {p["name"] for p in params}
            genexpr_valid = self._validate_genexpr_code(
                codebox_content, shader_name, param_names
            )
            valid &= genexpr_valid

        return valid, params

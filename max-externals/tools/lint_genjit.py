#!/usr/bin/env python3
"""
Red-team validation linter for .genjit files.

Validates that .genjit files conform to Max/MSP's JSON patcher format.
This linter is adversarial and catches issues that would cause Max to fail loading.

Usage:
    python lint_genjit.py code/*.genjit
    python lint_genjit.py --strict code/sr.saturation.genjit
    python lint_genjit.py --verbose code/*.genjit

Exit codes:
    0: All validations passed
    1: One or more validations failed
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


class ValidationError:
    """Represents a validation error with severity and location."""

    def __init__(
        self,
        message: str,
        severity: str = "ERROR",
        location: str | None = None,
    ):
        self.message = message
        self.severity = severity
        self.location = location

    def __str__(self) -> str:
        if self.location:
            return f"  [{self.severity}] {self.location}: {self.message}"
        return f"  [{self.severity}] {self.message}"


class GenjitLinter:
    """Validates .genjit files against Max/MSP JSON patcher format."""

    def __init__(self, strict: bool = False, verbose: bool = False):
        self.strict = strict
        self.verbose = verbose
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def error(self, message: str, location: str | None = None) -> None:
        """Add an error."""
        self.errors.append(ValidationError(message, "ERROR", location))

    def warning(self, message: str, location: str | None = None) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(message, "WARNING", location))

    def info(self, message: str, location: str | None = None) -> None:
        """Add an info message (only shown in verbose mode)."""
        if self.verbose:
            self.warnings.append(ValidationError(message, "INFO", location))

    def validate_file(self, filepath: Path) -> bool:
        """
        Validate a single .genjit file.

        Returns:
            True if valid, False otherwise

        """
        self.errors.clear()
        self.warnings.clear()

        if not filepath.exists():
            self.error(f"File does not exist: {filepath}")
            return False

        if filepath.suffix != ".genjit":
            self.error(f"File must have .genjit extension, got: {filepath.suffix}")
            return False

        # Read and parse JSON
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            self.error(f"Failed to read file: {e}")
            return False

        # Check if it's JSON at all
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self.error(
                f"Invalid JSON syntax: {e.msg} at line {e.lineno}, col {e.colno}"
            )
            # Check if it looks like GenExpr code instead of JSON
            if content.strip().startswith(("//", "/*", "Param", "out")):
                self.error(
                    "File appears to be GenExpr code, not Max JSON patcher format. "
                    "Must be a JSON file with 'patcher' structure."
                )
            return False

        # Validate JSON structure
        valid = True
        valid &= self._validate_structure(data)

        if not valid:
            return False

        # Continue with detailed validations
        patcher = data.get("patcher", {})
        valid &= self._validate_patcher_fields(patcher)
        valid &= self._validate_boxes(patcher.get("boxes", []))
        valid &= self._validate_patchlines(
            patcher.get("lines", []), patcher.get("boxes", [])
        )
        valid &= self._validate_codebox(patcher.get("boxes", []))
        valid &= self._validate_params(patcher.get("boxes", []))
        valid &= self._validate_connections(
            patcher.get("lines", []), patcher.get("boxes", [])
        )

        return valid

    def _validate_structure(self, data: Any) -> bool:
        """Validate top-level JSON structure."""
        if not isinstance(data, dict):
            self.error("Top-level must be a JSON object")
            return False

        if "patcher" not in data:
            self.error(
                "Missing required top-level key 'patcher'. "
                "File must be in Max JSON patcher format."
            )
            return False

        if not isinstance(data["patcher"], dict):
            self.error("'patcher' must be an object")
            return False

        return True

    def _validate_patcher_fields(self, patcher: dict[str, Any]) -> bool:
        """Validate required patcher fields."""
        valid = True

        required_fields = ["fileversion", "appversion", "boxes", "lines"]
        for field in required_fields:
            if field not in patcher:
                self.error(f"Missing required patcher field: '{field}'")
                valid = False

        # Validate fileversion
        if "fileversion" in patcher:
            if not isinstance(patcher["fileversion"], int):
                self.error("'fileversion' must be an integer")
                valid = False
            elif patcher["fileversion"] != 1:
                self.warning(f"'fileversion' is {patcher['fileversion']}, expected 1")

        # Validate appversion
        if "appversion" in patcher:
            if not isinstance(patcher["appversion"], dict):
                self.error("'appversion' must be an object")
                valid = False
            else:
                for key in ["major", "minor", "revision"]:
                    if key not in patcher["appversion"]:
                        self.warning(f"Missing appversion field: '{key}'")

        # Validate boxes and lines are arrays
        if "boxes" in patcher and not isinstance(patcher["boxes"], list):
            self.error("'boxes' must be an array")
            valid = False

        if "lines" in patcher and not isinstance(patcher["lines"], list):
            self.error("'lines' must be an array")
            valid = False

        return valid

    def _validate_boxes(self, boxes: list[dict[str, Any]]) -> bool:
        """Validate boxes array."""
        valid = True

        if not boxes:
            self.error(
                "'boxes' array is empty - must contain at least input, output, and codebox"
            )
            return False

        # Check each box has required structure
        for i, item in enumerate(boxes):
            if not isinstance(item, dict):
                self.error(f"Box {i} must be an object")
                valid = False
                continue

            if "box" not in item:
                self.error(f"Box {i} missing 'box' wrapper")
                valid = False
                continue

            box = item["box"]
            if not isinstance(box, dict):
                self.error(f"Box {i} 'box' must be an object")
                valid = False
                continue

            # Check required box fields
            required = ["id", "maxclass", "numinlets", "numoutlets"]
            for field in required:
                if field not in box:
                    self.error(
                        f"Box {i} ({box.get('id', '?')}) missing field: '{field}'"
                    )
                    valid = False

            # Validate id format
            if "id" in box and (
                not isinstance(box["id"], str) or not box["id"].startswith("obj-")
            ):
                self.error(
                    f"Box {i} has invalid id format: '{box.get('id')}' "
                    "(must be string starting with 'obj-')"
                )
                valid = False

            # Validate maxclass
            if "maxclass" in box:
                valid_classes = ["newobj", "codebox", "comment"]
                if box["maxclass"] not in valid_classes:
                    self.warning(
                        f"Box {i} ({box.get('id')}) has unusual maxclass: '{box['maxclass']}' "
                        f"(expected one of {valid_classes})"
                    )

            # Validate inlet/outlet counts
            for field in ["numinlets", "numoutlets"]:
                if field in box and not isinstance(box[field], int):
                    self.error(
                        f"Box {i} ({box.get('id')}) '{field}' must be an integer"
                    )
                    valid = False
                elif field in box and box[field] < 0:
                    self.error(
                        f"Box {i} ({box.get('id')}) '{field}' cannot be negative"
                    )
                    valid = False

        return valid

    def _validate_patchlines(
        self, lines: list[dict[str, Any]], boxes: list[dict[str, Any]]
    ) -> bool:
        """Validate patchlines structure."""
        valid = True

        # Extract valid box IDs
        box_ids = set()
        for item in boxes:
            if "box" in item and "id" in item["box"]:
                box_ids.add(item["box"]["id"])

        for i, item in enumerate(lines):
            if not isinstance(item, dict):
                self.error(f"Patchline {i} must be an object")
                valid = False
                continue

            if "patchline" not in item:
                self.error(f"Patchline {i} missing 'patchline' wrapper")
                valid = False
                continue

            patchline = item["patchline"]
            if not isinstance(patchline, dict):
                self.error(f"Patchline {i} 'patchline' must be an object")
                valid = False
                continue

            # Check required fields
            required = ["source", "destination"]
            for field in required:
                if field not in patchline:
                    self.error(f"Patchline {i} missing field: '{field}'")
                    valid = False
                    continue

                # Validate format: [box_id, outlet_number]
                value = patchline[field]
                if not isinstance(value, list) or len(value) != 2:
                    self.error(
                        f"Patchline {i} '{field}' must be array of [box_id, outlet_num], "
                        f"got: {value}"
                    )
                    valid = False
                    continue

                box_id, outlet = value
                if not isinstance(box_id, str):
                    self.error(
                        f"Patchline {i} '{field}' box_id must be string, got: {box_id}"
                    )
                    valid = False

                if not isinstance(outlet, int):
                    self.error(
                        f"Patchline {i} '{field}' outlet must be integer, got: {outlet}"
                    )
                    valid = False

                # Validate box ID exists
                if box_id not in box_ids:
                    self.error(
                        f"Patchline {i} '{field}' references non-existent box: '{box_id}'"
                    )
                    valid = False

        return valid

    def _validate_codebox(self, boxes: list[dict[str, Any]]) -> bool:
        """Validate codebox requirements."""
        valid = True

        codeboxes = [
            item["box"]
            for item in boxes
            if "box" in item and item["box"].get("maxclass") == "codebox"
        ]

        if not codeboxes:
            self.error("Missing required codebox object (maxclass: 'codebox')")
            return False

        if len(codeboxes) > 1:
            self.error(f"Found {len(codeboxes)} codebox objects, expected exactly 1")
            valid = False

        codebox = codeboxes[0]

        # Check for code field
        if "code" not in codebox:
            self.error("Codebox missing required 'code' field")
            return False

        if not isinstance(codebox["code"], str):
            self.error("Codebox 'code' must be a string")
            return False

        code = codebox["code"]
        if not code or not code.strip():
            self.error("Codebox 'code' is empty")
            return False

        # Detect shader type: GenExpr or GLSL
        # A file is GLSL if it has GLSL markers AND doesn't look like GenExpr
        has_glsl_markers = bool(
            re.search(r"<jit\.gl\.pix>", code)
            or re.search(r"#version\s+\d+", code)
            or "gl_FragColor" in code
            or "texture2DRect" in code
            or "uniform sampler" in code
        )
        has_genexpr_markers = bool(
            re.search(r"\bin1\b", code)
            or re.search(r"\bout1\s*=", code)
            or re.search(r"\bout\s*=", code)
            or re.search(r"sample\s*\(", code)
        )

        # If has GenExpr markers, treat as GenExpr even if it has some GLSL
        # Otherwise, if has GLSL markers, treat as GLSL
        is_glsl = has_glsl_markers and not has_genexpr_markers

        if is_glsl:
            # GLSL shader validation
            # Check for gl_FragColor assignment (not just in comments)
            has_gl_frag_color = bool(re.search(r"gl_FragColor\s*=", code))
            if not has_gl_frag_color:
                self.error(
                    "GLSL shader does not assign to 'gl_FragColor'. "
                    "No output will be generated."
                )
                valid = False

            # Check for texture sampling
            has_texture_input = bool(
                re.search(r"texture2DRect\s*\(", code)
                or re.search(r"texture2D\s*\(", code)
                or re.search(r"texture\s*\(", code)
            )
            if not has_texture_input:
                self.warning(
                    "GLSL shader does not appear to sample input texture. "
                    "This may cause issues."
                )

            self.info("Detected GLSL shader format")
        else:
            # GenExpr shader validation
            if "in1" not in code:
                self.warning(
                    "GenExpr code does not reference 'in1' (input texture). "
                    "This may cause issues."
                )

            # Check for output assignment
            has_output = bool(
                re.search(r"\bout1\s*=", code) or re.search(r"\bout\s*=", code)
            )
            if not has_output:
                self.error(
                    "GenExpr code does not assign to 'out1' or 'out'. "
                    "No output will be generated."
                )
                valid = False

            self.info("Detected GenExpr shader format")

        # Check inlet/outlet counts match code complexity
        if "numinlets" in codebox:
            if codebox["numinlets"] < 1:
                self.error("Codebox must have at least 1 inlet (for input texture)")
                valid = False

        if "numoutlets" in codebox:
            if codebox["numoutlets"] != 1:
                self.warning(
                    f"Codebox has {codebox['numoutlets']} outlets, typically should be 1"
                )

        return valid

    def _validate_params(self, boxes: list[dict[str, Any]]) -> bool:
        """Validate param objects."""
        valid = True

        param_boxes = [
            item["box"]
            for item in boxes
            if "box" in item
            and item["box"].get("maxclass") == "newobj"
            and isinstance(item["box"].get("text"), str)
            and item["box"]["text"].startswith("param ")
        ]

        if not param_boxes:
            self.info("No param objects found (shader may not have parameters)")
            return True

        param_pattern = re.compile(r"^param\s+(\w+)\s+([\d.\-\se]+)$")

        for i, param_box in enumerate(param_boxes):
            text = param_box.get("text", "")

            # Validate param text format
            match = param_pattern.match(text)
            if not match:
                self.error(
                    f"Param {i} has invalid format: '{text}'. "
                    "Expected: 'param <name> <value>' or 'param <name> <default> <min> <max>'"
                )
                valid = False
                continue

            param_name = match.group(1)
            param_values = match.group(2)

            # Validate inlet/outlet counts
            if param_box.get("numinlets") != 0:
                self.error(
                    f"Param '{param_name}' has {param_box.get('numinlets')} inlets, "
                    "must be 0"
                )
                valid = False

            if param_box.get("numoutlets") != 1:
                self.error(
                    f"Param '{param_name}' has {param_box.get('numoutlets')} outlets, "
                    "must be 1"
                )
                valid = False

        return valid

    def _validate_connections(
        self, lines: list[dict[str, Any]], boxes: list[dict[str, Any]]
    ) -> bool:
        """Validate patchline connections form valid graph."""
        valid = True

        # Build box lookup
        box_map = {}
        for item in boxes:
            if "box" in item and "id" in item["box"]:
                box_map[item["box"]["id"]] = item["box"]

        # Find key objects
        input_boxes = [
            (box_id, box)
            for box_id, box in box_map.items()
            if box.get("maxclass") == "newobj" and box.get("text") == "in 1"
        ]

        output_boxes = [
            (box_id, box)
            for box_id, box in box_map.items()
            if box.get("maxclass") == "newobj" and box.get("text") == "out 1"
        ]

        codebox_boxes = [
            (box_id, box)
            for box_id, box in box_map.items()
            if box.get("maxclass") == "codebox"
        ]

        # Validate required objects exist
        if not input_boxes:
            self.error(
                "Missing required 'in 1' object (maxclass: newobj, text: 'in 1')"
            )
            return False

        if not output_boxes:
            self.error(
                "Missing required 'out 1' object (maxclass: newobj, text: 'out 1')"
            )
            return False

        if not codebox_boxes:
            # Already checked in _validate_codebox
            return False

        if len(input_boxes) > 1:
            self.warning(
                f"Found {len(input_boxes)} 'in 1' objects, typically should be 1"
            )

        if len(output_boxes) > 1:
            self.warning(
                f"Found {len(output_boxes)} 'out 1' objects, typically should be 1"
            )

        # Build connection graph
        connections = []
        for item in lines:
            if "patchline" not in item:
                continue
            patchline = item["patchline"]
            if "source" in patchline and "destination" in patchline:
                src_id, src_outlet = patchline["source"]
                dst_id, dst_inlet = patchline["destination"]
                connections.append((src_id, src_outlet, dst_id, dst_inlet))

        # Validate input -> codebox connection
        input_id = input_boxes[0][0]
        codebox_id = codebox_boxes[0][0]

        input_to_codebox = [
            c for c in connections if c[0] == input_id and c[2] == codebox_id
        ]

        if not input_to_codebox:
            self.error(
                f"No connection from input ('{input_id}') to codebox ('{codebox_id}'). "
                "Input must connect to codebox inlet 0."
            )
            valid = False
        else:
            # Check it connects to inlet 0
            inlet = input_to_codebox[0][3]
            if inlet != 0:
                self.warning(
                    f"Input connects to codebox inlet {inlet}, typically should be inlet 0"
                )

        # Validate codebox -> output connection
        output_id = output_boxes[0][0]

        codebox_to_output = [
            c for c in connections if c[0] == codebox_id and c[2] == output_id
        ]

        if not codebox_to_output:
            self.error(
                f"No connection from codebox ('{codebox_id}') to output ('{output_id}'). "
                "Codebox must connect to output."
            )
            valid = False

        # Validate param objects exist (they don't need patchline connections)
        # In Gen, params are available by name to codeboxes without explicit connections
        param_boxes = [
            (box_id, box)
            for box_id, box in box_map.items()
            if box.get("maxclass") == "newobj"
            and isinstance(box.get("text"), str)
            and box["text"].startswith("param ")
        ]

        # Just log the params we found (they don't need connections)
        if param_boxes:
            self.info(f"Found {len(param_boxes)} param object(s)")
            for param_id, param_box in param_boxes:
                param_text = param_box.get("text", "")
                self.info(f"  - {param_text}")

        return valid

    def print_results(self, filepath: Path) -> None:
        """Print validation results."""
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if has_errors:
            print(f"FAIL: {filepath}")
        elif has_warnings and self.strict:
            print(f"FAIL: {filepath} (warnings in strict mode)")
        else:
            print(f"PASS: {filepath}")

        for error in self.errors:
            print(error)

        for warning in self.warnings:
            print(warning)

        if not has_errors and not has_warnings:
            print("  All validations passed ✓")

        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Red-team validation linter for .genjit files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lint_genjit.py code/*.genjit
  python lint_genjit.py --strict code/sr.saturation.genjit
  python lint_genjit.py --verbose code/*.genjit

Exit codes:
  0: All validations passed
  1: One or more validations failed
        """,
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more .genjit files to validate",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation info",
    )

    args = parser.parse_args()

    linter = GenjitLinter(strict=args.strict, verbose=args.verbose)

    all_passed = True
    for filepath in args.files:
        passed = linter.validate_file(filepath)
        linter.print_results(filepath)

        if not passed or (args.strict and len(linter.warnings) > 0):
            all_passed = False

    if all_passed:
        print(f"✓ All {len(args.files)} file(s) passed validation")
        return 0
    else:
        failed_count = sum(
            1
            for f in args.files
            if not linter.validate_file(f) or (args.strict and len(linter.warnings) > 0)
        )
        print(f"✗ {failed_count} file(s) failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())

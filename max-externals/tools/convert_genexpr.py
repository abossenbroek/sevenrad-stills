#!/usr/bin/env python3
"""
Convert plain text GenExpr files to Max/MSP JSON patcher format.

This script converts .genjit files from plain text GenExpr code to the
JSON patcher format that Max/MSP expects. The JSON format includes
param objects, codebox, input/output objects, and patchlines.

Usage:
    python convert_genexpr.py code/*.genjit
    python convert_genexpr.py code/sr.saturation.genjit --backup
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def parse_genexpr(content: str) -> tuple[list[tuple[str, str]], str]:
    """
    Parse plain text GenExpr file.

    Args:
        content: The raw GenExpr file content

    Returns:
        Tuple of (params, code) where:
        - params: List of (name, default_value) tuples
        - code: The GenExpr code (everything after param declarations)

    """
    lines = content.split("\n")
    params = []
    code_lines = []
    in_header_comment = True
    past_params = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the start
        if not stripped and not past_params:
            continue

        # Skip header comment block
        if in_header_comment:
            if (
                stripped.startswith("/**")
                or stripped.startswith("*")
                or stripped.startswith("//")
            ):
                continue
            if stripped.endswith("*/"):
                continue
            in_header_comment = False

        # Parse Param declarations
        param_match = re.match(r"^Param\s+(\w+)\s*\(\s*([^)]+)\s*\)\s*;?\s*$", stripped)
        if param_match:
            name = param_match.group(1)
            default_raw = param_match.group(2)
            # Handle multiple values (min, max) - just take the first (default)
            default_parts = [p.strip() for p in default_raw.split(",")]
            default_value = default_parts[0]
            # Clean up the default value
            try:
                # Try to format as number
                if "." in default_value:
                    default_value = str(float(default_value))
                else:
                    default_value = str(int(default_value))
            except ValueError:
                pass  # Keep as string
            params.append((name, default_value))
            continue

        # If we hit a non-param, non-comment line, we're in the code section
        if stripped and not stripped.startswith("//"):
            past_params = True

        # Collect code lines (after params)
        if past_params:
            code_lines.append(line)

    # Join code lines and strip leading/trailing whitespace
    code = "\n".join(code_lines).strip()

    return params, code


def count_texture_inputs(code: str) -> int:
    """
    Count how many texture inputs (in1, in2, etc.) the code uses.

    Args:
        code: The GenExpr code

    Returns:
        Number of texture inputs used (minimum 1)

    """
    max_inlet = 1  # At least in1
    # Find all inN references
    for match in re.finditer(r"\bin(\d+)\b", code):
        inlet_num = int(match.group(1))
        max_inlet = max(max_inlet, inlet_num)
    return max_inlet


def generate_patcher_json(params: list[tuple[str, str]], code: str, title: str) -> dict:
    """
    Generate Max/MSP JSON patcher format.

    Args:
        params: List of (name, default_value) tuples
        code: GenExpr code for the codebox
        title: Title for the patcher (usually filename)

    Returns:
        Dictionary in Max patcher JSON format

    Note:
        Params are NOT connected to codebox via patchlines.
        In Gen, params defined with 'param' objects are automatically
        available by name to all codeboxes without explicit connections.
        Codebox numinlets is based on texture inputs (in1, in2, etc.),
        NOT params.

    """
    boxes = []
    lines = []
    obj_id = 1

    # Count how many texture inputs the code uses
    num_texture_inputs = count_texture_inputs(code)

    # Layout configuration
    base_x = 50.0
    param_y = 30.0
    input_y = 60.0
    codebox_y = 120.0
    output_y = 480.0
    param_spacing = 120.0

    # Create input objects (in 1, in 2, etc.) for each texture input
    in_obj_ids = []
    for i in range(num_texture_inputs):
        in_obj_id = f"obj-{obj_id}"
        obj_id += 1
        in_obj_ids.append(in_obj_id)
        boxes.append(
            {
                "box": {
                    "fontname": "Arial",
                    "fontsize": 12.0,
                    "id": in_obj_id,
                    "maxclass": "newobj",
                    "numinlets": 0,
                    "numoutlets": 1,
                    "outlettype": [""],
                    "patching_rect": [base_x + (i * 80.0), input_y, 30.0, 20.0],
                    "text": f"in {i + 1}",
                }
            }
        )

    # Create param objects
    param_obj_ids = []
    for i, (name, default) in enumerate(params):
        param_obj_id = f"obj-{obj_id}"
        obj_id += 1
        param_obj_ids.append(param_obj_id)

        # Format param text
        param_text = f"param {name} {default}"

        boxes.append(
            {
                "box": {
                    "fontname": "Arial",
                    "fontsize": 12.0,
                    "id": param_obj_id,
                    "maxclass": "newobj",
                    "numinlets": 0,
                    "numoutlets": 1,
                    "outlettype": [""],
                    "patching_rect": [
                        base_x + 80.0 + (i * param_spacing),
                        param_y,
                        100.0,
                        20.0,
                    ],
                    "text": param_text,
                }
            }
        )

    # Create codebox
    codebox_id = f"obj-{obj_id}"
    obj_id += 1

    # Number of inlets = number of texture inputs (NOT params)
    # Params are accessed by name, not via patchlines

    # Fix output variable: Max Gen expects 'out1', not 'out'
    # Replace 'out =' with 'out1 =' (but not 'out1 =' which is already correct)
    fixed_code = re.sub(r"\bout\s*=", "out1 =", code)

    # Format code for JSON (use \r\n as in official files)
    formatted_code = fixed_code.replace("\n", "\r\n")

    boxes.append(
        {
            "box": {
                "code": formatted_code,
                "fontname": "Arial",
                "fontsize": 12.0,
                "id": codebox_id,
                "maxclass": "codebox",
                "numinlets": num_texture_inputs,
                "numoutlets": 1,
                "outlettype": [""],
                "patching_rect": [base_x, codebox_y, 500.0, 340.0],
            }
        }
    )

    # Create output object (out 1)
    out_obj_id = f"obj-{obj_id}"
    obj_id += 1
    boxes.append(
        {
            "box": {
                "fontname": "Arial",
                "fontsize": 12.0,
                "id": out_obj_id,
                "maxclass": "newobj",
                "numinlets": 1,
                "numoutlets": 0,
                "patching_rect": [base_x, output_y, 37.0, 20.0],
                "text": "out 1",
            }
        }
    )

    # Create patchlines
    # Connect all texture inputs to codebox inlets
    for i, in_obj_id in enumerate(in_obj_ids):
        lines.append(
            {
                "patchline": {
                    "destination": [codebox_id, i],
                    "disabled": 0,
                    "hidden": 0,
                    "source": [in_obj_id, 0],
                }
            }
        )

    # NOTE: Params are NOT connected to codebox via patchlines.
    # They are automatically available by name to the codebox.

    # Connect codebox to output
    lines.append(
        {
            "patchline": {
                "destination": [out_obj_id, 0],
                "disabled": 0,
                "hidden": 0,
                "source": [codebox_id, 0],
            }
        }
    )

    # Build the full patcher structure
    patcher = {
        "patcher": {
            "fileversion": 1,
            "appversion": {"major": 8, "minor": 6, "revision": 0},
            "rect": [100.0, 100.0, 650.0, 550.0],
            "bgcolor": [0.9, 0.9, 0.9, 0.9],
            "bglocked": 0,
            "openinpresentation": 0,
            "default_fontsize": 12.0,
            "default_fontface": 0,
            "default_fontname": "Arial",
            "gridonopen": 0,
            "gridsize": [15.0, 15.0],
            "gridsnaponopen": 0,
            "statusbarvisible": 0,
            "toolbarvisible": 1,
            "boxanimatetime": 200,
            "imprint": 0,
            "enablehscroll": 1,
            "enablevscroll": 1,
            "devicewidth": 0.0,
            "description": "",
            "digest": "",
            "tags": "",
            "title": title,
            "boxes": boxes,
            "lines": lines,
            "dependency_cache": [
                {"name": "codebox.mxo", "type": "iLaX"},
                {"name": "param.mxo", "type": "iLaX"},
            ],
        }
    }

    return patcher


def is_json_patcher(content: str) -> bool:
    """Check if content is already in JSON patcher format."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(content)
        return "patcher" in data
    except json.JSONDecodeError:
        return False


def convert_file(filepath: Path, backup: bool = True) -> bool:
    """
    Convert a single GenExpr file to JSON patcher format.

    Args:
        filepath: Path to the .genjit file
        backup: If True, create a .genjit.bak backup

    Returns:
        True if conversion was successful

    """
    print(f"Processing: {filepath.name}")

    content = filepath.read_text()

    # Check if already in JSON format
    if is_json_patcher(content):
        print(f"  Skipping: already in JSON patcher format")
        return True

    # Parse the GenExpr content
    try:
        params, code = parse_genexpr(content)
    except Exception as e:
        print(f"  Error parsing: {e}")
        return False

    if not code:
        print(f"  Error: no code found after parsing")
        return False

    print(f"  Found {len(params)} param(s): {[p[0] for p in params]}")

    # Generate JSON patcher
    title = filepath.name
    patcher_json = generate_patcher_json(params, code, title)

    # Create backup if requested
    if backup:
        backup_path = filepath.with_suffix(".genjit.bak")
        shutil.copy2(filepath, backup_path)
        print(f"  Backup: {backup_path.name}")

    # Write the JSON output
    json_content = json.dumps(patcher_json, indent="\t")
    filepath.write_text(json_content)
    print(f"  Converted successfully")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert GenExpr text files to Max/MSP JSON patcher format"
    )
    parser.add_argument(
        "files", nargs="+", type=Path, help="GenExpr files to convert (.genjit)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .bak backup files (default: True)",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup files"
    )

    args = parser.parse_args()
    backup = not args.no_backup

    success_count = 0
    fail_count = 0

    for filepath in args.files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            fail_count += 1
            continue

        if filepath.suffix != ".genjit":
            print(f"Skipping non-.genjit file: {filepath}")
            continue

        if convert_file(filepath, backup=backup):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())

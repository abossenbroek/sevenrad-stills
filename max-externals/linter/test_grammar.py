#!/usr/bin/env python3
"""Test script to verify the GenExpr Lark grammar can parse all production shaders."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lark import Lark, exceptions


def extract_code_from_genjit(filepath: Path) -> str:
    """Extract the code field from a .genjit JSON file."""
    with open(filepath) as f:
        data: dict[str, Any] = json.load(f)

    # Navigate to the codebox object and extract the code field
    for box in data["patcher"]["boxes"]:
        if "box" in box and box["box"].get("maxclass") == "codebox":
            code: str = box["box"]["code"]
            return code

    raise ValueError(f"No codebox found in {filepath}")


def main() -> None:
    # Path to grammar file
    grammar_path = Path(__file__).parent / "grammars" / "genexpr.lark"

    # Path to shader directory
    code_dir = Path(__file__).parent.parent / "code"

    # Load the Lark grammar
    print(f"Loading grammar from: {grammar_path}")
    parser = Lark.open(str(grammar_path), parser="lalr")

    # Find all .genjit files
    shader_files = sorted(code_dir.glob("*.genjit"))
    print(f"\nFound {len(shader_files)} shader files\n")

    # Test each shader
    failed = []
    passed = []

    for shader_file in shader_files:
        shader_name = shader_file.stem
        try:
            # Extract code from JSON
            code = extract_code_from_genjit(shader_file)

            # Parse the code
            parser.parse(code)

            print(f"✓ {shader_name}")
            passed.append(shader_name)

        except exceptions.LarkError as e:
            print(f"✗ {shader_name}")
            print(f"  Error: {e}")
            failed.append((shader_name, str(e)))
        except Exception as e:
            print(f"✗ {shader_name}")
            print(f"  Unexpected error: {e}")
            failed.append((shader_name, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed shaders:")
        for name, error in failed:
            print(f"  - {name}")
            print(f"    {error[:100]}...")
        sys.exit(1)
    else:
        print("\nAll shaders parsed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

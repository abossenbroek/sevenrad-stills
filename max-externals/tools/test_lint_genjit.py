#!/usr/bin/env python3
"""
Test suite for genjit linter.

Creates various malformed .genjit files and verifies the linter catches them.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def create_valid_genjit() -> dict:
    """Create a valid .genjit structure."""
    return {
        "patcher": {
            "fileversion": 1,
            "appversion": {"major": 8, "minor": 0, "revision": 0},
            "boxes": [
                {
                    "box": {
                        "id": "obj-1",
                        "maxclass": "newobj",
                        "numinlets": 0,
                        "numoutlets": 1,
                        "outlettype": [""],
                        "patching_rect": [78.0, 28.0, 30.0, 20.0],
                        "text": "in 1",
                    }
                },
                {
                    "box": {
                        "id": "obj-2",
                        "maxclass": "newobj",
                        "numinlets": 0,
                        "numoutlets": 1,
                        "outlettype": [""],
                        "patching_rect": [290.0, 32.0, 99.0, 20.0],
                        "text": "param factor 1.0",
                    }
                },
                {
                    "box": {
                        "id": "obj-3",
                        "maxclass": "codebox",
                        "code": "// GenExpr shader\ncolor = sample(in1, norm);\nout1 = color * factor;\n",
                        "numinlets": 2,
                        "numoutlets": 1,
                        "outlettype": [""],
                        "patching_rect": [78.0, 90.0, 444.0, 337.0],
                    }
                },
                {
                    "box": {
                        "id": "obj-4",
                        "maxclass": "newobj",
                        "numinlets": 1,
                        "numoutlets": 0,
                        "patching_rect": [78.0, 442.0, 37.0, 20.0],
                        "text": "out 1",
                    }
                },
            ],
            "lines": [
                {
                    "patchline": {
                        "source": ["obj-1", 0],
                        "destination": ["obj-3", 0],
                        "disabled": 0,
                        "hidden": 0,
                    }
                },
                {
                    "patchline": {
                        "source": ["obj-2", 0],
                        "destination": ["obj-3", 1],
                        "disabled": 0,
                        "hidden": 0,
                    }
                },
                {
                    "patchline": {
                        "source": ["obj-3", 0],
                        "destination": ["obj-4", 0],
                        "disabled": 0,
                        "hidden": 0,
                    }
                },
            ],
            "dependency_cache": [
                {"name": "codebox.mxo", "type": "iLaX"},
                {"name": "param.mxo", "type": "iLaX"},
            ],
        }
    }


def test_case(name: str, data: dict | str, should_fail: bool = True) -> bool:
    """
    Test a single case.

    Args:
        name: Test case name
        data: JSON data or plain text
        should_fail: Whether linter should fail (True) or pass (False)

    Returns:
        True if test passed, False otherwise

    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".genjit", delete=False) as f:
        if isinstance(data, dict):
            json.dump(data, f)
        else:
            f.write(data)
        filepath = Path(f.name)

    try:
        result = subprocess.run(
            ["python", "tools/lint_genjit.py", str(filepath)],
            capture_output=True,
            text=True,
        )

        passed = result.returncode == 0
        test_passed = passed != should_fail

        if test_passed:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            print(
                f"  Expected {'FAIL' if should_fail else 'PASS'}, got {'PASS' if passed else 'FAIL'}"
            )
            if result.stdout:
                print(f"  Output: {result.stdout[:200]}")

        return test_passed

    finally:
        filepath.unlink()


def main() -> int:
    """Run all tests."""
    print("Running genjit linter test suite...\n")

    tests_passed = 0
    tests_total = 0

    # Test 1: Valid genjit file
    tests_total += 1
    if test_case("Valid GenExpr shader", create_valid_genjit(), should_fail=False):
        tests_passed += 1

    # Test 2: Not JSON at all
    tests_total += 1
    if test_case(
        "Not JSON - plain text",
        "This is not JSON at all\nJust plain text",
        should_fail=True,
    ):
        tests_passed += 1

    # Test 3: GenExpr code instead of JSON
    tests_total += 1
    if test_case(
        "GenExpr code instead of JSON",
        "// GenExpr shader\nParam factor(1.0);\nout1 = sample(in1, norm);",
        should_fail=True,
    ):
        tests_passed += 1

    # Test 4: Missing patcher key
    tests_total += 1
    if test_case("Missing 'patcher' key", {"wrong_key": {}}, should_fail=True):
        tests_passed += 1

    # Test 5: Missing fileversion
    tests_total += 1
    data = create_valid_genjit()
    del data["patcher"]["fileversion"]
    if test_case("Missing fileversion", data, should_fail=True):
        tests_passed += 1

    # Test 6: Missing boxes
    tests_total += 1
    data = create_valid_genjit()
    del data["patcher"]["boxes"]
    if test_case("Missing boxes array", data, should_fail=True):
        tests_passed += 1

    # Test 7: Empty boxes
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["boxes"] = []
    if test_case("Empty boxes array", data, should_fail=True):
        tests_passed += 1

    # Test 8: Missing 'in 1' object
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["boxes"] = [
        b for b in data["patcher"]["boxes"] if b["box"].get("text") != "in 1"
    ]
    if test_case("Missing 'in 1' object", data, should_fail=True):
        tests_passed += 1

    # Test 9: Missing 'out 1' object
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["boxes"] = [
        b for b in data["patcher"]["boxes"] if b["box"].get("text") != "out 1"
    ]
    if test_case("Missing 'out 1' object", data, should_fail=True):
        tests_passed += 1

    # Test 10: Missing codebox
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["boxes"] = [
        b for b in data["patcher"]["boxes"] if b["box"].get("maxclass") != "codebox"
    ]
    if test_case("Missing codebox", data, should_fail=True):
        tests_passed += 1

    # Test 11: Codebox with empty code
    tests_total += 1
    data = create_valid_genjit()
    for box in data["patcher"]["boxes"]:
        if box["box"].get("maxclass") == "codebox":
            box["box"]["code"] = ""
    if test_case("Codebox with empty code", data, should_fail=True):
        tests_passed += 1

    # Test 12: Codebox without output assignment
    tests_total += 1
    data = create_valid_genjit()
    for box in data["patcher"]["boxes"]:
        if box["box"].get("maxclass") == "codebox":
            box["box"]["code"] = "// No output\ncolor = sample(in1, norm);\n"
    if test_case("Codebox without output", data, should_fail=True):
        tests_passed += 1

    # Test 13: Invalid patchline reference
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["lines"][0]["patchline"]["source"] = ["obj-999", 0]
    if test_case("Invalid patchline source reference", data, should_fail=True):
        tests_passed += 1

    # Test 14: No connection from input to codebox
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["lines"] = [
        line
        for line in data["patcher"]["lines"]
        if line["patchline"]["destination"][0] != "obj-3"
        or line["patchline"]["source"][0] != "obj-1"
    ]
    if test_case("No input-to-codebox connection", data, should_fail=True):
        tests_passed += 1

    # Test 15: No connection from codebox to output
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["lines"] = [
        line
        for line in data["patcher"]["lines"]
        if line["patchline"]["destination"][0] != "obj-4"
    ]
    if test_case("No codebox-to-output connection", data, should_fail=True):
        tests_passed += 1

    # Test 16: Param with wrong inlet count
    tests_total += 1
    data = create_valid_genjit()
    for box in data["patcher"]["boxes"]:
        if box["box"].get("text", "").startswith("param"):
            box["box"]["numinlets"] = 1  # Should be 0
    if test_case("Param with wrong inlet count", data, should_fail=True):
        tests_passed += 1

    # Test 17: Valid GLSL shader
    tests_total += 1
    data = create_valid_genjit()
    for box in data["patcher"]["boxes"]:
        if box["box"].get("maxclass") == "codebox":
            box["box"]["code"] = """
<jit.gl.pix>
<program name="fp" type="fragment">
uniform sampler2DRect tex0;
varying vec2 texcoord0;
void main() {
    gl_FragColor = texture2DRect(tex0, texcoord0);
}
</program>
</jit.gl.pix>
"""
    if test_case("Valid GLSL shader", data, should_fail=False):
        tests_passed += 1

    # Test 18: GLSL without gl_FragColor
    tests_total += 1
    data = create_valid_genjit()
    for box in data["patcher"]["boxes"]:
        if box["box"].get("maxclass") == "codebox":
            box["box"]["code"] = """<jit.gl.pix>
<program name="fp" type="fragment">
#version 120
void main() {
    // Missing gl_FragColor assignment
}
</program>
</jit.gl.pix>"""
    if test_case("GLSL without gl_FragColor", data, should_fail=True):
        tests_passed += 1

    # Test 19: Invalid box ID format
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["boxes"][0]["box"]["id"] = "invalid-id"
    if test_case("Invalid box ID format", data, should_fail=True):
        tests_passed += 1

    # Test 20: Patchline with invalid format
    tests_total += 1
    data = create_valid_genjit()
    data["patcher"]["lines"][0]["patchline"]["source"] = "obj-1"  # Should be array
    if test_case("Patchline with invalid format", data, should_fail=True):
        tests_passed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {tests_total - tests_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

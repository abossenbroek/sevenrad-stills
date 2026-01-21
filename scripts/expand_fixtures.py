#!/usr/bin/env python3
"""Expand all .toe fixtures to .toe.dir format for CI testing.

This script:
1. Expands all .toe files in the fixtures directory using toeexpand
2. Creates a zip archive of all .toe.dir directories for use in CI

Requirements:
- TouchDesigner must be installed (provides toeexpand command)
- Run this script locally before committing for CI

Usage:
    python scripts/expand_fixtures.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "docs/touchdesigner/fixtures/projects"
OUTPUT_ZIP = FIXTURES_DIR / "fixtures-expanded.zip"


def find_toeexpand() -> str | None:
    """Find the toeexpand executable."""
    # Common locations for toeexpand
    candidates = [
        "toeexpand",  # In PATH
        "/Applications/TouchDesigner.app/Contents/MacOS/toeexpand",  # macOS
        "C:/Program Files/Derivative/TouchDesigner/bin/toeexpand.exe",  # Windows
    ]

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--help"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 or b"usage" in result.stdout.lower():
                return candidate
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue

    return None


def expand_toe_file(toeexpand: str, toe_file: Path) -> bool:
    """Expand a single .toe file to .toe.dir format."""
    toe_dir = toe_file.parent / (toe_file.name + ".dir")

    if toe_dir.exists():
        print(f"  Skipping (already expanded): {toe_file.name}")
        return True

    print(f"  Expanding: {toe_file.name}")
    try:
        result = subprocess.run(
            [toeexpand, str(toe_file)],
            capture_output=True,
            timeout=60,
            cwd=toe_file.parent,
        )
        # toeexpand writes success messages to stderr, so check if .toe.dir was created
        if toe_dir.exists():
            return True
        # Only report error if directory wasn't created
        if result.returncode != 0:
            print(f"    ERROR (exit {result.returncode}): {result.stderr.decode()}")
        else:
            print(f"    ERROR: Directory not created after expansion")
        return False
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Timeout expanding {toe_file.name}")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def create_zip(toe_dirs: list[Path]) -> None:
    """Create a zip archive of all .toe.dir directories."""
    print(f"\nCreating zip archive: {OUTPUT_ZIP.name}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for toe_dir in toe_dirs:
            for file in toe_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(FIXTURES_DIR)
                    zf.write(file, arcname)
                    print(f"  Added: {arcname}")

    size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
    print(f"\nCreated: {OUTPUT_ZIP} ({size_mb:.1f} MB)")


def main() -> int:
    """Main entry point."""
    print("TouchDesigner Fixture Expansion Script")
    print("=" * 40)

    # Find toeexpand
    toeexpand = find_toeexpand()
    if not toeexpand:
        print(
            "\nERROR: toeexpand not found. Please install TouchDesigner "
            "or add toeexpand to your PATH."
        )
        return 1

    print(f"Using toeexpand: {toeexpand}")

    # Find all .toe files
    if not FIXTURES_DIR.exists():
        print(f"\nERROR: Fixtures directory not found: {FIXTURES_DIR}")
        return 1

    toe_files = sorted(FIXTURES_DIR.glob("*.toe"))
    print(f"\nFound {len(toe_files)} .toe files in {FIXTURES_DIR}")

    if not toe_files:
        print("No .toe files to expand.")
        return 0

    # Expand all .toe files
    print("\nExpanding .toe files:")
    failed = []
    for toe_file in toe_files:
        if not expand_toe_file(toeexpand, toe_file):
            failed.append(toe_file.name)

    if failed:
        print(f"\nWARNING: Failed to expand {len(failed)} files:")
        for name in failed:
            print(f"  - {name}")

    # Collect all .toe.dir directories
    toe_dirs = sorted(FIXTURES_DIR.glob("*.toe.dir"))
    print(f"\nFound {len(toe_dirs)} .toe.dir directories")

    if not toe_dirs:
        print("No .toe.dir directories to zip.")
        return 1

    # Create zip archive
    create_zip(toe_dirs)

    print("\nDone! Next steps:")
    print(f"  1. Verify the zip: unzip -l {OUTPUT_ZIP}")
    print("  2. Commit the zip: git add -f", OUTPUT_ZIP.name)
    print("  3. Push to trigger CI")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

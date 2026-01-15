#!/usr/bin/env python3
"""
TouchDesigner Test Harness Build Script

Collapses shader_test_harness.toe.dir to shader_test_harness.toe using
TouchDesigner's toecollapse utility. Includes rebuild detection to skip
unnecessary rebuilds.

Usage:
    python build_test_harness.py          # Rebuild only if needed
    python build_test_harness.py --force  # Force rebuild
    python build_test_harness.py --expand # Expand .toe back to .toe.dir

Requirements:
    - TouchDesigner installed at /Applications/TouchDesigner.app (macOS)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# TouchDesigner tool paths (macOS)
TD_APP = Path("/Applications/TouchDesigner.app/Contents/MacOS")
TOECOLLAPSE = TD_APP / "toecollapse"
TOEEXPAND = TD_APP / "toeexpand"

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent / "fixtures" / "projects"
HARNESS_DIR = PROJECT_DIR / "shader_test_harness.toe.dir"
HARNESS_TOE = PROJECT_DIR / "shader_test_harness.toe"


def check_td_tools() -> bool:
    """Verify TouchDesigner tools are available."""
    if not TOECOLLAPSE.exists():
        print(f"ERROR: toecollapse not found at {TOECOLLAPSE}")
        print("Ensure TouchDesigner is installed at /Applications/TouchDesigner.app")
        return False
    return True


def needs_rebuild() -> bool:
    """Check if .toe.dir has been modified since .toe was last built."""
    if not HARNESS_TOE.exists():
        return True

    if not HARNESS_DIR.exists():
        print(f"ERROR: Source directory not found: {HARNESS_DIR}")
        return False

    toe_mtime = HARNESS_TOE.stat().st_mtime

    for f in HARNESS_DIR.rglob("*"):
        if f.is_file() and f.stat().st_mtime > toe_mtime:
            return True

    return False


def regenerate_toc() -> None:
    """Regenerate the .toc manifest file from current .toe.dir contents."""
    toc_path = PROJECT_DIR / "shader_test_harness.toe.toc"

    files = []
    for f in HARNESS_DIR.rglob("*"):
        if f.is_file() and f.name != ".DS_Store":
            rel_path = f.relative_to(HARNESS_DIR)
            files.append(str(rel_path))

    files.sort()
    toc_path.write_text("\n".join(files) + "\n")


def collapse(verbose: bool = False) -> int:
    """Collapse .toe.dir to .toe binary format."""
    if not HARNESS_DIR.exists():
        print(f"ERROR: Source directory not found: {HARNESS_DIR}")
        return 1

    if not check_td_tools():
        return 1

    # Regenerate .toc manifest (CRITICAL - toecollapse needs this!)
    if verbose:
        print("Regenerating .toc manifest...")
    regenerate_toc()

    if verbose:
        print(f"Collapsing: {HARNESS_DIR}")
        print(f"       To: {HARNESS_TOE}")

    # toecollapse takes single argument: the .toe.dir directory
    # Output is automatically the same name without .dir suffix
    result = subprocess.run(
        [str(TOECOLLAPSE), str(HARNESS_DIR)],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,  # Run from project directory
    )

    if result.returncode != 0:
        print(f"ERROR: toecollapse failed:")
        print(result.stderr)
        return 1

    print(f"Built: {HARNESS_TOE}")
    return 0


def expand(verbose: bool = False) -> int:
    """Expand .toe to .toe.dir text format."""
    if not HARNESS_TOE.exists():
        print(f"ERROR: Binary file not found: {HARNESS_TOE}")
        return 1

    if not check_td_tools():
        return 1

    if verbose:
        print(f"Expanding: {HARNESS_TOE}")
        print(f"       To: {HARNESS_DIR}")

    result = subprocess.run(
        [str(TOEEXPAND), str(HARNESS_TOE)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: toeexpand failed:")
        print(result.stderr)
        return 1

    print(f"Expanded: {HARNESS_DIR}")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build TouchDesigner shader test harness from .toe.dir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build if needed
    python build_test_harness.py

    # Force rebuild
    python build_test_harness.py --force

    # Expand .toe back to .toe.dir (after TD edits)
    python build_test_harness.py --expand
        """,
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild even if up to date",
    )

    parser.add_argument(
        "--expand",
        "-e",
        action="store_true",
        help="Expand .toe to .toe.dir (reverse operation)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.expand:
        return expand(args.verbose)

    if args.force or needs_rebuild():
        return collapse(args.verbose)
    else:
        print(f"Up to date: {HARNESS_TOE}")
        print("Use --force to rebuild anyway")
        return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Graph-based linter for .maxhelp files.

Validates that help patchers have correct structure, initialization order,
complete signal flow, proper UI components for shader parameters, and proper
UI layout using networkx graph analysis.

Validation Features:
    - Context naming conventions (underscores, not dots)
    - OpenGL context initialization order
    - Complete GPU texture signal flow
    - Parameter UI controls and connections
    - UI component overlap detection
    - Dead code detection
    - Feedback loop validation
    - Trigger order validation

Usage:
    python lint_maxhelp.py help/*.maxhelp
    python lint_maxhelp.py --strict help/sr.bandswap.maxhelp
    python lint_maxhelp.py --verbose help/*.maxhelp

Exit codes:
    0: All validations passed
    1: One or more validations failed

This module is a backward-compatible wrapper that imports the modular
linter implementation from max_linter.validators.maxhelp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add path to linter package if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "linter" / "src"))

# Import from the modular linter package
# Re-export for backward compatibility
from max_linter.constants import JITTER_DISPLAY_SINKS, KNOWN_MAX_OBJECTS
from max_linter.lint_error import LintError
from max_linter.lint_graph import LintGraph
from max_linter.types import (
    TYPE_COMPATIBLE,
    JitterType,
    Severity,
    types_compatible,
)
from max_linter.validators.maxhelp import MaxhelpLinter

__all__ = [
    "JITTER_DISPLAY_SINKS",
    "KNOWN_MAX_OBJECTS",
    "TYPE_COMPATIBLE",
    "JitterType",
    "LintError",
    "LintGraph",
    "MaxhelpLinter",
    "Severity",
    "types_compatible",
]


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate .maxhelp files with graph-based analysis"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Path(s) to .maxhelp files to validate",
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
        help="Show info messages and OK status",
    )
    args = parser.parse_args()

    linter = MaxhelpLinter(strict=args.strict, verbose=args.verbose)
    has_errors = False
    files_checked = 0
    files_failed = 0

    for filepath in args.files:
        # Expand directory to all .maxhelp files, or use single file
        files = list(filepath.glob("*.maxhelp")) if filepath.is_dir() else [filepath]

        for f in files:
            files_checked += 1
            linter.validate_file(f)
            linter.print_results(f)

            if linter.has_errors():
                has_errors = True
                files_failed += 1

    # Summary
    print(f"\nValidated {files_checked} file(s), {files_failed} with issues")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

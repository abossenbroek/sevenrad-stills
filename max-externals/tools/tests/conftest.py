# ruff: noqa: S101
"""
Common test fixtures and imports for MaxhelpLinter tests.

This module provides shared fixtures and imports used across all test modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from lint_maxhelp import (
    TYPE_COMPATIBLE,
    JitterType,
    LintGraph,
    MaxhelpLinter,
    types_compatible,
)

# Export for test modules
__all__ = [
    "TYPE_COMPATIBLE",
    "Any",
    "JitterType",
    "LintGraph",
    "MaxhelpLinter",
    "Path",
    "create_test_patcher",
    "json",
    "nx",
    "pytest",
    "tempfile",
    "types_compatible",
]


def create_test_patcher(
    boxes: list[dict[str, Any]], lines: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create a minimal test patcher JSON structure."""
    if lines is None:
        lines = []

    return {
        "patcher": {
            "fileversion": 1,
            "boxes": [{"box": box} for box in boxes],
            "lines": [{"patchline": line} for line in lines],
        }
    }

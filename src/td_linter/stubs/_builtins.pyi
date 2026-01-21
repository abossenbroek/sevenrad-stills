"""
TouchDesigner built-in objects stubs for type checking.

These are the special global objects available in TouchDesigner Python scripts.

Note: Names follow TouchDesigner API conventions (camelCase) which differ from PEP 8.
"""
# ruff: noqa: N802, N803, N816, ANN401

from typing import Any, Callable, overload

from td import UI, AbsTime, App, Op, Par, Project

# Global operator lookup function
@overload
def op(path: str) -> Op | None: ...
@overload
def op(path: str, *paths: str) -> list[Op | None]: ...
def ops(*patterns: str) -> list[Op]:
    """
    Find operators matching patterns.

    Examples:
        ops('*')
        ops('text*', 'geo*')

    """
    ...

# Current operator
me: Op
"""Reference to the current operator (the one containing this script)."""

# Parent operator
parent: Callable[[], Op | None]
"""Get the parent operator. Usage: parent()"""

# Module extensions
mod: Any
"""Access to module extensions (MOD class)."""

ext: Any
"""Access to extensions (EXT class)."""

# Parameters
par: Any
"""Quick access to parameters. Usage: par.paramname"""

pars: list[Par]
"""List of all parameters on the current operator."""

# Storage
storage: dict[str, Any]
"""Operator storage dictionary."""

def fetch(key: str, default: Any = None, storeDefault: bool = False) -> Any:
    """Fetch a value from storage."""
    ...

def store(key: str, value: Any) -> None:
    """Store a value in storage."""
    ...

# Project/root
project: Project
"""Current project object."""

root: Op
"""Root operator (/)."""

# Time
absTime: AbsTime
"""Absolute time object for frame/time queries."""

# Application
app: App
"""Application object with version/system info."""

# UI
ui: UI
"""User interface object."""

# System info
sysinfo: Any
"""System information object."""

monitors: Any
"""Monitor configuration object."""

# Parent parameters (for extensions)
ipar: Any
"""Internal parent parameters."""

iop: Any
"""Internal parent operators."""

# Functions
def run(
    cmd: str,
    *args: Any,
    delayFrames: int = 0,
    delayMilliSeconds: int = 0,
    delayRef: Op | None = None,
) -> None:
    """Run a command with optional delay."""
    ...

def cook(op: Op | None = None) -> None:
    """Force cook an operator."""
    ...

def debug(*args: Any) -> None:
    """Print debug message to textport."""
    ...

def passive(op: Op) -> None:
    """Mark operator as passive (won't cause recooks)."""
    ...

def var(name: str, default: Any = None) -> Any:
    """Get a project variable."""
    ...

def vardict() -> dict[str, Any]:
    """Get all project variables as dictionary."""
    ...

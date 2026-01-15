"""
TouchDesigner td module stubs for type checking.

These stubs provide type information for the main TouchDesigner Python API.
Use with mypy, pyright, or IDE autocompletion.

Note: Names follow TouchDesigner API conventions (camelCase) which differ from PEP 8.
"""
# ruff: noqa: N802, N803, ANN401

from typing import Any, Iterator, Literal, overload

# Operator families
OperatorFamily = Literal["TOP", "CHOP", "SOP", "DAT", "COMP", "MAT", "POP"]

class Par:
    """TouchDesigner parameter object."""

    @property
    def val(self) -> Any:
        """Current parameter value."""
        ...

    @val.setter
    def val(self, value: Any) -> None: ...
    @property
    def eval(self) -> Any:
        """Evaluated parameter value."""
        ...

    @property
    def expr(self) -> str:
        """Parameter expression string."""
        ...

    @expr.setter
    def expr(self, value: str) -> None: ...
    @property
    def mode(self) -> int:
        """Parameter mode (0=constant, 49=expression, etc.)."""
        ...

    @mode.setter
    def mode(self, value: int) -> None: ...
    @property
    def name(self) -> str:
        """Parameter name."""
        ...

    @property
    def label(self) -> str:
        """Parameter label."""
        ...

    @property
    def default(self) -> Any:
        """Default parameter value."""
        ...

    @property
    def min(self) -> float:
        """Minimum parameter value."""
        ...

    @property
    def max(self) -> float:
        """Maximum parameter value."""
        ...

    @property
    def owner(self) -> "Op":
        """Owner operator of this parameter."""
        ...

class Op:
    """TouchDesigner operator base class."""

    @property
    def name(self) -> str:
        """Operator name."""
        ...

    @name.setter
    def name(self, value: str) -> None: ...
    @property
    def path(self) -> str:
        """Full operator path."""
        ...

    @property
    def parent(self) -> "Op | None":
        """Parent operator."""
        ...

    @property
    def type(self) -> str:
        """Operator type name."""
        ...

    @property
    def family(self) -> OperatorFamily:
        """Operator family (TOP, CHOP, etc.)."""
        ...

    @property
    def storage(self) -> dict[str, Any]:
        """Operator storage dictionary."""
        ...

    @property
    def par(self) -> "ParGroup":
        """Access to operator parameters."""
        ...

    @property
    def pars(self) -> list[Par]:
        """List of all parameters."""
        ...

    @overload
    def op(self, path: str) -> "Op | None": ...
    @overload
    def op(self, path: str, *paths: str) -> "list[Op | None]": ...
    def ops(self, *patterns: str) -> list["Op"]:
        """Find operators matching patterns."""
        ...

    def cook(self, force: bool = False) -> None:
        """Force cook the operator."""
        ...

    def destroy(self) -> None:
        """Destroy the operator."""
        ...

    def copy(self, dest: "Op", name: str | None = None) -> "Op":
        """Copy operator to destination."""
        ...

class ParGroup:
    """Group of parameters accessible by name."""

    def __getattr__(self, name: str) -> Par: ...
    def __iter__(self) -> Iterator[Par]: ...

class AbsTime:
    """Absolute time object."""

    @property
    def frame(self) -> float:
        """Current frame number."""
        ...

    @property
    def seconds(self) -> float:
        """Current time in seconds."""
        ...

    @property
    def realSeconds(self) -> float:
        """Real-time seconds since startup."""
        ...

    @property
    def step(self) -> int:
        """Current step (ticks)."""
        ...

class Project:
    """Project object."""

    @property
    def name(self) -> str:
        """Project name."""
        ...

    @property
    def folder(self) -> str:
        """Project folder path."""
        ...

    @property
    def saveVersion(self) -> str:
        """Version project was saved with."""
        ...

    @property
    def paths(self) -> dict[str, str]:
        """Project paths dictionary."""
        ...

class App:
    """Application object."""

    @property
    def version(self) -> str:
        """TouchDesigner version string."""
        ...

    @property
    def build(self) -> str:
        """TouchDesigner build number."""
        ...

    @property
    def launchTime(self) -> float:
        """Launch time in seconds since epoch."""
        ...

    @property
    def osName(self) -> str:
        """Operating system name."""
        ...

class UI:
    """UI object."""

    @property
    def status(self) -> str:
        """Status bar text."""
        ...

    @status.setter
    def status(self, value: str) -> None: ...
    def messageBox(self, title: str, message: str, buttons: list[str] = ...) -> int:
        """Display a message box."""
        ...

# Module-level functions
def run(
    cmd: str,
    *args: Any,
    delayFrames: int = 0,
    delayMilliSeconds: int = 0,
    delayRef: Op | None = None,
) -> None:
    """Run a command with optional delay."""
    ...

def passive(op: Op) -> None:
    """Mark operator as passive."""
    ...

def debug(*args: Any) -> None:
    """Print debug message."""
    ...

def var(name: str, default: Any = None) -> Any:
    """Get a project variable."""
    ...

def vardict() -> dict[str, Any]:
    """Get all project variables."""
    ...

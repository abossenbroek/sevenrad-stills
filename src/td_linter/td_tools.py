"""
TouchDesigner tool discovery and integration utilities.

Provides functions to locate TouchDesigner installations and run
toeexpand/toecollapse commands for .toe file handling.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class TouchDesignerNotFoundError(Exception):
    """Raised when TouchDesigner installation cannot be found."""

    pass


class ToeExpandError(Exception):
    """Raised when toeexpand fails."""

    pass


class ToeCollapseError(Exception):
    """Raised when toecollapse fails."""

    pass


def _get_common_td_locations() -> list[Path]:
    """
    Get common TouchDesigner installation locations for the current platform.

    Returns:
        List of potential TouchDesigner binary directories.

    """
    system = platform.system()
    locations: list[Path] = []

    if system == "Darwin":  # macOS
        # Check /Applications for TouchDesigner*.app
        apps_dir = Path("/Applications")
        if apps_dir.exists():
            # Find all TouchDesigner app bundles, sorted by name (newest versions last)
            td_apps = sorted(apps_dir.glob("TouchDesigner*.app"))
            for app in reversed(td_apps):  # Prefer newer versions
                bin_path = app / "Contents" / "MacOS"
                if bin_path.exists():
                    locations.append(bin_path)

    elif system == "Windows":
        # Check Program Files locations
        program_files = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        ]
        for pf in program_files:
            derivative_dir = pf / "Derivative"
            if derivative_dir.exists():
                # Find all TouchDesigner installations
                td_dirs = sorted(derivative_dir.glob("TouchDesigner*"))
                for td_dir in reversed(td_dirs):
                    bin_path = td_dir / "bin"
                    if bin_path.exists():
                        locations.append(bin_path)

    elif system == "Linux":
        # Check common Linux installation locations
        linux_locations = [
            Path("/opt"),
            Path.home() / ".local" / "share" / "Derivative",
            Path("/usr/local"),
        ]
        for base in linux_locations:
            if base.exists():
                td_dirs = sorted(base.glob("TouchDesigner*"))
                for td_dir in reversed(td_dirs):
                    bin_path = td_dir / "bin"
                    if bin_path.exists():
                        locations.append(bin_path)
                    # Some installations have binaries directly in the folder
                    if (td_dir / "toeexpand").exists():
                        locations.append(td_dir)

    return locations


def find_touchdesigner(td_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find TouchDesigner installation directory containing toeexpand/toecollapse.

    Search order:
    1. Explicit td_path argument (if provided)
    2. TOUCHDESIGNER_PATH environment variable
    3. Common installation locations for the current platform
    4. PATH lookup for toeexpand executable

    Args:
        td_path: Optional explicit path to TouchDesigner bin directory.

    Returns:
        Path to directory containing toeexpand/toecollapse, or None if not found.

    """
    # 1. Check explicit path
    if td_path is not None:
        td_path = Path(td_path)
        if _has_td_tools(td_path):
            return td_path
        # Maybe they passed the app bundle, not the MacOS dir
        macos_path = td_path / "Contents" / "MacOS"
        if macos_path.exists() and _has_td_tools(macos_path):
            return macos_path

    # 2. Check environment variable
    env_path = os.environ.get("TOUCHDESIGNER_PATH")
    if env_path:
        env_path = Path(env_path)
        if _has_td_tools(env_path):
            return env_path
        macos_path = env_path / "Contents" / "MacOS"
        if macos_path.exists() and _has_td_tools(macos_path):
            return macos_path

    # 3. Check common locations
    for location in _get_common_td_locations():
        if _has_td_tools(location):
            return location

    # 4. Check PATH
    toeexpand_path = shutil.which("toeexpand")
    if toeexpand_path:
        return Path(toeexpand_path).parent

    return None


def _has_td_tools(directory: Path) -> bool:
    """Check if directory contains toeexpand and toecollapse."""
    if not directory.is_dir():
        return False

    system = platform.system()
    if system == "Windows":
        return (directory / "toeexpand.exe").exists() and (
            directory / "toecollapse.exe"
        ).exists()
    else:
        return (directory / "toeexpand").exists() and (
            directory / "toecollapse"
        ).exists()


def get_toeexpand(td_path: Optional[Path] = None) -> Optional[Path]:
    """
    Get path to toeexpand executable.

    Args:
        td_path: Optional explicit path to TouchDesigner bin directory.

    Returns:
        Path to toeexpand executable, or None if not found.

    """
    td_dir = find_touchdesigner(td_path)
    if td_dir is None:
        return None

    system = platform.system()
    exe_name = "toeexpand.exe" if system == "Windows" else "toeexpand"
    exe_path = td_dir / exe_name

    return exe_path if exe_path.exists() else None


def get_toecollapse(td_path: Optional[Path] = None) -> Optional[Path]:
    """
    Get path to toecollapse executable.

    Args:
        td_path: Optional explicit path to TouchDesigner bin directory.

    Returns:
        Path to toecollapse executable, or None if not found.

    """
    td_dir = find_touchdesigner(td_path)
    if td_dir is None:
        return None

    system = platform.system()
    exe_name = "toecollapse.exe" if system == "Windows" else "toecollapse"
    exe_path = td_dir / exe_name

    return exe_path if exe_path.exists() else None


def expand_toe(
    toe_path: Path,
    output_dir: Optional[Path] = None,
    td_path: Optional[Path] = None,
    timeout: int = 60,
) -> Path:
    """
    Expand a .toe file to a .toe.dir directory.

    Args:
        toe_path: Path to the .toe file to expand.
        output_dir: Optional output directory. If not specified, creates
                   .toe.dir in the same directory as the .toe file.
        td_path: Optional explicit path to TouchDesigner bin directory.
        timeout: Command timeout in seconds.

    Returns:
        Path to the expanded .toe.dir directory.

    Raises:
        TouchDesignerNotFoundError: If toeexpand cannot be found.
        ToeExpandError: If expansion fails.
        FileNotFoundError: If toe_path doesn't exist.

    """
    toe_path = Path(toe_path).resolve()

    if not toe_path.exists():
        raise FileNotFoundError(f"File not found: {toe_path}")

    if not toe_path.suffix == ".toe":
        raise ValueError(f"Expected .toe file, got: {toe_path}")

    toeexpand = get_toeexpand(td_path)
    if toeexpand is None:
        raise TouchDesignerNotFoundError(
            "TouchDesigner not found. Set TOUCHDESIGNER_PATH environment variable "
            "or use --td-path option to specify the installation directory."
        )

    # Determine output directory
    if output_dir is None:
        # Default: create .toe.dir next to .toe file
        toe_dir = toe_path.parent / f"{toe_path.name}.dir"
    else:
        toe_dir = Path(output_dir) / f"{toe_path.name}.dir"

    # Run toeexpand
    try:
        result = subprocess.run(
            [str(toeexpand), str(toe_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=toe_path.parent,  # Run in same directory as .toe file
        )

        # toeexpand returns exit code 1 even on success, so check if directory exists
        if not toe_dir.exists():
            raise ToeExpandError(
                f"toeexpand failed to create {toe_dir}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )

        return toe_dir

    except subprocess.TimeoutExpired:
        raise ToeExpandError(f"toeexpand timed out after {timeout} seconds")
    except OSError as e:
        raise ToeExpandError(f"Failed to run toeexpand: {e}")


def collapse_toe(
    toe_dir: Path,
    td_path: Optional[Path] = None,
    timeout: int = 60,
) -> Path:
    """
    Collapse a .toe.dir directory back to a .toe file.

    Args:
        toe_dir: Path to the .toe.dir directory to collapse.
        td_path: Optional explicit path to TouchDesigner bin directory.
        timeout: Command timeout in seconds.

    Returns:
        Path to the collapsed .toe file.

    Raises:
        TouchDesignerNotFoundError: If toecollapse cannot be found.
        ToeCollapseError: If collapse fails.
        FileNotFoundError: If toe_dir doesn't exist.

    """
    toe_dir = Path(toe_dir).resolve()

    if not toe_dir.exists():
        raise FileNotFoundError(f"Directory not found: {toe_dir}")

    if not toe_dir.name.endswith(".toe.dir"):
        raise ValueError(f"Expected .toe.dir directory, got: {toe_dir}")

    toecollapse = get_toecollapse(td_path)
    if toecollapse is None:
        raise TouchDesignerNotFoundError(
            "TouchDesigner not found. Set TOUCHDESIGNER_PATH environment variable "
            "or use --td-path option to specify the installation directory."
        )

    # Determine output .toe file path
    toe_path = toe_dir.parent / toe_dir.name[:-4]  # Remove ".dir" suffix

    # Run toecollapse
    try:
        result = subprocess.run(
            [str(toecollapse), str(toe_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=toe_dir.parent,
        )

        if result.returncode != 0:
            raise ToeCollapseError(
                f"toecollapse failed with exit code {result.returncode}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )

        if not toe_path.exists():
            raise ToeCollapseError(
                f"toecollapse did not create {toe_path}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )

        return toe_path

    except subprocess.TimeoutExpired:
        raise ToeCollapseError(f"toecollapse timed out after {timeout} seconds")
    except OSError as e:
        raise ToeCollapseError(f"Failed to run toecollapse: {e}")

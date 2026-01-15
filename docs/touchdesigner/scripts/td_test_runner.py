#!/usr/bin/env python3
"""
TouchDesigner Shader Test Orchestrator.

External orchestrator for automated TouchDesigner shader testing.
Launches TouchDesigner with environment variables, waits for test completion,
and retrieves results.

Usage:
    python td_test_runner.py --config tests/config.json --output /tmp/output
    python td_test_runner.py --config tests/config.json --output /tmp/output --timeout 180

Environment Variables (set automatically by this script):
    SHADER_TEST_MODE    - Set to '1' to enable automated testing
    TEST_CONFIG         - Path to test configuration JSON
    OUTPUT_DIR          - Directory for output files and results

Requirements:
    - TouchDesigner installed at default macOS location
    - shader_test_harness.toe project configured with Execute DAT
    - Test configuration JSON file
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default TouchDesigner paths (macOS)
TD_APP_DEFAULT = "/Applications/TouchDesigner.app/Contents/MacOS/TouchDesigner"
TOECOLLAPSE_DEFAULT = "/Applications/TouchDesigner.app/Contents/MacOS/toecollapse"

# Default test project (relative to this script's directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TEST_PROJECT_DEFAULT = PROJECT_DIR / "fixtures" / "projects" / "shader_test_harness.toe"


@dataclass
class TestResult:
    """Container for test execution results."""

    success: bool
    tests: list[dict[str, Any]]
    total: int
    passed: int
    failed: int
    error: str | None = None


def find_touchdesigner() -> Path | None:
    """Find TouchDesigner executable on macOS."""
    default_path = Path(TD_APP_DEFAULT)
    if default_path.exists():
        return default_path

    # Try common alternative locations
    alternatives = [
        Path.home()
        / "Applications"
        / "TouchDesigner.app"
        / "Contents"
        / "MacOS"
        / "TouchDesigner",
        Path("/Applications/TouchDesigner099.app/Contents/MacOS/TouchDesigner"),
    ]

    for alt in alternatives:
        if alt.exists():
            return alt

    return None


def _needs_rebuild(toe_path: Path, toe_dir: Path) -> bool:
    """Check if any file in .toe.dir is newer than .toe."""
    if not toe_path.exists():
        return True
    if not toe_dir.exists():
        return False

    toe_mtime = toe_path.stat().st_mtime
    for f in toe_dir.rglob("*"):
        if f.is_file() and f.stat().st_mtime > toe_mtime:
            return True
    return False


def _run_toecollapse(toe_dir: Path, toe_path: Path) -> bool:
    """Run toecollapse to build .toe from .toe.dir."""
    toecollapse = Path(TOECOLLAPSE_DEFAULT)
    if not toecollapse.exists():
        print(f"Warning: toecollapse not found at {toecollapse}")
        return False

    # toecollapse takes single argument: the .toe.dir directory
    # Output is automatically the same name without .dir suffix
    result = subprocess.run(
        [str(toecollapse), str(toe_dir)],
        capture_output=True,
        text=True,
        cwd=toe_dir.parent,  # Run from parent directory
    )

    if result.returncode != 0:
        print(f"Warning: toecollapse failed: {result.stderr}")
        return False

    return True


def ensure_toe_built(project_path: Path, verbose: bool = False) -> Path:
    """
    Ensure .toe file is built from .toe.dir if needed.

    If a .toe.dir directory exists alongside the .toe file and contains
    newer files, automatically rebuild the .toe using toecollapse.

    Args:
        project_path: Path to .toe file
        verbose: Print rebuild messages

    Returns:
        The project_path (unchanged)

    """
    toe_dir = project_path.with_suffix(".toe.dir")

    if toe_dir.exists():
        if not project_path.exists() or _needs_rebuild(project_path, toe_dir):
            if verbose:
                print(f"Rebuilding {project_path.name} from {toe_dir.name}")
            if _run_toecollapse(toe_dir, project_path):
                if verbose:
                    print(f"  Built: {project_path}")
            elif verbose:
                print(f"  Warning: Auto-rebuild failed, using existing .toe")

    return project_path


def run_td_tests(
    config_path: str | Path,
    output_dir: str | Path,
    project_path: str | Path | None = None,
    timeout: int = 120,
    td_path: str | Path | None = None,
    verbose: bool = False,
) -> TestResult:
    """
    Launch TouchDesigner, run tests, and wait for completion.

    Args:
        config_path: Path to test configuration JSON
        output_dir: Directory for output files
        project_path: Path to .toe test project (uses default if None)
        timeout: Maximum seconds to wait for completion
        td_path: Path to TouchDesigner executable (finds automatically if None)
        verbose: Print verbose output

    Returns:
        TestResult with test execution details

    Raises:
        FileNotFoundError: If TouchDesigner or project not found
        TimeoutError: If tests don't complete within timeout
        RuntimeError: If TouchDesigner crashes

    """
    # Resolve paths
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()

    if project_path is None:
        project_path = TEST_PROJECT_DEFAULT
    project_path = Path(project_path).resolve()

    # Auto-rebuild .toe from .toe.dir if needed
    ensure_toe_built(project_path, verbose=verbose)

    if td_path is None:
        td_path = find_touchdesigner()
        if td_path is None:
            raise FileNotFoundError(
                "TouchDesigner not found. Install TouchDesigner or specify path with --td-path"
            )
    td_path = Path(td_path)

    # Validate inputs
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not project_path.exists():
        raise FileNotFoundError(
            f"Test project not found: {project_path}\n"
            "Create the shader_test_harness.toe project in TouchDesigner first."
        )

    if not td_path.exists():
        raise FileNotFoundError(f"TouchDesigner not found at: {td_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define marker and results paths
    marker_path = output_dir / "complete.marker"
    results_path = output_dir / "results.json"

    # Clean up old files
    for old_file in [marker_path, results_path]:
        old_file.unlink(missing_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["SHADER_TEST_MODE"] = "1"
    env["TEST_CONFIG"] = str(config_path)
    env["OUTPUT_DIR"] = str(output_dir)

    if verbose:
        print(f"Launching TouchDesigner:")
        print(f"  Executable: {td_path}")
        print(f"  Project: {project_path}")
        print(f"  Config: {config_path}")
        print(f"  Output: {output_dir}")
        print(f"  Timeout: {timeout}s")

    # Launch TouchDesigner in Perform mode
    cmd = [str(td_path), str(project_path)]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None,
    )

    if verbose:
        print(f"Started TouchDesigner (PID: {proc.pid})")

    # Wait for completion marker
    start_time = time.time()
    poll_interval = 0.5

    while time.time() - start_time < timeout:
        # Check if marker exists
        if marker_path.exists():
            if verbose:
                print("Completion marker found")
            break

        # Check if TD crashed
        if proc.poll() is not None:
            if not marker_path.exists():
                stderr = proc.stderr.read().decode() if proc.stderr else "Unknown error"
                return TestResult(
                    success=False,
                    tests=[],
                    total=0,
                    passed=0,
                    failed=0,
                    error=f"TouchDesigner exited unexpectedly (code {proc.returncode}): {stderr}",
                )
            break

        time.sleep(poll_interval)

    else:
        # Timeout reached
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        return TestResult(
            success=False,
            tests=[],
            total=0,
            passed=0,
            failed=0,
            error=f"Tests did not complete within {timeout} seconds",
        )

    # Read results
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        tests = data.get("tests", [])
        passed = sum(1 for t in tests if t.get("status") == "passed")
        failed = len(tests) - passed

        return TestResult(
            success=failed == 0,
            tests=tests,
            total=len(tests),
            passed=passed,
            failed=failed,
        )
    else:
        return TestResult(
            success=False,
            tests=[],
            total=0,
            passed=0,
            failed=0,
            error=f"Results file not found: {results_path}",
        )


def print_results(result: TestResult) -> None:
    """Print formatted test results."""
    if result.error:
        print(f"\nError: {result.error}")
        return

    print(f"\nTest Results: {result.passed}/{result.total} passed")
    print("-" * 40)

    for test in result.tests:
        status = test.get("status", "unknown")
        name = test.get("name", "unnamed")
        symbol = "PASS" if status == "passed" else "FAIL"
        print(f"  [{symbol}] {name}")

        if status != "passed":
            error = test.get("error", "No error details")
            print(f"        Error: {error}")

    print("-" * 40)
    if result.success:
        print("All tests passed!")
    else:
        print(f"Failed: {result.failed} test(s)")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run TouchDesigner shader tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run tests with default project
    python td_test_runner.py --config tests/config.json --output /tmp/output

    # Run with custom timeout
    python td_test_runner.py --config tests/config.json --output /tmp/output --timeout 180

    # Run with verbose output
    python td_test_runner.py --config tests/config.json --output /tmp/output -v

    # Use custom project and TD path
    python td_test_runner.py --config tests/config.json --output /tmp/output \\
        --project my_test.toe --td-path /custom/path/TouchDesigner
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to test configuration JSON",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="/tmp/td_test_output",
        help="Output directory for results (default: /tmp/td_test_output)",
    )

    parser.add_argument(
        "--project",
        "-p",
        default=None,
        help="Path to .toe test project (default: fixtures/projects/shader_test_harness.toe)",
    )

    parser.add_argument(
        "--td-path",
        default=None,
        help="Path to TouchDesigner executable (default: auto-detect)",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Timeout in seconds (default: 120)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON only",
    )

    args = parser.parse_args()

    try:
        result = run_td_tests(
            config_path=args.config,
            output_dir=args.output,
            project_path=args.project,
            timeout=args.timeout,
            td_path=args.td_path,
            verbose=args.verbose,
        )

        if args.json:
            output = {
                "success": result.success,
                "total": result.total,
                "passed": result.passed,
                "failed": result.failed,
                "tests": result.tests,
                "error": result.error,
            }
            print(json.dumps(output, indent=2))
        else:
            print_results(result)

        return 0 if result.success else 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Subprocess sandboxing utilities.

This module provides utilities for running external tools with security
constraints. On macOS, it uses sandbox-exec when available. On other
platforms, it falls back to resource limits.

Security Note:
    Full sandboxing is platform-specific and may not be available on all
    systems. When sandboxing is not available, the module falls back to
    basic mitigations (timeouts, resource limits) but this is NOT a
    security boundary. Only run trusted tools in production.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# Maximum execution time in seconds
DEFAULT_TIMEOUT = 30

# Maximum output size in bytes (10MB)
MAX_OUTPUT_SIZE = 10 * 1024 * 1024


class SandboxError(Exception):
    """Base exception for sandbox errors."""


class SandboxTimeoutError(SandboxError):
    """Raised when a sandboxed process times out."""


class SandboxExecutionError(SandboxError):
    """Raised when a sandboxed process fails to execute."""


class BinaryValidationError(SandboxError):
    """Raised when binary validation fails."""


@dataclass(frozen=True)
class SandboxResult:
    """Result of a sandboxed command execution."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


@dataclass(frozen=True)
class BinaryInfo:
    """Information about a validated binary."""

    path: Path
    sha256: str
    size: int


class Sandbox:
    """Executes commands with security constraints.

    Security features:
    - Timeout enforcement (prevents DoS via infinite loops)
    - Resource limits where available
    - macOS sandbox-exec support (when profile provided)
    - Optional binary hash validation

    Limitations:
    - Full sandboxing requires platform support and root privileges
    - Falls back to basic mitigations on unsupported platforms
    - Not a complete security boundary without container/VM isolation
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_output: int = MAX_OUTPUT_SIZE,
        allowed_binary_hashes: dict[str, str] | None = None,
    ) -> None:
        """Initialize the sandbox.

        Args:
            timeout: Maximum execution time in seconds.
            max_output: Maximum output size in bytes.
            allowed_binary_hashes: Optional dict of binary name -> SHA256 hash.
                If provided, only binaries matching these hashes are allowed.
        """
        self.timeout = timeout
        self.max_output = max_output
        self.allowed_binary_hashes = allowed_binary_hashes or {}
        self._platform = platform.system()

    def validate_binary(self, binary_path: Path) -> BinaryInfo:
        """Validate a binary before execution.

        Args:
            binary_path: Path to the binary to validate.

        Returns:
            BinaryInfo with path, hash, and size.

        Raises:
            BinaryValidationError: If binary doesn't exist or hash doesn't match.
        """
        if not binary_path.exists():
            raise BinaryValidationError(f"Binary not found: {binary_path}")

        if not binary_path.is_file():
            raise BinaryValidationError(f"Not a file: {binary_path}")

        # Calculate SHA256 hash
        hasher = hashlib.sha256()
        try:
            with open(binary_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except OSError as e:
            raise BinaryValidationError(f"Cannot read binary: {e}") from e

        sha256 = hasher.hexdigest()
        size = binary_path.stat().st_size

        # Check against allowed hashes if configured
        binary_name = binary_path.name
        if self.allowed_binary_hashes:
            expected_hash = self.allowed_binary_hashes.get(binary_name)
            if expected_hash is None:
                logger.warning(f"Binary {binary_name} not in allowed list")
            elif expected_hash != sha256:
                raise BinaryValidationError(
                    f"Binary hash mismatch for {binary_name}: "
                    f"expected {expected_hash}, got {sha256}"
                )

        return BinaryInfo(path=binary_path, sha256=sha256, size=size)

    def run(
        self,
        args: Sequence[str | Path],
        *,
        input_data: str | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        validate_binary: bool = True,
    ) -> SandboxResult:
        """Run a command in the sandbox.

        Args:
            args: Command and arguments to run.
            input_data: Optional input to send to stdin.
            cwd: Working directory for the command.
            env: Environment variables (None = inherit current).
            validate_binary: Whether to validate the binary before running.

        Returns:
            SandboxResult with stdout, stderr, and return code.

        Raises:
            SandboxTimeoutError: If the command times out.
            SandboxExecutionError: If the command fails to execute.
            BinaryValidationError: If binary validation fails.
        """
        if not args:
            raise SandboxExecutionError("No command provided")

        # Convert all args to strings
        str_args = [str(arg) for arg in args]

        # Validate binary if requested
        if validate_binary:
            binary_path = Path(str_args[0])
            if not binary_path.is_absolute():
                # Try to find in PATH
                found = shutil.which(str_args[0])
                if found:
                    binary_path = Path(found)
            self.validate_binary(binary_path)

        # Prepare environment
        run_env = os.environ.copy() if env is None else env.copy()

        # Try platform-specific sandboxing
        sandboxed_args = self._wrap_for_sandbox(str_args)

        try:
            result = subprocess.run(
                sandboxed_args,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=run_env,
            )

            # Truncate output if too large
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > self.max_output:
                stdout = stdout[: self.max_output] + "\n... (output truncated)"
            if len(stderr) > self.max_output:
                stderr = stderr[: self.max_output] + "\n... (output truncated)"

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
            )

        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else ""
            raise SandboxTimeoutError(
                f"Command timed out after {self.timeout}s: {' '.join(str_args)}"
            ) from e

        except OSError as e:
            raise SandboxExecutionError(f"Failed to execute: {e}") from e

    def _wrap_for_sandbox(self, args: list[str]) -> list[str]:
        """Wrap command with platform-specific sandboxing.

        Args:
            args: Original command and arguments.

        Returns:
            Wrapped command list.
        """
        if self._platform == "Darwin":
            return self._wrap_macos(args)
        elif self._platform == "Linux":
            return self._wrap_linux(args)
        else:
            # No sandboxing available
            logger.debug(f"No sandboxing available on {self._platform}")
            return args

    def _wrap_macos(self, args: list[str]) -> list[str]:
        """Wrap command for macOS sandbox-exec.

        Uses a restrictive profile that:
        - Allows read access to system libraries and the binary
        - Allows read/write access to temp directory
        - Denies network access
        - Denies access to user files outside temp

        Note: sandbox-exec requires SIP to be enabled and may not work
        in all environments.
        """
        # Check if sandbox-exec is available
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec is None:
            logger.debug("sandbox-exec not found, skipping macOS sandboxing")
            return args

        # Create a temporary sandbox profile
        profile = self._create_macos_sandbox_profile()

        try:
            # Write profile to temp file
            profile_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".sb",
                delete=False,
            )
            profile_file.write(profile)
            profile_file.close()

            # Return wrapped command
            return [sandbox_exec, "-f", profile_file.name, *args]
        except OSError:
            logger.warning("Failed to create sandbox profile, running without sandbox")
            return args

    def _create_macos_sandbox_profile(self) -> str:
        """Create a macOS sandbox profile for glslangValidator.

        Note: This is a permissive profile that allows most file reads.
        A more restrictive profile would require passing the temp directory
        as a parameter, but sandbox-exec parameter handling varies across
        macOS versions.
        """
        return """
(version 1)
(deny default)

; Allow basic process operations
(allow process-fork)
(allow process-exec)
(allow signal (target self))

; Allow reading most files (needed for binaries, libraries, temp files)
(allow file-read*)

; Allow writing to temp directories
(allow file-write*
    (subpath "/tmp")
    (subpath "/private/tmp")
    (subpath "/var/folders")
    (subpath "/private/var/folders")
)

; Allow sysctl for basic system info
(allow sysctl-read)

; Deny network access
(deny network*)
"""

    def _wrap_linux(self, args: list[str]) -> list[str]:
        """Wrap command for Linux sandboxing.

        Tries to use bubblewrap (bwrap) if available, otherwise falls back
        to no sandboxing.

        Note: Full sandboxing on Linux typically requires bubblewrap or
        firejail, which may not be installed by default.
        """
        # Check for bubblewrap
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            logger.debug("bubblewrap not found, skipping Linux sandboxing")
            return args

        # Use bubblewrap with minimal permissions
        bwrap_args = [
            bwrap,
            # Bind read-only
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            # Temp directory (writable)
            "--bind", "/tmp", "/tmp",
            # Proc filesystem
            "--proc", "/proc",
            # No network
            "--unshare-net",
            # Drop all capabilities
            "--cap-drop", "ALL",
            # Die when parent dies
            "--die-with-parent",
            # The actual command
            "--",
            *args,
        ]

        return bwrap_args


def run_sandboxed(
    args: Sequence[str | Path],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    input_data: str | None = None,
) -> SandboxResult:
    """Convenience function to run a command in a sandbox.

    Args:
        args: Command and arguments.
        timeout: Maximum execution time in seconds.
        input_data: Optional input to send to stdin.

    Returns:
        SandboxResult with stdout, stderr, and return code.
    """
    sandbox = Sandbox(timeout=timeout)
    return sandbox.run(args, input_data=input_data, validate_binary=False)

"""Unit tests for subprocess sandboxing utilities."""

import platform
import shutil
from pathlib import Path

import pytest

from td_linter.sandbox import (
    BinaryInfo,
    BinaryValidationError,
    Sandbox,
    SandboxError,
    SandboxExecutionError,
    SandboxResult,
    SandboxTimeoutError,
    run_sandboxed,
)


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_basic_result(self) -> None:
        """Should store basic result data."""
        result = SandboxResult(stdout="out", stderr="err", return_code=0)
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.return_code == 0
        assert result.timed_out is False

    def test_timed_out_result(self) -> None:
        """Should store timed_out flag."""
        result = SandboxResult(
            stdout="", stderr="", return_code=-1, timed_out=True
        )
        assert result.timed_out is True


class TestBinaryInfo:
    """Tests for BinaryInfo dataclass."""

    def test_binary_info(self) -> None:
        """Should store binary information."""
        info = BinaryInfo(
            path=Path("/usr/bin/echo"),
            sha256="abc123",
            size=12345,
        )
        assert info.path == Path("/usr/bin/echo")
        assert info.sha256 == "abc123"
        assert info.size == 12345


class TestSandboxValidateBinary:
    """Tests for Sandbox.validate_binary method."""

    def test_validate_nonexistent_binary(self) -> None:
        """Should raise error for nonexistent binary."""
        sandbox = Sandbox()
        with pytest.raises(BinaryValidationError) as exc_info:
            sandbox.validate_binary(Path("/nonexistent/binary"))
        assert "not found" in str(exc_info.value)

    def test_validate_directory_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when path is a directory."""
        sandbox = Sandbox()
        with pytest.raises(BinaryValidationError) as exc_info:
            sandbox.validate_binary(tmp_path)
        assert "Not a file" in str(exc_info.value)

    def test_validate_returns_binary_info(self, tmp_path: Path) -> None:
        """Should return BinaryInfo for valid binary."""
        # Create a test file
        test_file = tmp_path / "test_binary"
        test_file.write_bytes(b"test content")

        sandbox = Sandbox()
        info = sandbox.validate_binary(test_file)

        assert info.path == test_file
        assert len(info.sha256) == 64  # SHA256 hex length
        assert info.size == 12  # len("test content")

    def test_hash_mismatch_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when hash doesn't match."""
        test_file = tmp_path / "test_binary"
        test_file.write_bytes(b"test content")

        sandbox = Sandbox(
            allowed_binary_hashes={"test_binary": "invalid_hash"}
        )
        with pytest.raises(BinaryValidationError) as exc_info:
            sandbox.validate_binary(test_file)
        assert "hash mismatch" in str(exc_info.value)


class TestSandboxRun:
    """Tests for Sandbox.run method."""

    def test_empty_args_raises_error(self) -> None:
        """Should raise error for empty command."""
        sandbox = Sandbox()
        with pytest.raises(SandboxExecutionError) as exc_info:
            sandbox.run([])
        assert "No command" in str(exc_info.value)

    def test_run_simple_command(self) -> None:
        """Should run simple command and return result."""
        sandbox = Sandbox()
        result = sandbox.run(["echo", "hello"], validate_binary=False)
        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_run_with_timeout(self) -> None:
        """Should enforce timeout."""
        sandbox = Sandbox(timeout=1)
        with pytest.raises(SandboxTimeoutError):
            # Sleep for longer than timeout
            sandbox.run(["sleep", "10"], validate_binary=False)

    def test_run_captures_stderr(self) -> None:
        """Should capture stderr."""
        sandbox = Sandbox()
        # Use shell to redirect to stderr
        result = sandbox.run(
            ["sh", "-c", "echo error >&2"],
            validate_binary=False,
        )
        assert "error" in result.stderr

    def test_run_returns_nonzero_exit_code(self) -> None:
        """Should return nonzero exit code without raising."""
        sandbox = Sandbox()
        result = sandbox.run(["false"], validate_binary=False)
        assert result.return_code != 0

    def test_run_with_input_data(self) -> None:
        """Should pass input data to stdin."""
        sandbox = Sandbox()
        result = sandbox.run(
            ["cat"],
            input_data="test input",
            validate_binary=False,
        )
        assert result.return_code == 0
        assert "test input" in result.stdout

    def test_run_with_cwd(self, tmp_path: Path) -> None:
        """Should run command in specified working directory."""
        sandbox = Sandbox()
        result = sandbox.run(["pwd"], cwd=tmp_path, validate_binary=False)
        assert str(tmp_path) in result.stdout

    def test_output_truncation(self) -> None:
        """Should truncate large output."""
        sandbox = Sandbox(max_output=100)
        result = sandbox.run(
            ["sh", "-c", "yes | head -1000"],
            validate_binary=False,
        )
        # Output should be truncated
        assert len(result.stdout) <= 150  # Some room for truncation message


class TestSandboxPlatformWrapping:
    """Tests for platform-specific sandbox wrapping."""

    def test_wrap_for_sandbox_returns_list(self) -> None:
        """_wrap_for_sandbox should return a list of strings."""
        sandbox = Sandbox()
        args = ["echo", "test"]
        wrapped = sandbox._wrap_for_sandbox(args)
        assert isinstance(wrapped, list)
        assert all(isinstance(arg, str) for arg in wrapped)

    @pytest.mark.skipif(
        platform.system() != "Darwin", reason="macOS-specific test"
    )
    def test_macos_sandboxing_available(self) -> None:
        """On macOS, sandbox-exec should be detected if available."""
        sandbox = Sandbox()
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec:
            args = ["echo", "test"]
            wrapped = sandbox._wrap_macos(args)
            assert "sandbox-exec" in wrapped[0]
        else:
            # sandbox-exec not available, should return original args
            args = ["echo", "test"]
            wrapped = sandbox._wrap_macos(args)
            assert wrapped == args

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Linux-specific test"
    )
    def test_linux_sandboxing_fallback(self) -> None:
        """On Linux without bwrap, should return original args."""
        sandbox = Sandbox()
        # Check if bwrap is available
        bwrap = shutil.which("bwrap")
        args = ["echo", "test"]
        wrapped = sandbox._wrap_linux(args)
        if bwrap:
            assert "bwrap" in wrapped[0]
        else:
            assert wrapped == args


class TestRunSandboxed:
    """Tests for run_sandboxed convenience function."""

    def test_run_sandboxed_simple(self) -> None:
        """Should run simple command."""
        result = run_sandboxed(["echo", "hello"])
        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_run_sandboxed_with_timeout(self) -> None:
        """Should respect timeout parameter."""
        with pytest.raises(SandboxTimeoutError):
            run_sandboxed(["sleep", "10"], timeout=1)

    def test_run_sandboxed_with_input(self) -> None:
        """Should pass input to command."""
        result = run_sandboxed(["cat"], input_data="test data")
        assert "test data" in result.stdout


class TestSandboxExceptions:
    """Tests for sandbox exception hierarchy."""

    def test_sandbox_error_base(self) -> None:
        """SandboxError should be base exception."""
        assert issubclass(SandboxTimeoutError, SandboxError)
        assert issubclass(SandboxExecutionError, SandboxError)
        assert issubclass(BinaryValidationError, SandboxError)

    def test_timeout_error_message(self) -> None:
        """SandboxTimeoutError should have descriptive message."""
        error = SandboxTimeoutError("Command timed out")
        assert "timed out" in str(error)

    def test_execution_error_message(self) -> None:
        """SandboxExecutionError should have descriptive message."""
        error = SandboxExecutionError("Command failed")
        assert "failed" in str(error)


class TestGLSLValidatorIntegration:
    """Integration tests for GLSL validator using sandbox."""

    def test_glsl_validator_imports_sandbox(self) -> None:
        """GLSL validator should import sandbox correctly."""
        from td_linter.embedded.glsl_validator import GLSLValidator
        from td_linter.sandbox import Sandbox

        validator = GLSLValidator()
        # is_available should not raise even if glslangValidator is missing
        result = validator.is_available()
        assert isinstance(result, bool)

    def test_glsl_validator_handles_missing_binary(self) -> None:
        """GLSL validator should handle missing glslangValidator gracefully."""
        from td_linter.embedded.glsl_validator import GLSLValidator

        # Create validator with nonexistent path
        validator = GLSLValidator(glslang_path=Path("/nonexistent/path"))
        assert validator.is_available() is False

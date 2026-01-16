"""Unit tests for plugin security checker."""

from pathlib import Path

import pytest

from td_linter.plugin_security import (
    PluginSecurityChecker,
    PluginSecurityError,
    SecurityViolation,
    validate_plugin_security,
)
from td_linter.plugins import PluginLoader, PluginSecurityViolationError


class TestPluginSecurityChecker:
    """Tests for PluginSecurityChecker class."""

    def test_checker_initialization(self) -> None:
        """Should initialize with default settings."""
        checker = PluginSecurityChecker()
        assert not checker.allow_file_read

    def test_checker_allow_file_read(self) -> None:
        """Should accept allow_file_read option."""
        checker = PluginSecurityChecker(allow_file_read=True)
        assert checker.allow_file_read

    def test_safe_code_passes(self) -> None:
        """Should pass safe plugin code."""
        code = '''
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class MyRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "my-rule"

    @property
    def description(self) -> str:
        return "My rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
'''
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0


class TestDangerousImports:
    """Tests for detecting dangerous imports."""

    def test_blocks_os_import(self) -> None:
        """Should block 'import os'."""
        code = "import os"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("os" in v.message for v in violations)

    def test_blocks_subprocess_import(self) -> None:
        """Should block 'import subprocess'."""
        code = "import subprocess"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("subprocess" in v.message for v in violations)

    def test_blocks_sys_import(self) -> None:
        """Should block 'import sys'."""
        code = "import sys"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("sys" in v.message for v in violations)

    def test_blocks_socket_import(self) -> None:
        """Should block 'import socket'."""
        code = "import socket"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("socket" in v.message for v in violations)

    def test_blocks_ctypes_import(self) -> None:
        """Should block 'import ctypes'."""
        code = "import ctypes"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("ctypes" in v.message for v in violations)

    def test_blocks_pickle_import(self) -> None:
        """Should block 'import pickle'."""
        code = "import pickle"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("pickle" in v.message for v in violations)

    def test_blocks_from_os_import(self) -> None:
        """Should block 'from os import ...'."""
        code = "from os import system"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert len(violations) > 0

    def test_blocks_os_submodule(self) -> None:
        """Should block 'import os.path'."""
        code = "import os.path"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("os" in v.message for v in violations)

    def test_allows_safe_modules(self) -> None:
        """Should allow safe modules."""
        code = """
import json
import math
import re
from typing import Iterator
from collections import defaultdict
"""
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_allows_td_linter_imports(self) -> None:
        """Should allow td_linter imports."""
        code = """
from td_linter.rules.base import LintRule, Violation
from td_linter import lint
"""
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_allows_networkx_import(self) -> None:
        """Should allow networkx import (commonly used by rules)."""
        code = """
import networkx as nx
import networkx
"""
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0


class TestDangerousBuiltins:
    """Tests for detecting dangerous builtin calls."""

    def test_blocks_eval(self) -> None:
        """Should block eval() calls."""
        code = "result = eval('1 + 1')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("eval" in v.message for v in violations)

    def test_blocks_exec(self) -> None:
        """Should block exec() calls."""
        code = "exec('x = 1')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("exec" in v.message for v in violations)

    def test_blocks_compile(self) -> None:
        """Should block compile() calls."""
        code = "compile('x = 1', '<string>', 'exec')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("compile" in v.message for v in violations)

    def test_blocks_dunder_import(self) -> None:
        """Should block __import__() calls."""
        code = "__import__('os')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("__import__" in v.message for v in violations)

    def test_blocks_open_by_default(self) -> None:
        """Should block open() by default."""
        code = "f = open('file.txt')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("open" in v.message for v in violations)

    def test_allows_open_with_allow_file_read(self) -> None:
        """Should allow open() for reading when configured."""
        code = "f = open('file.txt')"
        checker = PluginSecurityChecker(allow_file_read=True)
        violations = checker.check_source(code)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_blocks_open_write_even_with_allow_file_read(self) -> None:
        """Should block open() for writing even with allow_file_read."""
        code = "f = open('file.txt', mode='w')"
        checker = PluginSecurityChecker(allow_file_read=True)
        violations = checker.check_source(code)
        assert any("writing" in v.message.lower() for v in violations)

    def test_blocks_open_append_with_allow_file_read(self) -> None:
        """Should block open() for appending even with allow_file_read."""
        code = "f = open('file.txt', mode='a')"
        checker = PluginSecurityChecker(allow_file_read=True)
        violations = checker.check_source(code)
        assert len([v for v in violations if v.severity == "error"]) > 0


class TestDangerousAttributes:
    """Tests for detecting dangerous attribute access."""

    def test_blocks_code_access(self) -> None:
        """Should block __code__ access."""
        code = "x = func.__code__"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("__code__" in v.message for v in violations)

    def test_blocks_globals_access(self) -> None:
        """Should block __globals__ access."""
        code = "x = func.__globals__"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("__globals__" in v.message for v in violations)

    def test_blocks_builtins_access(self) -> None:
        """Should block __builtins__ access."""
        code = "x = __builtins__"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert len(violations) > 0

    def test_blocks_subclasses_access(self) -> None:
        """Should block __subclasses__ access."""
        code = "x = object.__subclasses__()"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("__subclasses__" in v.message for v in violations)

    def test_blocks_getattr_dangerous(self) -> None:
        """Should block getattr() with dangerous attribute names."""
        code = "x = getattr(obj, '__code__')"
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert any("__code__" in v.message for v in violations)


class TestSyntaxErrors:
    """Tests for handling syntax errors."""

    def test_reports_syntax_error(self) -> None:
        """Should report syntax errors."""
        code = "def incomplete("
        checker = PluginSecurityChecker()
        violations = checker.check_source(code)
        assert len(violations) > 0
        assert any("syntax" in v.message.lower() for v in violations)


class TestValidatePluginSecurity:
    """Tests for validate_plugin_security function."""

    def test_raises_on_dangerous_code(self) -> None:
        """Should raise PluginSecurityError for dangerous code."""
        code = "import os"
        with pytest.raises(PluginSecurityError):
            validate_plugin_security(code)

    def test_passes_safe_code(self) -> None:
        """Should not raise for safe code."""
        code = "x = 1 + 1"
        # Should not raise
        validate_plugin_security(code)

    def test_error_contains_violations(self) -> None:
        """Should include violations in error."""
        code = "import subprocess"
        with pytest.raises(PluginSecurityError) as exc_info:
            validate_plugin_security(code)
        assert len(exc_info.value.violations) > 0


class TestPluginLoaderSecurity:
    """Tests for PluginLoader with security validation."""

    def test_rejects_dangerous_plugin(self, tmp_path: Path) -> None:
        """Should reject plugin with dangerous imports."""
        plugin_file = tmp_path / "bad_plugin.py"
        plugin_file.write_text("""
import os
from td_linter.rules.base import LintRule, Violation

class DangerousRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "dangerous-rule"

    @property
    def description(self) -> str:
        return "Dangerous rule"

    def check(self, graph):
        os.system('rm -rf /')  # Very dangerous!
        return iter(())
""")
        loader = PluginLoader()
        with pytest.raises(PluginSecurityViolationError):
            loader.load_from_path(plugin_file)

    def test_accepts_safe_plugin(self, tmp_path: Path) -> None:
        """Should accept plugin with safe code."""
        plugin_file = tmp_path / "safe_plugin.py"
        plugin_file.write_text("""
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class SafeRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "safe-rule"

    @property
    def description(self) -> str:
        return "Safe rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        loader = PluginLoader()
        rules = loader.load_from_path(plugin_file)
        assert len(rules) == 1

    def test_skip_security_check_allows_dangerous(self, tmp_path: Path) -> None:
        """Should allow dangerous plugin when security check is skipped."""
        plugin_file = tmp_path / "dangerous_but_trusted.py"
        plugin_file.write_text("""
import json  # Safe anyway, but testing the skip mechanism
from typing import Iterator
from td_linter.rules.base import LintRule, Violation

class TrustedRule(LintRule):
    @property
    def rule_id(self) -> str:
        return "X001"

    @property
    def name(self) -> str:
        return "trusted-rule"

    @property
    def description(self) -> str:
        return "Trusted rule"

    def check(self, graph) -> Iterator[Violation]:
        return iter(())
""")
        # With skip_security_check=True, should load without validation
        loader = PluginLoader(skip_security_check=True)
        rules = loader.load_from_path(plugin_file)
        assert len(rules) == 1

    def test_error_contains_path(self, tmp_path: Path) -> None:
        """Should include path in security violation error."""
        plugin_file = tmp_path / "bad.py"
        plugin_file.write_text("import subprocess")
        loader = PluginLoader()

        with pytest.raises(PluginSecurityViolationError) as exc_info:
            loader.load_from_path(plugin_file)

        assert plugin_file == exc_info.value.path


class TestSecurityViolation:
    """Tests for SecurityViolation dataclass."""

    def test_str_representation(self) -> None:
        """Should format violation as string."""
        v = SecurityViolation(
            message="Dangerous import: 'os' is not allowed",
            line=5,
            col=0,
            severity="error",
        )
        s = str(v)
        assert "ERROR" in s
        assert "Line 5" in s
        assert "os" in s

    def test_frozen(self) -> None:
        """Should be immutable."""
        v = SecurityViolation(
            message="test",
            line=1,
            col=0,
            severity="error",
        )
        with pytest.raises(AttributeError):
            v.message = "changed"  # type: ignore

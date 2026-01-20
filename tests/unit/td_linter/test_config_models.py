"""Unit tests for Pydantic configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from td_linter.config_models import (
    ConfigFileSizeError,
    LintConfigModel,
    PluginModuleConfig,
    PluginPathConfig,
    RuleConfigModel,
    Severity,
    validate_config_file_size,
)
from td_linter.rules.loader import ConfigLoader, ConfigValidationError


class TestSeverityEnum:
    """Tests for Severity enum."""

    def test_severity_values(self) -> None:
        """Should have expected values."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


class TestRuleConfigModel:
    """Tests for RuleConfigModel."""

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        config = RuleConfigModel()
        assert config.enabled is True
        assert config.severity == Severity.ERROR
        assert config.options == {}

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = RuleConfigModel(
            enabled=False,
            severity=Severity.WARNING,
            options={"max_depth": 10},
        )
        assert config.enabled is False
        assert config.severity == Severity.WARNING
        assert config.options == {"max_depth": 10}

    def test_rejects_extra_fields(self) -> None:
        """Should reject extra fields."""
        with pytest.raises(ValidationError):
            RuleConfigModel(unknown_field="value")  # type: ignore


class TestPluginPathConfig:
    """Tests for PluginPathConfig."""

    def test_valid_path(self) -> None:
        """Should accept .py file paths."""
        config = PluginPathConfig(path="/path/to/rules.py")
        assert config.path == "/path/to/rules.py"

    def test_rejects_non_py_extension(self) -> None:
        """Should reject non-.py files."""
        with pytest.raises(ValidationError) as exc_info:
            PluginPathConfig(path="/path/to/rules.txt")
        assert ".py" in str(exc_info.value)


class TestPluginModuleConfig:
    """Tests for PluginModuleConfig."""

    def test_valid_module_name(self) -> None:
        """Should accept valid module names."""
        config = PluginModuleConfig(module="my_company.td_rules")
        assert config.module == "my_company.td_rules"

    def test_simple_module_name(self) -> None:
        """Should accept simple module names."""
        config = PluginModuleConfig(module="my_rules")
        assert config.module == "my_rules"

    def test_rejects_invalid_module_name(self) -> None:
        """Should reject invalid module names."""
        with pytest.raises(ValidationError):
            PluginModuleConfig(module="invalid-module-name")

        with pytest.raises(ValidationError):
            PluginModuleConfig(module="")


class TestLintConfigModel:
    """Tests for LintConfigModel."""

    def test_minimal_config(self) -> None:
        """Should accept empty config."""
        config = LintConfigModel()
        assert config.version == "1.0.0"
        assert config.extends is None
        assert config.rules == {}

    def test_full_config(self) -> None:
        """Should accept full configuration."""
        config = LintConfigModel(
            version="2.0.0",
            extends="recommended",
            select=["S", "C001"],
            ignore=["F"],
            rules={
                "S001": RuleConfigModel(severity=Severity.WARNING),
            },
            plugins=[{"path": "/rules.py"}],
        )
        assert config.version == "2.0.0"
        assert config.extends == "recommended"
        assert config.select == ["S", "C001"]
        assert config.ignore == ["F"]
        assert "S001" in config.rules

    def test_rejects_invalid_version(self) -> None:
        """Should reject invalid version format."""
        with pytest.raises(ValidationError):
            LintConfigModel(version="1.0")  # Missing patch

        with pytest.raises(ValidationError):
            LintConfigModel(version="v1.0.0")  # Has prefix

    def test_rejects_invalid_preset(self) -> None:
        """Should reject invalid preset names."""
        with pytest.raises(ValidationError):
            LintConfigModel(extends="unknown_preset")  # type: ignore

    def test_rejects_invalid_select_pattern(self) -> None:
        """Should reject invalid select patterns."""
        with pytest.raises(ValidationError) as exc_info:
            LintConfigModel(select=["X001"])  # Invalid category
        assert "Invalid pattern" in str(exc_info.value)

    def test_rejects_invalid_rule_id(self) -> None:
        """Should reject invalid rule IDs in rules dict."""
        with pytest.raises(ValidationError) as exc_info:
            LintConfigModel(rules={"invalid_id": RuleConfigModel()})
        assert "Invalid rule ID" in str(exc_info.value)

    def test_rejects_extra_fields(self) -> None:
        """Should reject extra fields."""
        with pytest.raises(ValidationError):
            LintConfigModel(unknown_option="value")  # type: ignore

    def test_accepts_multiple_presets(self) -> None:
        """Should accept list of presets."""
        config = LintConfigModel(extends=["recommended", "strict"])
        assert config.extends == ["recommended", "strict"]

    def test_validates_plugin_config(self) -> None:
        """Should validate plugin configurations."""
        # Valid path plugin
        config = LintConfigModel(plugins=[{"path": "/my/rules.py"}])
        assert len(config.plugins) == 1

        # Valid module plugin
        config = LintConfigModel(plugins=[{"module": "my_rules"}])
        assert len(config.plugins) == 1

        # Invalid: missing path and module
        with pytest.raises(ValidationError):
            LintConfigModel(plugins=[{}])

        # Invalid: both path and module
        with pytest.raises(ValidationError):
            LintConfigModel(plugins=[{"path": "/rules.py", "module": "rules"}])

        # Invalid: path without .py
        with pytest.raises(ValidationError):
            LintConfigModel(plugins=[{"path": "/rules.txt"}])


class TestConfigFileSizeValidation:
    """Tests for config file size validation."""

    def test_accepts_small_file(self, tmp_path: Path) -> None:
        """Should accept files under size limit."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text("version: 1.0.0\n")
        # Should not raise
        validate_config_file_size(config_file)

    def test_rejects_large_file(self, tmp_path: Path) -> None:
        """Should reject files over size limit."""
        config_file = tmp_path / "td-linter.yaml"
        # Write more than 1MB
        config_file.write_text("x" * (1024 * 1024 + 1))

        with pytest.raises(ConfigFileSizeError) as exc_info:
            validate_config_file_size(config_file)

        assert exc_info.value.size > exc_info.value.max_size


class TestConfigLoaderWithPydantic:
    """Tests for ConfigLoader with Pydantic validation."""

    def test_rejects_unknown_fields(self, tmp_path: Path) -> None:
        """Should reject config with unknown fields."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text(
            """
version: "1.0.0"
unknown_field: value
"""
        )
        loader = ConfigLoader()

        with pytest.raises(ConfigValidationError) as exc_info:
            loader.load(config_file)

        assert "unknown_field" in str(exc_info.value).lower()

    def test_rejects_wrong_type(self, tmp_path: Path) -> None:
        """Should reject config with wrong types."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text(
            """
version: 123
"""
        )
        loader = ConfigLoader()

        with pytest.raises(ConfigValidationError):
            loader.load(config_file)

    def test_rejects_invalid_severity(self, tmp_path: Path) -> None:
        """Should reject invalid severity values."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text(
            """
version: "1.0.0"
rules:
  S001:
    severity: critical
"""
        )
        loader = ConfigLoader()

        with pytest.raises(ConfigValidationError):
            loader.load(config_file)

    def test_accepts_valid_config(self, tmp_path: Path) -> None:
        """Should accept valid configuration."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text(
            """
version: "1.0.0"
extends: recommended
rules:
  S001:
    enabled: false
    severity: warning
"""
        )
        loader = ConfigLoader()
        config = loader.load(config_file)

        assert config.version == "1.0.0"

    def test_size_limit_enforced(self, tmp_path: Path) -> None:
        """Should reject files over size limit."""
        config_file = tmp_path / "td-linter.yaml"
        config_file.write_text("x" * (1024 * 1024 + 1))

        loader = ConfigLoader()

        with pytest.raises(ConfigValidationError) as exc_info:
            loader.load(config_file)

        assert "exceeds maximum" in str(exc_info.value)

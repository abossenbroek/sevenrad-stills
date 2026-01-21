"""
Unit tests for rule configuration loader (TDL-041).

These tests verify that the config loader correctly:
- Loads configuration from YAML files
- Resolves preset configurations
- Handles select/ignore patterns
- Merges configurations properly
"""

from pathlib import Path

import pytest
import yaml
from td_linter.rules.loader import (
    ConfigLoader,
    ConfigValidationError,
    LintConfig,
    RuleConfig,
    load_config,
)


class TestPresetLoading:
    """Tests for loading preset configurations."""

    def test_load_recommended_preset(self) -> None:
        """Recommended preset should be loadable."""
        loader = ConfigLoader()
        config = loader._load_preset("recommended")

        assert isinstance(config, LintConfig)
        # Check that core rules are configured
        assert "S001" in config.rules
        assert "C001" in config.rules
        assert config.rules["C001"].enabled is True
        assert config.rules["C001"].severity == "error"

    def test_load_strict_preset(self) -> None:
        """Strict preset should enable all rules."""
        loader = ConfigLoader()
        config = loader._load_preset("strict")

        # All rules should be enabled in strict mode
        assert config.rules["G002"].enabled is True
        assert config.rules["F001"].enabled is True

    def test_load_minimal_preset(self) -> None:
        """Minimal preset should only enable critical rules."""
        loader = ConfigLoader()
        config = loader._load_preset("minimal")

        # Critical rules enabled
        assert config.rules["S001"].enabled is True
        assert config.rules["C001"].enabled is True
        # Non-critical rules disabled
        assert config.rules["T001"].enabled is False
        assert config.rules["G001"].enabled is False

    def test_load_pedantic_preset(self) -> None:
        """Pedantic preset should have stricter settings."""
        loader = ConfigLoader()
        config = loader._load_preset("pedantic")

        # All rules enabled
        assert config.rules["F001"].enabled is True
        # Stricter thresholds
        assert config.rules["F001"].options.get("max_depth") == 5
        # Upgraded severities
        assert config.rules["C002"].severity == "error"

    def test_unknown_preset_raises_error(self) -> None:
        """Loading unknown preset should raise ConfigValidationError."""
        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError, match="Unknown preset"):
            loader._load_preset("unknown")


class TestConfigFromFile:
    """Tests for loading configuration from YAML files."""

    def test_load_simple_config(self, tmp_path: Path) -> None:
        """Simple config file should load correctly."""
        config_content = {
            "version": "1.0.0",
            "extends": "recommended",
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        assert config.version == "1.0.0"
        # Should have rules from recommended preset
        assert "S001" in config.rules

    def test_load_config_with_rule_overrides(self, tmp_path: Path) -> None:
        """Rule overrides should be applied on top of preset."""
        config_content = {
            "version": "1.0.0",
            "extends": "recommended",
            "rules": {
                "C001": {"severity": "warning"},
                "F001": {"enabled": True, "options": {"max_depth": 20}},
            },
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # Override should apply
        assert config.rules["C001"].severity == "warning"
        # Options should be set
        assert config.rules["F001"].enabled is True
        assert config.rules["F001"].options["max_depth"] == 20

    def test_load_config_with_multiple_presets(self, tmp_path: Path) -> None:
        """Multiple presets should be merged in order."""
        config_content = {
            "version": "1.0.0",
            "extends": ["minimal", "recommended"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # recommended (later) should override minimal
        assert config.rules["G001"].enabled is True


class TestSelectIgnore:
    """Tests for select/ignore pattern handling."""

    def test_select_category(self, tmp_path: Path) -> None:
        """Selecting a category should enable only those rules."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "select": ["S"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # S rules should be enabled
        assert config.rules["S001"].enabled is True
        assert config.rules["S002"].enabled is True
        # Other categories disabled
        assert config.rules["C001"].enabled is False
        assert config.rules["G001"].enabled is False

    def test_select_specific_rules(self, tmp_path: Path) -> None:
        """Selecting specific rule IDs should work."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "select": ["S001", "C001"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # Selected rules enabled
        assert config.rules["S001"].enabled is True
        assert config.rules["C001"].enabled is True
        # Other rules disabled
        assert config.rules["S002"].enabled is False
        assert config.rules["G001"].enabled is False

    def test_ignore_category(self, tmp_path: Path) -> None:
        """Ignoring a category should disable those rules."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "ignore": ["F"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # F rules should be disabled
        assert config.rules["F001"].enabled is False
        assert config.rules["F002"].enabled is False
        # Other rules still enabled
        assert config.rules["S001"].enabled is True

    def test_ignore_specific_rules(self, tmp_path: Path) -> None:
        """Ignoring specific rule IDs should work."""
        config_content = {
            "version": "1.0.0",
            "extends": "strict",
            "ignore": ["C002", "G002"],
        }
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        config = loader._load_from_file(config_file)

        # Ignored rules disabled
        assert config.rules["C002"].enabled is False
        assert config.rules["G002"].enabled is False
        # Other rules still enabled
        assert config.rules["C001"].enabled is True
        assert config.rules["G001"].enabled is True


class TestConfigDiscovery:
    """Tests for config file discovery."""

    def test_explicit_config_path(self, tmp_path: Path) -> None:
        """Explicit config path should be used."""
        config_content = {"version": "2.0.0", "extends": "minimal"}
        config_file = tmp_path / "custom.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(config_file)
        assert config.version == "2.0.0"

    def test_default_to_recommended(self) -> None:
        """Without config file, should use recommended preset."""
        # Use a path that doesn't have a config file
        config = load_config(None)

        # Should have recommended preset settings
        assert "S001" in config.rules
        assert config.rules["S001"].enabled is True


class TestLintConfigMethods:
    """Tests for LintConfig helper methods."""

    def test_is_rule_enabled_explicit(self) -> None:
        """is_rule_enabled should return explicit setting."""
        config = LintConfig(
            rules={
                "S001": RuleConfig(enabled=True),
                "S002": RuleConfig(enabled=False),
            }
        )

        assert config.is_rule_enabled("S001") is True
        assert config.is_rule_enabled("S002") is False

    def test_is_rule_enabled_default(self) -> None:
        """is_rule_enabled should default to True for unknown rules."""
        config = LintConfig(rules={})

        assert config.is_rule_enabled("unknown") is True

    def test_get_rule_severity(self) -> None:
        """get_rule_severity should return configured severity."""
        config = LintConfig(
            rules={
                "S001": RuleConfig(severity="error"),
                "S002": RuleConfig(severity="warning"),
            }
        )

        assert config.get_rule_severity("S001") == "error"
        assert config.get_rule_severity("S002") == "warning"
        assert config.get_rule_severity("unknown") == "error"  # default

    def test_get_rule_options(self) -> None:
        """get_rule_options should return configured options."""
        config = LintConfig(
            rules={
                "F001": RuleConfig(options={"max_depth": 15}),
            }
        )

        assert config.get_rule_options("F001") == {"max_depth": 15}
        assert config.get_rule_options("unknown") == {}


class TestInvalidConfig:
    """Tests for invalid configuration handling."""

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Invalid YAML should raise error."""
        import yaml

        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content:")

        loader = ConfigLoader()
        with pytest.raises(yaml.scanner.ScannerError):
            loader._load_from_file(config_file)

    def test_invalid_schema_raises_error(self, tmp_path: Path) -> None:
        """Config violating schema should raise ConfigValidationError."""
        config_content = {"extends": "nonexistent_preset"}
        config_file = tmp_path / "td-linter.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loader = ConfigLoader()
        with pytest.raises(ConfigValidationError):
            loader._load_from_file(config_file)
